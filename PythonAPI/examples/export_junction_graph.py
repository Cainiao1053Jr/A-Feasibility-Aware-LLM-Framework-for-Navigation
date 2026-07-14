#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_junction_graph.py

将 CARLA 地图导出为 GeoJSON，仅保留：
  - 每个交叉路口一个代表点（Point）：该路口内所有 waypoint 的 XYZ 均值
  - 路口间的连通边（LineString）：从路口出口沿车道走到下一个路口，记录实际路径

用法：
  python export_junction_graph.py [--host HOST] [--port PORT] [--out OUT.geojson]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import carla

# ---------------------------------------------------------------------------
# 节点：每个路口一个代表点
# ---------------------------------------------------------------------------

def build_junction_representatives(
    carla_map: carla.Map,
    sample_step: float = 2.0,
) -> Dict[int, Tuple[float, float, float]]:
    """
    扫全图 waypoint，按 junction.id 分组，取均值作为代表点。
    返回 {junction_id: (x, y, z)}。
    """
    grouped: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)

    for wp in carla_map.generate_waypoints(sample_step):
        if not wp.is_junction:
            continue
        j = wp.get_junction()
        if j is None:
            continue
        loc = wp.transform.location
        grouped[j.id].append((loc.x, loc.y, loc.z))

    reps: Dict[int, Tuple[float, float, float]] = {}
    for jid, pts in grouped.items():
        n = len(pts)
        reps[jid] = (
            sum(p[0] for p in pts) / n,
            sum(p[1] for p in pts) / n,
            sum(p[2] for p in pts) / n,
        )
    return reps


# ---------------------------------------------------------------------------
# 边：拓扑路段的起点在某路口内，终点沿车道走到下一个路口
# ---------------------------------------------------------------------------

def _walk_to_next_junction(
    start_wp: carla.Waypoint,
    junction_reps: Dict[int, Tuple[float, float, float]],
    step: float,
    max_steps: int,
) -> Tuple[Optional[int], List[Tuple[float, float, float]]]:
    """
    从 start_wp 沿车道向前走，直到进入某个已知路口。
    返回 (junction_id, 折线坐标列表)；若未找到则返回 (None, [])。
    """
    loc0 = start_wp.transform.location
    coords: List[Tuple[float, float, float]] = [(loc0.x, loc0.y, loc0.z)]
    cur = start_wp

    for _ in range(max_steps):
        if cur.is_junction:
            j = cur.get_junction()
            if j and j.id in junction_reps:
                return j.id, coords
            return None, []           # 进入了未知路口，放弃
        nexts = cur.next(step)
        if not nexts:
            return None, []
        cur = nexts[0]
        loc = cur.transform.location
        coords.append((loc.x, loc.y, loc.z))

    return None, []


def build_junction_edges(
    carla_map: carla.Map,
    junction_reps: Dict[int, Tuple[float, float, float]],
    step: float = 2.0,
    max_walk_steps: int = 600,
) -> List[Dict]:
    """
    利用 get_topology() 寻找路口间的连通边。

    策略：
      - 对每条拓扑路段 (wp_a, wp_b)：
          * 若 wp_a 属于某路口（is_junction == True），则该路口是源节点；
          * 从 wp_b 向前走，找到下一个路口作为目标节点；
          * 若源 ≠ 目标且都在已知路口集合中，则记录边。
      - 以 (min_id, max_id) 为键去重，保留路径最长的折线做代表。

    返回 edge 列表，每条边：
      {"from": jid_a, "to": jid_b, "length_m": float, "coords": [(x,y,z),...]}
    """
    merged: Dict[Tuple[int, int], Dict] = {}

    for wp_a, wp_b in carla_map.get_topology():
        # 1. 源路口
        if not wp_a.is_junction:
            continue
        j_src = wp_a.get_junction()
        if j_src is None or j_src.id not in junction_reps:
            continue
        src_id = j_src.id

        # 2. 目标路口：从 wp_b 开始往前走
        #    wp_b 本身已经脱离了路口（若在路口内则直接判断）
        if wp_b.is_junction:
            j_dst = wp_b.get_junction()
            if j_dst is None or j_dst.id not in junction_reps:
                continue
            dst_id = j_dst.id
            loc = wp_b.transform.location
            coords = [(loc.x, loc.y, loc.z)]
        else:
            dst_id, coords = _walk_to_next_junction(
                wp_b, junction_reps, step, max_walk_steps
            )
            if dst_id is None:
                continue

        if src_id == dst_id:
            continue

        # 在折线两端插入路口代表点，使线段真正连接两个路口质心
        sx, sy, sz = junction_reps[src_id]
        dx, dy, dz = junction_reps[dst_id]
        full_coords = [(sx, sy, sz)] + coords + [(dx, dy, dz)]

        length = sum(
            ((full_coords[i][0] - full_coords[i-1][0]) ** 2 +
             (full_coords[i][1] - full_coords[i-1][1]) ** 2) ** 0.5
            for i in range(1, len(full_coords))
        )

        key = (min(src_id, dst_id), max(src_id, dst_id))
        if key not in merged or length > merged[key]["length_m"]:
            merged[key] = {
                "from": src_id,
                "to": dst_id,
                "length_m": length,
                "coords": full_coords,
            }

    return list(merged.values())


# ---------------------------------------------------------------------------
# GeoJSON 导出（节点 + 边 合并为一个 FeatureCollection）
# ---------------------------------------------------------------------------

def export_geojson(
    junction_reps: Dict[int, Tuple[float, float, float]],
    edges: List[Dict],
    out_path: str,
) -> None:
    features = []

    # Point：路口代表点
    for jid, (x, y, z) in sorted(junction_reps.items()):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [x, y, z],
            },
            "properties": {
                "junction_id": jid,
            },
        })

    # LineString：路口间连通边
    for e in edges:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[x, y, z] for x, y, z in e["coords"]],
            },
            "properties": {
                "from_junction": e["from"],
                "to_junction": e["to"],
                "length_m": round(e["length_m"], 2),
            },
        })

    fc = {"type": "FeatureCollection", "features": features}
    Path(out_path).write_text(
        json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[OK] {out_path}  "
        f"nodes={len(junction_reps)}  edges={len(edges)}"
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export CARLA map as GeoJSON: one node per junction + connectivity edges."
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--timeout", type=float, default=10.0, help="Client timeout (s)")
    parser.add_argument(
        "--step", type=float, default=2.0,
        help="Waypoint sampling interval in meters (default: 2.0)"
    )
    parser.add_argument(
        "--max-walk", type=int, default=600,
        help="Max steps when walking from junction exit to next junction (default: 600)"
    )
    parser.add_argument(
        "--out", default="custom_junction_graph.geojson",
        help="Output GeoJSON file path (default: junction_graph.geojson)"
    )
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    carla_map = world.get_map()
    print(f"[*] Map: {carla_map.name}")

    print("[*] Building junction representative points...")
    junction_reps = build_junction_representatives(carla_map, sample_step=args.step)
    print(f"    -> {len(junction_reps)} junctions")

    print("[*] Building junction connectivity edges...")
    edges = build_junction_edges(
        carla_map, junction_reps,
        step=args.step,
        max_walk_steps=args.max_walk,
    )
    print(f"    -> {len(edges)} edges")

    print("[*] Exporting GeoJSON...")
    export_geojson(junction_reps, edges, args.out)


if __name__ == "__main__":
    main()
