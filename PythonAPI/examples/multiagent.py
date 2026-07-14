import carla
import random
import math
import re
import sys
import os
import json
import csv
import threading
from pathlib import Path
from datetime import datetime
from junction_graph_matrix import load_graph

try:
    import pygame
    from pygame.locals import K_x, K_ESCAPE, KMOD_CTRL, K_q, K_w, K_a, K_s, K_d, K_l, K_t, K_1, K_2, K_y, K_u, K_i, K_o, K_h
except ImportError:
    raise RuntimeError('cannot invoke pygame, please install pip install pygame')

try:
    from llmutil import llmutils as _llmutils
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False

try:
    from agent_comm import AgentPool
    _AGENT_COMM_AVAILABLE = True
except ImportError:
    _AGENT_COMM_AVAILABLE = False

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

from carlaLocal.agents.navigation.behavior_agent import BehaviorAgent
from carlaLocal.agents.navigation.global_route_planner import GlobalRoutePlanner
from carlaLocal.agents.navigation.local_planner import RoadOption


def yaw_from_two_points(p0: carla.Location, p1: carla.Location) -> float:
    dx = p1.x - p0.x
    dy = p1.y - p0.y
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def angle_diff_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def move_back_along_yaw(loc: carla.Location, yaw_deg: float, dist: float) -> carla.Location:
    rot = carla.Rotation(pitch=0.0, yaw=yaw_deg, roll=0.0)
    fwd = rot.get_forward_vector()
    return carla.Location(
        x=loc.x - fwd.x * dist,
        y=loc.y - fwd.y * dist,
        z=loc.z - fwd.z * dist,
    )


def snap_to_driving_lane_with_yaw(
        cmap: carla.Map,
        loc: carla.Location,
        desired_yaw_deg: float,
        max_dir_diff_deg: float = 90.0,
) -> carla.Location:
    wp = cmap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        return loc

    yaw_lane = wp.transform.rotation.yaw
    diff = abs(angle_diff_deg(desired_yaw_deg, yaw_lane))

    if diff > max_dir_diff_deg:
        opposite_lane_id = -wp.lane_id
        opp_wp = None
        try:
            opp_wp = cmap.get_waypoint_xodr(wp.road_id, opposite_lane_id, wp.s)
        except RuntimeError:
            opp_wp = None

        if opp_wp is not None:
            yaw_opp = opp_wp.transform.rotation.yaw
            diff_opp = abs(angle_diff_deg(desired_yaw_deg, yaw_opp))
            if diff_opp < diff:
                wp = opp_wp
        else:
            search_dir = yaw_lane - 90
            if search_dir > 180:
                search_dir -= 360
            if search_dir < -180:
                search_dir += 360
            new_loc = move_back_along_yaw(loc, search_dir, -7)
            t_opp_wp = cmap.get_waypoint(
                new_loc, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            if t_opp_wp is not None:
                wp = t_opp_wp

    return wp.transform.location


def grp_trace_points(grp: GlobalRoutePlanner, points: list,
                     forbidden_areas: list = None) -> list:
    if len(points) < 2:
        return []
    import networkx as nx
    use_avoidance = bool(forbidden_areas)
    route = []
    for i in range(len(points) - 1):
        try:
            if use_avoidance:
                seg = grp.trace_route_avoid_areas(points[i], points[i + 1], forbidden_areas)
            else:
                seg = grp.trace_route(points[i], points[i + 1])
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError, KeyError) as e:
            return []
        if i > 0 and seg and route:
            a = seg[0][0]
            b = route[-1][0]
            if a.road_id == b.road_id and a.lane_id == b.lane_id:
                seg = seg[1:]
        route += seg
    return route


def debug_draw_route(world: carla.World, route_wp_ro: list, life_time: float = 15.0):
    dbg = world.debug
    for wp, _ in route_wp_ro:
        dbg.draw_point(
            wp.transform.location + carla.Location(z=0.2),
            size=0.05,
            color=carla.Color(0, 255, 255),
            life_time=life_time,
        )


def route_from_sparse(
        vehicle: carla.Vehicle,
        cmap: carla.Map,
        grp: GlobalRoutePlanner,
        world: carla.World,
        agent: BehaviorAgent,
        anchors: list,
        forbidden_areas: list = None,
        draw: bool = True,
) -> list:
    if not anchors or len(anchors) < 2:
        print('[Route] At least 2 anchor points are required.')
        return []

    init_yaw = vehicle.get_transform().rotation.yaw
    yaws = [init_yaw]
    for i in range(len(anchors) - 1):
        yaws.append(yaw_from_two_points(anchors[i], anchors[i + 1]))

    pts = []
    for yaw_index, p in enumerate(anchors):
        if yaw_index < len(yaws) - 1 and yaw_index != 0:
            back_p = move_back_along_yaw(p, yaws[yaw_index + 1], -17.0)
        elif yaw_index == len(yaws) - 1:
            back_p = move_back_along_yaw(p, yaws[yaw_index], 17.0)
        else:
            back_p = p

        if yaw_index == len(yaws) - 1:
            snapped = snap_to_driving_lane_with_yaw(cmap, back_p, yaws[yaw_index])
        else:
            snapped = snap_to_driving_lane_with_yaw(cmap, back_p, yaws[yaw_index + 1])
        pts.append(snapped)

    dbg = world.debug
    for locp in anchors:
        dbg.draw_point(locp + carla.Location(z=0.2), size=0.05,
                       color=carla.Color(0, 255, 0), life_time=120.0)
    for loc in pts:
        dbg.draw_point(loc + carla.Location(z=0.2), size=0.05,
                       color=carla.Color(255, 0, 0), life_time=120.0)

    if forbidden_areas:
        for fa_loc, fa_radius in forbidden_areas:
            for angle in range(0, 360, 15):
                rad = math.radians(angle)
                p = carla.Location(
                    x=fa_loc.x + fa_radius * math.cos(rad),
                    y=fa_loc.y + fa_radius * math.sin(rad),
                    z=fa_loc.z + 0.2,
                )
                dbg.draw_point(p, size=0.08,
                               color=carla.Color(255, 140, 0), life_time=15.0)

    route_wp_ro = grp_trace_points(grp, pts, forbidden_areas=forbidden_areas)
    if not route_wp_ro:
        return []

    if draw:
        debug_draw_route(world, route_wp_ro, life_time=600.0)

    global_plan = [
        (wp, ro if ro is not None else RoadOption.LANEFOLLOW)
        for wp, ro in route_wp_ro
    ]
    agent.set_global_plan(global_plan)
    print(f'[Route] Route set with {len(global_plan)} waypoints'
          + (f', avoiding {len(forbidden_areas)} forbidden area(s).' if forbidden_areas else '.'))
    return global_plan


def route_from_interpolation(
        vehicle: carla.Vehicle,
        cmap: carla.Map,
        world: carla.World,
        agent: BehaviorAgent,
        anchors: list,
        step_m: float = 2.0,
        draw: bool = True,
        check_continuity: bool = False,
) -> list:
    """
    Generate a route by linear interpolation between sparse anchors output by the
    LLM, without going through GlobalRoutePlanner.

    Points are interpolated evenly at step_m spacing between adjacent anchors;
    each interpolated point is snapped to the nearest driving-lane Waypoint, and
    the result is set as the global_plan of the BehaviorAgent.

    Args:
        anchors           : list[carla.Location], sparse anchor coordinates from the LLM (>= 2)
        step_m            : interpolation step (meters), controls waypoint density, default 2.0
        draw              : whether to render the interpolated waypoints as green dots
        check_continuity  : whether to drop jump points farther than 3 x step_m from
                            the previous waypoint, preventing get_waypoint from
                            snapping to other nearby roads, default True

    Returns:
        list[(carla.Waypoint, RoadOption)], the interpolated route; empty list on failure.
    """
    if not anchors or len(anchors) < 2:
        print('[Interpolation] At least 2 anchor points are required.')
        return []

    global_plan = []
    dbg = world.debug
    continuity_threshold = step_m * 3

    for seg in range(len(anchors) - 1):
        a = anchors[seg]
        b = anchors[seg + 1]
        dx = b.x - a.x
        dy = b.y - a.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-3:
            continue

        n_steps = max(1, int(dist / step_m))
        for i in range(n_steps):
            t = i / n_steps
            loc = carla.Location(
                x=a.x + dx * t,
                y=a.y + dy * t,
                z=0,
            )
            wp = cmap.get_waypoint(loc, project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
            if wp is None:
                continue
            if check_continuity and global_plan:
                prev_wp = global_plan[-1][0]
                if wp.transform.location.distance(prev_wp.transform.location) > continuity_threshold:
                    continue
            global_plan.append((wp, RoadOption.LANEFOLLOW))
            if draw:
                dbg.draw_point(
                    wp.transform.location + carla.Location(z=0.3),
                    size=0.07,
                    color=carla.Color(50, 220, 50),
                    life_time=600.0,
                )

    last_loc = carla.Location(x=anchors[-1].x, y=anchors[-1].y, z=0)
    last_wp = cmap.get_waypoint(last_loc, project_to_road=True,
                                lane_type=carla.LaneType.Driving)
    if last_wp is not None:
        global_plan.append((last_wp, RoadOption.LANEFOLLOW))
        if draw:
            dbg.draw_point(
                last_wp.transform.location + carla.Location(z=0.3),
                size=0.07,
                color=carla.Color(50, 220, 50),
                life_time=600.0,
            )

    if not global_plan:
        print('[Interpolation] No interpolated point could be snapped to a driving lane.')
        return []

    agent.set_global_plan(global_plan)
    print(f'[Interpolation] Route set with {len(global_plan)} interpolated waypoints.')
    return global_plan


def route_from_linear(
        vehicle: carla.Vehicle,
        cmap: carla.Map,
        world: carla.World,
        agent: BehaviorAgent,
        anchors: list,
        step_m: float = 2.0,
        draw: bool = True,
        check_continuity: bool = False,
) -> list:
    """
    Generate a route by linear interpolation between sparse anchors output by the
    LLM, without going through GlobalRoutePlanner.
    Instead of snapping to the nearest road centerline with project_to_road=True, it:
      1. First queries with project_to_road=True using the interpolated point's XY
         coordinates (z=0), only to obtain the road surface height z_road;
      2. Then queries with project_to_road=False at (x, y, z_road), returning a
         Waypoint only if the point itself lies within some lane, without forcing
         the XY onto the nearest road center, avoiding cross-road jumps.

    Args:
        anchors          : list[carla.Location], sparse anchor coordinates from the LLM (>= 2)
        step_m           : interpolation step (meters), controls waypoint density, default 2.0
        draw             : whether to render the interpolated waypoints as green dots
        check_continuity : whether to drop jump points farther than 3 x step_m from
                           the previous waypoint, default True

    Returns:
        list[(carla.Waypoint, RoadOption)], the interpolated route; empty list on failure.
    """
    if not anchors or len(anchors) < 2:
        print('[Linear Interpolation] Requires 2 anchors at least')
        return []

    global_plan = []
    dbg = world.debug
    continuity_threshold = step_m * 3

    def _snap_no_project(x: float, y: float):
        """Get the road surface height from XY, then fetch the Waypoint with project_to_road=False."""
        ref = cmap.get_waypoint(
            carla.Location(x=x, y=y, z=0),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ref is None:
            return None
        return cmap.get_waypoint(
            carla.Location(x=x, y=y, z=ref.transform.location.z),
            project_to_road=False,
            lane_type=carla.LaneType.Driving,
        )

    for seg in range(len(anchors) - 1):
        a = anchors[seg]
        b = anchors[seg + 1]
        dx = b.x - a.x
        dy = b.y - a.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-3:
            continue

        n_steps = max(1, int(dist / step_m))
        for i in range(n_steps):
            t = i / n_steps
            wp = _snap_no_project(a.x + dx * t, a.y + dy * t)
            if wp is None:
                continue
            if check_continuity and global_plan:
                prev_wp = global_plan[-1][0]
                if wp.transform.location.distance(prev_wp.transform.location) > continuity_threshold:
                    continue
            global_plan.append((wp, RoadOption.LANEFOLLOW))
            if draw:
                dbg.draw_point(
                    wp.transform.location + carla.Location(z=0.3),
                    size=0.07,
                    color=carla.Color(50, 220, 50),
                    life_time=600.0,
                )

    last_wp = _snap_no_project(anchors[-1].x, anchors[-1].y)
    if last_wp is not None:
        global_plan.append((last_wp, RoadOption.LANEFOLLOW))
        if draw:
            dbg.draw_point(
                last_wp.transform.location + carla.Location(z=0.3),
                size=0.07,
                color=carla.Color(50, 220, 50),
                life_time=600.0,
            )

    if not global_plan:
        print('[Linear Interpolation] No point available on driving lane')
        return []

    agent.set_global_plan(global_plan)
    print(f'[Linear Interpolation] Route set, with {len(global_plan)} points')
    return global_plan


# ==============================================================================
# -- Path collision check ------------------------------------------------------
# ==============================================================================

def check_path_collisions(
        paths: list,
        min_dist: float = 7.0,
        path_names: list = None,
) -> str:
    """
    Check for collision risk between multiple completed paths.

    Assumes adjacent waypoints on each path take the same amount of time, so the
    same index corresponds to the same moment. For every pair of paths, compares
    positions at the same index; if the distance is below min_dist, a collision
    risk is recorded along with the time step and location.

    Args
    ----
    paths      : list[list[(carla.Waypoint, RoadOption)]]
                 Paths per vehicle, each the return value of a route_from_* function.
    min_dist   : float, collision distance threshold in meters, default 7.0.
    path_names : list[str], readable names of the paths, default ["path_0", "path_1", ...].

    Returns
    -------
    str: multi-line text describing collision risks, ready to be inserted into an
         LLM prompt; empty string if there is no collision.
    """
    n = len(paths)
    if n < 2:
        return ""

    if path_names is None:
        path_names = [f"path_{i}" for i in range(n)]

    collision_blocks = []

    for i in range(n):
        for j in range(i + 1, n):
            pa, pb = paths[i], paths[j]
            name_a, name_b = path_names[i], path_names[j]

            common_len = min(len(pa), len(pb))
            if common_len == 0:
                continue

            events = []
            for idx in range(common_len):
                loc_a = pa[idx][0].transform.location
                loc_b = pb[idx][0].transform.location
                dx = loc_a.x - loc_b.x
                dy = loc_a.y - loc_b.y
                dz = loc_a.z - loc_b.z
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist < min_dist:
                    events.append((idx, dist, loc_a, loc_b))

            if not events:
                continue

            segments = []
            seg_start = events[0][0]
            seg_end   = events[0][0]
            seg_min_d = events[0][1]
            for idx, dist, la, lb in events[1:]:
                if idx == seg_end + 1:
                    seg_end   = idx
                    seg_min_d = min(seg_min_d, dist)
                else:
                    segments.append((seg_start, seg_end, seg_min_d))
                    seg_start = idx
                    seg_end   = idx
                    seg_min_d = dist
            segments.append((seg_start, seg_end, seg_min_d))

            lines = [
                f"Collision risk between {name_a} and {name_b}: "
                f"{len(events)} time step(s) closer than {min_dist:.1f}m "
                f"(path lengths: {len(pa)} vs {len(pb)})."
            ]
            for seg_s, seg_e, seg_d in segments:
                closest = min(
                    (e for e in events if seg_s <= e[0] <= seg_e),
                    key=lambda e: e[1]
                )
                _, d, la, lb = closest
                mid_x = (la.x + lb.x) / 2
                mid_y = (la.y + lb.y) / 2
                if seg_s == seg_e:
                    step_desc = f"step {seg_s}"
                else:
                    step_desc = f"steps {seg_s}-{seg_e}"
                lines.append(
                    f"  {step_desc}: min_dist={seg_d:.2f}m "
                    f"near ({mid_x:.1f}, {mid_y:.1f})"
                )

            collision_blocks.append("\n".join(lines))

    if not collision_blocks:
        return ""

    header = (
        f"[Collision Check] threshold={min_dist:.1f}m, "
        f"{len(collision_blocks)} conflict pair(s) detected:\n"
    )
    return header + "\n\n".join(collision_blocks)


# ==============================================================================
# -- LLM graph-based navigation ------------------------------------------------
# ==============================================================================

def nearest_node(graph, loc: carla.Location) -> int:
    """Return the index of the JunctionGraph node closest to loc in 2D distance."""
    best_idx, best_dist = 0, float('inf')
    for node in graph.nodes:
        d = math.sqrt((node.x - loc.x) ** 2 + (node.y - loc.y) ** 2)
        if d < best_dist:
            best_dist = d
            best_idx = node.idx
    return best_idx


def forbidden_node_indices(graph, forbidden_areas: list) -> list:
    """
    Return the list of node indices that fall inside any forbidden area.
    forbidden_areas: list[(carla.Location, float)], each item is (center, radius)
    """
    result = []
    for node in graph.nodes:
        for center, radius in (forbidden_areas or []):
            d = math.sqrt((node.x - center.x) ** 2 + (node.y - center.y) ** 2)
            if d <= radius:
                result.append(node.idx)
                break
    return result


def _print_graph_debug(graph) -> None:
    """
    Print the adjacency matrix to the terminal (for debugging, not sent to the LLM).
    Column width and row-label width adapt to the node count so that column
    indices align vertically with their values.
    """
    n = graph.size
    col_w = len(str(n - 1))
    cell_w = col_w + 1
    prefix_w = col_w + 3

    header = ' ' * prefix_w + ''.join(f'{j:{cell_w}d}' for j in range(n))
    sep    = ' ' * prefix_w + '-' * (n * cell_w)
    print(f'[LLM Graph Matrix] {n}x{n} adjacency matrix (0=not connected, 1=directly connected):')
    print(header)
    print(sep)
    for i, row in enumerate(graph.matrix):
        vals = ''.join(f'{v:{cell_w}d}' for v in row)
        print(f'{i:{col_w}d} | {vals}')
    print()


def graph_to_llm_context(graph, use_adj_list: bool = True) -> str:
    """
    Encode the JunctionGraph as connectivity text for LLM path planning.
    Contains no coordinates -- nodes are referred to by index (0 to N-1).

    Args
    ----
    use_adj_list : bool, default False
        False (default): adjacency matrix format.
            Column width adapts to the node count; column indices align
            strictly with their values.
        True: adjacency list format.
            Lists each node's neighbor indices directly, with no row/column
            alignment ambiguity; suitable when there are many nodes or to
            reduce LLM misreading.
    """
    n = graph.size

    if use_adj_list:
        lines = [
            f"N={n} nodes (indices 0 to {n - 1})",
            "",
            "Adjacency list — for each node, the indices of nodes it is DIRECTLY connected to.",
            "IMPORTANT: only edges listed here exist; any pair NOT listed is NOT connected.",
            "",
        ]
        print("neighbours: ")
        for i in range(n):
            lines.append(f"  {i:3d}: {graph.neighbors(i)}")
            print(f"  {i:3d}: {graph.neighbors(i)}")
        return "\n".join(lines)

    # ---- Adjacency matrix format (default) ----
    col_w = len(str(n - 1))
    cell_w = col_w + 1
    prefix_w = col_w + 3

    col_header = ' ' * prefix_w + ''.join(f'{j:{cell_w}d}' for j in range(n))
    sep        = ' ' * prefix_w + '-' * (n * cell_w)
    lines = [
        f"N={n} nodes (indices 0 to {n - 1})",
        "",
        "Adjacency matrix — matrix[row][col]=1 means the two nodes are directly connected:",
        col_header,
        sep,
    ]
    print("matrix to send: ")
    for i, row in enumerate(graph.matrix):
        vals = ''.join(f'{v:{cell_w}d}' for v in row)
        lines.append(f'{i:{col_w}d} | {vals}')
        print(f'{i:{col_w}d} | {vals}')
    return "\n".join(lines)


def graph_to_llm_context_coords(graph) -> str:
    """
    Encode the JunctionGraph as text with real coordinates for LLM path planning.
    Nodes are identified by (x, y) coordinates instead of indices; neighbors are
    also listed as coordinates.
    Expected LLM output format: path = [(x1, y1), (x2, y2), ...]

    Difference from graph_to_llm_context
    ------------------------------------
    graph_to_llm_context        : exposes only abstract node indices; the LLM
                                  cannot perceive spatial positions.
    graph_to_llm_context_coords : exposes coordinates directly, so the LLM can
                                  reason about distance and direction without a
                                  local index<->coordinate mapping.
    """
    n = graph.size
    lines = [
        f"Road graph with {n} nodes, each identified by real-world (x, y) coordinates.",
        "",
        "Adjacency list — each line shows a node coordinate and its directly connected neighbor coordinates.",
        "IMPORTANT: only connections listed here exist; any pair NOT listed is NOT connected.",
        "",
    ]
    print("[LLM Coords] node coordinate adjacency list:")
    for i in range(n):
        x, y, _ = graph.coord(i)
        neighbors = graph.neighbors(i)
        nb_coords = []
        for nb in neighbors:
            nx_, ny_, _ = graph.coord(nb)
            nb_coords.append(f"({nx_:.1f}, {ny_:.1f})")
        lines.append(f"  ({x:.1f}, {y:.1f}) -> {nb_coords}")
        print(f"  ({x:.1f}, {y:.1f}) -> {nb_coords}")
    return "\n".join(lines)


def _load_geojson(geojson_path: str) -> dict:
    """Read a GeoJSON file; supports relative paths (based on this script's directory)."""
    path = Path(geojson_path)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_geojson_points(geojson_path: str) -> list:
    """
    Extract junction point information from a GeoJSON file.
    Returns a list of dicts, each containing junction_id, x, y.
    """
    data = _load_geojson(geojson_path)
    points = []
    for feature in data.get("features", []):
        if feature.get("geometry", {}).get("type") == "Point":
            coords = feature["geometry"]["coordinates"]
            junction_id = feature.get("properties", {}).get("junction_id")
            points.append({
                "junction_id": junction_id,
                "x": coords[0],
                "y": coords[1],
            })
    return points


def extract_geojson_road_segments(geojson_path: str) -> list:
    """
    Extract road segment information from a GeoJSON file, keeping only the first
    and last endpoints of each segment.
    Returns a list of dicts, each containing from_junction, to_junction, length_m,
    start (x, y), end (x, y).
    """
    data = _load_geojson(geojson_path)
    segments = []
    for feature in data.get("features", []):
        if feature.get("geometry", {}).get("type") == "LineString":
            coords = feature["geometry"]["coordinates"]
            if len(coords) < 2:
                continue
            props = feature.get("properties", {})
            segments.append({
                "from_junction": props.get("from_junction"),
                "to_junction":   props.get("to_junction"),
                "length_m":      props.get("length_m"),
                "start": (coords[0][0],  coords[0][1]),
                "end":   (coords[-1][0], coords[-1][1]),
            })
    return segments


def graph_to_llm_context_geojson_points(geojson_path: str) -> str:
    """
    Extract junction points and road segments from a GeoJSON file and generate
    map text for LLM path planning.
    Segments keep only their two endpoints (intermediate interpolated points removed).
    Expected LLM output format: path = [(x1, y1), (x2, y2), ...]
    """
    points   = extract_geojson_points(geojson_path)
    segments = extract_geojson_road_segments(geojson_path)

    lines = [
        f"Road map with {len(points)} junction nodes and {len(segments)} road segments.",
        "Nodes are identified by real-world (x, y) coordinates.",
        "",
        "Junction nodes:",
    ]
    for p in points:
        lines.append(f"  junction {p['junction_id']}: ({p['x']:.1f}, {p['y']:.1f})")

    lines += [
        "",
        "Road segments (each segment is defined by its two endpoints only):",
        "  format: junction A (xA, yA) -- junction B (xB, yB)  [length]",
    ]
    for s in segments:
        sx, sy = s["start"]
        ex, ey = s["end"]
        length_str = f"{s['length_m']:.1f}m" if s["length_m"] is not None else "?"
        lines.append(
            f"  junction {s['from_junction']} ({sx:.1f}, {sy:.1f})"
            f" -- junction {s['to_junction']} ({ex:.1f}, {ey:.1f})"
            f"  [{length_str}]"
        )

    print(f"[LLM GeoJSON] {len(points)} junctions, {len(segments)} road segments loaded from {geojson_path}")
    return "\n".join(lines)


def _log_index_lookup(label: str, locs_and_indices: list) -> None:
    """
    Print a "coordinate -> node index" lookup table to the terminal
    (not sent to the LLM, for debugging only).
    locs_and_indices: list[(carla.Location | None, int)]
    """
    print(f'[LLM Mapping] {label}:')
    for loc, idx in locs_and_indices:
        if loc is not None:
            print(f'    ({loc.x:8.2f}, {loc.y:8.2f})  ->  node {idx}')
        else:
            print(f'    (start)  ->  node {idx}')


def parse_llm_node_path(text: str, graph, silent: bool = False) -> list:
    """
    Parse a node-index path from an LLM reply and return the corresponding
    list of carla.Location.

    Expected LLM output format (either works):
      path = [0, 3, 5, 8]
      [0, 3, 5, 8]

    Returns list[carla.Location] in node-index order; empty list on parse failure.
    """
    m = re.search(r'path\s*=\s*\[([^\]]+)\]', text)
    if not m:
        m = re.search(r'\[([0-9,\s]+)\]', text)
    if not m:
        if not silent:
            print('[LLM] Node path list not found')
        return []

    try:
        indices = [int(x.strip()) for x in m.group(1).split(',') if x.strip().isdigit()]
    except Exception as e:
        print(f'[LLM] Failed to parse node indices: {e}')
        return []

    print(f'[LLM Mapping] index -> coord (LLM output to coords):')
    locations = []
    for idx in indices:
        try:
            x, y, z = graph.coord(idx)
            print(f'    node {idx}  ->  ({x:.2f}, {y:.2f})')
            locations.append(carla.Location(x=x, y=y, z=z))
        except IndexError:
            print(f'    node {idx}  ->  [Out of range, skipped]')
    return locations


def parse_unified_llm_paths(text: str, graph) -> tuple:
    """
    Parse two paths from a single-LLM unified planning reply.

    Expected format (one per line):
        path1 = [<idx>, <idx>, ...]
        path2 = [<idx>, <idx>, ...]

    Returns (list[carla.Location], list[carla.Location]);
    the corresponding list is empty if a path fails to parse.
    """
    def _extract(tag: str) -> list:
        m = re.search(rf'{tag}\s*=\s*\[([^\]]+)\]', text)
        if not m:
            print(f'[Unified Planning] {tag} list not found')
            return []
        try:
            indices = [int(x.strip()) for x in m.group(1).split(',')
                       if x.strip().isdigit()]
        except Exception as e:
            print(f'[Unified Planning] Failed to parse {tag} indices: {e}')
            return []
        locs = []
        for idx in indices:
            try:
                x, y, z = graph.coord(idx)
                locs.append(carla.Location(x=x, y=y, z=z))
            except IndexError:
                print(f'[Unified Planning] {tag} node {idx} out of range, skipped')
        return locs

    locs1 = _extract('path1')
    locs2 = _extract('path2')
    print(f'[Unified Planning] Parsing done: path1={len(locs1)} nodes, path2={len(locs2)} nodes')
    return locs1, locs2


def parse_llm_coord_path(text: str, silent: bool = False) -> list:
    """
    Parse a coordinate path from an LLM reply and return a list of carla.Location.
    Used in prompt_format="coords" mode -- the LLM outputs coordinates directly,
    no index mapping required.

    Expected LLM output format (either works):
      path = [(x1, y1), (x2, y2), ...]
      [(x1, y1), (x2, y2), ...]

    Returns list[carla.Location] with z=0 (CARLA determines the road height);
    empty list on parse failure.
    """
    m = re.search(r'path\s*=\s*\[([^\]]+)\]', text)
    if not m:
        m = re.search(r'\[([^\]]+)\]', text)
    if not m:
        if not silent:
            print('[LLM-Coords] Coordinate path list not found')
        return []

    content = m.group(1)
    pairs = re.findall(
        r'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)(?:\s*,\s*-?\d+(?:\.\d+)?)?\s*\)',
        content,
    )
    if not pairs:
        if not silent:
            print('[LLM-Coords] No valid (x, y) coordinate pairs found')
        return []

    print('[LLM-Coords] Parsed coordinate sequence:')
    locations = []
    for x_str, y_str in pairs:
        x, y = float(x_str), float(y_str)
        print(f'    ({x:.2f}, {y:.2f})')
        locations.append(carla.Location(x=x, y=y, z=0.0))

    print(f'[LLM-Coords] {len(locations)} coordinate points in total')
    return locations


def parse_unified_llm_paths_coords(text: str) -> tuple:
    """
    Parse two coordinate paths from a single-LLM unified planning reply (coords mode).

    Expected format (one per line):
        path1 = [(x, y), (x, y), ...]
        path2 = [(x, y), (x, y), ...]

    Returns (list[carla.Location], list[carla.Location]);
    the corresponding list is empty if a path fails to parse.
    """
    def _extract(tag: str) -> list:
        m = re.search(
            rf'{tag}\s*=\s*\[([^\]]+)\]',
            text,
        )
        if not m:
            print(f'[Unified Planning-Coords] {tag} coordinate list not found')
            return []
        content = m.group(1)
        pairs = re.findall(
            r'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)(?:\s*,\s*-?\d+(?:\.\d+)?)?\s*\)',
            content,
        )
        locs = []
        for x_str, y_str in pairs:
            locs.append(carla.Location(x=float(x_str), y=float(y_str), z=0.0))
        return locs

    locs1 = _extract('path1')
    locs2 = _extract('path2')
    print(f'[Unified Planning-Coords] Parsing done: path1={len(locs1)} points, path2={len(locs2)} points')
    return locs1, locs2


def query_llm_for_navigation_graph(
        vehicle: carla.Vehicle,
        graph,
        destination: carla.Location = None,
        waypoints: list = None,
        forbidden_areas: list = None,
        custom_system_prompt: str = None,
        custom_user_prompt: str = None,
        additional_prompt: str = None,
        prompt_format: str = "index_list",
):
    """
    Asynchronously send a navigation request to the LLM based on the
    JunctionGraph, supporting single-destination, multi-waypoint tour, and
    fully custom prompts.

    Working modes (choose one)
    --------------------------
    destination mode (single destination)
        Pass destination; the LLM plans the shortest path from the current
        position to that destination.

    waypoints mode (multi-waypoint tour)
        Pass waypoints (list[carla.Location]); the LLM plans a tour that visits
        all waypoints in turn and finally returns to the start.

    Custom prompt
    -------------
    custom_system_prompt : str, optional
        Completely replaces the built-in system message. If omitted, the default
        system message matching the current mode is used.

    custom_user_prompt : str, optional
        Appended as an extra user message after the auto-generated graph context.
        Useful for adding task descriptions, constraints, or examples without
        manually assembling the graph text.

        Example:
            custom_user_prompt=(
                "Priority: minimize total path length. "
                "If two paths have equal length, prefer the one with fewer turns."
            )

    Args
    ----
    vehicle          : the currently controlled vehicle (used to get position and heading)
    graph            : JunctionGraph (loaded via load_graph())
    destination      : destination carla.Location (destination mode)
    waypoints        : list[carla.Location], waypoints to visit in turn (tour mode)
    forbidden_areas  : list[(carla.Location, float)], forbidden areas, each (center, radius)
    custom_system_prompt : custom text replacing the default system message
    custom_user_prompt   : custom text appended after the auto-generated user message
    additional_prompt    : history from a previous failed plan, prompting the LLM to re-plan
    prompt_format    : str, how path nodes are represented in the prompt, one of:
        "index_list"   (default) adjacency list + node indices, LLM outputs path = [idx, ...]
        "index_matrix" adjacency matrix + node indices, LLM outputs path = [idx, ...]
        "coords"       real-coordinate adjacency list, no index mapping, LLM outputs path = [(x,y), ...]

    Returns
    -------
    concurrent.futures.Future; once done, call
    parse_llm_node_path(text, graph) to get a list[carla.Location].
    Returns None if llmutil is not installed.

    Usage examples
    --------------
    # Single destination
    future = query_llm_for_navigation_graph(
        vehicle, graph,
        destination=carla.Location(80.0, -100.0, 0.0),
        forbidden_areas=forbidden_areas,
    )

    # Multi-waypoint tour with an extra custom constraint
    future = query_llm_for_navigation_graph(
        vehicle, graph,
        waypoints=[
            carla.Location(30.0,  20.0, 0.0),
            carla.Location(80.0, -50.0, 0.0),
            carla.Location(10.0, -80.0, 0.0),
        ],
        forbidden_areas=forbidden_areas,
        custom_user_prompt="Prefer paths that avoid sharp U-turns.",
    )

    # Fully custom prompt (only the auto-generated graph context is kept)
    future = query_llm_for_navigation_graph(
        vehicle, graph,
        custom_system_prompt="You are a logistics route optimizer...",
        custom_user_prompt="Visit all delivery points optimally and return to depot.",
        waypoints=[...],
    )

    # Main-loop check
    if future is not None and future.done():
        text, _ = future.result(timeout=0)
        anchors = parse_llm_node_path(text, graph)
        if anchors:
            route_from_sparse(vehicle, cmap, grp, world, agent,
                              anchors=anchors,
                              forbidden_areas=forbidden_areas)
    """
    if not _LLM_AVAILABLE:
        print('[LLM] llmutil is not installed; LLM navigation is unavailable.')
        return None
    if destination is None and not waypoints:
        print('[LLM] Either destination or waypoints must be provided.')
        return None

    cur = vehicle.get_transform().location

    # ------------------------------------------------------------------ #
    # 1 & 2. Build the system / user messages according to prompt_format
    # ------------------------------------------------------------------ #

    if prompt_format == "coords":
        # ============================================================== #
        # coords mode: use real coordinates directly, no index mapping
        # LLM output format: path = [(x1, y1), (x2, y2), ...]
        # ============================================================== #
        graph_ctx = graph_to_llm_context_geojson_points("custom_junction_graph.geojson")

        if forbidden_areas:
            fa_desc = [
                f"({fa_loc.x:.1f}, {fa_loc.y:.1f}) radius {fa_r:.1f}m"
                for fa_loc, fa_r in forbidden_areas
            ]
            forbidden_str = f"Forbidden areas (avoid nodes inside these circles): {fa_desc}"
        else:
            forbidden_str = "No forbidden areas."

        if waypoints:
            wp_coords = [f"({wp.x:.1f}, {wp.y:.1f})" for wp in waypoints]
            print(f'[LLM-Coords] Tour request: start ({cur.x:.1f},{cur.y:.1f}), waypoints {wp_coords}')

            default_system = (
                "You are an autonomous vehicle path planner working on a road graph "
                "where every node is identified by its real-world (x, y) coordinate.\n"
                "Plan a round-trip route that:\n"
                "  1. Starts at the given start coordinate.\n"
                "  2. Visits ALL required waypoint coordinates (choose the visit order yourself).\n"
                "  3. Returns to the start coordinate at the end.\n"
                "  4. Only use coordinates from the provided junction node list.\n"
                "  5. Avoids routing through any forbidden area.\n"
                "  6. Minimises the total number of nodes visited.\n"
                "Output EXACTLY one line — no explanation, no code block:\n"
                "  path = [(x1, y1), (x2, y2), ..., (xn, yn)]\n"
                "The first and last coordinate must both equal the start coordinate."
            )
            auto_user = (
                f"{graph_ctx}\n\n"
                f"Start coordinate (= return coordinate): ({cur.x:.1f}, {cur.y:.1f})\n"
                f"Required waypoint coordinates ({len(waypoints)} total): {wp_coords}\n"
                f"{forbidden_str}\n\n"
                f"Plan the shortest round-trip from ({cur.x:.1f}, {cur.y:.1f}) "
                f"that visits all of {wp_coords} and returns to ({cur.x:.1f}, {cur.y:.1f})."
            )

        else:
            print(f'[LLM-Coords] Single-destination request: '
                  f'({cur.x:.1f},{cur.y:.1f}) -> ({destination.x:.1f},{destination.y:.1f})')

            default_system = (
                "You are an autonomous vehicle path planner working on a road graph "
                "where every node is identified by its real-world (x, y) coordinate.\n"
                "Find the shortest path from the start coordinate to the destination coordinate.\n"
                "Rules:\n"
                "  - Only use coordinates from the provided junction node list.\n"
                "  - Avoid routing through any forbidden area.\n"
                "Output EXACTLY one line — no explanation, no code block:\n"
                "  path = [(x1, y1), (x2, y2), ..., (xn, yn)]\n"
                "The first coordinate must equal the start; the last must equal the destination."
            )
            auto_user = (
                f"{graph_ctx}\n\n"
                f"Start coordinate: ({cur.x:.1f}, {cur.y:.1f})\n"
                f"Destination coordinate: ({destination.x:.1f}, {destination.y:.1f})\n"
                f"{forbidden_str}\n\n"
                f"Find the shortest valid path from ({cur.x:.1f}, {cur.y:.1f}) "
                f"to ({destination.x:.1f}, {destination.y:.1f})."
            )

    else:
        # ============================================================== #
        # index mode (index_list / index_matrix): coordinate -> index mapping,
        # the LLM works with indices
        # LLM output format: path = [idx, idx, ...]
        # ============================================================== #
        use_adj_list = (prompt_format != "index_matrix")

        start_idx = nearest_node(graph, cur)
        forbidden_nodes = forbidden_node_indices(graph, forbidden_areas)

        _log_index_lookup("Start Point", [(cur, start_idx)])
        if forbidden_nodes:
            print(f'[LLM Mapping] Forbidden nodes index: {forbidden_nodes}')

        _print_graph_debug(graph)
        graph_ctx = graph_to_llm_context(graph, use_adj_list=use_adj_list)

        forbidden_str = (
            f"Forbidden node indices (must NOT appear in path): {forbidden_nodes}"
            if forbidden_nodes else "No forbidden nodes."
        )

        if waypoints:
            wp_nodes = [nearest_node(graph, wp) for wp in waypoints]
            _log_index_lookup("Tour waypoints", list(zip(waypoints, wp_nodes)))

            default_system = (
                "You are an autonomous vehicle path planner working on an abstract road graph.\n"
                "Nodes are identified by integer indices only — no coordinates are given.\n"
                "Plan a round-trip route that:\n"
                "  1. Starts at the given start node.\n"
                "  2. Visits ALL required waypoint nodes (choose the visit order yourself).\n"
                "  3. Returns to the start node at the end.\n"
                "  4. Only uses edges that exist in the adjacency matrix (value = 1).\n"
                "  5. Does NOT pass through any forbidden node.\n"
                "  6. Minimize the usage of nodes.\n"
                "Output EXACTLY one line — no explanation, no code block:\n"
                "  path = [<idx>, <idx>, ..., <idx>]\n"
                "The first and last index must both equal the start node."
            )
            auto_user = (
                f"{graph_ctx}\n\n"
                f"Start node (= return node): {start_idx}\n"
                f"Required waypoint nodes ({len(wp_nodes)} total): {wp_nodes}\n"
                f"{forbidden_str}\n\n"
                f"Plan the shortest round-trip from node {start_idx} "
                f"that visits all of {wp_nodes} and returns to node {start_idx}."
            )
            print(f'[LLM] Tour request: start {start_idx}, waypoint nodes {wp_nodes}, forbidden nodes {forbidden_nodes}')

        else:
            end_idx = nearest_node(graph, destination)
            _log_index_lookup("Destination", [(destination, end_idx)])

            default_system = (
                "You are an autonomous vehicle path planner working on an abstract road graph.\n"
                "Nodes are identified by integer indices only — no coordinates are given.\n"
                "Find the shortest path from the start node to the destination node.\n"
                "Rules:\n"
                "  - Only traverse edges that exist in the adjacency matrix (value = 1).\n"
                "  - Do NOT include any forbidden node in the path.\n"
                "Output EXACTLY one line — no explanation, no code block:\n"
                "  path = [<idx>, <idx>, ..., <idx>]\n"
                "The first index must equal the start node; the last must equal the destination node."
            )
            auto_user = (
                f"{graph_ctx}\n\n"
                f"Start node: {start_idx}\n"
                f"Destination node: {end_idx}\n"
                f"{forbidden_str}\n\n"
                f"Find the shortest valid path from node {start_idx} to node {end_idx}."
            )
            print(f'[LLM] Single-destination request: node {start_idx} -> {end_idx}, forbidden nodes {forbidden_nodes}')

    # ------------------------------------------------------------------ #
    # 3. Assemble the message list
    # ------------------------------------------------------------------ #
    system_content = custom_system_prompt if custom_system_prompt is not None else default_system

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": auto_user},
    ]

    if custom_user_prompt is not None:
        messages.append({"role": "user", "content": custom_user_prompt})

    if additional_prompt is not None:
        messages.append({
            "role": "user",
            "content": (
                f"The previous attempt produced an invalid path. "
                f"History: {additional_prompt}. "
                f"Please generate a different valid path."
            ),
        })

    return _llmutils.query_llm_async(messages, seed=0)


# ==============================================================================
# -- Keyboard control ----------------------------------------------------------
# ==============================================================================

class KeyboardController:
    """
    Listen to pygame keyboard events and convert key presses into command flags.
      W / S     - move the destination forward / backward along the Y axis
      A / D     - move the destination left / right along the X axis
      X         - trigger path planning with the current destination (direct BehaviorAgent planning)
      L         - trigger LLM single-destination navigation with the current destination
      T         - trigger LLM multi-waypoint tour planning (waypoints defined in main())
      H         - remove all vehicles and respawn at spawn_pos1/spawn_pos2
      ESC / Q   - quit

    Each WASD press moves the destination by MOVE_STEP meters.
    """

    MOVE_STEP = 5.0  # distance moved per key press (meters)

    def __init__(self, init_destination: carla.Location):
        self.destination = carla.Location(
            x=init_destination.x,
            y=init_destination.y,
            z=init_destination.z,
        )
        self.active_vehicle = 1   # which vehicle is being controlled (1 or 2)
        self._flags = {
            'go': False,                   # X key: direct planning
            'llm_go': False,               # L key: LLM single destination
            'llm_tour': False,             # T key: LLM multi-waypoint tour
            'llm_multi_go': False,         # Y key: dual-agent negotiated planning
            'llm_unified_go': False,       # U key: single-LLM unified planning for both vehicles
            'llm_multi_tour': False,       # I key: dual-agent shared-waypoint tour
            'llm_unified_tour': False,     # O key: single-LLM shared-waypoint tour for both vehicles
            'reset': False,                # H key: remove and respawn all vehicles
            'destination_changed': False,
            'quit': False,
        }

    def parse_events(self):
        """Process all pygame events; update and return this frame's command flag dict."""
        self._flags['go'] = False
        self._flags['llm_go'] = False
        self._flags['llm_tour'] = False
        self._flags['llm_multi_go'] = False
        self._flags['llm_unified_go'] = False
        self._flags['llm_multi_tour'] = False
        self._flags['llm_unified_tour'] = False
        self._flags['reset'] = False
        self._flags['destination_changed'] = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._flags['quit'] = True
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event.key)

        return dict(self._flags)

    def _handle_keydown(self, key):
        moved = False
        if key == K_w:
            self.destination.y += self.MOVE_STEP
            moved = True
        elif key == K_s:
            self.destination.y -= self.MOVE_STEP
            moved = True
        elif key == K_a:
            self.destination.x += self.MOVE_STEP
            moved = True
        elif key == K_d:
            self.destination.x -= self.MOVE_STEP
            moved = True
        elif key == K_x:
            self._flags['go'] = True
            print('[Keyboard] X pressed -> direct planning')
        elif key == K_l:
            self._flags['llm_go'] = True
            print('[Keyboard] L pressed -> LLM single-destination navigation')
        elif key == K_t:
            self._flags['llm_tour'] = True
            print('[Keyboard] T pressed -> LLM multi-waypoint tour')
        elif key == K_y:
            self._flags['llm_multi_go'] = True
            print('[Keyboard] Y pressed -> dual-agent negotiated planning')
        elif key == K_u:
            self._flags['llm_unified_go'] = True
            print('[Keyboard] U pressed -> single-LLM unified planning for both vehicles')
        elif key == K_i:
            self._flags['llm_multi_tour'] = True
            print('[Keyboard] I pressed -> dual-agent shared-waypoint tour')
        elif key == K_o:
            self._flags['llm_unified_tour'] = True
            print('[Keyboard] O pressed -> single-LLM shared-waypoint tour for both vehicles')
        elif key == K_h:
            self._flags['reset'] = True
            print('[Keyboard] H pressed -> remove and respawn all vehicles')
        elif key == K_1:
            self.active_vehicle = 1
            print('[Keyboard] Switched to vehicle 1')
        elif key == K_2:
            self.active_vehicle = 2
            print('[Keyboard] Switched to vehicle 2')
        elif key == K_ESCAPE or (key == K_q and pygame.key.get_mods() & KMOD_CTRL):
            self._flags['quit'] = True

        if moved:
            self._flags['destination_changed'] = True
            print(f'[Destination] x={self.destination.x:.1f}  y={self.destination.y:.1f}  z={self.destination.z:.1f}')


# ==============================================================================
# -- Vehicle spawning ----------------------------------------------------------
# ==============================================================================

def get_road_spawn_transforms(cmap: carla.Map, count: int = 30) -> list:
    """
    Sample road waypoints from the map topology and return a list of
    carla.Transform usable as spawn positions.
    Prefers the map's preset spawn points; if empty (common with custom xodr),
    takes road-segment midpoints from the topology.
    """
    preset = cmap.get_spawn_points()
    if preset:
        return preset

    topology = cmap.get_topology()   # list[(entry_wp, exit_wp)]
    transforms = []
    for entry_wp, exit_wp in topology:
        mid_loc = carla.Location(
            x=(entry_wp.transform.location.x + exit_wp.transform.location.x) / 2,
            y=(entry_wp.transform.location.y + exit_wp.transform.location.y) / 2,
            z=(entry_wp.transform.location.z + exit_wp.transform.location.z) / 2,
        )
        wp = cmap.get_waypoint(mid_loc, project_to_road=True,
                               lane_type=carla.LaneType.Driving)
        if wp is not None:
            # raise slightly to avoid ground collision
            tf = wp.transform
            tf.location.z += 0.5
            transforms.append(tf)

    random.shuffle(transforms)
    return transforms[:count]


def spawn_vehicle(world, blueprint_library, cmap: carla.Map = None,
                  spawn_location: carla.Location = None):
    """
    Spawn a vehicle.nissan.micra on the road and return the actor object.

    spawn_location : carla.Location, optional.
        If provided, the coordinate is snapped to the nearest driving lane
        before spawning; otherwise a random spawn point is chosen from the map.
    """
    blueprint = blueprint_library.find('vehicle.nissan.micra')
    if blueprint.has_attribute('color'):
        color = random.choice(blueprint.get_attribute('color').recommended_values)
        blueprint.set_attribute('color', color)
    blueprint.set_attribute('role_name', 'agent_vehicle')

    if cmap is None:
        cmap = world.get_map()

    if spawn_location is not None:
        wp = cmap.get_waypoint(spawn_location, project_to_road=True,
                               lane_type=carla.LaneType.Driving)
        if wp is None:
            print(f'[Error] The specified position ({spawn_location.x:.1f}, {spawn_location.y:.1f}) '
                  f'cannot be snapped to a driving lane.')
            return None
        tf = wp.transform
        tf.location.z += 0.5   # raise slightly to avoid ground collision
        vehicle = world.try_spawn_actor(blueprint, tf)
        if vehicle is not None:
            print(f'[Spawn] Spawned at the specified position: ({tf.location.x:.1f}, {tf.location.y:.1f})')
            return vehicle
        print(f'[Error] Failed to spawn at the specified position (possibly overlapping another actor).')
        return None

    spawn_transforms = get_road_spawn_transforms(cmap)
    if not spawn_transforms:
        print('[Error] Failed to obtain road spawn points from the map.')
        return None

    random.shuffle(spawn_transforms)
    for tf in spawn_transforms:
        vehicle = world.try_spawn_actor(blueprint, tf)
        if vehicle is not None:
            return vehicle

    print('[Error] Spawning failed at all road positions.')
    return None


# ==============================================================================
# -- Experiment logging --------------------------------------------------------
# ==============================================================================

_EXPERIMENTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")


def calc_path_length_m(plan: list) -> float:
    """
    Compute the total path length in meters.
    plan: list[(carla.Waypoint, RoadOption)]
    """
    if not plan or len(plan) < 2:
        return 0.0
    total = 0.0
    for i in range(len(plan) - 1):
        a = plan[i][0].transform.location
        b = plan[i + 1][0].transform.location
        total += a.distance(b)
    return total


def save_experiment_log(task_name: str, data: dict) -> None:
    """
    Save experiment data to:
      experiments/<task_name>/<YYYY-MM-DD_HH-MM-SS>/report.txt
    Also append one row to the global CSV:
      experiments/results.csv

    data fields:
      iterations         : int   number of iterations
      avg_llm_time       : float average total LLM time per agent (seconds)
      path1_length       : float vehicle 1 path length (meters)
      path2_length       : float vehicle 2 path length (meters)
      collision_detected : bool  whether a collision was detected while driving
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_dir  = os.path.join(_EXPERIMENTS_ROOT, task_name)
    run_dir   = os.path.join(task_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    p1  = data.get("path1_length", 0.0)
    p2  = data.get("path2_length", 0.0)
    avg_path = (p1 + p2) / 2.0 if (p1 + p2) > 0 else 0.0

    # ── Text report ──────────────────────────────────────────────────────────
    report_lines = [
        f"Task            : {task_name}",
        f"Timestamp       : {timestamp}",
        f"Iterations      : {data.get('iterations', 0)}",
        f"Avg LLM Time(s) : {data.get('avg_llm_time', 0.0):.2f}",
        f"Path1 Length(m) : {p1:.1f}",
        f"Path2 Length(m) : {p2:.1f}",
        f"Avg Path Len(m) : {avg_path:.1f}",
        f"Collision Det.  : {data.get('collision_detected', False)}",
    ]
    report_path = os.path.join(run_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    # ── Global CSV (one row appended per run) ────────────────────────────────
    csv_path   = os.path.join(_EXPERIMENTS_ROOT, "results.csv")
    csv_fields = [
        "timestamp", "task", "iterations",
        "avg_llm_time_s", "path1_length_m", "path2_length_m",
        "avg_path_length_m", "collision_detected",
    ]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp":         timestamp,
            "task":              task_name,
            "iterations":        data.get("iterations", 0),
            "avg_llm_time_s":    f"{data.get('avg_llm_time', 0.0):.2f}",
            "path1_length_m":    f"{p1:.1f}",
            "path2_length_m":    f"{p2:.1f}",
            "avg_path_length_m": f"{avg_path:.1f}",
            "collision_detected": data.get("collision_detected", False),
        })

    print(f"[Experiment Log] Saved -> {run_dir}")
    print(f"[Experiment Log] CSV   -> {csv_path}")


def main():
    actor_list = []
    world = None
    graph = load_graph("custom_junction_graph.geojson")
    pygame.init()
    pygame.display.set_mode((400, 100))
    pygame.display.set_caption('multiagent - WASD move destination, X confirm navigation, ESC/Ctrl+Q quit')
    _init_dest = carla.Location(0.0, 0.0, 0.0)
    keyboard = KeyboardController(_init_dest)
    spawn_pos1 = carla.Location(365, 109, 0.2)
    spawn_pos2 = carla.Location(362, -55, 0.2)

    try:
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(10.0)

        xodr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'custom.xodr')
        with open(xodr_path, 'r', encoding='utf-8') as f:
            xodr_data = f.read()
        world = client.generate_opendrive_world(
            xodr_data,
            carla.OpendriveGenerationParameters(
                vertex_distance=2.0,
                max_road_length=500.0,
                wall_height=1.0,
                additional_width=0.6,
                smooth_junctions=True,
                enable_mesh_visibility=True,
            ),
        )

        blueprint_library = world.get_blueprint_library()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1  # 20 FPS # original as 0.05
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
        cmap = world.get_map()

        vehicle = spawn_vehicle(world, blueprint_library, cmap, spawn_pos1)
        if vehicle is None:
            return
        actor_list.append(vehicle)

        vehicle2 = spawn_vehicle(world, blueprint_library, cmap, spawn_pos2)
        if vehicle2 is None:
            return
        actor_list.append(vehicle2)

        world.tick()

        grp = GlobalRoutePlanner(cmap, sampling_resolution=2.0)

        agent = BehaviorAgent(vehicle, behavior='normal')
        agent.follow_speed_limits(False)
        agent.set_target_speed(40)

        agent2 = BehaviorAgent(vehicle2, behavior='normal')
        agent2.follow_speed_limits(False)
        agent2.set_target_speed(40)
        forbidden_areas = [
            (carla.Location(50.0, 50.0, 0.0), 20.0),
        ]

        keyboard.destination = carla.Location(
            x=vehicle.get_location().x,
            y=vehicle.get_location().y,
            z=0.0,
        )
        print(f'[Destination] Start x={keyboard.destination.x:.1f}  y={keyboard.destination.y:.1f}')

        # tour_waypoints = [ # 5 point case
        #     carla.Location( 120.0,  187.0, 0.0),
        #     carla.Location(271.0, 31.0, 0.0),
        #     carla.Location(-57.0, 25.0, 0.0),
        #     carla.Location(120.0, -118.0, 0.0),
        #     carla.Location(-277.0, 88.0, 0.0),
        # ]

        # tour_waypoints = [  # 2 point case
        #     carla.Location(120.0, 187.0, 0.0),
        #     carla.Location(-277.0, 88.0, 0.0),
        # ]

        tour_waypoints = [ # 9 point case
            carla.Location( 120.0,  187.0, 0.0),
            carla.Location(271.0, 31.0, 0.0),
            carla.Location(-57.0, 25.0, 0.0),
            carla.Location(120.0, -118.0, 0.0),
            carla.Location(-277.0, 88.0, 0.0),
            carla.Location(-189.0, -118.0, 0.0),
            carla.Location(-161.0, 176.0, 0.0),
            carla.Location(-63.0, -63.0, 0.0),
            carla.Location(266.0, 104.0, 0.0),
        ]

        tour_custom_user_prompt = (
            ""
        )

        # -------------------------------------------------------
        # LLM path execution mode switch
        #   False (default): route_from_sparse -- projected onto the road network via GlobalRoutePlanner
        #   True           : route_from_interpolation -- linear interpolation between LLM nodes, then snapped to lanes
        # -------------------------------------------------------
        use_interpolation_path = True

        # -------------------------------------------------------
        # How path nodes are represented in the LLM prompt (choose one)
        #   "index_list"   : list input
        #                    LLM outputs path = [idx, idx, ...]
        #   "index_matrix" : matrix input
        #                    LLM outputs path = [idx, idx, ...]
        #   "coords"       : use real coords, baseline
        #                    LLM outputs path = [(x, y), (x, y), ...]
        # -------------------------------------------------------
        WAYPOINT_PROMPT_FORMAT = "index_list"

        agent_active  = False
        llm_future    = None
        llm_waiting   = False
        agent2_active = False
        llm_future2   = None
        llm_waiting2  = False

        tour_waypoints_2 = [
            carla.Location(-277.0,  88.0, 0.0),
            carla.Location( 120.0, -118.0, 0.0),
            carla.Location(  50.0,  -60.0, 0.0),
        ]

        multi_dest_1 = carla.Location(-290.0,  -66.0, 0.0)
        multi_dest_2 = carla.Location( -161.0,  176.0, 0.0)

        multi_pool    = None
        multi_future  = None
        multi_waiting = False

        #uni_dest_1      = carla.Location(-184.0,  -65.0, 0.0) # abandoned
        #uni_dest_2      = carla.Location( -278.0,  89.0, 0.0)
        uni_future      = None
        uni_waiting     = False
        uni_retry_count = 0
        MAX_UNI_RETRIES = 20
        uni_messages    = []

        multi_tour_pool    = None
        multi_tour_future  = None
        multi_tour_waiting = False
        _tour_collision_approved = [False]
        _tour_replanning_agent   = [None]
        _tour_agent_loc_paths    = {}
        _tour_agent_paths_lock   = threading.Lock()

        uni_tour_future      = None
        uni_tour_waiting     = False
        uni_tour_retry_count = 0
        MAX_UNI_TOUR_RETRIES = 20
        uni_tour_messages    = []
        uni_tour_total_time  = 0.0

        RUNTIME_COLLISION_DIST_M  = 7.0
        STUCK_SPEED_THRESHOLD     = 0.125
        STUCK_TIMEOUT_TICKS       = int(5.0 / 0.05)
        pending_experiment        = None
        runtime_collision_det     = False
        _exp_vehicles_active      = False
        _stuck_stationary_ticks   = 0
        _pending_auto_reset       = False
        _auto_restart_tour        = False
        _auto_restart_uni_tour    = False

        while True:
            world.tick()
            flags = keyboard.parse_events()
            av = keyboard.active_vehicle

            if flags['quit']:
                break
            if flags['reset'] or _pending_auto_reset:
                for _a in actor_list:
                    if _a is not None and _a.is_alive:
                        _a.destroy()
                actor_list.clear()
                world.tick()

                vehicle = spawn_vehicle(world, blueprint_library, cmap, spawn_pos1)
                if vehicle is None:
                    break
                actor_list.append(vehicle)

                vehicle2 = spawn_vehicle(world, blueprint_library, cmap, spawn_pos2)
                if vehicle2 is None:
                    break
                actor_list.append(vehicle2)
                world.tick()

                agent = BehaviorAgent(vehicle, behavior='normal')
                agent.follow_speed_limits(False)
                agent.set_target_speed(40)

                agent2 = BehaviorAgent(vehicle2, behavior='normal')
                agent2.follow_speed_limits(False)
                agent2.set_target_speed(40)

                agent_active  = False
                agent2_active = False
                llm_future    = None
                llm_waiting   = False
                llm_future2   = None
                llm_waiting2  = False
                multi_pool    = None
                multi_future  = None
                multi_waiting = False
                uni_future      = None
                uni_waiting     = False
                uni_retry_count = 0
                uni_messages    = []
                multi_tour_pool          = None
                multi_tour_future        = None
                multi_tour_waiting       = False
                _tour_collision_approved = [False]
                _tour_replanning_agent   = [None]
                _tour_agent_loc_paths    = {}
                _tour_agent_paths_lock   = threading.Lock()
                uni_tour_future          = None
                uni_tour_waiting         = False
                uni_tour_retry_count     = 0
                uni_tour_messages        = []
                uni_tour_total_time      = 0.0
                pending_experiment        = None
                runtime_collision_det     = False
                _exp_vehicles_active      = False
                _stuck_stationary_ticks   = 0
                _pending_auto_reset       = False

            if flags['destination_changed']:
                dest = keyboard.destination
                color = carla.Color(255, 255, 0) if av == 1 else carla.Color(0, 255, 255)
                world.debug.draw_point(
                    carla.Location(dest.x, dest.y, dest.z + 0.3),
                    size=0.15, color=color, life_time=2.0,
                )

            def _draw_forbidden():
                for fn_idx in forbidden_node_indices(graph, forbidden_areas):
                    fx, fy, fz = graph.coord(fn_idx)
                    world.debug.draw_point(
                        carla.Location(fx, fy, fz + 0.5),
                        size=0.12, color=carla.Color(160, 0, 200), life_time=10.0,
                    )

            cur_vehicle = vehicle  if av == 1 else vehicle2
            cur_agent   = agent    if av == 1 else agent2

            # -- X key: direct BehaviorAgent planning (for the active vehicle) --
            if flags['go']:
                destination_loc = keyboard.destination
                print(f'[Nav-Vehicle{av}] Destination: x={destination_loc.x:.1f}  y={destination_loc.y:.1f}')
                plan = route_from_sparse(
                    vehicle=cur_vehicle,
                    cmap=cmap, grp=grp, world=world, agent=cur_agent,
                    anchors=[cur_vehicle.get_location(), destination_loc],
                    forbidden_areas=forbidden_areas, draw=True,
                )
                if av == 1:
                    agent_active  = bool(plan)
                else:
                    agent2_active = bool(plan)

            # -- L key: LLM single-destination navigation (for the active vehicle) --
            cur_waiting = llm_waiting if av == 1 else llm_waiting2
            if flags['llm_go'] and not cur_waiting:
                llm_dest = keyboard.destination
                print(f'[LLM-Single-Vehicle{av}] Destination: x={llm_dest.x:.1f}  y={llm_dest.y:.1f}')
                _draw_forbidden()
                fut = query_llm_for_navigation_graph(
                    vehicle=cur_vehicle, graph=graph,
                    destination=llm_dest, forbidden_areas=forbidden_areas,
                    prompt_format=WAYPOINT_PROMPT_FORMAT,
                )
                if fut is not None:
                    if av == 1:
                        llm_future  = fut;  llm_waiting  = True
                    else:
                        llm_future2 = fut;  llm_waiting2 = True

            # -- T key: both vehicles submit their own LLM tour plans at once --
            if flags['llm_tour'] and not llm_waiting and not llm_waiting2:
                print(f'[LLM-Tour] Vehicle1 waypoints: {len(tour_waypoints)}  Vehicle2 waypoints: {len(tour_waypoints_2)}')
                for wp in tour_waypoints:
                    world.debug.draw_point(carla.Location(wp.x, wp.y, wp.z + 0.5),
                                           size=0.15, color=carla.Color(255, 165, 0), life_time=15.0)
                for wp in tour_waypoints_2:
                    world.debug.draw_point(carla.Location(wp.x, wp.y, wp.z + 0.5),
                                           size=0.15, color=carla.Color(0, 200, 255), life_time=15.0)
                _draw_forbidden()
                # Submit both async requests in the same frame; they run concurrently in the background
                fut1 = query_llm_for_navigation_graph(
                    vehicle=vehicle,  graph=graph, waypoints=tour_waypoints,
                    forbidden_areas=forbidden_areas,
                    custom_user_prompt=tour_custom_user_prompt or None,
                    prompt_format=WAYPOINT_PROMPT_FORMAT,
                )
                fut2 = query_llm_for_navigation_graph(
                    vehicle=vehicle2, graph=graph, waypoints=tour_waypoints_2,
                    forbidden_areas=forbidden_areas,
                    prompt_format=WAYPOINT_PROMPT_FORMAT,
                )
                if fut1 is not None:
                    llm_future  = fut1;  llm_waiting  = True
                if fut2 is not None:
                    llm_future2 = fut2;  llm_waiting2 = True

            # -- Y key: both vehicles plan via independent LLM agents negotiating through agent_comm to avoid collisions --
            if flags['llm_multi_go'] and not multi_waiting:
                if not _AGENT_COMM_AVAILABLE:
                    print('[LLM-Multi] agent_comm invalid, cannot start Multi-Agent negotiation')
                else:
                    print('[LLM-Multi] Start Multi-Agent Planning Negotiation...')
                    _draw_forbidden()

                    # Build the graph context (both agents share the same graph)
                    forbidden_nodes = forbidden_node_indices(graph, forbidden_areas)

                    if WAYPOINT_PROMPT_FORMAT == "coords":
                        graph_ctx = graph_to_llm_context_geojson_points("custom_junction_graph.geojson")
                        loc1 = vehicle.get_location()
                        loc2 = vehicle2.get_location()
                        if forbidden_areas:
                            fa_desc = [
                                f"({fl.x:.1f}, {fl.y:.1f}) radius {fr:.1f}m"
                                for fl, fr in forbidden_areas
                            ]
                            forbidden_str = f"Forbidden areas (avoid nodes inside): {fa_desc}"
                        else:
                            forbidden_str = "No forbidden areas."
                        start_desc_1 = f"({loc1.x:.1f}, {loc1.y:.1f})"
                        end_desc_1   = f"({multi_dest_1.x:.1f}, {multi_dest_1.y:.1f})"
                        start_desc_2 = f"({loc2.x:.1f}, {loc2.y:.1f})"
                        end_desc_2   = f"({multi_dest_2.x:.1f}, {multi_dest_2.y:.1f})"
                        path_fmt_hint = "path = [(x1, y1), (x2, y2), ..., (xn, yn)]"
                        first_last_rule_1 = (
                            f"First coordinate = {start_desc_1}; "
                            f"last coordinate = {end_desc_1}."
                        )
                        first_last_rule_2 = (
                            f"First coordinate = {start_desc_2}; "
                            f"last coordinate = {end_desc_2}."
                        )
                        print(f'[LLM-Multi-Coords] vehicle1: {start_desc_1} -> {end_desc_1}')
                        print(f'[LLM-Multi-Coords] vehicle2: {start_desc_2} -> {end_desc_2}')
                    else:
                        use_adj_list = (WAYPOINT_PROMPT_FORMAT != "index_matrix")
                        graph_ctx = graph_to_llm_context(graph, use_adj_list=use_adj_list)
                        forbidden_str = (
                            f"Forbidden node indices (must NOT appear in path): {forbidden_nodes}"
                            if forbidden_nodes else "No forbidden nodes."
                        )
                        start_idx_1 = nearest_node(graph, vehicle.get_location())
                        end_idx_1   = nearest_node(graph, multi_dest_1)
                        start_idx_2 = nearest_node(graph, vehicle2.get_location())
                        end_idx_2   = nearest_node(graph, multi_dest_2)
                        start_desc_1 = str(start_idx_1)
                        end_desc_1   = str(end_idx_1)
                        start_desc_2 = str(start_idx_2)
                        end_desc_2   = str(end_idx_2)
                        path_fmt_hint = "path = [<idx>, <idx>, ..., <idx>]"
                        first_last_rule_1 = (
                            f"First index = your start node; last index = your destination node."
                        )
                        first_last_rule_2 = first_last_rule_1
                        print(f'[LLM-Multi] vehicle1: nodes {start_idx_1} -> {end_idx_1}')
                        print(f'[LLM-Multi] vehicle2: nodes {start_idx_2} -> {end_idx_2}')

                    world.debug.draw_point(multi_dest_1 + carla.Location(z=0.5),
                                           size=0.2, color=carla.Color(255, 255, 0), life_time=60.0)
                    world.debug.draw_point(multi_dest_2 + carla.Location(z=0.5),
                                           size=0.2, color=carla.Color(0, 200, 255), life_time=60.0)

                    _multi_sys = (
                        "You are {self_name} (name: '{self_id}'), an autonomous vehicle path planner "
                        "on an abstract road graph. Another agent named '{other_id}' is planning a "
                        "separate route for a different vehicle on the SAME graph simultaneously.\n"
                        "You MUST use the send_message tool to coordinate with {other_id} and avoid "
                        "routing through the same critical nodes at the same time.\n\n"
                        "Negotiation protocol:\n"
                        "  1. Compute your shortest candidate path.\n"
                        "  2. Send {other_id} your candidate path via send_message.\n"
                        "  3. In the next turn you will receive {other_id}'s reply as a user message.\n"
                        "  4. If paths share nodes, negotiate: one agent takes an alternative route.\n"
                        "  5. After agreeing, output EXACTLY one line — no explanation, no code block:\n"
                        f"       {path_fmt_hint}\n\n"
                        "Hard rules:\n"
                        "  - Only use coordinates from the provided junction node list.\n"
                        "  - Never include a forbidden node/area.\n"
                        "  - {first_last_rule}"
                    )

                    system_prompt_1 = _multi_sys.format(
                        self_name="Agent 1", self_id="agent1", other_id="agent2",
                        first_last_rule=first_last_rule_1,
                    )
                    system_prompt_2 = _multi_sys.format(
                        self_name="Agent 2", self_id="agent2", other_id="agent1",
                        first_last_rule=first_last_rule_2,
                    )

                    user_msg_1 = (
                        f"{graph_ctx}\n\n"
                        f"Start: {start_desc_1}\n"
                        f"Destination: {end_desc_1}\n"
                        f"{forbidden_str}\n\n"
                        f"Find the shortest valid path from {start_desc_1} to {end_desc_1}. "
                        f"Coordinate with agent2 first to resolve any shared-node conflicts."
                    )
                    user_msg_2 = (
                        f"{graph_ctx}\n\n"
                        f"Start: {start_desc_2}\n"
                        f"Destination: {end_desc_2}\n"
                        f"{forbidden_str}\n\n"
                        f"Find the shortest valid path from {start_desc_2} to {end_desc_2}. "
                        f"Coordinate with agent1 first to resolve any shared-node conflicts."
                    )

                    _agent_loc_paths: dict = {}
                    _agent_paths_lock = threading.Lock()
                    _collision_approved  = [False]
                    _replanning_agent    = [None]

                    def _locs_to_plan_for_check(locs):
                        plan = []
                        for loc in locs:
                            wp = cmap.get_waypoint(
                                carla.Location(x=loc.x, y=loc.y, z=0),
                                project_to_road=True,
                                lane_type=carla.LaneType.Driving,
                            )
                            if wp is not None:
                                plan.append((wp, RoadOption.LANEFOLLOW))
                        return plan

                    def _collision_interceptor(from_ag: str, to_ag: str, content: str) -> str:
                        """
                        Mailbox interceptor: parse the path proposal in the sender's
                        message, run a collision check against the receiver's last
                        proposal, and append the result to the end of the message.
                        A real collision check only runs when both sides have paths;
                        if either side has not proposed yet, return an explicit PENDING
                        notice to stop the sender from finalizing early.
                        """
                        if WAYPOINT_PROMPT_FORMAT == "coords":
                            from_locs = parse_llm_coord_path(content, silent=True)
                        else:
                            from_locs = parse_llm_node_path(content, graph, silent=True)
                        with _agent_paths_lock:
                            if from_locs:
                                _agent_loc_paths[from_ag] = from_locs
                                _collision_approved[0] = False   # path updated, re-check needed
                            stored_from_locs = list(_agent_loc_paths.get(from_ag, []))
                            to_locs          = list(_agent_loc_paths.get(to_ag, []))
                            cur_replanning   = _replanning_agent[0]

                        if not stored_from_locs:
                            return (
                                "[Real-time Collision Check] PENDING: Your message does not contain a path proposal. "
                                "Please include your proposed path (e.g. path = [n1, n2, ...]) "
                                "and send it to the other agent before finalizing."
                            )

                        if not to_locs:
                            return (
                                f"[Real-time Collision Check] PENDING: {to_ag} has not proposed a path yet. "
                                f"Collision check cannot be performed until {to_ag} submits their path. "
                                f"Please wait for {to_ag}'s response before finalizing your plan."
                            )

                        if not from_locs and cur_replanning is not None and cur_replanning != from_ag:
                            return (
                                f"[Real-time Collision Check] WAITING — {cur_replanning} is still revising their path. "
                                f"Your path is unchanged. Continue holding your current path."
                            )

                        plan_from = _locs_to_plan_for_check(stored_from_locs)
                        plan_to   = _locs_to_plan_for_check(to_locs)
                        if not plan_from or not plan_to:
                            return (
                                "[Real-time Collision Check] WARNING: Could not convert one or both paths to "
                                "waypoints. Please verify your path node indices are valid."
                            )

                        report = check_path_collisions(
                            [plan_from, plan_to],
                            path_names=[from_ag, to_ag],
                        )
                        print("collision interceptor used")
                        if report:
                            with _agent_paths_lock:
                                _collision_approved[0] = False
                                prev = _replanning_agent[0]
                                if prev is None:
                                    _replanning_agent[0] = from_ag
                                elif prev == from_ag:
                                    _replanning_agent[0] = to_ag
                                cur = _replanning_agent[0]
                            if cur == from_ag:
                                return (
                                    f"[Real-time Collision Check] CONFLICT DETECTED — paths NOT approved.\n{report}\n"
                                    f">>> {from_ag}: Please revise your path and resubmit to {to_ag}.\n"
                                    f">>> {to_ag}: Hold your current path. Wait for {from_ag}'s revised proposal before making any changes."
                                )
                            elif prev == from_ag:
                                return (
                                    f"[Real-time Collision Check] CONFLICT STILL DETECTED after {from_ag}'s revision.\n{report}\n"
                                    f">>> {from_ag}: Your revision still conflicts — wait while {to_ag} replans.\n"
                                    f">>> {to_ag}: {from_ag} revised their path but conflict remains. Please revise YOUR path now and resubmit."
                                )
                            else:
                                return (
                                    f"[Real-time Collision Check] CONFLICT — {to_ag} is currently replanning.\n{report}\n"
                                    f">>> {from_ag}: {to_ag} is revising their path based on your update. Please wait.\n"
                                    f">>> {to_ag}: {from_ag} has updated their path. Please revise yours based on this latest proposal."
                                )
                        with _agent_paths_lock:
                            _collision_approved[0] = True
                            _replanning_agent[0] = None
                        return (
                            "[Real-time Collision Check] APPROVED — no collision risk detected. "
                            "Both paths are safe. You may now finalize your plan."
                        )

                    def _finalization_guard(agent_name: str, final_text: str) -> str:
                        with _agent_paths_lock:
                            approved = _collision_approved[0]
                            other = "agent2" if agent_name == "agent1" else "agent1"
                            other_has_path = bool(_agent_loc_paths.get(other))
                            self_has_path  = bool(_agent_loc_paths.get(agent_name))
                            cur_replanning = _replanning_agent[0]

                        if approved:
                            return ""

                        if not self_has_path:
                            return (
                                "You have not yet submitted a path proposal. "
                                f"Please propose a complete path and send it to {other} via send_message "
                                "so that a collision check can be performed."
                            )
                        if not other_has_path:
                            return (
                                f"Collision check is PENDING because {other} has not submitted a path yet. "
                                f"Please send your current path to {other} again and wait for their path proposal "
                                "before finalizing your plan."
                            )
                        if cur_replanning is not None and cur_replanning != agent_name:
                            return (
                                f"{cur_replanning} is currently revising their path to resolve the conflict. "
                                f"Please send a brief acknowledgment to {cur_replanning} via send_message "
                                f"(e.g., 'Acknowledged — holding my current path, waiting for your revision.'). "
                                f"Do NOT include a revised path in that message."
                            )
                        return (
                            "Collision check has NOT been approved yet — your paths may conflict. "
                            f"Please revise your path and resubmit to {other} to resolve all conflicts."
                        )

                    multi_pool = AgentPool(message_interceptor=_collision_interceptor,
                                           finalization_guard=_finalization_guard)
                    multi_pool.add_agent("agent1", system_prompt_1)
                    multi_pool.add_agent("agent2", system_prompt_2)
                    multi_pool.send_task("agent1", user_msg_1)
                    multi_pool.send_task("agent2", user_msg_2)

                    multi_future  = multi_pool.run_all_async()
                    multi_waiting = True
                    print('[LLM-Multi] 2 Agents started; collision check interceptor active.')

            if multi_waiting and multi_future is not None and multi_future.done():
                try:
                    results = multi_future.result(timeout=0)

                    if not _collision_approved[0]:
                        print('[LLM-Multi] WARNING: Agents finished but collision check was never approved. '
                              'Paths will NOT be applied to avoid potential collision.')
                        multi_future  = None
                        multi_waiting = False
                        multi_pool    = None
                        continue

                    print('[LLM-Multi] Negotiation complete and collision check approved, applying routes...')

                    text1 = results.get("agent1", "")
                    text2 = results.get("agent2", "")
                    LLM_DRAW_LT = 600.0

                    if WAYPOINT_PROMPT_FORMAT == "coords":
                        anchors1 = parse_llm_coord_path(text1) if text1 else []
                        anchors2 = parse_llm_coord_path(text2) if text2 else []
                    else:
                        anchors1 = parse_llm_node_path(text1, graph) if text1 else []
                        anchors2 = parse_llm_node_path(text2, graph) if text2 else []

                    if anchors1:
                        if use_interpolation_path:
                            plan1 = route_from_linear(vehicle=vehicle, cmap=cmap,
                                                      world=world, agent=agent,
                                                      anchors=anchors1, draw=True)
                        else:
                            plan1 = route_from_sparse(vehicle=vehicle, cmap=cmap, grp=grp,
                                                      world=world, agent=agent,
                                                      anchors=anchors1,
                                                      forbidden_areas=forbidden_areas, draw=True)
                        for loc in anchors1:
                            world.debug.draw_point(loc + carla.Location(z=0.8), size=0.08,
                                                   color=carla.Color(255, 220, 0),
                                                   life_time=LLM_DRAW_LT)
                        for i in range(len(anchors1) - 1):
                            world.debug.draw_line(
                                anchors1[i] + carla.Location(z=0.8),
                                anchors1[i + 1] + carla.Location(z=0.8),
                                thickness=0.08, color=carla.Color(255, 180, 0),
                                life_time=LLM_DRAW_LT)
                        agent_active = bool(plan1)
                        print(f'[LLM-Multi] vehicle1 Route {"Activated" if agent_active else "Failed"}, '
                              f'{len(plan1) if plan1 else 0} Way Points')
                    else:
                        print('[LLM-Multi] vehicle1 Failed to parse a valide route')

                    if anchors2:
                        if use_interpolation_path:
                            plan2 = route_from_linear(vehicle=vehicle2, cmap=cmap,
                                                      world=world, agent=agent2,
                                                      anchors=anchors2, draw=True)
                        else:
                            plan2 = route_from_sparse(vehicle=vehicle2, cmap=cmap, grp=grp,
                                                      world=world, agent=agent2,
                                                      anchors=anchors2,
                                                      forbidden_areas=forbidden_areas, draw=True)
                        for loc in anchors2:
                            world.debug.draw_point(loc + carla.Location(z=0.8), size=0.08,
                                                   color=carla.Color(0, 210, 255),
                                                   life_time=LLM_DRAW_LT)
                        for i in range(len(anchors2) - 1):
                            world.debug.draw_line(
                                anchors2[i] + carla.Location(z=0.8),
                                anchors2[i + 1] + carla.Location(z=0.8),
                                thickness=0.08, color=carla.Color(0, 180, 255),
                                life_time=LLM_DRAW_LT)
                        agent2_active = bool(plan2)
                        print(f'[LLM-Multi] vehicle2 Route {"Activated" if agent2_active else "Failed"}, '
                              f'{len(plan2) if plan2 else 0} Way Points')
                    else:
                        print('[LLM-Multi] vehicle2 failed to parse a valide route')

                except Exception as e:
                    print(f'[LLM-Multi] Error in negotiation as: {e}')

                multi_future  = None
                multi_waiting = False
                multi_pool    = None

            if flags['llm_unified_go'] and not uni_waiting:
                if not _LLM_AVAILABLE:
                    print('[Unified Planning] LLM invalide')
                else:
                    print('[Unified Planning] Starting Unified LLM Planning...')
                    _draw_forbidden()

                    forbidden_nodes = forbidden_node_indices(graph, forbidden_areas)

                    if WAYPOINT_PROMPT_FORMAT == "coords":
                        graph_ctx = graph_to_llm_context_geojson_points("custom_junction_graph.geojson")
                        loc1 = vehicle.get_location()
                        loc2 = vehicle2.get_location()
                        if forbidden_areas:
                            fa_desc = [
                                f"({fl.x:.1f}, {fl.y:.1f}) radius {fr:.1f}m"
                                for fl, fr in forbidden_areas
                            ]
                            forbidden_str = f"Forbidden areas (avoid nodes inside): {fa_desc}"
                        else:
                            forbidden_str = "No forbidden areas."
                        v1_start = f"({loc1.x:.1f}, {loc1.y:.1f})"
                        v1_end   = f"({multi_dest_1.x:.1f}, {multi_dest_1.y:.1f})"
                        v2_start = f"({loc2.x:.1f}, {loc2.y:.1f})"
                        v2_end   = f"({multi_dest_2.x:.1f}, {multi_dest_2.y:.1f})"
                        print(f'[Unified Planning-Coords] vehicle1: {v1_start} -> {v1_end}')
                        print(f'[Unified Planning-Coords] vehicle2: {v2_start} -> {v2_end}')
                        system_msg = (
                            "You are an autonomous vehicle path planner working on a road graph "
                            "where every node is identified by its real-world (x, y) coordinate.\n"
                            "Plan paths for TWO vehicles simultaneously so they avoid physical collision.\n\n"
                            "Output EXACTLY two lines — no explanation, no code block:\n"
                            "  path1 = [(x1, y1), (x2, y2), ..., (xn, yn)]\n"
                            "  path2 = [(x1, y1), (x2, y2), ..., (xn, yn)]\n\n"
                            "Rules:\n"
                            "  - Only use coordinates from the provided junction node list.\n"
                            "  - Avoid routing through any forbidden area.\n"
                            "  - path1: first coordinate = vehicle1 start, last = vehicle1 destination.\n"
                            "  - path2: first coordinate = vehicle2 start, last = vehicle2 destination.\n"
                            "  - Minimise the number of time steps where both vehicles occupy spatially close positions."
                        )
                        user_msg = (
                            f"{graph_ctx}\n\n"
                            f"Vehicle 1 — Start: {v1_start}, Destination: {v1_end}\n"
                            f"Vehicle 2 — Start: {v2_start}, Destination: {v2_end}\n"
                            f"{forbidden_str}\n\n"
                            f"Plan both paths. Make sure the vehicles do not converge at the same positions at the same time step."
                        )
                    else:
                        use_adj_list = (WAYPOINT_PROMPT_FORMAT != "index_matrix")
                        graph_ctx = graph_to_llm_context(graph, use_adj_list=use_adj_list)
                        forbidden_str = (
                            f"Forbidden node indices (must NOT appear in either path): {forbidden_nodes}"
                            if forbidden_nodes else "No forbidden nodes."
                        )
                        start_idx_1 = nearest_node(graph, vehicle.get_location())
                        end_idx_1   = nearest_node(graph, multi_dest_1)
                        start_idx_2 = nearest_node(graph, vehicle2.get_location())
                        end_idx_2   = nearest_node(graph, multi_dest_2)
                        print(f'[Unified Planning] vehicle1: node {start_idx_1} -> {end_idx_1}')
                        print(f'[Unified Planning] vehicle2: node {start_idx_2} -> {end_idx_2}')
                        system_msg = (
                            "You are an autonomous vehicle path planner working on an abstract road graph.\n"
                            "Plan paths for TWO vehicles simultaneously so they avoid physical collision.\n"
                            "Nodes are identified by integer indices only — no coordinates.\n\n"
                            "Output EXACTLY two lines — no explanation, no code block:\n"
                            "  path1 = [<idx>, <idx>, ..., <idx>]\n"
                            "  path2 = [<idx>, <idx>, ..., <idx>]\n\n"
                            "Rules:\n"
                            "  - Only traverse edges listed in the adjacency list.\n"
                            "  - Never include a forbidden node in either path.\n"
                            "  - path1: first index = vehicle1 start, last = vehicle1 destination.\n"
                            "  - path2: first index = vehicle2 start, last = vehicle2 destination.\n"
                            "  - Minimise the number of time steps where both vehicles occupy spatially close nodes."
                        )
                        user_msg = (
                            f"{graph_ctx}\n\n"
                            f"Vehicle 1 — Start node: {start_idx_1}, Destination node: {end_idx_1}\n"
                            f"Vehicle 2 — Start node: {start_idx_2}, Destination node: {end_idx_2}\n"
                            f"{forbidden_str}\n\n"
                            f"Plan both paths. Make sure the vehicles do not converge on the same nodes at the same time step."
                        )

                    world.debug.draw_point(multi_dest_1 + carla.Location(z=0.5),
                                           size=0.2, color=carla.Color(255, 255, 0), life_time=60.0)
                    world.debug.draw_point(multi_dest_2 + carla.Location(z=0.5),
                                           size=0.2, color=carla.Color(0, 200, 255), life_time=60.0)

                    uni_messages    = [
                        {"role": "system", "content": system_msg},
                        {"role": "user",   "content": user_msg},
                    ]
                    uni_future      = _llmutils.query_llm_async(uni_messages, seed=0)
                    uni_waiting     = True
                    uni_retry_count = 0
                    print('[Unified Planning] Request sent, waiting for response...')

            if uni_waiting and uni_future is not None and uni_future.done():
                _reset_uni = True
                try:
                    text, elapsed = uni_future.result(timeout=0)
                    print(f'[Unified Planning] LLM replies in {elapsed:.2f}s, checking collision...')

                    if WAYPOINT_PROMPT_FORMAT == "coords":
                        locs1, locs2 = parse_unified_llm_paths_coords(text)
                    else:
                        locs1, locs2 = parse_unified_llm_paths(text, graph)

                    def _to_wp_plan(locs):
                        plan = []
                        for loc in locs:
                            wp = cmap.get_waypoint(
                                carla.Location(x=loc.x, y=loc.y, z=0),
                                project_to_road=True,
                                lane_type=carla.LaneType.Driving,
                            )
                            if wp is not None:
                                plan.append((wp, RoadOption.LANEFOLLOW))
                        return plan

                    plan1_chk = _to_wp_plan(locs1) if locs1 else []
                    plan2_chk = _to_wp_plan(locs2) if locs2 else []

                    collision_report = ""
                    if plan1_chk and plan2_chk:
                        collision_report = check_path_collisions(
                            [plan1_chk, plan2_chk],
                            path_names=["vehicle1", "vehicle2"],
                        )
                        if collision_report:
                            print(f'[Unified Planning] Collision test result:\n{collision_report}')
                        else:
                            print('[Unified Planning] Collision test passed')
                    else:
                        print('[Unified Planning] Incompleted Route')

                    if collision_report and uni_retry_count < MAX_UNI_RETRIES:
                        uni_retry_count += 1
                        print(f'[Unified Planning] {uni_retry_count}/{MAX_UNI_RETRIES} th iteration...')
                        uni_messages.append({"role": "assistant", "content": text})
                        if WAYPOINT_PROMPT_FORMAT == "coords":
                            _retry_fmt = (
                                "  path1 = [(x1, y1), ..., (xn, yn)]\n"
                                "  path2 = [(x1, y1), ..., (xn, yn)]"
                            )
                        else:
                            _retry_fmt = (
                                "  path1 = [<idx>, <idx>, ..., <idx>]\n"
                                "  path2 = [<idx>, <idx>, ..., <idx>]"
                            )
                        uni_messages.append({"role": "user", "content": (
                            f"The paths you proposed have collision risks detected by a "
                            f"physics-level waypoint check:\n\n{collision_report}\n\n"
                            f"Please revise both paths to eliminate all listed conflicts.\n"
                            f"Output EXACTLY two lines — no explanation:\n"
                            f"{_retry_fmt}"
                        )})
                        uni_future  = _llmutils.query_llm_async(uni_messages, seed=uni_retry_count)
                        _reset_uni  = False

                    else:
                        if collision_report:
                            print(f'[Unified Planning] Reach max iterations as {MAX_UNI_RETRIES}, forcing applying')
                        else:
                            print('[Unified Planning] Collision test passed, applying...')

                        LLM_DRAW_LT = 600.0

                        if locs1:
                            if use_interpolation_path:
                                plan1 = route_from_linear(vehicle=vehicle, cmap=cmap,
                                                          world=world, agent=agent,
                                                          anchors=locs1, draw=True)
                            else:
                                plan1 = route_from_sparse(vehicle=vehicle, cmap=cmap, grp=grp,
                                                          world=world, agent=agent, anchors=locs1,
                                                          forbidden_areas=forbidden_areas, draw=True)
                            for loc in locs1:
                                world.debug.draw_point(loc + carla.Location(z=0.8), size=0.08,
                                                       color=carla.Color(255, 220, 0),
                                                       life_time=LLM_DRAW_LT)
                            for i in range(len(locs1) - 1):
                                world.debug.draw_line(
                                    locs1[i] + carla.Location(z=0.8),
                                    locs1[i + 1] + carla.Location(z=0.8),
                                    thickness=0.08, color=carla.Color(255, 180, 0),
                                    life_time=LLM_DRAW_LT)
                            agent_active = bool(plan1)
                            print(f'[Unified Planning] vehicle1 path {"activated" if agent_active else "failed"}, '
                                  f'{len(plan1) if plan1 else 0} waypoints in total.')
                        else:
                            print('[Unified Planning] vehicle1 failed to parse a valid route.')

                        if locs2:
                            if use_interpolation_path:
                                plan2 = route_from_linear(vehicle=vehicle2, cmap=cmap,
                                                          world=world, agent=agent2,
                                                          anchors=locs2, draw=True)
                            else:
                                plan2 = route_from_sparse(vehicle=vehicle2, cmap=cmap, grp=grp,
                                                          world=world, agent=agent2, anchors=locs2,
                                                          forbidden_areas=forbidden_areas, draw=True)
                            for loc in locs2:
                                world.debug.draw_point(loc + carla.Location(z=0.8), size=0.08,
                                                       color=carla.Color(0, 210, 255),
                                                       life_time=LLM_DRAW_LT)
                            for i in range(len(locs2) - 1):
                                world.debug.draw_line(
                                    locs2[i] + carla.Location(z=0.8),
                                    locs2[i + 1] + carla.Location(z=0.8),
                                    thickness=0.08, color=carla.Color(0, 180, 255),
                                    life_time=LLM_DRAW_LT)
                            agent2_active = bool(plan2)
                            print(f'[Unified Planning] vehicle2 path {"activated" if agent2_active else "failed"}, '
                                  f'{len(plan2) if plan2 else 0} waypoints in total.')
                        else:
                            print('[Unified Planning] vehicle2 failed to parse a valid route.')

                except Exception as e:
                    print(f'[Unified Planning] error: {e}')

                if _reset_uni:
                    uni_future      = None
                    uni_waiting     = False
                    uni_retry_count = 0
                    uni_messages    = []

            if (flags['llm_multi_tour'] or _auto_restart_tour) and not multi_tour_waiting:
                _auto_restart_tour = False
                if not _AGENT_COMM_AVAILABLE:
                    print('[LLM-MultiTour] agent_comm invalid, cannot start Multi-Agent tour')
                else:
                    print('[LLM-MultiTour] Starting Multi-Agent shared-waypoint tour...')
                    _draw_forbidden()

                    forbidden_nodes = forbidden_node_indices(graph, forbidden_areas)

                    for wp in tour_waypoints:
                        world.debug.draw_point(carla.Location(wp.x, wp.y, wp.z + 0.5),
                                               size=0.15, color=carla.Color(255, 165, 0), life_time=30.0)

                    if WAYPOINT_PROMPT_FORMAT == "coords":
                        graph_ctx = graph_to_llm_context_geojson_points("custom_junction_graph.geojson")
                        loc1 = vehicle.get_location()
                        loc2 = vehicle2.get_location()
                        _snap_idx1 = nearest_node(graph, loc1)
                        _snap_idx2 = nearest_node(graph, loc2)
                        _sx1, _sy1, _ = graph.coord(_snap_idx1)
                        _sx2, _sy2, _ = graph.coord(_snap_idx2)
                        _snapped_tour = []
                        for _twp in tour_waypoints:
                            _ti = nearest_node(graph, _twp)
                            _tx, _ty, _ = graph.coord(_ti)
                            _snapped_tour.append(f"({_tx:.1f}, {_ty:.1f})")
                        wp_coords = _snapped_tour
                        if forbidden_areas:
                            fa_desc = [
                                f"({fl.x:.1f}, {fl.y:.1f}) radius {fr:.1f}m"
                                for fl, fr in forbidden_areas
                            ]
                            forbidden_str = f"Forbidden areas (avoid nodes inside): {fa_desc}"
                        else:
                            forbidden_str = "No forbidden areas."
                        start_desc_1 = f"({_sx1:.1f}, {_sy1:.1f})"
                        start_desc_2 = f"({_sx2:.1f}, {_sy2:.1f})"
                        path_fmt_hint = "path = [(x1, y1), (x2, y2), ..., (xn, yn)]"
                        first_last_rule = "First and last coordinate must both equal your vehicle's start coordinate."
                        wp_desc_1 = f"required waypoint coordinates ({len(tour_waypoints)} total): {wp_coords}"
                        wp_desc_2 = wp_desc_1
                        print(f'[LLM-MultiTour-Coords] v1 actual: ({loc1.x:.1f}, {loc1.y:.1f}) -> snapped to node {_snap_idx1}: {start_desc_1}')
                        print(f'[LLM-MultiTour-Coords] v2 actual: ({loc2.x:.1f}, {loc2.y:.1f}) -> snapped to node {_snap_idx2}: {start_desc_2}')
                        print(f'[LLM-MultiTour-Coords] shared waypoints: {wp_coords}')
                    else:
                        use_adj_list = (WAYPOINT_PROMPT_FORMAT != "index_matrix")
                        graph_ctx = graph_to_llm_context(graph, use_adj_list=use_adj_list)
                        forbidden_str = (
                            f"Forbidden node indices (must NOT appear in path): {forbidden_nodes}"
                            if forbidden_nodes else "No forbidden nodes."
                        )
                        start_idx_1 = nearest_node(graph, vehicle.get_location())
                        start_idx_2 = nearest_node(graph, vehicle2.get_location())
                        tour_wp_nodes = [nearest_node(graph, wp) for wp in tour_waypoints]
                        start_desc_1 = str(start_idx_1)
                        start_desc_2 = str(start_idx_2)
                        path_fmt_hint = "path = [<idx>, <idx>, ..., <idx>]"
                        first_last_rule = "First and last index must both equal your start node."
                        wp_desc_1 = f"required waypoint nodes ({len(tour_wp_nodes)} total): {tour_wp_nodes}"
                        wp_desc_2 = wp_desc_1
                        print(f'[LLM-MultiTour] v1 start: node {start_idx_1}, v2 start: node {start_idx_2}')
                        print(f'[LLM-MultiTour] shared waypoint nodes: {tour_wp_nodes}')

                    _multi_tour_sys = (
                        "You are {self_name} (name: '{self_id}'), an autonomous vehicle path planner "
                        "on a road graph. Another agent named '{other_id}' is planning a separate "
                        "round-trip tour for a different vehicle on the SAME graph simultaneously.\n"
                        "BOTH vehicles must visit ALL of the listed waypoints, but may visit them in any order.\n"
                        "You MUST use the send_message tool to coordinate with {other_id} and avoid "
                        "routing through the same critical nodes at the same time.\n\n"
                        "Negotiation protocol:\n"
                        "  1. Plan your shortest round-trip candidate path visiting all required waypoints.\n"
                        "  2. Send {other_id} your candidate path via send_message.\n"
                        "  3. In the next turn you will receive {other_id}'s reply as a user message.\n"
                        "  4. If paths share nodes at the same time steps, negotiate: adjust visit order or detour.\n"
                        "  5. After agreeing, output EXACTLY one line — no explanation, no code block:\n"
                        "       {path_fmt_hint}\n\n"
                        "Hard rules:\n"
                        "  - Only use nodes/coordinates from the provided junction node list.\n"
                        "  - Never include a forbidden node/area.\n"
                        "  - Your path must start AND end at your vehicle's start position.\n"
                        "  - All required waypoints must appear in your path.\n"
                        "  - When changing path for collision avoidance, move some of the current points to a different place instead of wander between points if could.\n"
                        "  - {first_last_rule}"
                    )

                    system_prompt_1 = _multi_tour_sys.format(
                        self_name="Agent 1", self_id="agent1", other_id="agent2",
                        path_fmt_hint=path_fmt_hint, first_last_rule=first_last_rule,
                    )
                    system_prompt_2 = _multi_tour_sys.format(
                        self_name="Agent 2", self_id="agent2", other_id="agent1",
                        path_fmt_hint=path_fmt_hint, first_last_rule=first_last_rule,
                    )

                    user_msg_1 = (
                        f"{graph_ctx}\n\n"
                        f"Vehicle start position: {start_desc_1}\n"
                        f"Shared {wp_desc_1}\n"
                        f"{forbidden_str}\n\n"
                        f"Plan the shortest round-trip from {start_desc_1} visiting all required waypoints "
                        f"and returning to {start_desc_1}. "
                        f"Coordinate with agent2 first to resolve any shared-node conflicts."
                    )
                    user_msg_2 = (
                        f"{graph_ctx}\n\n"
                        f"Vehicle start position: {start_desc_2}\n"
                        f"Shared {wp_desc_2}\n"
                        f"{forbidden_str}\n\n"
                        f"Plan the shortest round-trip from {start_desc_2} visiting all required waypoints "
                        f"and returning to {start_desc_2}. "
                        f"Coordinate with agent1 first to resolve any shared-node conflicts."
                    )

                    _tour_agent_loc_paths    = {}
                    _tour_agent_paths_lock   = threading.Lock()
                    _tour_collision_approved = [False]
                    _tour_replanning_agent   = [None]

                    def _locs_to_plan_tour(locs):
                        plan = []
                        for loc in locs:
                            wp = cmap.get_waypoint(
                                carla.Location(x=loc.x, y=loc.y, z=0),
                                project_to_road=True,
                                lane_type=carla.LaneType.Driving,
                            )
                            if wp is not None:
                                plan.append((wp, RoadOption.LANEFOLLOW))
                        return plan

                    def _tour_collision_interceptor(from_ag: str, to_ag: str, content: str) -> str:
                        if WAYPOINT_PROMPT_FORMAT == "coords":
                            from_locs = parse_llm_coord_path(content, silent=True)
                        else:
                            from_locs = parse_llm_node_path(content, graph, silent=True)
                        with _tour_agent_paths_lock:
                            if from_locs:
                                _tour_agent_loc_paths[from_ag] = from_locs
                                _tour_collision_approved[0] = False
                            stored_from_locs = list(_tour_agent_loc_paths.get(from_ag, []))
                            to_locs          = list(_tour_agent_loc_paths.get(to_ag, []))
                            cur_replanning   = _tour_replanning_agent[0]

                        if not stored_from_locs:
                            return (
                                "[Real-time Collision Check] PENDING: Your message does not contain a path proposal. "
                                "Please include your proposed path and send it to the other agent before finalizing."
                            )
                        if not to_locs:
                            return (
                                f"[Real-time Collision Check] PENDING: {to_ag} has not proposed a path yet. "
                                f"Please wait for {to_ag}'s response before finalizing your plan."
                            )

                        if not from_locs and cur_replanning is not None and cur_replanning != from_ag:
                            return (
                                f"[Real-time Collision Check] WAITING — {cur_replanning} is still revising their path. "
                                f"Your path is unchanged. Continue holding your current path."
                            )

                        plan_from = _locs_to_plan_tour(stored_from_locs)
                        plan_to   = _locs_to_plan_tour(to_locs)
                        if not plan_from or not plan_to:
                            return (
                                "[Real-time Collision Check] WARNING: Could not convert one or both paths to waypoints."
                            )

                        report = check_path_collisions(
                            [plan_from, plan_to],
                            path_names=[from_ag, to_ag],
                        )
                        print("[MultiTour] collision interceptor used")
                        if report:
                            with _tour_agent_paths_lock:
                                _tour_collision_approved[0] = False
                                prev = _tour_replanning_agent[0]
                                if prev is None:
                                    _tour_replanning_agent[0] = from_ag
                                elif prev == from_ag:
                                    _tour_replanning_agent[0] = to_ag
                                cur = _tour_replanning_agent[0]
                            if cur == from_ag:
                                return (
                                    f"[Real-time Collision Check] CONFLICT DETECTED — paths NOT approved.\n{report}\n"
                                    f">>> {from_ag}: Please revise your path and resubmit to {to_ag}.\n"
                                    f">>> {to_ag}: Hold your current path. Wait for {from_ag}'s revised proposal before making any changes."
                                )
                            elif prev == from_ag:
                                return (
                                    f"[Real-time Collision Check] CONFLICT STILL DETECTED after {from_ag}'s revision.\n{report}\n"
                                    f">>> {from_ag}: Your revision still conflicts — wait while {to_ag} replans.\n"
                                    f">>> {to_ag}: {from_ag} revised their path but conflict remains. Please revise YOUR path now and resubmit."
                                )
                            else:
                                return (
                                    f"[Real-time Collision Check] CONFLICT — {to_ag} is currently replanning.\n{report}\n"
                                    f">>> {from_ag}: {to_ag} is revising their path based on your update. Please wait.\n"
                                    f">>> {to_ag}: {from_ag} has updated their path. Please revise yours based on this latest proposal."
                                )
                        with _tour_agent_paths_lock:
                            _tour_collision_approved[0] = True
                            _tour_replanning_agent[0] = None
                        return (
                            "[Real-time Collision Check] APPROVED — no collision risk detected. "
                            "Both paths are safe. You may now finalize your plan."
                        )

                    def _tour_finalization_guard(agent_name: str, final_text: str) -> str:
                        with _tour_agent_paths_lock:
                            approved = _tour_collision_approved[0]
                            other = "agent2" if agent_name == "agent1" else "agent1"
                            other_has_path = bool(_tour_agent_loc_paths.get(other))
                            self_has_path  = bool(_tour_agent_loc_paths.get(agent_name))
                            cur_replanning = _tour_replanning_agent[0]

                        if approved:
                            return ""
                        if not self_has_path:
                            return (
                                "You have not yet submitted a path proposal. "
                                f"Please propose a complete round-trip path and send it to {other} via send_message."
                            )
                        if not other_has_path:
                            return (
                                f"Collision check is PENDING because {other} has not submitted a path yet. "
                                f"Please resend your path to {other} and wait for their proposal."
                            )
                        if cur_replanning is not None and cur_replanning != agent_name:
                            return (
                                f"{cur_replanning} is currently revising their path to resolve the conflict. "
                                f"Please send a brief acknowledgment to {cur_replanning} via send_message "
                                f"(e.g., 'Acknowledged — holding my current path, waiting for your revision.'). "
                                f"Do NOT include a revised path in that message."
                            )
                        return (
                            "Collision check has NOT been approved yet — your paths may conflict. "
                            f"Please revise your path and resubmit to {other} to resolve all conflicts."
                        )

                    multi_tour_pool = AgentPool(
                        message_interceptor=_tour_collision_interceptor,
                        finalization_guard=_tour_finalization_guard,
                    )
                    multi_tour_pool.add_agent("agent1", system_prompt_1)
                    multi_tour_pool.add_agent("agent2", system_prompt_2)
                    multi_tour_pool.send_task("agent1", user_msg_1)
                    multi_tour_pool.send_task("agent2", user_msg_2)

                    multi_tour_future  = multi_tour_pool.run_all_async()
                    multi_tour_waiting = True
                    print('[LLM-MultiTour] 2 Agents started; collision check interceptor active.')

            if multi_tour_waiting and multi_tour_future is not None and multi_tour_future.done():
                try:
                    results = multi_tour_future.result(timeout=0)

                    if not _tour_collision_approved[0]:
                        print('[LLM-MultiTour] WARNING: Agents finished but collision check was never approved. '
                              'Paths will NOT be applied.')
                        multi_tour_future  = None
                        multi_tour_waiting = False
                        multi_tour_pool    = None
                        continue

                    print('[LLM-MultiTour] Negotiation complete and collision approved, applying routes...')
                    text1 = results.get("agent1", "")
                    text2 = results.get("agent2", "")
                    LLM_DRAW_LT = 600.0
                    plan1, plan2 = [], []

                    if WAYPOINT_PROMPT_FORMAT == "coords":
                        anchors1 = parse_llm_coord_path(text1) if text1 else []
                        anchors2 = parse_llm_coord_path(text2) if text2 else []
                    else:
                        anchors1 = parse_llm_node_path(text1, graph) if text1 else []
                        anchors2 = parse_llm_node_path(text2, graph) if text2 else []

                    if anchors1:
                        if use_interpolation_path:
                            plan1 = route_from_linear(vehicle=vehicle, cmap=cmap,
                                                      world=world, agent=agent,
                                                      anchors=anchors1, draw=True)
                        else:
                            plan1 = route_from_sparse(vehicle=vehicle, cmap=cmap, grp=grp,
                                                      world=world, agent=agent, anchors=anchors1,
                                                      forbidden_areas=forbidden_areas, draw=True)
                        for loc in anchors1:
                            world.debug.draw_point(loc + carla.Location(z=0.8), size=0.08,
                                                   color=carla.Color(255, 220, 0), life_time=LLM_DRAW_LT)
                        for i in range(len(anchors1) - 1):
                            world.debug.draw_line(
                                anchors1[i] + carla.Location(z=0.8),
                                anchors1[i + 1] + carla.Location(z=0.8),
                                thickness=0.08, color=carla.Color(255, 180, 0), life_time=LLM_DRAW_LT)
                        agent_active = bool(plan1)
                        print(f'[LLM-MultiTour] vehicle1 route {"activated" if agent_active else "failed"}, '
                              f'{len(plan1) if plan1 else 0} waypoints')
                    else:
                        print('[LLM-MultiTour] vehicle1 failed to parse a valid route')

                    if anchors2:
                        if use_interpolation_path:
                            plan2 = route_from_linear(vehicle=vehicle2, cmap=cmap,
                                                      world=world, agent=agent2,
                                                      anchors=anchors2, draw=True)
                        else:
                            plan2 = route_from_sparse(vehicle=vehicle2, cmap=cmap, grp=grp,
                                                      world=world, agent=agent2, anchors=anchors2,
                                                      forbidden_areas=forbidden_areas, draw=True)
                        for loc in anchors2:
                            world.debug.draw_point(loc + carla.Location(z=0.8), size=0.08,
                                                   color=carla.Color(0, 210, 255), life_time=LLM_DRAW_LT)
                        for i in range(len(anchors2) - 1):
                            world.debug.draw_line(
                                anchors2[i] + carla.Location(z=0.8),
                                anchors2[i + 1] + carla.Location(z=0.8),
                                thickness=0.08, color=carla.Color(0, 180, 255), life_time=LLM_DRAW_LT)
                        agent2_active = bool(plan2)
                        print(f'[LLM-MultiTour] vehicle2 route {"activated" if agent2_active else "failed"}, '
                              f'{len(plan2) if plan2 else 0} waypoints')
                    else:
                        print('[LLM-MultiTour] vehicle2 failed to parse a valid route')

                    _mt_stats  = multi_tour_pool.get_stats()
                    _mt_iters  = max(s["iterations"] for s in _mt_stats.values()) if _mt_stats else 0
                    _mt_time   = (sum(s["total_llm_time"] for s in _mt_stats.values()) / len(_mt_stats)
                                  ) if _mt_stats else 0.0
                    pending_experiment = {
                        "task":         "llm_multi_tour",
                        "iterations":   _mt_iters,
                        "avg_llm_time": _mt_time,
                        "path1_length": calc_path_length_m(plan1),
                        "path2_length": calc_path_length_m(plan2),
                    }
                    runtime_collision_det = False
                    _exp_vehicles_active  = False
                    print(f'[Experiment] multi_tour data ready: iters={_mt_iters}, '
                          f'avg_llm_time={_mt_time:.2f}s, '
                          f'path1={calc_path_length_m(plan1):.1f}m, '
                          f'path2={calc_path_length_m(plan2):.1f}m')

                except Exception as e:
                    print(f'[LLM-MultiTour] Error: {e}')

                multi_tour_future  = None
                multi_tour_waiting = False
                multi_tour_pool    = None

            if (flags['llm_unified_tour'] or _auto_restart_uni_tour) and not uni_tour_waiting:
                _auto_restart_uni_tour = False
                if not _LLM_AVAILABLE:
                    print('[UnifiedTour] LLM invalid')
                else:
                    print('[UnifiedTour] Starting Unified LLM shared-waypoint tour planning...')
                    _draw_forbidden()

                    forbidden_nodes = forbidden_node_indices(graph, forbidden_areas)

                    for wp in tour_waypoints:
                        world.debug.draw_point(carla.Location(wp.x, wp.y, wp.z + 0.5),
                                               size=0.15, color=carla.Color(255, 165, 0), life_time=30.0)

                    if WAYPOINT_PROMPT_FORMAT == "coords":
                        graph_ctx = graph_to_llm_context_geojson_points("custom_junction_graph.geojson")
                        loc1 = vehicle.get_location()
                        loc2 = vehicle2.get_location()
                        _snap_idx1 = nearest_node(graph, loc1)
                        _snap_idx2 = nearest_node(graph, loc2)
                        _sx1, _sy1, _ = graph.coord(_snap_idx1)
                        _sx2, _sy2, _ = graph.coord(_snap_idx2)
                        _snapped_tour = []
                        for _twp in tour_waypoints:
                            _ti = nearest_node(graph, _twp)
                            _tx, _ty, _ = graph.coord(_ti)
                            _snapped_tour.append(f"({_tx:.1f}, {_ty:.1f})")
                        wp_coords = _snapped_tour
                        if forbidden_areas:
                            fa_desc = [
                                f"({fl.x:.1f}, {fl.y:.1f}) radius {fr:.1f}m"
                                for fl, fr in forbidden_areas
                            ]
                            forbidden_str = f"Forbidden areas (avoid nodes inside): {fa_desc}"
                        else:
                            forbidden_str = "No forbidden areas."
                        v1_start = f"({_sx1:.1f}, {_sy1:.1f})"
                        v2_start = f"({_sx2:.1f}, {_sy2:.1f})"
                        print(f'[UnifiedTour-Coords] v1 actual: ({loc1.x:.1f}, {loc1.y:.1f}) -> snapped to node {_snap_idx1}: {v1_start}')
                        print(f'[UnifiedTour-Coords] v2 actual: ({loc2.x:.1f}, {loc2.y:.1f}) -> snapped to node {_snap_idx2}: {v2_start}')
                        print(f'[UnifiedTour-Coords] shared waypoints (snapped): {wp_coords}')
                        system_msg = (
                            "You are an autonomous vehicle path planner working on a road graph "
                            "where every node is identified by its real-world (x, y) coordinate.\n"
                            "Plan round-trip tours for TWO vehicles simultaneously so they avoid physical collision.\n"
                            "BOTH vehicles must visit ALL of the listed waypoints, but may visit them in any order.\n\n"
                            "Output EXACTLY two lines — no explanation, no code block:\n"
                            "  path1 = [(x1, y1), (x2, y2), ..., (xn, yn)]\n"
                            "  path2 = [(x1, y1), (x2, y2), ..., (xn, yn)]\n\n"
                            "Rules:\n"
                            "  - Only use coordinates from the provided junction node list.\n"
                            "  - Avoid routing through any forbidden area.\n"
                            "  - path1: first and last coordinate = vehicle1 start coordinate (round trip).\n"
                            "  - path2: first and last coordinate = vehicle2 start coordinate (round trip).\n"
                            "  - Both paths must include ALL required waypoints.\n"
                            "  - Minimise the number of time steps where both vehicles occupy spatially close positions."
                        )
                        user_msg = (
                            f"{graph_ctx}\n\n"
                            f"Vehicle 1 — Start (= return): {v1_start}\n"
                            f"Vehicle 2 — Start (= return): {v2_start}\n"
                            f"Required waypoints (both vehicles, {len(tour_waypoints)} total): {wp_coords}\n"
                            f"{forbidden_str}\n\n"
                            f"Plan both round-trip tours visiting all required waypoints. "
                            f"Make sure the vehicles do not converge at the same positions at the same time step."
                        )
                    else:
                        use_adj_list = (WAYPOINT_PROMPT_FORMAT != "index_matrix")
                        graph_ctx = graph_to_llm_context(graph, use_adj_list=use_adj_list)
                        forbidden_str = (
                            f"Forbidden node indices (must NOT appear in either path): {forbidden_nodes}"
                            if forbidden_nodes else "No forbidden nodes."
                        )
                        start_idx_1 = nearest_node(graph, vehicle.get_location())
                        start_idx_2 = nearest_node(graph, vehicle2.get_location())
                        tour_wp_nodes = [nearest_node(graph, wp) for wp in tour_waypoints]
                        print(f'[UnifiedTour] v1 start: node {start_idx_1}, v2 start: node {start_idx_2}')
                        print(f'[UnifiedTour] shared waypoint nodes: {tour_wp_nodes}')
                        system_msg = (
                            "You are an autonomous vehicle path planner working on an abstract road graph.\n"
                            "Plan round-trip tours for TWO vehicles simultaneously so they avoid physical collision.\n"
                            "Nodes are identified by integer indices only — no coordinates.\n"
                            "BOTH vehicles must visit ALL of the listed waypoints, but may visit them in any order.\n\n"
                            "Output EXACTLY two lines — no explanation, no code block:\n"
                            "  path1 = [<idx>, <idx>, ..., <idx>]\n"
                            "  path2 = [<idx>, <idx>, ..., <idx>]\n\n"
                            "Rules:\n"
                            "  - Only traverse edges listed in the adjacency list.\n"
                            "  - Never include a forbidden node in either path.\n"
                            "  - path1: first and last index = vehicle1 start node (round trip).\n"
                            "  - path2: first and last index = vehicle2 start node (round trip).\n"
                            "  - Both paths must visit ALL required waypoint nodes.\n"
                            "  - Minimise the number of time steps where both vehicles occupy spatially close nodes."
                        )
                        user_msg = (
                            f"{graph_ctx}\n\n"
                            f"Vehicle 1 — Start node (= return): {start_idx_1}\n"
                            f"Vehicle 2 — Start node (= return): {start_idx_2}\n"
                            f"Required waypoint nodes (both vehicles, {len(tour_wp_nodes)} total): {tour_wp_nodes}\n"
                            f"{forbidden_str}\n\n"
                            f"Plan both round-trip tours visiting all required waypoints. "
                            f"Make sure the vehicles do not converge on the same nodes at the same time step."
                        )

                    uni_tour_messages    = [
                        {"role": "system", "content": system_msg},
                        {"role": "user",   "content": user_msg},
                    ]
                    uni_tour_future      = _llmutils.query_llm_async(uni_tour_messages, seed=0)
                    uni_tour_waiting     = True
                    uni_tour_retry_count = 0
                    print('[UnifiedTour] Request sent, waiting for response...')

            if uni_tour_waiting and uni_tour_future is not None and uni_tour_future.done():
                _reset_uni_tour = True
                try:
                    text, elapsed = uni_tour_future.result(timeout=0)
                    uni_tour_total_time += elapsed
                    print(f'[UnifiedTour] LLM replies in {elapsed:.2f}s (total {uni_tour_total_time:.2f}s), checking collision...')

                    if WAYPOINT_PROMPT_FORMAT == "coords":
                        locs1, locs2 = parse_unified_llm_paths_coords(text)
                    else:
                        locs1, locs2 = parse_unified_llm_paths(text, graph)

                    def _to_wp_plan_tour(locs):
                        plan = []
                        for loc in locs:
                            wp = cmap.get_waypoint(
                                carla.Location(x=loc.x, y=loc.y, z=0),
                                project_to_road=True,
                                lane_type=carla.LaneType.Driving,
                            )
                            if wp is not None:
                                plan.append((wp, RoadOption.LANEFOLLOW))
                        return plan

                    plan1_chk = _to_wp_plan_tour(locs1) if locs1 else []
                    plan2_chk = _to_wp_plan_tour(locs2) if locs2 else []

                    collision_report = ""
                    if plan1_chk and plan2_chk:
                        collision_report = check_path_collisions(
                            [plan1_chk, plan2_chk],
                            path_names=["vehicle1", "vehicle2"],
                        )
                        if collision_report:
                            print(f'[UnifiedTour] Collision detected:\n{collision_report}')
                        else:
                            print('[UnifiedTour] Collision check passed')
                    else:
                        print('[UnifiedTour] Incomplete routes, skipping collision check')

                    if collision_report and uni_tour_retry_count < MAX_UNI_TOUR_RETRIES:
                        uni_tour_retry_count += 1
                        print(f'[UnifiedTour] Retry {uni_tour_retry_count}/{MAX_UNI_TOUR_RETRIES}...')
                        uni_tour_messages.append({"role": "assistant", "content": text})
                        if WAYPOINT_PROMPT_FORMAT == "coords":
                            _retry_fmt = (
                                "  path1 = [(x1, y1), ..., (xn, yn)]\n"
                                "  path2 = [(x1, y1), ..., (xn, yn)]"
                            )
                        else:
                            _retry_fmt = (
                                "  path1 = [<idx>, <idx>, ..., <idx>]\n"
                                "  path2 = [<idx>, <idx>, ..., <idx>]"
                            )
                        uni_tour_messages.append({"role": "user", "content": (
                            f"The paths you proposed have collision risks detected:\n\n{collision_report}\n\n"
                            f"Please revise both round-trip paths to eliminate all listed conflicts. "
                            f"Both paths must still visit ALL required waypoints.\n"
                            f"Output EXACTLY two lines — no explanation:\n"
                            f"{_retry_fmt}"
                        )})
                        uni_tour_future  = _llmutils.query_llm_async(uni_tour_messages, seed=uni_tour_retry_count)
                        _reset_uni_tour  = False

                    else:
                        if collision_report:
                            print(f'[UnifiedTour] Max retries ({MAX_UNI_TOUR_RETRIES}) reached, forcing apply')
                        else:
                            print('[UnifiedTour] Collision check passed, applying...')

                        LLM_DRAW_LT = 600.0
                        plan1, plan2 = [], []

                        if locs1:
                            if use_interpolation_path:
                                plan1 = route_from_linear(vehicle=vehicle, cmap=cmap,
                                                          world=world, agent=agent,
                                                          anchors=locs1, draw=True)
                            else:
                                plan1 = route_from_sparse(vehicle=vehicle, cmap=cmap, grp=grp,
                                                          world=world, agent=agent, anchors=locs1,
                                                          forbidden_areas=forbidden_areas, draw=True)
                            for loc in locs1:
                                world.debug.draw_point(loc + carla.Location(z=0.8), size=0.08,
                                                       color=carla.Color(255, 220, 0), life_time=LLM_DRAW_LT)
                            for i in range(len(locs1) - 1):
                                world.debug.draw_line(
                                    locs1[i] + carla.Location(z=0.8),
                                    locs1[i + 1] + carla.Location(z=0.8),
                                    thickness=0.08, color=carla.Color(255, 180, 0), life_time=LLM_DRAW_LT)
                            agent_active = bool(plan1)
                            print(f'[UnifiedTour] vehicle1 path {"activated" if agent_active else "failed"}, '
                                  f'{len(plan1) if plan1 else 0} waypoints')
                        else:
                            print('[UnifiedTour] vehicle1 no valid path parsed')

                        if locs2:
                            if use_interpolation_path:
                                plan2 = route_from_linear(vehicle=vehicle2, cmap=cmap,
                                                          world=world, agent=agent2,
                                                          anchors=locs2, draw=True)
                            else:
                                plan2 = route_from_sparse(vehicle=vehicle2, cmap=cmap, grp=grp,
                                                          world=world, agent=agent2, anchors=locs2,
                                                          forbidden_areas=forbidden_areas, draw=True)
                            for loc in locs2:
                                world.debug.draw_point(loc + carla.Location(z=0.8), size=0.08,
                                                       color=carla.Color(0, 210, 255), life_time=LLM_DRAW_LT)
                            for i in range(len(locs2) - 1):
                                world.debug.draw_line(
                                    locs2[i] + carla.Location(z=0.8),
                                    locs2[i + 1] + carla.Location(z=0.8),
                                    thickness=0.08, color=carla.Color(0, 180, 255), life_time=LLM_DRAW_LT)
                            agent2_active = bool(plan2)
                            print(f'[UnifiedTour] vehicle2 path {"activated" if agent2_active else "failed"}, '
                                  f'{len(plan2) if plan2 else 0} waypoints')
                        else:
                            print('[UnifiedTour] vehicle2 no valid path parsed')

                        pending_experiment = {
                            "task":         "llm_unified_tour",
                            "iterations":   uni_tour_retry_count + 1,
                            "avg_llm_time": uni_tour_total_time,
                            "path1_length": calc_path_length_m(plan1),
                            "path2_length": calc_path_length_m(plan2),
                        }
                        runtime_collision_det = False
                        _exp_vehicles_active  = False
                        print(f'[Experiment] unified_tour data ready: iters={uni_tour_retry_count + 1}, '
                              f'total_llm_time={uni_tour_total_time:.2f}s, '
                              f'path1={calc_path_length_m(plan1):.1f}m, '
                              f'path2={calc_path_length_m(plan2):.1f}m')

                except Exception as e:
                    print(f'[UnifiedTour] error: {e}')

                if _reset_uni_tour:
                    uni_tour_future      = None
                    uni_tour_waiting     = False
                    uni_tour_retry_count = 0
                    uni_tour_messages    = []
                    uni_tour_total_time  = 0.0

            def _apply_llm_result(fut, veh, agt, label):
                LLM_DRAW_LIFETIME = 600.0
                try:
                    text, elapsed = fut.result(timeout=0)
                    print(f'[LLM-{label}] Reply received in {elapsed:.2f}s')
                    if WAYPOINT_PROMPT_FORMAT == "coords":
                        anchors = parse_llm_coord_path(text)
                    else:
                        anchors = parse_llm_node_path(text, graph)
                    print(f'[LLM-{label}] Parsed {len(anchors)} nodes')
                    for loc in anchors:
                        world.debug.draw_point(loc + carla.Location(z=0.8),
                                               size=0.08, color=carla.Color(255, 20, 25),
                                               life_time=LLM_DRAW_LIFETIME)
                    for i in range(len(anchors) - 1):
                        world.debug.draw_line(
                            anchors[i]     + carla.Location(z=0.8),
                            anchors[i + 1] + carla.Location(z=0.8),
                            thickness=0.08, color=carla.Color(0, 180, 255),
                            life_time=LLM_DRAW_LIFETIME,
                        )
                    if anchors:
                        if use_interpolation_path:
                            plan = route_from_linear(vehicle=veh, cmap=cmap, world=world,
                                                     agent=agt, anchors=anchors, draw=True)
                        else:
                            plan = route_from_sparse(vehicle=veh, cmap=cmap, grp=grp,
                                                     world=world, agent=agt, anchors=anchors,
                                                     forbidden_areas=forbidden_areas, draw=True)
                        if plan:
                            print(f'[LLM-{label}] Route activated with {len(plan)} waypoints.')
                            return True
                        print(f'[LLM-{label}] Route planning failed.')
                    else:
                        print(f'[LLM-{label}] No valid path parsed.')
                except Exception as e:
                    print(f'[LLM-{label}] Error handling the reply: {e}')
                return False

            if llm_waiting and llm_future is not None and llm_future.done():
                agent_active = _apply_llm_result(llm_future, vehicle, agent, 'Vehicle1')
                llm_future  = None
                llm_waiting = False

            if llm_waiting2 and llm_future2 is not None and llm_future2.done():
                agent2_active = _apply_llm_result(llm_future2, vehicle2, agent2, 'Vehicle2')
                llm_future2  = None
                llm_waiting2 = False

            if agent_active:
                if agent.done():
                    print('[Autopilot] vehicle1 has reached destination')
                    agent_active = False
                else:
                    ctrl = agent.run_step()
                    ctrl.manual_gear_shift = False
                    vehicle.apply_control(ctrl)
            else:
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

            if agent2_active:
                if agent2.done():
                    print('[Autopilot] vehicle2 has reached destination')
                    agent2_active = False
                else:
                    ctrl2 = agent2.run_step()
                    ctrl2.manual_gear_shift = False
                    vehicle2.apply_control(ctrl2)
            else:
                vehicle2.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

            if pending_experiment is not None:
                if agent_active or agent2_active:
                    _exp_vehicles_active = True

                    _dist = vehicle.get_location().distance(vehicle2.get_location())
                    if _dist < RUNTIME_COLLISION_DIST_M:
                        if not runtime_collision_det:
                            print(f'[Collision] Vehicle distance while driving {_dist:.2f}m < {RUNTIME_COLLISION_DIST_M}m, '
                                  f'marking collision and ending statistics early')
                            runtime_collision_det = True
                            agent_active  = False
                            agent2_active = False

                    _vel1 = vehicle.get_velocity()
                    _vel2 = vehicle2.get_velocity()
                    _active_speeds = []
                    if agent_active:
                        _active_speeds.append(math.sqrt(_vel1.x**2 + _vel1.y**2 + _vel1.z**2))
                    if agent2_active:
                        _active_speeds.append(math.sqrt(_vel2.x**2 + _vel2.y**2 + _vel2.z**2))

                    if _active_speeds and all(s < STUCK_SPEED_THRESHOLD for s in _active_speeds):
                        _stuck_stationary_ticks += 1
                        if _stuck_stationary_ticks >= STUCK_TIMEOUT_TICKS:
                            print(f'[Stuck] All active vehicles stationary for '
                                  f'{_stuck_stationary_ticks * 0.05:.1f}s, judged stuck, '
                                  f'ending statistics early (path lengths use the full planned values)')
                            runtime_collision_det    = True
                            agent_active             = False
                            agent2_active            = False
                            _stuck_stationary_ticks  = 0
                            _pending_auto_reset      = True
                            _finished_task = pending_experiment["task"]
                            if _finished_task == "llm_unified_tour":
                                _auto_restart_uni_tour = True
                            else:
                                _auto_restart_tour = True
                    else:
                        _stuck_stationary_ticks = 0


                if _exp_vehicles_active and not agent_active and not agent2_active:
                    _finished_task          = pending_experiment["task"]
                    pending_experiment["collision_detected"] = runtime_collision_det
                    save_experiment_log(_finished_task, pending_experiment)
                    pending_experiment      = None
                    runtime_collision_det   = False
                    _exp_vehicles_active    = False
                    _stuck_stationary_ticks = 0
                    _pending_auto_reset     = True
                    if _finished_task == "llm_unified_tour":
                        _auto_restart_uni_tour = True
                    else:
                        _auto_restart_tour = True



    except KeyboardInterrupt:
        print('\n[Exit] Interrupted by Ctrl+C.')

    finally:
        for actor in actor_list:
            if actor is not None and actor.is_alive:
                actor.destroy()

        if world is not None:
            try:
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)
            except Exception:
                pass

        print('[Done] All actors destroyed.')
        pygame.quit()


if __name__ == '__main__':
    main()
