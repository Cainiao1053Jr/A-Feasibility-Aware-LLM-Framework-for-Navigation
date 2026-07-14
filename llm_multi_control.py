"""
llm_multi_control.py

LLM-controlled autonomous navigation system for two Crazyflie drones.
Supports two planning modes (switched via PLANNING_MODE):
  - "single_llm"  : one LLM plans paths for both drones simultaneously
  - "multi_agent" : two agents communicate and negotiate, then each plans its own path (see agent_comm.py)
"""
import math
import os
import re
import time
import threading
import contextlib
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

from position_controller import PositionController
from llmutil import llmutils as _llmutils
from agent_comm import AgentPool


DRONES = [
    {
        #"uri":   "radio://0/3/2M/E7E7E7E7E7", #real
        "uri":   "udp://0.0.0.0:19850", #simulate
        "name":  "drone_1",
        "start": (0.0, 0.0, 0.3),
    },
    {
        #"uri":   "radio://0/4/2M/E7E7E7E7E8", #real
        "uri":   "udp://0.0.0.0:19851", #simulate
        "name":  "drone_2",
        "start": (1.0, 0.0, 0.3),
    },
]

DRONE_DESTINATIONS = {
    "drone_1": (1.0, 1, 0.3),
    "drone_2": (0.0, 1, 0.3),
}

# Planning mode:
#   "single_llm"  — one LLM plans paths for both drones simultaneously
#   "multi_agent" — two agents communicate and negotiate, then each plans its own path
PLANNING_MODE = "multi_agent"

DEFAULT_HEIGHT = 0.3

# Mission mode:
#   "goto" — fly to individual destinations (using DRONE_DESTINATIONS)
#   "tour" — pass through a list of waypoints in order (both drones share the same waypoint list but have different start points)
MISSION_MODE = "goto"

# Waypoint list for tour mode (each drone starts from its own start point and visits these waypoints in order)
TOUR_WAYPOINTS = [
    (1.0, 0.0, 0.3),
    (1.0, 1.0, 0.3),
    (0.0, 1.0, 0.3),
    (0.0, 0.0, 0.3),
]

# Obstacle list, each entry is (center_x, center_y, center_z, radius_m)
OBSTACLES = [
    (0.5, 0.25, 0.3, 0.08),
    (0.3, 0.7,  0.3, 0.06),
]

# Flight velocity (m/s)
FLIGHT_VELOCITY = 0.2

# Preview mode: True = only run LLM planning + output trajectory plot, no hardware connection, no takeoff
DRY_RUN = True


def build_obstacle_context(obstacles: list) -> str:
    if not obstacles:
        return "No obstacles."
    lines = ["Obstacles (sphere: center_x, center_y, center_z, radius_m):"]
    for i, obs in enumerate(obstacles):
        cx, cy, cz, r = obs
        lines.append(f"  [{i}] center=({cx:.3f}, {cy:.3f}, {cz:.3f}), radius={r:.3f}m")
    return "\n".join(lines)


def _create_log_session() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join("logs", ts)
    os.makedirs(session_dir, exist_ok=True)
    print(f"[Log] Session directory: {session_dir}")
    return session_dir


def parse_llm_path(text: str, label: str = "path") -> list:
    m = re.search(rf'{re.escape(label)}\s*=\s*\[([^\]]+)\]', text, re.DOTALL)
    if not m:
        m = re.search(r'\[([^\]]+)\]', text, re.DOTALL)
    if not m:
        print(f'[LLM] Path list not found (label={label})')
        return []

    inner = m.group(1)
    tuples = re.findall(
        r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)',
        inner,
    )
    if not tuples:
        print(f'[LLM] No valid coordinate tuples found in the path list (label={label})')
        return []

    path = [(float(x), float(y), float(z)) for x, y, z in tuples]
    print(f'[LLM] Parsed {len(path)} waypoints (label={label}):')
    for i, pt in enumerate(path):
        print(f'    [{i}] ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})')
    return path


def parse_dual_llm_paths(text: str, names: list) -> dict:
    results = {}
    for i, name in enumerate(names, start=1):
        path = parse_llm_path(text, label=f"path_{i}")
        if not path:
            # Fallback: try using the drone name
            path = parse_llm_path(text, label=name)
        results[name] = path
    return results


def _build_messages_dual_goto(
    drones: list,
    destinations: dict,
    obstacles: list,
) -> list:
    obstacle_ctx = build_obstacle_context(obstacles)
    flight_z = drones[0].start[2]

    system_msg = (
        "You are a multi-drone path planner operating in a 2D horizontal plane (units: meters).\n"
        "Both drones fly at the SAME FIXED altitude — the z coordinate NEVER changes.\n"
        "Plan collision-free paths for BOTH drones simultaneously.\n"
        "Each drone must avoid all obstacles AND avoid getting within 0.3m of the other drone.\n\n"
        f"Output EXACTLY two lines in this format (no extra explanation needed, but brief reasoning is fine):\n"
        f"  path_1 = [(x1,y1,{flight_z:.3f}), (x2,y2,{flight_z:.3f}), ...]\n"
        f"  path_2 = [(x1,y1,{flight_z:.3f}), (x2,y2,{flight_z:.3f}), ...]\n\n"
        "Rules:\n"
        "  - Every waypoint MUST have z = fixed altitude (do NOT change z).\n"
        "  - The last waypoint of each path must equal its destination exactly.\n"
        "  - Keep at least 0.1m clearance from every obstacle in the XY plane.\n"
        "  - The two drone paths must not come within 0.3m of each other at any point.\n"
        "  - Use the minimum number of waypoints necessary."
    )

    drone_info = ""
    for i, agent in enumerate(drones, start=1):
        dest = destinations[agent.name]
        drone_info += (
            f"Drone {i} ({agent.name}):\n"
            f"  Current position: ({agent.start[0]:.3f}, {agent.start[1]:.3f}, {agent.start[2]:.3f})\n"
            f"  Destination:      ({dest[0]:.3f}, {dest[1]:.3f}, {dest[2]:.3f})\n\n"
        )

    user_msg = (
        f"{drone_info}"
        f"Fixed flight altitude (z): {flight_z:.3f} m\n\n"
        f"{obstacle_ctx}\n\n"
        "Plan collision-free paths for both drones. "
        "Output path_1 for drone_1 and path_2 for drone_2."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]


def _build_messages_dual_tour(
    drones: list,
    waypoints: list,
    obstacles: list,
) -> list:
    """Build the LLM message list for dual-drone tour planning (single-LLM mode)."""
    obstacle_ctx = build_obstacle_context(obstacles)
    flight_z = drones[0].start[2]
    wp_str = "\n".join(
        f"  [{i}] ({wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f})"
        for i, wp in enumerate(waypoints)
    )

    system_msg = (
        "You are a multi-drone path planner operating in a 2D horizontal plane (units: meters).\n"
        "Both drones fly at the SAME FIXED altitude — the z coordinate NEVER changes.\n"
        "Plan collision-free paths for BOTH drones to visit all required waypoints in order.\n"
        "Each drone must avoid all obstacles AND keep 0.3m separation from the other drone.\n\n"
        f"Output EXACTLY two lines:\n"
        f"  path_1 = [(x1,y1,{flight_z:.3f}), ...]\n"
        f"  path_2 = [(x1,y1,{flight_z:.3f}), ...]\n\n"
        "Rules:\n"
        "  - Every waypoint MUST have z = fixed altitude.\n"
        "  - Each path must visit ALL required waypoints IN ORDER.\n"
        "  - Keep at least 0.1m clearance from every obstacle.\n"
        "  - The two drone paths must not come within 0.1m of each other.\n"
        "  - You may add intermediate avoidance waypoints."
    )

    drone_info = ""
    for i, agent in enumerate(drones, start=1):
        drone_info += (
            f"Drone {i} ({agent.name}): "
            f"start=({agent.start[0]:.3f}, {agent.start[1]:.3f}, {agent.start[2]:.3f})\n"
        )

    user_msg = (
        f"{drone_info}\n"
        f"Fixed flight altitude (z): {flight_z:.3f} m\n\n"
        f"Required waypoints (both drones visit all in order):\n{wp_str}\n\n"
        f"{obstacle_ctx}\n\n"
        "Plan collision-free paths for both drones."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]


def plan_single_llm(drones: list, destinations: dict, obstacles: list,
                    mission_mode: str, tour_waypoints: list,
                    session_dir: str = None) -> dict:

    if mission_mode == "goto":
        messages = _build_messages_dual_goto(drones, destinations, obstacles)
    else:
        messages = _build_messages_dual_tour(drones, tour_waypoints, obstacles)

    future = _llmutils.query_llm_async(messages, seed=0)

    while not future.done():
        print("[SingleLLM] Waiting for LLM response...", flush=True)
        time.sleep(2.0)

    try:
        text, elapsed = future.result(timeout=0)
        print(f"[SingleLLM] LLM response received, took {elapsed:.2f}s")
        print(f"[SingleLLM] Raw LLM response:\n{text}")
        names = [d.name for d in drones]
        paths = parse_dual_llm_paths(text, names)
        for name, path in paths.items():
            if not path:
                print(f"[SingleLLM] Path parsing failed for {name}")
        if session_dir:
            log_path = os.path.join(session_dir, "single_llm_log.txt")
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("=== Single LLM Planning Log ===\n")
                f.write(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Elapsed: {elapsed:.2f}s\n\n")
                f.write("=== Request ===\n\n")
                for msg in messages:
                    f.write(f"[{msg['role'].upper()}]\n{msg['content']}\n\n")
                f.write("=== LLM Response ===\n\n")
                f.write(text)
            print(f"[SingleLLM] Log saved: {log_path}")
        return paths
    except Exception as e:
        print(f"[SingleLLM] Error while processing LLM response: {e}")
        return {d.name: [] for d in drones}

def _build_agent_system_prompt(agent_name: str, other_name: str) -> str:
    """Build the system prompt for a single agent."""
    return (
        f"You are {agent_name}, an autonomous drone path planner.\n"
        f"You must plan a collision-free path for your drone in a 2D horizontal plane (units: meters).\n"
        f"The z coordinate (altitude) is FIXED and must NEVER change.\n\n"
        f"There is another drone named {other_name} also planning its path.\n"
        f"You can communicate with {other_name} using the send_message tool to:\n"
        f"  - Share your draft path so the other drone can avoid conflicts.\n"
        f"  - Negotiate if your paths might collide (within 0.1m of each other).\n"
        f"  - Agree on final paths that keep at least 0.1m separation at all times.\n\n"
        f"Workflow:\n"
        f"  1. Draft your path based on your start, destination, and obstacles.\n"
        f"  2. Send your draft path to {other_name} for conflict checking.\n"
        f"  3. Review {other_name}'s draft. If paths conflict, negotiate alternatives.\n"
        f"  4. Once both paths are conflict-free, output your final path as:\n"
        f"       path = [(x1,y1,z), (x2,y2,z), ..., (xn,yn,z)]\n\n"
        f"Rules:\n"
        f"  - Every waypoint MUST keep z = your fixed altitude.\n"
        f"  - Last waypoint must equal your destination exactly.\n"
        f"  - Keep at least 0.1m clearance from every obstacle.\n"
        f"  - Keep at least 0.5m separation from the other drone's path.\n"
        f"  - Use minimum waypoints necessary."
    )


def _build_agent_task_message(agent_name: str, start: tuple,
                               destination: tuple, obstacles: list) -> str:
    obstacle_ctx = build_obstacle_context(obstacles)
    flight_z = start[2]
    return (
        f"Your drone: {agent_name}\n"
        f"Current position: ({start[0]:.3f}, {start[1]:.3f}, {start[2]:.3f})\n"
        f"Destination:      ({destination[0]:.3f}, {destination[1]:.3f}, {destination[2]:.3f})\n"
        f"Fixed flight altitude (z): {flight_z:.3f} m\n\n"
        f"{obstacle_ctx}\n\n"
        f"Begin planning. Draft your path, share it with the other agent, "
        f"negotiate if needed, then output your final path as:\n"
        f"  path = [(x1,y1,{flight_z:.3f}), ..., (xn,yn,{flight_z:.3f})]"
    )


def _format_collision_feedback(before: dict, after: dict) -> str:
    d1b, d2b = before["drone1"], before["drone2"]
    d1a, d2a = after["drone1"],  after["drone2"]
    collision_type = after.get("type", "drone_drone")

    if collision_type == "obstacle":
        obs = after["obstacle"]
        header = (
            f"[Obstacle Collision Detected] {after['who']} will enter obstacle [{after['obstacle_idx']}] "
            f"(center=({obs[0]:.3f},{obs[1]:.3f},{obs[2]:.3f}), radius={obs[3]:.3f}m). "
            "The path must detour around this obstacle."
        )
    else:
        header = (
            "[Drone-Drone Near-Collision Detected] "
            "The two drones come within unsafe distance of each other."
        )

    return (
        f"{header}\n"
        f"Last safe moment  (t={before['t']:.2f}s):\n"
        f"  drone_1: ({d1b[0]:.3f}, {d1b[1]:.3f}, {d1b[2]:.3f})\n"
        f"  drone_2: ({d2b[0]:.3f}, {d2b[1]:.3f}, {d2b[2]:.3f})\n"
        f"Collision moment  (t={after['t']:.2f}s):\n"
        f"  drone_1: ({d1a[0]:.3f}, {d1a[1]:.3f}, {d1a[2]:.3f})\n"
        f"  drone_2: ({d2a[0]:.3f}, {d2a[1]:.3f}, {d2a[2]:.3f})\n"
        "Please revise YOUR path to avoid this collision zone, "
        "then re-output your final path:\n"
        "  path = [(x1,y1,z), ..., (xn,yn,z)]"
    )


class CollisionGuard:
    """
    AgentPool finalization_guard: runs collision detection after both drones submit their final paths.

    Workflow
    --------
    1. The first agent to submit its path → the path is stored, and a "waiting" message is returned (forcing the agent to resubmit).
    2. The second agent to submit its path → both paths are available, run detect_collision.
       - No collision: an empty string is returned as feedback, and both agents may finish.
       - Collision: a collision report is returned as feedback, and both agents are blocked and must replan.
    3. After both agents have read this round's result, the guard automatically resets for the next round of checking.
    """

    def __init__(self, drones: list, obstacles: list = None,
                 velocity: float = FLIGHT_VELOCITY):
        self._drones    = {d.name: d for d in drones}
        self._obstacles = obstacles or []
        self._velocity  = velocity
        self._names    = [d.name for d in drones]
        self._n        = len(self._names)
        self._lock     = threading.Lock()
        self._reset()

    def _reset(self):
        self._paths:      dict = {}
        self._feedback:   dict = {}
        self._check_done: bool = False
        self._done_count: int  = 0

    def __call__(self, agent_name: str, final_text: str) -> str:
        path = parse_llm_path(final_text, label="path")
        if not path:
            return (
                "Path parsing failed. Please re-output your final path in the format:\n"
                "  path = [(x1,y1,z), ..., (xn,yn,z)]"
            )

        with self._lock:
            self._paths[agent_name] = path

            if len(self._paths) < self._n:
                return (
                    "Waiting for the other drone to submit its final path. "
                    "Please re-confirm and re-output your final path:\n"
                    "  path = [(x1,y1,z), ..., (xn,yn,z)]"
                )

            if not self._check_done:
                self._check_done = True
                n1, n2 = self._names
                collide, before, after = detect_collision(
                    self._paths[n1], self._drones[n1].start,
                    self._paths[n2], self._drones[n2].start,
                    velocity=self._velocity,
                    obstacles=self._obstacles,
                )
                if collide:
                    fb = _format_collision_feedback(before, after)
                    self._feedback = {name: fb for name in self._names}
                else:
                    print("[CollisionGuard] Collision check passed, paths are valid")
                    self._feedback = {name: "" for name in self._names}

            fb = self._feedback.get(agent_name, "")

            self._done_count += 1
            if self._done_count >= self._n:
                self._reset()

            return fb


def plan_multi_agent(drones: list, destinations: dict, obstacles: list,
                     session_dir: str = None) -> dict:
    print("\n[MultiAgent] Starting dual-agent negotiated planning...")

    names = [d.name for d in drones]
    pool = AgentPool(finalization_guard=CollisionGuard(drones, obstacles=obstacles),
                     log_dir=session_dir)

    for i, agent in enumerate(drones):
        other_name = names[1 - i]
        system_prompt = _build_agent_system_prompt(agent.name, other_name)
        pool.add_agent(agent.name, system_prompt)

    for agent in drones:
        dest = destinations[agent.name]
        task_msg = _build_agent_task_message(agent.name, agent.start, dest, obstacles)
        pool.send_task(agent.name, task_msg)

    print("[MultiAgent] The two agents are negotiating in parallel, please wait...")
    raw_results = pool.run_all(timeout=300.0)

    paths = {}
    for name, text in raw_results.items():
        print(f"[MultiAgent] Final response from {name}:\n{text}\n")
        path = parse_llm_path(text, label="path")
        if not path:
            print(f"[MultiAgent] Path parsing failed for {name}")
        paths[name] = path

    return paths


_DRONE_COLORS = ["royalblue", "darkorange"]


def plot_dual_trajectory(
    drones: list,
    paths: dict,
    obstacles: list,
    planning_mode: str,
    required_waypoints: list = None,
    save_dir: str = ".",
) -> str:
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")

    for obs in obstacles:
        cx, cy, _cz, r = obs
        circle = plt.Circle((cx, cy), r, color="tomato", alpha=0.5, zorder=3)
        ax.add_patch(circle)
        margin = plt.Circle((cx, cy), r + 0.1, color="orange",
                             alpha=0.15, fill=True, zorder=2)
        ax.add_patch(margin)
        margin_edge = plt.Circle((cx, cy), r + 0.1, color="orange",
                                  fill=False, linestyle="--", linewidth=1.2, zorder=4)
        ax.add_patch(margin_edge)
        ax.text(cx, cy, f"obs\nr={r}m", ha="center", va="center",
                fontsize=7, color="white", fontweight="bold", zorder=5)

    all_pts = []
    for idx, agent in enumerate(drones):
        color = _DRONE_COLORS[idx % len(_DRONE_COLORS)]
        path  = paths.get(agent.name, [])
        start = agent.start

        xs = [start[0]] + [p[0] for p in path]
        ys = [start[1]] + [p[1] for p in path]
        ax.plot(xs, ys, "-o", color=color, linewidth=2,
                markersize=5, label=f"{agent.name} path", zorder=6)

        for i, (px, py) in enumerate(zip(xs[1:], ys[1:]), start=1):
            ax.annotate(str(i), (px, py), textcoords="offset points",
                        xytext=(6, 4), fontsize=7, color=color)

        ax.scatter(start[0], start[1], c=color, marker="*", s=280,
                   label=f"{agent.name} start ({start[0]:.2f},{start[1]:.2f})",
                   zorder=8, edgecolors="black", linewidths=0.6)

        if path:
            end = path[-1]
            ax.scatter(end[0], end[1], c=color, marker="X", s=200,
                       label=f"{agent.name} dest ({end[0]:.2f},{end[1]:.2f})",
                       zorder=8, edgecolors="black", linewidths=0.6)

        all_pts += [start] + path

    if required_waypoints:
        rwx = [w[0] for w in required_waypoints]
        rwy = [w[1] for w in required_waypoints]
        ax.scatter(rwx, rwy, c="purple", marker="^", s=90,
                   label="Waypoints", zorder=7)
        for i, wp in enumerate(required_waypoints):
            ax.annotate(f"WP{i}", (wp[0], wp[1]), textcoords="offset points",
                        xytext=(6, 4), fontsize=7, color="purple")
        all_pts += required_waypoints

    if all_pts:
        all_x = [p[0] for p in all_pts]
        all_y = [p[1] for p in all_pts]
        pad = max(
            max(all_x) - min(all_x),
            max(all_y) - min(all_y),
            0.5,
        ) * 0.25 + 0.4
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    mode_label = "Single LLM" if planning_mode == "single_llm" else "Multi-Agent"
    flight_z = drones[0].start[2]
    ax.set_title(
        f"Dual-Drone Path Preview  [{mode_label}]  (z = {flight_z:.2f} m)",
        fontsize=13,
    )
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/trajectory_dual_{planning_mode}_{ts}.png"
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"[Trajectory Plot] Saved: {filename}")
    return filename


def estimate_position_at_time(
    t: float,
    path: list,
    start: tuple,
    velocity: float = FLIGHT_VELOCITY,
    sleep_time: float = 3.0,
    turn_rate: float = 72.0,
) -> tuple:
    if not path:
        return start
    if t <= 0:
        return start

    segments = []
    current_t   = 0.0
    current_pos = tuple(start)
    current_yaw = 0.0

    for wp in path:
        dx = wp[0] - current_pos[0]
        dy = wp[1] - current_pos[1]
        dz = wp[2] - current_pos[2]
        horiz_dist = math.sqrt(dx ** 2 + dy ** 2)

        if horiz_dist > 1e-3:
            target_angle = math.degrees(math.atan2(dy, dx))
            delta        = (target_angle - current_yaw + 180) % 360 - 180
            turn_angle   = abs(delta)

            if turn_angle > 0.1:
                move_dur = turn_angle / turn_rate
                seg_dur  = move_dur + sleep_time
                segments.append((
                    current_t, current_t + seg_dur,
                    current_pos, current_pos,
                    move_dur, "turn",
                ))
                current_t += seg_dur

            current_yaw = target_angle % 360

            move_dur = horiz_dist / velocity
            seg_dur  = move_dur + sleep_time
            end_pos  = (wp[0], wp[1], current_pos[2])
            segments.append((
                current_t, current_t + seg_dur,
                current_pos, end_pos,
                move_dur, "forward",
            ))
            current_t  += seg_dur
            current_pos = end_pos

        if abs(dz) > 1e-3:
            move_dur = abs(dz) / velocity
            seg_dur  = move_dur + sleep_time
            end_pos  = (current_pos[0], current_pos[1], wp[2])
            segments.append((
                current_t, current_t + seg_dur,
                current_pos, end_pos,
                move_dur, "vertical",
            ))
            current_t  += seg_dur
            current_pos = end_pos

    for t_start, t_end, pos_s, pos_e, move_dur, seg_type in segments:
        if t_start <= t <= t_end:
            if seg_type == "turn":
                return pos_s
            dt    = t - t_start
            alpha = min(dt / move_dur, 1.0) if move_dur > 1e-9 else 1.0
            x = pos_s[0] + alpha * (pos_e[0] - pos_s[0])
            y = pos_s[1] + alpha * (pos_e[1] - pos_s[1])
            z = pos_s[2] + alpha * (pos_e[2] - pos_s[2])
            return (x, y, z)

    return current_pos



def _path_duration(
    path: list,
    start: tuple,
    velocity: float = FLIGHT_VELOCITY,
    sleep_time: float = 3.0,
    turn_rate: float = 72.0,
) -> float:
    if not path:
        return 0.0

    current_t   = 0.0
    current_pos = tuple(start)
    current_yaw = 0.0

    for wp in path:
        dx = wp[0] - current_pos[0]
        dy = wp[1] - current_pos[1]
        dz = wp[2] - current_pos[2]
        horiz_dist = math.sqrt(dx ** 2 + dy ** 2)

        if horiz_dist > 1e-3:
            target_angle = math.degrees(math.atan2(dy, dx))
            delta        = (target_angle - current_yaw + 180) % 360 - 180
            turn_angle   = abs(delta)

            if turn_angle > 0.1:
                current_t += turn_angle / turn_rate + sleep_time

            current_yaw  = target_angle % 360
            current_t   += horiz_dist / velocity + sleep_time
            current_pos  = (wp[0], wp[1], current_pos[2])

        if abs(dz) > 1e-3:
            current_t  += abs(dz) / velocity + sleep_time
            current_pos = (current_pos[0], current_pos[1], wp[2])

    return current_t


def detect_collision(
    path1: list,
    start1: tuple,
    path2: list,
    start2: tuple,
    dt: float = 0.1,
    velocity: float = FLIGHT_VELOCITY,
    sleep_time: float = 3.0,
    turn_rate: float = 72.0,
    threshold: float = 0.5,
    obstacles: list = None,
) -> tuple:
    """
    Detect collisions between the two drones' paths (drone-drone + with obstacles) at time step dt.

    The detection interval is [0, min(total flight time of both)].

    Returns:
        (collide, before, after)
        collide : bool
        before/after both contain a 'type' field:
          'drone_drone' — the two drones are too close
          'obstacle'    — hit an obstacle (additionally contains 'who', 'obstacle_idx', 'obstacle')
        If there is no collision, both before and after are None.
    """
    dur1  = _path_duration(path1, start1, velocity, sleep_time, turn_rate)
    dur2  = _path_duration(path2, start2, velocity, sleep_time, turn_rate)
    t_max = min(dur1, dur2)

    print(
        f"[Collision Check] drone1={dur1:.1f}s  drone2={dur2:.1f}s  "
        f"detection interval=[0, {t_max:.1f}s]  dt={dt}s  threshold={threshold}m"
        + (f"  obstacle count={len(obstacles)}" if obstacles else "")
    )

    pos1_prev = estimate_position_at_time(0.0, path1, start1, velocity, sleep_time, turn_rate)
    pos2_prev = estimate_position_at_time(0.0, path2, start2, velocity, sleep_time, turn_rate)
    t_prev    = 0.0
    t         = dt

    while t <= t_max + 1e-9:
        pos1 = estimate_position_at_time(t, path1, start1, velocity, sleep_time, turn_rate)
        pos2 = estimate_position_at_time(t, path2, start2, velocity, sleep_time, turn_rate)

        dist = math.sqrt(
            (pos1[0] - pos2[0]) ** 2 +
            (pos1[1] - pos2[1]) ** 2 +
            (pos1[2] - pos2[2]) ** 2
        )
        if dist < threshold:
            print(
                f"[Collision Check] Drone-drone collision! t={t:.2f}s  dist={dist:.3f}m\n"
                f"  before (t={t_prev:.2f}s): drone1={tuple(round(v,3) for v in pos1_prev)}"
                f"  drone2={tuple(round(v,3) for v in pos2_prev)}\n"
                f"  after  (t={t:.2f}s): drone1={tuple(round(v,3) for v in pos1)}"
                f"  drone2={tuple(round(v,3) for v in pos2)}"
            )
            before = {"t": t_prev, "drone1": pos1_prev, "drone2": pos2_prev,
                      "type": "drone_drone"}
            after  = {"t": t,      "drone1": pos1,      "drone2": pos2,
                      "type": "drone_drone"}
            return True, before, after

        if obstacles:
            for oi, obs in enumerate(obstacles):
                cx, cy, cz, r = obs
                for label, pos in (("drone1", pos1), ("drone2", pos2)):
                    d = math.sqrt(
                        (pos[0] - cx) ** 2 +
                        (pos[1] - cy) ** 2 +
                        (pos[2] - cz) ** 2
                    )
                    if d < r:
                        print(
                            f"[Collision Check] {label} hit obstacle [{oi}]! "
                            f"t={t:.2f}s  dist_to_center={d:.3f}m  r={r}m"
                        )
                        before = {"t": t_prev, "drone1": pos1_prev, "drone2": pos2_prev,
                                  "type": "obstacle", "who": label,
                                  "obstacle_idx": oi, "obstacle": obs}
                        after  = {"t": t,      "drone1": pos1,      "drone2": pos2,
                                  "type": "obstacle", "who": label,
                                  "obstacle_idx": oi, "obstacle": obs}
                        return True, before, after

        pos1_prev, pos2_prev = pos1, pos2
        t_prev = t
        t     += dt

    print("[Collision Check] No collision detected")
    return False, None, None



class DroneAgent:

    def __init__(self, config: dict):
        self.uri   = config["uri"]
        self.name  = config["name"]
        self.start = config["start"]

        self._scf = None
        self._mc  = None
        self._pc  = None

    @property
    def position(self) -> tuple:
        if self._pc is None:
            return self.start
        return self._pc.position

    def connect_and_takeoff(self, exit_stack: contextlib.ExitStack):
        print(f"[{self.name}] Connecting: {self.uri}")
        self._scf = exit_stack.enter_context(SyncCrazyflie(self.uri))
        print(f"[{self.name}] Connected, taking off to {DEFAULT_HEIGHT:.2f}m ...")
        self._mc = exit_stack.enter_context(
            MotionCommander(self._scf, default_height=DEFAULT_HEIGHT)
        )
        time.sleep(1.5)
        self._pc = PositionController(
            self._mc,
            initial_x=self.start[0],
            initial_y=self.start[1],
            initial_z=self.start[2],
        )
        print(f"[{self.name}] taking off, current pos: {self._pc.position}")

    def execute_path(self, waypoints: list):
        if not waypoints:
            print(f"[{self.name}] empty path, skipping")
            return
        if self._pc is None:
            print(f"[{self.name}] PositionController not initialized")
            return

        print(f"[{self.name}] start executing path of {len(waypoints)} points")
        for i, wp in enumerate(waypoints):
            x, y, z = wp
            print(f"[{self.name}] → waypoint [{i}]: ({x:.3f}, {y:.3f}, {z:.3f})")
            self._pc.goto(x, y, z, velocity=FLIGHT_VELOCITY)
            time.sleep(0.5)

        print(f"[{self.name}] Path execution complete, final position: {self._pc.position}")


# ==============================================================================
# -- MissionRunner -------------------------------------------------------------
# ==============================================================================

class MissionRunner:
    """
    Mission scheduler: manages takeoff, LLM path planning, flight execution, and landing for two drones.

    Supports two planning modes:
      - "single_llm"  : one LLM plans paths for both drones simultaneously
      - "multi_agent" : two agents negotiate with each other, then each plans its own path
    """

    def __init__(
        self,
        drones_config: list,
        obstacles: list,
        mission_mode: str,
        destinations: dict = None,
        tour_waypoints: list = None,
        planning_mode: str = "single_llm",
        dry_run: bool = False,
    ):
        self.agents        = [DroneAgent(cfg) for cfg in drones_config]
        self.obstacles     = obstacles
        self.mission_mode  = mission_mode
        self.destinations  = destinations or {}
        self.tour_waypoints = tour_waypoints or []
        self.planning_mode = planning_mode
        self.dry_run       = dry_run

    def run(self):
        session_dir = _create_log_session()

        if self.dry_run:
            paths = self._plan_paths(session_dir)
            plot_dual_trajectory(
                drones=self.agents,
                paths=paths,
                obstacles=self.obstacles,
                planning_mode=self.planning_mode,
                required_waypoints=self.tour_waypoints if self.mission_mode == "tour" else None,
                save_dir=session_dir,
            )
            print(f"[MissionRunner] Preview complete, files saved to {session_dir}")
            return

        cflib.crtp.init_drivers()

        ready_events = {a.name: threading.Event() for a in self.agents}
        go_event     = threading.Event()
        paths        = {}
        errors       = []

        def run_agent(agent):
            try:
                with contextlib.ExitStack() as stack:
                    agent.connect_and_takeoff(stack)
                    ready_events[agent.name].set()
                    go_event.wait()  
                    agent.execute_path(paths.get(agent.name, []))
            except Exception as e:
                errors.append((agent.name, e))
                ready_events[agent.name].set()

        threads = [
            threading.Thread(
                target=run_agent,
                args=(agent,),
                name=f"drone-{agent.name}",
                daemon=False,
            )
            for agent in self.agents
        ]

        for t in threads:
            t.start()

        for agent in self.agents:
            ready_events[agent.name].wait()

        if errors:
            print(f"[MissionRunner] A drone failed to take off, aborting mission: {errors}")
            go_event.set()
            for t in threads:
                t.join()
            return

        planned = self._plan_paths(session_dir)
        paths.update(planned)

        plot_dual_trajectory(
            drones=self.agents,
            paths=paths,
            obstacles=self.obstacles,
            planning_mode=self.planning_mode,
            required_waypoints=self.tour_waypoints if self.mission_mode == "tour" else None,
            save_dir=session_dir,
        )

        print("[MissionRunner] Both drones are ready, departing in sync!")
        go_event.set()

        for t in threads:
            t.join()

        if errors:
            print(f"[MissionRunner] Mission finished with errors: {errors}")
        else:
            print("[MissionRunner] All drones landed, connections closed.")

    def _plan_paths(self, session_dir: str = None) -> dict:
        print(f"\n[MissionRunner] Planning mode: {self.planning_mode.upper()}")

        if self.planning_mode == "single_llm":
            return plan_single_llm(
                drones=self.agents,
                destinations=self.destinations,
                obstacles=self.obstacles,
                mission_mode=self.mission_mode,
                tour_waypoints=self.tour_waypoints,
                session_dir=session_dir,
            )
        elif self.planning_mode == "multi_agent":
            if self.mission_mode != "goto":
                print("[MissionRunner] multi_agent mode currently only supports goto, switching automatically")
            return plan_multi_agent(
                drones=self.agents,
                destinations=self.destinations,
                obstacles=self.obstacles,
                session_dir=session_dir,
            )
        else:
            print(f"[MissionRunner] Unknown planning mode: {self.planning_mode}, returning empty paths")
            return {agent.name: [] for agent in self.agents}



def main():
    print("=" * 60)
    print("  Multiagent LLM Path Planning System")
    print("=" * 60)
    print(f"  Planning Mode:{PLANNING_MODE}")
    print(f"  Mission Mode: {MISSION_MODE}")
    print(f"  Previewing: {DRY_RUN}")
    print("=" * 60)

    runner = MissionRunner(
        drones_config=DRONES,
        obstacles=OBSTACLES,
        mission_mode=MISSION_MODE,
        destinations=DRONE_DESTINATIONS,
        tour_waypoints=TOUR_WAYPOINTS,
        planning_mode=PLANNING_MODE,
        dry_run=DRY_RUN,
    )
    runner.run()


if __name__ == "__main__":
    main()
