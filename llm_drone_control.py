"""
llm_drone_control.py

LLM-controlled Crazyflie drone autonomous navigation system.
- No keyboard input required: preset the mission in the CONFIGURATION section below and run directly.
- The map only describes obstacles (spherical); the LLM outputs a path in absolute coordinates.
- Flies point by point via PositionController, no interpolation needed.
- Runs a single drone, but the code structure supports multiple drones running simultaneously.
"""
import math
import re
import time
import contextlib
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Allows saving images even in headless environments
import matplotlib.pyplot as plt

import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

from position_controller import PositionController
from llmutil import llmutils as _llmutils


# Modify mission parameters here
# Drone list, each entry corresponds to one aircraft
# To add more drones, simply append an entry to the list
DRONES = [
    {
        "uri":   "radio://0/3/2M/E7E7E7E7E7", #real
        #"uri":   "udp://0.0.0.0:19850", #simulate
        "name":  "drone_1",
        # Initial world coordinates (meters) used by PositionController after takeoff
        # z is usually set to default_height
        "start": (0.0, 0.0, 0.45),
    },
    # {
    #     "uri":   "radio://0/90/2M/E7E7E7E7E8",
    #     "name":  "drone_2",
    #     "start": (1.0, 0.0, 0.5),
    # },
]

# Takeoff hover height (meters), MotionCommander default_height
DEFAULT_HEIGHT = 0.45

# Mission mode:
#   "goto"  — fly to a single destination
#   "tour"  — pass through a list of waypoints in order (the last waypoint is the final destination; does not automatically return to the start)
MISSION_MODE = "tour"

# Destination for goto mode (x, y, z)
DESTINATION = (0.25, -0.5, 0.45)

# Waypoint list for tour mode; the LLM plans a path from the start through each waypoint in order
TOUR_WAYPOINTS = [
    (0.8, 0.0, 0.45),
    (0.0, 0.8, 0.45),
    (0.0, 0.0, 0.45),
]

# Obstacle list, each entry is (center_x, center_y, center_z, radius_m)
# Spherical obstacles; the LLM must plan paths around them
OBSTACLES = [
    (0.0, 0.3, 0.45, 0.05),
    (0.6, 0.6, 0.45, 0.05),
    (0.4, -0.2, 0.45, 0.05)
]

#（m/s）
FLIGHT_VELOCITY = 0.3

# Preview mode: True = only run LLM planning + output trajectory plot, no hardware connection, no takeoff
DRY_RUN = False

# Maximum LLM retries after collision detection failure
MAX_COLLISION_RETRIES = 5


def build_obstacle_context(obstacles: list) -> str:
    if not obstacles:
        return "No obstacles."
    lines = ["Obstacles (sphere: center_x, center_y, center_z, radius_m):"]
    for i, obs in enumerate(obstacles):
        cx, cy, cz, r = obs
        lines.append(f"  [{i}] center=({cx:.3f}, {cy:.3f}, {cz:.3f}), radius={r:.3f}m")
    return "\n".join(lines)


def _build_messages_goto(
    drone_name: str,
    current_pos: tuple,
    destination: tuple,
    obstacles: list,
    custom_user_prompt: str = None,
    additional_prompt: str = None,
) -> list:
    obstacle_ctx = build_obstacle_context(obstacles)

    flight_z = current_pos[2]

    system_msg = (
        "You are a drone path planner operating in a 2D horizontal plane (units: meters).\n"
        "The drone flies at a FIXED altitude — the z coordinate NEVER changes.\n"
        "Plan a collision-free path from the current position to the destination\n"
        "by adjusting only x and y, keeping a safe margin from all obstacles.\n\n"
        #"Output EXACTLY one line — no explanation, no code block:\n"
        f"  path = [(x1,y1,{flight_z:.3f}), (x2,y2,{flight_z:.3f}), ..., (xn,yn,{flight_z:.3f})]\n\n"
        "Rules:\n"
        "  - Every waypoint MUST have z = current altitude (do NOT change z).\n"
        "  - The last waypoint must equal the destination exactly.\n"
        "  - Keep at least 0.1m clearance from every obstacle in the XY plane.\n"
        "  - Use the minimum number of waypoints necessary — do not add redundant points."
    )

    user_msg = (
        f"Drone: {drone_name}\n"
        f"Current position: ({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f})\n"
        f"Destination:      ({destination[0]:.3f}, {destination[1]:.3f}, {destination[2]:.3f})\n"
        f"Fixed flight altitude (z): {flight_z:.3f} m\n\n"
        f"{obstacle_ctx}\n\n"
        f"Plan a 2D collision-free path (x,y only) from current position to destination."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
    if custom_user_prompt:
        messages.append({"role": "user", "content": custom_user_prompt})
    if additional_prompt:
        messages.append({
            "role": "user",
            "content": (
                f"The previous attempt produced an invalid path.\n"
                f"History: {additional_prompt}\n"
                f"Please generate a different valid path."
            ),
        })
    return messages


def _build_messages_tour(
    drone_name: str,
    current_pos: tuple,
    waypoints: list,
    obstacles: list,
    custom_user_prompt: str = None,
    additional_prompt: str = None,
) -> list:
    obstacle_ctx = build_obstacle_context(obstacles)
    wp_str = "\n".join(
        f"  [{i}] ({wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f})"
        for i, wp in enumerate(waypoints)
    )

    flight_z = current_pos[2]

    system_msg = (
        "You are a drone path planner operating in a 2D horizontal plane (units: meters).\n"
        "The drone flies at a FIXED altitude — the z coordinate NEVER changes.\n"
        "Plan a collision-free path that visits all required waypoints in order\n"
        "by adjusting only x and y, keeping a safe margin from all obstacles.\n\n"
        #"Output EXACTLY one line — no explanation, no code block:\n"
        f"  path = [(x1,y1,{flight_z:.3f}), (x2,y2,{flight_z:.3f}), ..., (xn,yn,{flight_z:.3f})]\n\n"
        "Rules:\n"
        "  - Every waypoint MUST have z = current altitude (do NOT change z).\n"
        "  - The path must visit ALL required waypoints REGARDLESS OF ORDER.\n"
        "  - Each required waypoint must appear in the path exactly.\n"
        "  - Fly through the points with the shortest path"
        "  - Keep at least 0.1m clearance from every obstacle in the XY plane.\n"
        "  - You may add intermediate avoidance waypoints between required ones.\n"
        "  - Do not add unnecessary extra points where there are no obstacles.\n"
        "  - All coordinates must be real numbers (floats)."
    )

    user_msg = (
        f"Drone: {drone_name}\n"
        f"Current position: ({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f})\n"
        f"Fixed flight altitude (z): {flight_z:.3f} m\n\n"
        f"Required waypoints to visit in order:\n{wp_str}\n\n"
        f"{obstacle_ctx}\n\n"
        f"Plan a 2D collision-free path (x,y only) that visits all required waypoints in order."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
    if custom_user_prompt:
        messages.append({"role": "user", "content": custom_user_prompt})
    if additional_prompt:
        messages.append({
            "role": "user",
            "content": (
                f"The previous attempt produced an invalid path.\n"
                f"History: {additional_prompt}\n"
                f"Please generate a different valid path."
            ),
        })
    return messages


def query_llm_async(messages: list):
    return _llmutils.query_llm_async(messages, seed=0)


def parse_llm_path(text: str) -> list:
    m = re.search(r'path\s*=\s*\[([^\]]+)\]', text, re.DOTALL)
    if not m:
        m = re.search(r'\[([^\]]+)\]', text, re.DOTALL)
    if not m:
        print('[LLM] Path list not found')
        return []

    inner = m.group(1)
    tuples = re.findall(
        r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)',
        inner,
    )
    if not tuples:
        print('[LLM] No valid coordinate tuples found in the path list')
        return []

    path = [(float(x), float(y), float(z)) for x, y, z in tuples]
    print(f'[LLM] Parsed {len(path)} waypoints:')
    for i, pt in enumerate(path):
        print(f'    [{i}] ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})')
    return path

def _path_duration(
    path: list,
    start: tuple,
    velocity: float = FLIGHT_VELOCITY,
    sleep_time: float = 1.0,
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


def check_obstacle_collision(
    path: list,
    start: tuple,
    obstacles: list,
    dt: float = 0.1,
    velocity: float = FLIGHT_VELOCITY,
) -> tuple:
    if not path or not obstacles:
        return False, None

    dur = _path_duration(path, start, velocity)
    print(
        f"[Collision Check] Total path duration={dur:.1f}s  dt={dt}s  obstacle count={len(obstacles)}"
    )

    t = 0.0
    while t <= dur + 1e-9:
        pos = estimate_position_at_time(t, path, start, velocity)
        for oi, obs in enumerate(obstacles):
            cx, cy, cz, r = obs
            d = math.sqrt(
                (pos[0] - cx) ** 2 +
                (pos[1] - cy) ** 2 +
                (pos[2] - cz) ** 2
            )
            if d < r:
                print(
                    f"[Collision Check] Hit obstacle [{oi}]! "
                    f"t={t:.2f}s  dist_to_center={d:.3f}m  r={r}m"
                )
                return True, {
                    "t": t,
                    "pos": pos,
                    "obstacle_idx": oi,
                    "obstacle": obs,
                }
        t += dt

    print("[Collision Check] No collision with obstacles detected")
    return False, None


def _format_obstacle_feedback(info: dict) -> str:
    obs = info["obstacle"]
    oi  = info["obstacle_idx"]
    pos = info["pos"]
    t   = info["t"]
    return (
        f"[Obstacle Collision Detected] The planned path enters obstacle [{oi}] "
        f"(center=({obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f}), radius={obs[3]:.3f}m).\n"
        f"Collision point: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) at t={t:.2f}s.\n"
        f"Please replan a path that keeps at least 0.1m clearance from ALL obstacles.\n"
        f"Output the new path:\n"
        f"  path = [(x1,y1,z), ..., (xn,yn,z)]"
    )

def estimate_position_at_time(
    t: float,
    path: list,
    start: tuple,
    velocity: float = FLIGHT_VELOCITY,
    sleep_time: float = 1.0,
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
                move_dur  = turn_angle / turn_rate
                seg_dur   = move_dur + sleep_time
                segments.append((
                    current_t, current_t + seg_dur,
                    current_pos, current_pos,
                    move_dur, "turn",
                ))
                current_t += seg_dur

            current_yaw = target_angle % 360

            move_dur  = horiz_dist / velocity
            seg_dur   = move_dur + sleep_time
            end_pos   = (wp[0], wp[1], current_pos[2])
            segments.append((
                current_t, current_t + seg_dur,
                current_pos, end_pos,
                move_dur, "forward",
            ))
            current_t  += seg_dur
            current_pos = end_pos

        if abs(dz) > 1e-3:
            move_dur  = abs(dz) / velocity
            seg_dur   = move_dur + sleep_time
            end_pos   = (current_pos[0], current_pos[1], wp[2])
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

def plot_flight_trajectory(
    drone_name: str,
    start: tuple,
    path: list,
    obstacles: list,
    required_waypoints: list = None,
    save_dir: str = ".",
) -> str:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    for obs in obstacles:
        cx, cy, _cz, r = obs
        # Solid obstacle
        circle = plt.Circle((cx, cy), r, color="tomato", alpha=0.5, zorder=3)
        ax.add_patch(circle)

        margin = plt.Circle((cx, cy), r + 0.05, color="orange",
                             alpha=0.2, linestyle="--", fill=True, zorder=2)
        ax.add_patch(margin)
        margin_edge = plt.Circle((cx, cy), r + 0.05, color="orange",
                                  fill=False, linestyle="--", linewidth=1.2, zorder=4)
        ax.add_patch(margin_edge)
        ax.text(cx, cy, f"obs\nr={r}m", ha="center", va="center",
                fontsize=7, color="white", fontweight="bold", zorder=5)

    if path:
        xs = [start[0]] + [p[0] for p in path]
        ys = [start[1]] + [p[1] for p in path]
        ax.plot(xs, ys, "-o", color="royalblue", linewidth=2,
                markersize=5, label="LLM Path", zorder=6)
        # Waypoint numbering
        for i, (px, py) in enumerate(zip(xs[1:], ys[1:]), start=1):
            ax.annotate(str(i), (px, py), textcoords="offset points",
                        xytext=(6, 4), fontsize=7, color="royalblue")

    if required_waypoints:
        rwx = [w[0] for w in required_waypoints]
        rwy = [w[1] for w in required_waypoints]
        ax.scatter(rwx, rwy, c="darkorange", marker="^",
                   s=90, label="Waypoints", zorder=7)
        for i, wp in enumerate(required_waypoints):
            ax.annotate(f"WP{i}", (wp[0], wp[1]), textcoords="offset points",
                        xytext=(6, 4), fontsize=7, color="darkorange")

    ax.scatter(start[0], start[1], c="limegreen", marker="*", s=250,
               label=f"Start ({start[0]:.2f}, {start[1]:.2f})", zorder=8)
    if path:
        end = path[-1]
        ax.scatter(end[0], end[1], c="crimson", marker="*", s=250,
                   label=f"Destination ({end[0]:.2f}, {end[1]:.2f})", zorder=8)

    all_pts = [start] + (path or []) + (required_waypoints or [])
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
    flight_z = start[2]
    ax.set_title(f"[{drone_name}] Top-Down Path Preview  (z = {flight_z:.2f} m)", fontsize=13)
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/trajectory_{drone_name}_{ts}.png"
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f'[Trajectory Plot] Saved: {filename}')
    return filename

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
        print(f'[{self.name}] Connecting {self.uri}')
        self._scf = exit_stack.enter_context(SyncCrazyflie(self.uri))
        print(f'[{self.name}] Connected, taking off to {DEFAULT_HEIGHT:.2f}m ...')
        self._mc = exit_stack.enter_context(
            MotionCommander(self._scf, default_height=DEFAULT_HEIGHT)
        )
        time.sleep(3)
        self._pc = PositionController(
            self._mc,
            initial_x=self.start[0],
            initial_y=self.start[1],
            initial_z=self.start[2],
        )
        print(f'[{self.name}] taking off, current pos: {self._pc.position}')

    def execute_path(self, waypoints: list):
        if not waypoints:
            print(f'[{self.name}] Path is empty, skipping execution.')
            return
        if self._pc is None:
            print(f'[{self.name}] PositionController not initialized, please call connect_and_takeoff() first.')
            return

        print(f'[{self.name}] start moving through {len(waypoints)} waypoints')
        for i, wp in enumerate(waypoints):
            x, y, z = wp
            print(f'[{self.name}] → waypoints [{i}]: ({x:.3f}, {y:.3f}, {z:.3f})')
            self._pc.goto(x, y, z, velocity=FLIGHT_VELOCITY)
            time.sleep(1)

        print(f'[{self.name}] path completed, final position: {self._pc.position}')

class MissionRunner:
    def __init__(
        self,
        drones_config: list,
        obstacles: list,
        mission_mode: str,
        destination: tuple = None,
        tour_waypoints: list = None,
        dry_run: bool = False,
    ):
        self.agents         = [DroneAgent(cfg) for cfg in drones_config]
        self.obstacles      = obstacles
        self.mission_mode   = mission_mode
        self.destination    = destination
        self.tour_waypoints = tour_waypoints or []
        self.dry_run        = dry_run

    def run(self):
        if self.dry_run:
            print('[MissionRunner] *** Preview mode (DRY_RUN=True), not connecting to hardware ***')
            for agent in self.agents:
                self._run_agent(agent)
            print('[MissionRunner] Preview complete, trajectory plots saved to the current directory.')
            return

        cflib.crtp.init_drivers()

        with contextlib.ExitStack() as stack:
            for agent in self.agents:
                agent.connect_and_takeoff(stack)
            for agent in self.agents:
                self._run_agent(agent)
            print('[MissionRunner] All missions complete, preparing to land...')

        print('[MissionRunner] All drones landed, connections closed.')

    def _run_agent(self, agent: DroneAgent):
        if self.mission_mode == "goto":
            self._run_goto(agent)
        elif self.mission_mode == "tour":
            self._run_tour(agent)
        else:
            print(f'[MissionRunner] Unknown mission mode: {self.mission_mode}')

    def _run_goto(self, agent: DroneAgent):
        if self.destination is None:
            print(f'[{agent.name}] No destination set for goto mode.')
            return

        print(f'[{agent.name}] Mission mode: goto → {self.destination}')
        messages = _build_messages_goto(
            drone_name=agent.name,
            current_pos=agent.position,
            destination=self.destination,
            obstacles=self.obstacles,
        )
        path = self._query_and_parse(agent, messages)
        if path:
            print(f"path: {path}")
            plot_flight_trajectory(
                drone_name=agent.name,
                start=agent.position,
                path=path,
                obstacles=self.obstacles,
            )
            if not self.dry_run:
                agent.execute_path(path)

    def _run_tour(self, agent: DroneAgent):
        if not self.tour_waypoints:
            print(f'[{agent.name}] No waypoint list set for tour mode.')
            return

        print(f'[{agent.name}] Mission mode: tour, {len(self.tour_waypoints)} waypoints in total')
        messages = _build_messages_tour(
            drone_name=agent.name,
            current_pos=agent.position,
            waypoints=self.tour_waypoints,
            obstacles=self.obstacles,
        )
        path = self._query_and_parse(agent, messages)
        if path:
            plot_flight_trajectory(
                drone_name=agent.name,
                start=agent.position,
                path=path,
                obstacles=self.obstacles,
                required_waypoints=self.tour_waypoints,
            )
            if not self.dry_run:
                agent.execute_path(path)

    def _query_and_parse(self, agent: DroneAgent, messages: list) -> list:
        current_messages = list(messages)
        for attempt in range(MAX_COLLISION_RETRIES + 1):
            attempt_label = f"attempt {attempt + 1}/{MAX_COLLISION_RETRIES + 1}"
            print(f'[{agent.name}] Sending path planning request to LLM ({attempt_label})...')
            future = query_llm_async(current_messages)

            while not future.done():
                print(f'[{agent.name}] Waiting for LLM response...', flush=True)
                time.sleep(2.0)

            try:
                text, elapsed = future.result(timeout=0)
                print(f'[{agent.name}] LLM response received, took {elapsed:.2f}s')
                print(f'[{agent.name}] Raw LLM response:\n{text}')
                path = parse_llm_path(text)
                if not path:
                    print(f'[{agent.name}] Path parsing failed, the drone will keep hovering.')
                    return []

                collide, info = check_obstacle_collision(
                    path, agent.position, self.obstacles
                )
                if not collide:
                    return path

                if attempt < MAX_COLLISION_RETRIES:
                    feedback = _format_obstacle_feedback(info)
                    print(f'[{agent.name}] Obstacle collision detected, sending feedback to LLM for replanning...\n{feedback}')
                    current_messages = current_messages + [
                        {"role": "assistant", "content": text},
                        {"role": "user",      "content": feedback},
                    ]
                else:
                    print(
                        f'[{agent.name}] Maximum retries reached ({MAX_COLLISION_RETRIES}), '
                        f'using the last planned path (with collision risk).'
                    )
                    return path

            except Exception as e:
                print(f'[{agent.name}] Error while processing LLM response: {e}')
                return []

        return []

def main():
    runner = MissionRunner(
        drones_config=DRONES,
        obstacles=OBSTACLES,
        mission_mode=MISSION_MODE,
        destination=DESTINATION,
        tour_waypoints=TOUR_WAYPOINTS,
        dry_run=DRY_RUN,
    )
    runner.run()


if __name__ == "__main__":
    main()
