"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    O            : open/close all doors of vehicle
    T            : toggle vehicle's telemetry

    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import os
import random
import re
import sys
import weakref
from pathlib import Path
from collections import deque
import threading
import json
from typing import List, Tuple
import geopandas as gpd

from llmutil import llmutils

from carlaLocal.agents.navigation.global_route_planner import GlobalRoutePlanner
from carlaLocal.agents.navigation.basic_agent import BasicAgent
from carlaLocal.agents.navigation.behavior_agent import BehaviorAgent
from carlaLocal.agents.navigation.local_planner import RoadOption
#from autogen import ConversableAgent

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_e # added
    from pygame.locals import K_f
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_j
    from pygame.locals import K_k
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_t
    from pygame.locals import K_u
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_y
    from pygame.locals import K_z
    from pygame.locals import K_KP1
    from pygame.locals import K_KP2
    from pygame.locals import K_KP3
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')



# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

# SETTINGS

#avoided_area = [carla.Location(0,60,0)] #add more if you want
avoided_area = [(carla.Location(0,60,0),20)]
#destination = carla.Location(100, 50 ,0) #init
destination = carla.Location(-171, 85 ,0) #custom
map_file = "junction_points_with_edges.geojson"
"""
"junction_points_with_edges.geojson" for Town10
"map2_junction_points_with_edges.geojson" for Town03
"""
switch_map = True # default as Town10, turn true only when changing map
alternative_map = "Town03"
"""
"Town03"
"""


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, client, carla_world, hud, args):
        self.client = client
        self.world = carla_world
        self.sync = args.sync
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._gamma = args.gamma
        self.last_max_col = 0
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.show_vehicle_telemetry = False
        self.doors_are_open = False
        self.send_to_llm = False
        self.llm_response = None
        self.llm_text = None
        self.llm_all_paths = None
        self.llm_selected_paths = None
        self.should_move_car = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]
        resolution = 2.0  # 米；越小越密

        # 方案一（如你的 agents 版本支持）：
        self.grp = GlobalRoutePlanner(self.map, resolution)

        self.tm_port = 8000  # 选个不冲突的端口
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)

        self.agent = None
        self._agent_active = False

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
        if not blueprint_list:
            raise ValueError("Couldn't find any blueprints with the specified filters")
        blueprint = random.choice(blueprint_list)
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('terramechanics'):
            blueprint.set_attribute('terramechanics', 'true')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            try:
                # 你顶部已经有 from carlaLocal.agents.navigation.behavior_agent import BehaviorAgent
                self.agent = BehaviorAgent(self.player, behavior='normal')  # 'cautious'/'normal'/'aggressive'
                # 可选：速度策略
                self.agent.follow_speed_limits(False)  # 不强制跟随限速
                self.agent.set_target_speed(40)  # km/h：按需调整
                self.agent.ignore_traffic_lights(True)
            except Exception as e:
                self.hud.error(f"Init BehaviorAgent failed: {e}")
                self.agent = None
                self._agent_active = False
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
            tm = self.traffic_manager  # tm stuff
            veh = self.player
            tm.auto_lane_change(veh, True)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        self.hud.tick(self, clock)
        # if self.llm_response is not None and self.llm_response.done():
        #     print("1")
        #     try:
        #         text, elapsed = self.llm_response.result(timeout=0)
        #         self.llm_text = text
        #         if not text:
        #             text = "[no content]"
        #         print(f"[LLM DONE in {elapsed:.2f}s] {text[:20000]}")
        #     except Exception as e:
        #         print("[LLM ERROR]", e)
        #     finally:
        #         self.llm_response = None
        if getattr(self, "_agent_active", False) and self.agent is not None:
            try:
                #self.agent.update_information(self.world)
                control = self.agent.run_step()
                self.player.apply_control(control)
                if self.agent.done():
                    self._agent_active = False
                    self.hud.notification("BehaviorAgent reached destination.")
            except Exception as e:
                self._agent_active = False
                self.hud.error(f"BehaviorAgent error: {e}")

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()
    def get_current_waypoint(self):
        loc = self.player.get_transform().location
        return self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    def sample_forward_route(self, step_m=3.0, length_m=120.0):
        wp = self.get_current_waypoint()
        route = [wp]
        dist = 0.0
        while dist < length_m and route[-1] is not None:
            nxt = route[-1].next(step_m)
            if not nxt:
               break
            route.append(nxt[0])
            dist += step_m
        return route
    def apply_lateral_offset(self, route, offset_m=0.8):
        transforms = []
        for wp in route:
            tf = carla.Transform(wp.transform.location, wp.transform.rotation)
            right = tf.get_right_vector()
            tf.location += carla.Location(x=right.x*offset_m, y=right.y*offset_m, z=right.z*offset_m)
            transforms.append(tf)
        return transforms
    def force_lane_change(self, route, direction="left", max_steps=20):
        new_route = []
        for i, wp in enumerate(route):
            if i >= max_steps:
                new_route.extend(route[i:])
                break
            if direction == "left":
                adj = wp.get_left_lane()
            else:
                adj = wp.get_right_lane()
            if adj and adj.lane_type == carla.LaneType.Driving:
                new_route.append(adj)
            else:
                new_route.append(wp)
        return new_route
    def debug_draw_route(self, transforms, life_time=0.3):
        dbg = self.world.debug
        for tf in transforms:
            dbg.draw_point(tf.location, size=0.1, life_time=life_time, color=carla.Color(0, 255, 255))
    def _ensure_pp_state(self):
        if hasattr(self, "_pp_inited"):
            return
        self._pp_inited = True
        self._route_tfs = []       
        self._route_idx = 0    
        self._pp_active = False   
        self._pp_lookahead = 6.0  
        self._pp_target_speed = 12.0 
        self._speed_err_int = 0.0 
        self._last_speed = 0.0
    def send_path_to_tm(self, transforms):
        if not hasattr(self, "traffic_manager"):
            self.hud.error("Traffic Manager not available")
            return
        locs = []
        for tf in transforms:
            locs.append(tf.location if isinstance(tf, carla.Transform) else tf)
        try:
            self.traffic_manager.set_path(self.player, locs)
            #self.player.set_autopilot(True)
            self.player.set_autopilot(True, self.tm_port)
            self.hud.notification("Path sent to Traffic Manager.")
        except Exception as e:
            self.hud.error(f"TM path API not supported: {e}")

    #grp stuff
    def grp_trace_points(self, points):
        """
        points: list[carla.Location]
        return: list[(carla.Waypoint, RoadOption)]
        """
        if len(points) < 2:
            return []
        route = []
        for i in range(len(points) - 1):
            seg = self.grp.trace_route(points[i], points[i + 1])
            if i > 0 and seg and route:
                a = seg[0][0];
                b = route[-1][0]
                if a.road_id == b.road_id and a.lane_id == b.lane_id:
                    seg = seg[1:]
            route += seg
        return route
    def grp_trace_points_with_avoidance(self, points, forbidden_areas):
        """
        points: list[carla.Location]
        return: list[(carla.Waypoint, RoadOption)]
        """
        if len(points) < 2:
            return []
        route = []
        for i in range(len(points) - 1):
            seg = self.grp.trace_route_avoid_areas(points[i], points[i + 1], forbidden_areas)
            if i > 0 and seg and route:
                a = seg[0][0];
                b = route[-1][0]
                if a.road_id == b.road_id and a.lane_id == b.lane_id:
                    seg = seg[1:]
            route += seg
        return route

    def route_to_transforms(self, route):
        return [wp.transform for (wp, _ro) in route]

    def route_to_locations(self, route):
        return [wp.transform.location for (wp, _ro) in route]

    def load_points_from_json(self, json_path: Path):
        """
        支持：
          1) [{"x":..,"y":..,"z":..}, ...]
          2) [[x,y,z], ...] 或 [{"xyz":[x,y,z]}, ...]
        """
        data = json.loads(json_path.read_text(encoding="utf-8"))
        pts = []
        for it in data:
            if isinstance(it, dict) and all(k in it for k in ("x", "y", "z")):
                pts.append(carla.Location(x=float(it["x"]), y=float(it["y"]), z=float(it["z"])))
            elif isinstance(it, dict) and "xyz" in it:
                x, y, z = it["xyz"]
                pts.append(carla.Location(x=float(x), y=float(y), z=float(z)))
            elif isinstance(it, (list, tuple)) and len(it) >= 2:
                x, y = it[0], it[1]
                z = it[2] if len(it) > 2 else 0.0
                pts.append(carla.Location(x=float(x), y=float(y), z=float(z)))
        return pts

    def apply_grp_path(self, locations, use_tm=True, draw=True):
        route = self.grp_trace_points(locations)
        if not route:
            self.hud.error("GRP route empty (points too few / too far from roads?)")
            return
        tfs = self.route_to_transforms(route)
        if draw:
            self.debug_draw_route(tfs, life_time=2.0)
        if use_tm:
            self.send_path_to_tm(tfs)
        else:
            self.hud.notification("GRP route is ready (PID follow not wired here).")

    def l2(a: carla.Location, b: carla.Location) -> float:
        dx, dy, dz = a.x - b.x, a.y - b.y, a.z - b.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def densify_locations(wps: List[carla.Waypoint], spacing: float = 2.0) -> List[carla.Location]:
        if not wps:
            return []
        out: List[carla.Location] = []
        last = wps[0].transform.location
        out.append(last)
        for wp in wps[1:]:
            cur = wp.transform.location
            seg_len = World.l2(last, cur)
            if seg_len > 1e-6:
                n = max(1, int(seg_len // spacing))
                for k in range(1, n + 1):
                    t = min(1.0, k / n)
                    out.append(carla.Location(
                        x=last.x + (cur.x - last.x) * t,
                        y=last.y + (cur.y - last.y) * t,
                        z=last.z + (cur.z - last.z) * t,
                    ))
            last = cur
        return out

    # ---------- 只用 GRP 的核心封装 ----------
    def build_grp(world: carla.World, sampling_resolution: float = 2.0) -> GlobalRoutePlanner:
        wmap = world.get_map()
        grp = GlobalRoutePlanner(wmap, sampling_resolution)
        return grp

    def route_between(grp: GlobalRoutePlanner,
                      start: carla.Location,
                      goal: carla.Location) -> List[Tuple[carla.Waypoint, RoadOption]]:
        return grp.trace_route(start, goal)

    def route_through_anchors(grp: GlobalRoutePlanner,
                              anchors: List[carla.Location]) -> List[Tuple[carla.Waypoint, RoadOption]]:
        full: List[Tuple[carla.Waypoint, RoadOption]] = []
        for i in range(len(anchors) - 1):
            seg = World.route_between(grp, anchors[i], anchors[i + 1])
            if not seg:
                continue
            if full:
                # 去重：如果上段的最后一个 waypoint 与这段第一个相同，就从第二个开始拼
                if seg and full[-1][0].id == seg[0][0].id:
                    seg = seg[1:]
            full.extend(seg)
        return full

    def complete_path_locations(grp: GlobalRoutePlanner,
                                anchors: List[carla.Location],
                                output_spacing: float = 2.0) -> List[carla.Location]:
        plan = World.route_through_anchors(grp, anchors)
        wps = [wp for (wp, _opt) in plan]
        return World.densify_locations(wps, spacing=output_spacing)

    def snap_to_driving_lane(self, loc: carla.Location) -> carla.Location:
        wp = self.map.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        return wp.transform.location

    def vdot(a: carla.Vector3D, b: carla.Vector3D) -> float:
        return a.x * b.x + a.y * b.y + a.z * b.z

    def vnorm(v: carla.Vector3D) -> float:
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    def yaw_to_vec(yaw_deg: float) -> carla.Vector3D:
        # CARLA Yaw: 0°=X+, 90°=Y+, 单位：度
        rad = math.radians(yaw_deg)
        return carla.Vector3D(x=math.cos(rad), y=math.sin(rad), z=0.0)

    def vunit(v: carla.Vector3D) -> carla.Vector3D:
        n = World.vnorm(v) or 1.0
        return carla.Vector3D(v.x / n, v.y / n, 0.0)

    def forward_vec_of_wp(wp: carla.Waypoint) -> carla.Vector3D:
        return World.yaw_to_vec(wp.transform.rotation.yaw)

    def lane_neighbors(wp: carla.Waypoint, max_span: int = 4) -> List[carla.Waypoint]:
        out = [wp]

        cur = wp
        for _ in range(max_span):
            nxt = cur.get_left_lane()
            if nxt and nxt.lane_type == carla.LaneType.Driving:
                out.append(nxt)
                cur = nxt
            else:
                break

        cur = wp
        righties = []
        for _ in range(max_span):
            nxt = cur.get_right_lane()
            if nxt and nxt.lane_type == carla.LaneType.Driving:
                righties.append(nxt)
                cur = nxt
            else:
                break

        return out + righties

    def _angle_diff_deg(a: float, b: float) -> float:
        d = (a - b + 180.0) % 360.0 - 180.0
        return d

    def snap_to_driving_lane_with_yaw(
            self,
            loc: carla.Location,
            desired_yaw_deg: float,
            max_dir_diff_deg: float = 90.0
    ) -> carla.Location:
        wp = self.map.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if wp is None:
            return loc

        yaw_lane = wp.transform.rotation.yaw
        #print(f"yaw lene: {yaw_lane}")
        #print(f"desired yaw: {desired_yaw_deg}")
        diff = abs(World._angle_diff_deg(desired_yaw_deg, yaw_lane))
        #print(f"diff: {diff}")

        if diff > max_dir_diff_deg:
            opposite_lane_id = -wp.lane_id
            #print(f"switch to opposite at {loc.x}, {loc.y}")
            opp_wp = None
            try:
                opp_wp = self.map.get_waypoint_xodr(
                    wp.road_id,
                    opposite_lane_id,
                    wp.s
                )
            except RuntimeError:
                opp_wp = None

            if opp_wp is not None:
                yaw_opp = opp_wp.transform.rotation.yaw
                diff_opp = abs(World._angle_diff_deg(desired_yaw_deg, yaw_opp))
                #print(f"actually switching at {loc.x}, {loc.y}")
                if diff_opp < diff:
                    wp = opp_wp
            if opp_wp is None:
                #print("none opp_wp execption")
                yaw_opp_desire = -yaw_lane
                search_dir = yaw_lane-90
                if search_dir > 180:
                    search_dir -=360
                if search_dir <-180:
                    search_dir +=360
                new_loc = World.move_back_along_yaw(loc, search_dir, -7)
                t_opp_wp = self.map.get_waypoint(
                    new_loc,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )
                wp = t_opp_wp
                #print("none opp_wp case done")

        return wp.transform.location

    def _move_before_junction(
            self,
            wp: carla.Waypoint,
            step: float = 2.0,
            max_dist: float = 20.0
    ) -> carla.Waypoint:
        current = wp
        dist = 0.0

        while current.is_junction and dist < max_dist:
            prev_list = current.previous(step)
            if not prev_list:
                break
            current = prev_list[0]
            dist += step

        return current

    def snap_to_driving_lane_with_yaw_with_fallback(
            self,
            loc: carla.Location,
            desired_yaw_deg: float,
            max_dir_diff_deg: float = 60.0
    ) -> carla.Location:

        wp = self.map.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if wp is None:
            return loc

        yaw_lane = wp.transform.rotation.yaw
        diff = abs(World._angle_diff_deg(desired_yaw_deg, yaw_lane))

        if wp.is_junction:
            wp = self._move_before_junction(wp, step=1.0, max_dist=20.0)

        if diff > max_dir_diff_deg:
            opposite_lane_id = -wp.lane_id
            opp_wp = None
            try:
                opp_wp = self.map.get_waypoint_xodr(
                    wp.road_id,
                    opposite_lane_id,
                    wp.s
                )
            except RuntimeError:
                opp_wp = None

            if opp_wp is not None:
                yaw_opp = opp_wp.transform.rotation.yaw
                diff_opp = abs(World._angle_diff_deg(desired_yaw_deg, yaw_opp))
                if diff_opp < diff:
                    wp = opp_wp

        return wp.transform.location

    def yaw_from_two_points(p0: carla.Location, p1: carla.Location) -> float:
        dx = p1.x - p0.x
        dy = p1.y - p0.y

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0

        yaw_rad = math.atan2(dy, dx)  # -pi ~ pi
        yaw_deg = math.degrees(yaw_rad)  # -180 ~ 180

        return yaw_deg

    def snap_to_forward_waypoint(cmap: carla.Map,
                                 loc: carla.Location,
                                 d_global: carla.Vector3D,
                                 max_side_lanes: int = 4,
                                 cos_limit: float = 0.3) -> carla.Waypoint:
        base = cmap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        candidates = World.lane_neighbors(base, max_side_lanes)
        d = World.vunit(d_global)
        best, best_cos = None, -2.0
        for wp in candidates:
            f = World.vunit(World.forward_vec_of_wp(wp))
            c = World.vdot(f, d)
            if c > best_cos:
                best, best_cos = wp, c
        return best if best_cos >= cos_limit else best

    def yaw_from_two_points(p0: carla.Location, p1: carla.Location) -> float:
        dx = p1.x - p0.x
        dy = p1.y - p0.y

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0

        yaw_rad = math.atan2(dy, dx)  # -pi ~ pi
        yaw_deg = math.degrees(yaw_rad)  # -180 ~ 180

        return yaw_deg

    def move_back_along_yaw(loc: carla.Location, yaw_deg: float, dist: float) -> carla.Location:
        rot = carla.Rotation(pitch=0.0, yaw=yaw_deg, roll=0.0)
        fwd = rot.get_forward_vector()  # 车头方向的单位向量
        # 反向（后退）：减去 forward * dist
        return carla.Location(
            x=loc.x - fwd.x * dist,
            y=loc.y - fwd.y * dist,
            z=loc.z - fwd.z * dist
        )

    def route_from_sparse(self, anchors: list[carla.Location], simulate=False, draw=True, use_tm=False):
        if not anchors or len(anchors) < 2:
            self.hud.error("Need at least 2 anchors.")
            return []

        pts = []
        yaws = []
        player = self.player
        transform = player.get_transform()
        init_yaw = transform.rotation.yaw
        yaws.append(init_yaw)
        should_remove_last = False

        # remove last if too close
        if anchors[len(anchors) - 1].distance(anchors[len(anchors) - 2]) < 20: #too close then remove the last one
            should_remove_last = True
            # last_unit = anchors[len(anchors) - 1]
            # anchors.remove(last_unit)

        for i in range(len(anchors) - 1):
            new_yaw = World.yaw_from_two_points(anchors[i], anchors[i + 1])
            yaws.append(new_yaw)
        #print(f"yaws index{len(yaws)}")
        #print(f"anchor index{len(anchors)}")
        #print(f"yaws: {yaws}")
        #print(f"anchors : {anchors}")
        yaw_index = 0
        for p in anchors:
            #snapped = self.snap_to_driving_lane(p)
            if yaw_index<(len(yaws) - 1) and yaw_index !=0:
                back_p = World.move_back_along_yaw(p, yaws[yaw_index+1], -17.0) #move forward
            elif yaw_index==(len(yaws) - 1):
                back_p = World.move_back_along_yaw(p, yaws[yaw_index], 17.0)  # move backward
            else:
                back_p = p
            #snapped = self.snap_to_driving_lane_with_yaw(p,yaws[yaw_index])
            #snapped = self.snap_to_driving_lane_with_yaw_with_fallback(p, yaws[yaw_index])

            if(yaw_index==(len(yaws) - 1)):
                #snapped = back_p
                snapped = self.snap_to_driving_lane_with_yaw(back_p, yaws[yaw_index])
            else:
                snapped = self.snap_to_driving_lane_with_yaw(back_p, yaws[yaw_index + 1])  # move fwd
            pts.append(snapped)
            yaw_index +=1

        if pts[len(pts) - 1].distance(pts[len(pts) - 2]) < 20: #too close then remove the last one
            should_remove_last = True

        if should_remove_last:
            last_point = pts[len(pts)-1]
            pts.remove(last_point)

        for locp in anchors: # debug draw points
            self.world.debug.draw_point(
                locp + carla.Location(z=0.2),
                size=0.1,
                color=carla.Color(0, 255, 0),
                life_time=15.0,
                persistent_lines=False
            )

        for loc in pts: # debug draw points
            self.world.debug.draw_point(
                loc + carla.Location(z=0.2),
                size=0.1,
                color=carla.Color(255, 0, 0),  # 红色
                life_time=15.0,
                persistent_lines=False
            )

        route_wp_ro = self.grp_trace_points(pts)
        if not route_wp_ro:
            self.hud.error("GRP failed: empty route (points off-road or disconnected).")
            return []

        if draw:
            tfs = [wp.transform for (wp, _ro) in route_wp_ro]
            self.debug_draw_route(tfs, life_time=15.0)

        if simulate:
            print("well this is just simulation")
            return [(wp, (ro if ro is not None else RoadOption.LANEFOLLOW))
                           for (wp, ro) in route_wp_ro]
        if use_tm:
            locs = [wp.transform.location for (wp, _ro) in route_wp_ro]
            step = 5
            locs = [locs[i] for i in range(0, len(locs), step)] + ([locs[-1]] if locs else [])
            self.send_path_to_tm(locs)
            return [(wp, (ro if ro is not None else RoadOption.LANEFOLLOW))
                           for (wp, ro) in route_wp_ro]
        else:
            global_plan = [(wp, (ro if ro is not None else RoadOption.LANEFOLLOW))
                           for (wp, ro) in route_wp_ro]
            self._engage_behavior_agent(global_plan)
            return global_plan

    def _engage_behavior_agent(self, global_plan):
        if self.agent is None or (hasattr(self.agent, 'vehicle') and self.agent.vehicle.id != self.player.id):
            try:
                self.agent = BehaviorAgent(self.player, behavior='normal')
                self.agent.follow_speed_limits(False)
                self.agent.set_target_speed(40)
                self.agent.ignore_traffic_lights(True)
            except Exception as e:
                self.hud.error(f"Recreate BehaviorAgent failed: {e}")
                return

        try:
            self.player.set_autopilot(False)
        except Exception:
            pass

        try:
            self.agent.set_global_plan(global_plan)
            self._agent_active = True
            self.hud.notification("BehaviorAgent engaged.")
        except Exception as e:
            self.hud.error(f"BehaviorAgent engage failed: {e}")
            self._agent_active = False
    def route_from_sparse_avoided(self, anchors: list[carla.Location],forbidden_area , simulate=False, draw=True, use_tm=False):
        if not anchors or len(anchors) < 2:
            self.hud.error("Need at least 2 anchors.")
            return []

        raw_size = len(anchors)
        anchors = [anchors[0], anchors[raw_size - 1]] # select start and destination
        pts = []
        yaws = []
        player = self.player
        transform = player.get_transform()
        init_yaw = transform.rotation.yaw
        yaws.append(init_yaw)
        for i in range(len(anchors) - 1):
            new_yaw = World.yaw_from_two_points(anchors[i], anchors[i + 1])
            yaws.append(new_yaw)
        print(f"yaws index{len(yaws)}")
        print(f"anchor index{len(anchors)}")
        print(f"yaws: {yaws}")
        print(f"anchors : {anchors}")
        yaw_index = 0
        for p in anchors:
            #snapped = self.snap_to_driving_lane(p)
            if yaw_index<(len(yaws) - 1) and yaw_index !=0:
                back_p = World.move_back_along_yaw(p, yaws[yaw_index+1], -17.0) #move forward
            elif yaw_index==(len(yaws) - 1):
                back_p = World.move_back_along_yaw(p, yaws[yaw_index], 17.0)  # move backward
            else:
                back_p = p
            #snapped = self.snap_to_driving_lane_with_yaw(p,yaws[yaw_index])
            #snapped = self.snap_to_driving_lane_with_yaw_with_fallback(p, yaws[yaw_index])

            if(yaw_index==(len(yaws) - 1)):
                #snapped = back_p
                snapped = self.snap_to_driving_lane_with_yaw(back_p, yaws[yaw_index])
            else:
                snapped = self.snap_to_driving_lane_with_yaw(back_p, yaws[yaw_index + 1])  # move fwd
            pts.append(snapped)
            yaw_index +=1

        for locp in anchors: # debug draw points
            self.world.debug.draw_point(
                locp + carla.Location(z=0.2),
                size=0.1,
                color=carla.Color(0, 255, 0),
                life_time=5.0,
                persistent_lines=False
            )

        for loc in pts: # debug draw points
            self.world.debug.draw_point(
                loc + carla.Location(z=0.2),
                size=0.1,
                color=carla.Color(255, 0, 0),
                life_time=5.0,
                persistent_lines=False
            )

        route_wp_ro = self.grp_trace_points_with_avoidance(pts, forbidden_area)
        if not route_wp_ro:
            self.hud.error("GRP failed: empty route (points off-road or disconnected).")
            return []

        # 3) 画线
        if draw:
            tfs = [wp.transform for (wp, _ro) in route_wp_ro]
            self.debug_draw_route(tfs, life_time=5.0)

        # 4) 下发 TM
        if simulate:
            print("well this is just simulation")
            return [(wp, (ro if ro is not None else RoadOption.LANEFOLLOW))
                           for (wp, ro) in route_wp_ro]
        if use_tm:
            locs = [wp.transform.location for (wp, _ro) in route_wp_ro]
            step = 5
            locs = [locs[i] for i in range(0, len(locs), step)] + ([locs[-1]] if locs else [])
            self.send_path_to_tm(locs)
            return [(wp, (ro if ro is not None else RoadOption.LANEFOLLOW))
                           for (wp, ro) in route_wp_ro]
        else:
            #global_plan = [(wp.transform, ro) for (wp, ro) in route_wp_ro]
            global_plan = [(wp, (ro if ro is not None else RoadOption.LANEFOLLOW))
                           for (wp, ro) in route_wp_ro]
            #self.agent.set_global_plan(global_plan)
            self._engage_behavior_agent(global_plan)
            return global_plan

    def calc_global_plan_length(self, global_plan):
        if len(global_plan) < 2:
            return 0.0

        def to_location(p):
            # (waypoint, road_option)
            if isinstance(p, (tuple, list)):
                p = p[0]

            # carla.Waypoint
            if hasattr(p, "transform") and hasattr(p.transform, "location"):
                return p.transform.location

            # carla.Location
            if hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
                return p

            raise TypeError(f"Unsupported point type: {type(p)}, value={p}")

        total_length = 0.0
        prev_loc = to_location(global_plan[0])

        for i in range(1, len(global_plan)):
            curr_loc = to_location(global_plan[i])
            total_length += prev_loc.distance(curr_loc)
            prev_loc = curr_loc

        print("total length:", total_length)
        return total_length

    def path_history_to_prompt_text(self, all_paths):
        data = [
            [{"x": p.x, "y": p.y, "z": p.z} for p in path]
            for path in all_paths
        ]
        return json.dumps(data, ensure_ascii=False, indent=2)

    def parse_code(input_text: str):
        pattern = r"```python(.*?)```"
        matches = re.findall(pattern, input_text, re.DOTALL)
        if not matches:
            return None
        return "\n".join(matches)

    def _extract_all_anchors_rhs(code: str):

        out = []
        for m in re.finditer(r"\b(anchors\w*)\s*=\s*\[", code):
            name = m.group(1)
            i = m.end() - 1
            depth = 0
            for j, ch in enumerate(code[i:], start=i):
                if ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        out.append((name, code[i:j + 1]))
                        break
        return out

    def parse_all_anchors(input_text: str, env: dict):

        code = World.parse_code(input_text) or input_text
        pairs = World._extract_all_anchors_rhs(code)
        results = []
        for name, rhs in pairs:
            val = eval(rhs, {"__builtins__": {}}, env)  # 仅用于可信文本
            results.append(val)
        return results

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        self._ackermann_enabled = False
        self._ackermann_reverse = 1
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._ackermann_control = carla.VehicleAckermannControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)
        #radar_sensor_instance = RadarSensor(carla.Vehicle)

    def parse_events(self, client, world, clock, sync_mode):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                # elif event.key == K_c:
                #     world.next_weather()
                elif event.key == K_j:
                    base_route = world.sample_forward_route(step_m=3.0, length_m=120.0)
                    lc_route = world.force_lane_change(base_route, direction="left", max_steps=80)
                    tf_route = world.apply_lateral_offset(base_route, offset_m=0.06)
                    world.debug_draw_route(tf_route)
                    world.hud.notification("Drawn local route.")
                    #print(base_route)
                elif event.key == K_k:
                    # base_route = world.sample_forward_route(step_m=3.0, length_m=120.0)
                    # lc_route = world.force_lane_change(base_route, direction="left", max_steps=20)
                    # tf_route = world.apply_lateral_offset(lc_route, offset_m=0.6)
                    # world.debug_draw_route(tf_route, life_time=1.0)
                    # world.send_path_to_tm(tf_route)
                    if world.llm_all_paths is not None:
                        world.llm_selected_paths = world.llm_all_paths[0]
                        print("Selected route 1")

                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_y:
                    #RadarSensor.print_snapshot(world.radar_sensor)
                    if world.llm_all_paths is not None:
                        world.llm_selected_paths = world.llm_all_paths[1]
                        print("Selected route 2")
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    #world.camera_manager.next_sensor()
                    if world.llm_all_paths is not None:
                        world.llm_selected_paths = world.llm_all_paths[2]
                        print("Selected route 3")
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                elif event.key == K_o:
                    try:
                        if world.doors_are_open:
                            world.hud.notification("Closing Doors")
                            world.doors_are_open = False
                            world.player.close_door(carla.VehicleDoor.All)
                        else:
                            world.hud.notification("Opening doors")
                            world.doors_are_open = True
                            world.player.open_door(carla.VehicleDoor.All)
                    except Exception:
                        pass
                elif event.key == K_t:
                    if world.show_vehicle_telemetry:
                        world.player.show_debug_telemetry(False)
                        world.show_vehicle_telemetry = False
                        world.hud.notification("Disabled Vehicle Telemetry")
                    else:
                        try:
                            world.player.show_debug_telemetry(True)
                            world.show_vehicle_telemetry = True
                            world.hud.notification("Enabled Vehicle Telemetry")
                        except Exception:
                            pass
                elif event.key > K_0 and event.key <= K_9:
                    index_ctrl = 0
                    if pygame.key.get_mods() & KMOD_CTRL:
                        index_ctrl = 9
                    world.camera_manager.set_sensor(event.key - 1 - K_0 + index_ctrl)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.log")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.log'")
                    # replayer
                    client.replay_file("manual_recording.log", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_f: #mark unuse
                        # Toggle ackermann controller
                        # self._ackermann_enabled = not self._ackermann_enabled
                        # world.hud.show_ackermann_info(self._ackermann_enabled)
                        # world.hud.notification("Ackermann Controller %s" %
                        #                        ("Enabled" if self._ackermann_enabled else "Disabled"))
                        print(f"location{world.player.get_transform().location}")
                    if event.key == K_q:
                        if not self._ackermann_enabled:
                            self._control.gear = 1 if self._control.reverse else -1
                        else:
                            self._ackermann_reverse *= -1
                            # Reset ackermann control
                            self._ackermann_control = carla.VehicleAckermannControl()
                    elif event.key == K_m: #use here
                        # print("sending things to LLM!")
                        # prompt_path = (Path(__file__).parent / "prompts" / "navigation_prompt.txt")
                        # print(prompt_path)
                        # prompt_text = llmutils.parse_text_prompt(prompt_path)
                        # cur = world.player.get_transform().location
                        # dest = destination
                        # avoided = avoided_area
                        # map_points = gpd.read_file(map_file)
                        # player = world.player
                        # transform = player.get_transform()
                        # yaw = transform.rotation.yaw
                        # prompt_text = prompt_text + [{"role": "user", "content": f" Car current position at: {cur}, destination is: {dest}, user starts in yaw direction:{yaw}, avoided points are:{avoided} map points: {map_points}"}]
                        # print(prompt_text)
                        # world.llm_response = llmutils.query_llm_async(prompt_text, seed=0)
                        # print(world.llm_response)
                        self.send_to_llm(world)
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        if not self._autopilot_enabled and not sync_mode:
                            print("WARNING: You are currently in asynchronous mode and could "
                                  "experience some issues with the traffic simulation")
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                         # current_lights ^= carla.VehicleLightState.Interior
                         #spawns = world.map.get_spawn_points() # grp stuff
                         player = world.player
                         transform = player.get_transform()
                         yaw = transform.rotation.yaw
                         print(yaw)
                         # if not spawns:
                         #     world.hud.error("No spawn points on map.")
                         # else:
                         #     cur = world.player.get_transform().location
                         #     dst = random.choice(spawns).location
                         #     world.apply_grp_path([cur, dst], use_tm=True, draw=True)
                    elif event.key == K_KP1:
                        if world.llm_all_paths is not None:
                            world.llm_selected_paths = world.llm_all_paths[0]
                            print("Selected route 1")
                            try:
                                if world.llm_selected_paths is not None:
                                    world.route_from_sparse(world.llm_selected_paths, True, False)
                                    # world.llm_selected_paths = None
                                else:
                                    print("no path selected")
                            except Exception as e:
                                world.hud.error(f"Set path failed: {e}")
                    elif event.key == K_KP2:
                        if world.llm_all_paths is not None:
                            world.llm_selected_paths = world.llm_all_paths[1]
                            print("Selected route 2")
                            try:
                                if world.llm_selected_paths is not None:
                                    world.route_from_sparse(world.llm_selected_paths, True, False)
                                    # world.llm_selected_paths = None
                                else:
                                    print("no path selected")
                            except Exception as e:
                                world.hud.error(f"Set path failed: {e}")
                    elif event.key == K_KP3:
                        if world.llm_all_paths is not None:
                            world.llm_selected_paths = world.llm_all_paths[2]
                            print("Selected route 3")
                            try:
                                if world.llm_selected_paths is not None:
                                    world.route_from_sparse(world.llm_selected_paths, True, False)
                                    # world.llm_selected_paths = None
                                else:
                                    print("no path selected")
                            except Exception as e:
                                world.hud.error(f"Set path failed: {e}")
                    elif event.key == K_u: # set a path
                        try:
                            cur = world.player.get_transform().location
                            raw_points = [
                                {"x": 50.0, "y": -65.0, "z": 0.1}
                            ]
                            anchors = [cur,carla.Location(50.0, -65.0, 0.0)]

                            #locs = [carla.Location(p["x"], p["y"], p.get("z", 0.0)) for p in raw_points]
                            locs = [carla.Location(p["x"], p["y"], p.get("z", 0.0)) for p in raw_points]
                            waypoints = World.complete_path_locations(world.grp,anchors,2)

                            mmap = world.world.get_map()
                            wps = [
                                mmap.get_waypoint(
                                    location,
                                    project_to_road=True,
                                    lane_type=carla.LaneType.Driving
                                )
                                for location in locs
                            ]

                            #world.apply_grp_path(waypoints, use_tm=True, draw=True)

                            #Use this
                            #world.route_from_sparse(anchors)
                            if world.llm_selected_paths is not None:
                                world.route_from_sparse(world.llm_selected_paths)
                                #world.llm_selected_paths = None
                            else:
                                print("no path selected")

                            #self._autopilot_enabled = True

                        except Exception as e:
                            world.hud.error(f"Set path failed: {e}")
                    elif event.key == K_c: # actually moving
                        try:
                            if world.llm_selected_paths is not None:
                                world.route_from_sparse(world.llm_selected_paths, False)
                                # world.llm_selected_paths = None
                            else:
                                print("no path selected")
                        except Exception as e:
                            world.hud.error(f"Set path failed: {e}")
                    elif event.key == K_z: #preview LLM
                        try:
                            if world.llm_selected_paths is not None:
                                points = world.route_from_sparse(world.llm_selected_paths, True)
                                world.calc_global_plan_length(points)
                                #world.llm_selected_paths = None
                            else:
                                print("no path selected")
                        except Exception as e:
                            world.hud.error(f"Set path failed: {e}")
                        print(1)
                        #current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x: #preview A* with exclusive area
                        try:
                            if world.llm_selected_paths is not None:
                                #avoided = [(carla.Location(0.0, 70.0, 0.0),25)]  //raw
                                avoided = avoided_area
                                #avoided = []
                                #locs = [world.player.get_transform().location, destination]
                                world.route_from_sparse_avoided(world.llm_selected_paths, avoided, True)
                                #world.route_from_sparse_avoided(locs, avoided, False)
                                #world.llm_selected_paths = None
                            else:
                                locs = [world.player.get_transform().location, destination]
                                avoided = avoided_area
                                world.route_from_sparse_avoided(locs, avoided, False)
                                print("no path selected")
                        except Exception as e:
                            world.hud.error(f"Set path failed: {e}")
                        print(1)
                        #current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
                # Apply control
                if not self._ackermann_enabled:
                    world.player.apply_control(self._control)
                else:
                    world.player.apply_ackermann_control(self._ackermann_control)
                    # Update control to the last one applied by the ackermann controller.
                    self._control = world.player.get_control()
                    # Update hud with the newest ackermann control
                    world.hud.update_ackermann_control(self._ackermann_control)

            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
                world.player.apply_control(self._control)
    def send_to_llm(self, world, additional_prompt = None):
        print("sending things to LLM!")
        prompt_path = (Path(__file__).parent / "prompts" / "navigation_prompt.txt")
        print(prompt_path)
        prompt_text = llmutils.parse_text_prompt(prompt_path)
        cur = world.player.get_transform().location
        dest = destination
        avoided = avoided_area
        map_points = gpd.read_file(map_file)
        player = world.player
        transform = player.get_transform()
        yaw = transform.rotation.yaw
        prompt_text = prompt_text + [{"role": "user",
                                      "content": f" Car current position at: {cur}, destination is: {dest}, user starts in yaw direction:{yaw}, avoided points are:{avoided} map points: {map_points}"}]
        if additional_prompt is not None:
            prompt_text = prompt_text + [{"role": "user", "content": f"The past attempt failed to obtain a feasible path, the history is: {additional_prompt}, generate different paths"}]
        print(prompt_text)
        world.llm_response = llmutils.query_llm_async(prompt_text, seed=0)
        print(world.llm_response)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            if not self._ackermann_enabled:
                self._control.throttle = min(self._control.throttle + 0.1, 1.00)
            else:
                self._ackermann_control.speed += round(milliseconds * 0.005, 2) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            if not self._ackermann_enabled:
                self._control.brake = min(self._control.brake + 0.2, 1)
            else:
                self._ackermann_control.speed -= min(abs(self._ackermann_control.speed), round(milliseconds * 0.005, 2)) * self._ackermann_reverse
                self._ackermann_control.speed = max(0, abs(self._ackermann_control.speed)) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        if not self._ackermann_enabled:
            self._control.steer = round(self._steer_cache, 1)
            self._control.hand_brake = keys[K_SPACE]
        else:
            self._ackermann_control.steer = round(self._steer_cache, 1)

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self.llm_response = None

        self._show_ackermann_info = False
        self._ackermann_control = carla.VehicleAckermannControl()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        # if max_col>1000 and self.last_max_col < 10:
        #     CollisionSensor.onColission(self, world, max_col)
        self.last_max_col = max_col
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
            if self._show_ackermann_info:
                self._info_text += [
                    '',
                    'Ackermann Controller:',
                    '  Target speed: % 8.0f km/h' % (3.6*self._ackermann_control.speed),
                ]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))
        if self.llm_response is not None and self.llm_response.done():
            print("1")
            try:
                text, elapsed = self.llm_response.result(timeout=0)
                self.llm_text = text
                if not text:
                    text = "[no content]"
                print(f"[LLM DONE in {elapsed:.2f}s] {text}")
            except Exception as e:
                print("[LLM ERROR]", e)
            finally:
                self.llm_response = None

    def show_ackermann_info(self, enabled):
        self._show_ackermann_info = enabled

    def update_ackermann_control(self, ackermann_control):
        self._ackermann_control = ackermann_control

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        #radar_sensor_instance = RadarSensor(self.sensor)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)
    
    def onColission(self,world, collision):
        print("vehicle collided!")
        prompt_path = (Path(__file__).parent / "prompts" / "collision_prompt.txt")
        print(prompt_path)
        prompt_text = llmutils.parse_text_prompt(prompt_path)
        prompt_text = prompt_text + [{"role": "user", "content": f" Car Max Impact: {collision}"}]
        print(prompt_text)
        self.llm_response = llmutils.query_llm_async(prompt_text,seed=0)
        print(self.llm_response)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None

        # If the spawn object is not a vehicle, we cannot use the Lane Invasion Sensor
        if parent_actor.type_id.startswith("vehicle."):
            self._parent = parent_actor
            self.hud = hud
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid circular
            # reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z

        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug

        # ===== 新增：缓存最近一帧雷达数据（线程安全） =====
        self._lock = threading.Lock()
        self._last_frame = -1
        self._last_transform = None   # radar_data.transform
        self._last_points = None      # numpy array shape (N,4): [vel, alt(rad), azi(rad), depth]

        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=bound_x + 0.05, z=bound_z+0.05),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        pts = np.frombuffer(radar_data.raw_data, dtype=np.float32)
        pts = np.reshape(pts, (len(radar_data), 4))

        with self._lock:
            self._last_frame = radar_data.frame
            self._last_transform = radar_data.transform
            self._last_points = pts

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))
            
    def snapshot(self):
        with self._lock:
            if self._last_points is None:
                return None
            return {
                "frame": self._last_frame,
                "transform": self._last_transform,
                "points": self._last_points.copy()
            }
        
    def print_snapshot(self):
        snap = self.snapshot()
        if snap is None:
            print("[RADAR] No Data")
            return
        pts = snap["points"]
        v, alt, azi, dep = pts[:,0], pts[:,1], pts[:,2], pts[:,3]
        print(f"[RADAR] frame={snap['frame']} N={len(pts)} "
              f"vel=[{v.min():.2f},{v.max():.2f}] m/s "
              f"depth=[{dep.min():.2f},{dep.max():.2f}] m")
        alt_deg, azi_deg = np.degrees(alt), np.degrees(azi)
        for i in range(len(pts)):
            print(f"  #{i:05d} depth={dep[i]:.3f} m, vel={v[i]:.3f} m/s, "
                  f"az={azi_deg[i]:.2f}°, alt={alt_deg[i]:.2f}°")
            
    def save_csv(self, path="radar_latest.csv"):
        snap = self.snapshot()
        if snap is None:
            print("[RADAR] No saved data")
            return
        pts = snap["points"]
        out = np.column_stack([pts[:,3], pts[:,0], np.degrees(pts[:,2]), np.degrees(pts[:,1])])
        np.savetxt(path, out, delimiter=",",
                   header="depth_m,velocity_mps,azimuth_deg,altitude_deg",
                   comments="", fmt="%.6f")
        print(f"[RADAR] saved CSV -> {path}")

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        if not self._parent.type_id.startswith("walker.pedestrian"):
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), Attachment.Rigid)]
        else:
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=2.5, y=0.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=0, y=-2.5, z=-0.0), carla.Rotation(yaw=90.0)), Attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.CityScapesPalette, 'Camera Instance Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.Raw, 'Camera Instance Segmentation (Raw)', {}],
            ['sensor.camera.cosmos_visualization', cc.Raw, 'Cosmos Control Visualization', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}],
            ['sensor.camera.optical_flow', cc.Raw, 'Optical Flow', {}],
            ['sensor.camera.normals', cc.Raw, 'Camera Normals', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        elif self.sensors[self.index][0].startswith('sensor.camera.optical_flow'):
            image = image.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None
    send_to_llm_next_tick = False
    point_history = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2000.0)

        sim_world = client.get_world()
        if switch_map:
            print("map switched")
            #sim_world = client.load_world(alternative_map)
            with open('custom.xodr', 'r', encoding='utf-8') as f:
                xodr_data = f.read()
            sim_world = client.generate_opendrive_world(
                xodr_data,
                carla.OpendriveGenerationParameters(vertex_distance=2.0,
                max_road_length=500.0,
                wall_height=1.0,
                additional_width=0.6,
                smooth_junctions=True,
                enable_mesh_visibility=True
                )
            )
            #sim_world = client.load_world("Town03") # uncomment if you want to change map
        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                  "experience some issues with the traffic simulation")

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(client, sim_world, hud, args)
        controller = KeyboardControl(world, args.autopilot)

        world.traffic_manager = client.get_trafficmanager()
        if args.sync:
            world.traffic_manager.set_synchronous_mode(True)
        world.traffic_manager.ignore_lights_percentage(world.player, 100)
        world.traffic_manager.auto_lane_change(world.player, True)
        world.traffic_manager.vehicle_percentage_speed_difference(world.player, 0)
        world.traffic_manager.ignore_vehicles_percentage(world.player, 100)


        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()
        while True:
            if args.sync:
                sim_world.tick()
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock, args.sync):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()
            #print(world.llm_response)
            if world.llm_response is not None and world.llm_response.done():
                print("1")
                try:
                    text, elapsed = world.llm_response.result(timeout=0)
                    world.llm_text = text
                    if not text:
                        text = "[no content]"
                    #print(f"[LLM DONE in {elapsed:.2f}s] {text[:20000]}")
                    env = {"carla": carla}
                    all_paths = World.parse_all_anchors(text, env)
                    world.llm_all_paths = World.parse_all_anchors(text, env)
                    world.llm_selected_paths = world.llm_all_paths[0] #just for display
                    #print(f"All Paths {world.llm_all_paths}")
                    selected_path = None
                    cur = world.player.get_transform().location
                    dest = destination
                    dist = cur.distance(dest) * 3
                    if all_paths is not None:
                        num_of_paths = len(all_paths)
                        print(f"path length {num_of_paths}")
                        for i in range(num_of_paths):
                            path_point = world.route_from_sparse(all_paths[i], True, False)
                            path_len = world.calc_global_plan_length(path_point)
                            if path_len < dist and path_len < world.calc_global_plan_length(all_paths[i]) * 1.5:
                                selected_path = all_paths[i]
                                break

                    if selected_path is not None:
                        print("moving")
                        point_history = None
                        world.route_from_sparse(selected_path, False, True)
                    else:
                        #point_history = world.path_history_to_prompt_text(all_paths)
                        if point_history:
                            point_history += "\n\nPrevious paths:\n" + world.path_history_to_prompt_text(all_paths)
                        else:
                            point_history = "Previous paths:\n" + world.path_history_to_prompt_text(all_paths)
                        send_to_llm_next_tick = True
                except Exception as e:
                    print("[LLM ERROR]", e)
                finally:
                    world.llm_response = None
            if send_to_llm_next_tick:
                print("iterating")
                controller.send_to_llm(world, point_history)
                send_to_llm_next_tick = False

    finally:

        if original_settings:
            sim_world.apply_settings(original_settings)

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
