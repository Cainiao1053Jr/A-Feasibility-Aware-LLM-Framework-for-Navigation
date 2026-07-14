"""
position_controller.py

Coordinate control system: tracks the drone's current position and converts absolute coordinate commands into MotionCommander incremental controls.

World coordinate frame convention:
  x - the drone's initial heading at takeoff (forward)
  y - the drone's left at takeoff
  z - vertically up

Control strategy (body frame):
  1. Compute the world-frame vector from the current position to the target position.
  2. Derive the absolute yaw angle to turn to from the vector direction.
  3. Use turn_left / turn_right to rotate the body toward the target direction.
  4. Use forward to fly along the nose direction for the distance corresponding to the vector length.
  5. Handle the vertical direction separately with up / down.
  The initial heading is fixed to the world-frame +x direction (yaw = 0°).
"""

import math
import time
import cflib.crtp
import matplotlib.pyplot as plt
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander


class PositionController:
    """
    Crazyflie position controller based on absolute world coordinates.

    Usage example:
        with MotionCommander(scf, default_height=0.5) as mc:
            pc = PositionController(mc)
            pc.goto(1.0, 0.0, 0.5)   # turn toward +x, then fly forward 1m
            pc.goto(1.0, 1.0, 0.5)   # turn toward +y, then fly forward 1m
            pc.turn_to(90)            # rotate to absolute yaw 90°
            pc.goto(0.0, 0.0, 0.5)   # turn toward the origin, then fly back
    """

    DEFAULT_VELOCITY = 0.3  # m/s
    SLEEP_TIME = 1

    def __init__(
        self,
        motion_commander: MotionCommander,
        initial_x: float = 0.0,
        initial_y: float = 0.0,
        initial_z: float = 0.0,
        initial_yaw: float = 0.0,
        scf: SyncCrazyflie = None,
        sensor_log_period_ms: int = 100,
    ):
        self._mc = motion_commander
        self._x = float(initial_x)
        self._y = float(initial_y)
        self._z = float(initial_z)
        self._yaw = float(initial_yaw)

        self._sensor_positions: list[tuple[float, float]] = []
        self._log_config: LogConfig | None = None

        if scf is not None:
            self._start_sensor_logging(scf.cf, sensor_log_period_ms)

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def z(self) -> float:
        return self._z

    @property
    def yaw(self) -> float:
        return self._yaw

    @property
    def position(self) -> tuple:
        return (self._x, self._y, self._z)

    def goto(self, x: float, y: float, z: float, velocity: float = DEFAULT_VELOCITY):
        dx_world = x - self._x
        dy_world = y - self._y
        dz = z - self._z

        horiz_dist = math.sqrt(dx_world ** 2 + dy_world ** 2)

        if horiz_dist > 1e-3:
            target_angle = math.degrees(math.atan2(dy_world, dx_world))

            def _norm(a):
                return (a + 180) % 360 - 180

            candidates = [
                (_norm(target_angle),       'forward'),
                (_norm(target_angle + 180), 'back'),
                (_norm(target_angle - 90),  'left'),
                (_norm(target_angle + 90),  'right'),
            ]

            valid = [(yaw, cmd) for yaw, cmd in candidates if -90 <= yaw <= 90]

            desired_yaw, move_cmd = min(
                valid,
                key=lambda c: abs((c[0] - self._yaw + 180) % 360 - 180),
            )

            delta = (desired_yaw - self._yaw + 180) % 360 - 180

            if delta > 0.1:
                self._mc.turn_left(delta)
            elif delta < -0.1:
                self._mc.turn_right(abs(delta))
            self._yaw = desired_yaw

            time.sleep(1.5)

            print(f"[goto] turn {delta:.1f}°, {move_cmd} {horiz_dist:.3f}m, flight time: {horiz_dist/velocity:.2f}s")
            if move_cmd == 'forward':
                self._mc.forward(horiz_dist, velocity=velocity)
            elif move_cmd == 'back':
                self._mc.back(horiz_dist, velocity=velocity)
            elif move_cmd == 'left':
                self._mc.left(horiz_dist, velocity=velocity)
            else:
                self._mc.right(horiz_dist, velocity=velocity)
            self._x = x
            self._y = y
            time.sleep(1.5)

        if abs(dz) > 1e-3:
            vert_time = abs(dz) / velocity
            print(f"[goto] {'up' if dz > 0 else 'down'} {abs(dz):.3f}m, flight time: {vert_time:.2f}s")
            if dz > 0:
                self._mc.up(dz, velocity=velocity)
            else:
                self._mc.down(abs(dz), velocity=velocity)
            self._z = z

        if horiz_dist > 1e-3 or abs(dz) > 1e-3:
            self._log(f"goto ({x:.3f}, {y:.3f}, {z:.3f})")


    def move_to(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        velocity: float = DEFAULT_VELOCITY,
    ):
        tx = x if x is not None else self._x
        ty = y if y is not None else self._y
        tz = z if z is not None else self._z
        self.goto(tx, ty, tz, velocity=velocity)

    def turn_to(self, target_yaw: float):

        delta = (target_yaw - self._yaw + 180) % 360 - 180
        if abs(delta) < 0.1:
            return
        if delta > 0:
            self._mc.turn_left(abs(delta))
        else:
            self._mc.turn_right(abs(delta))
        self._yaw = target_yaw % 360
        self._log(f"turn_to {self._yaw:.1f} deg")

    def turn_by(self, delta_yaw: float):

        self.turn_to(self._yaw + delta_yaw)


    def set_position(self, x: float, y: float, z: float):
        self._x = float(x)
        self._y = float(y)
        self._z = float(z)

    def set_yaw(self, yaw: float):
        self._yaw = float(yaw) % 360

    def _start_sensor_logging(self, cf, period_ms: int):
        log_config = LogConfig(name="SensorPos", period_in_ms=period_ms)
        log_config.add_variable("stateEstimate.x", "float")
        log_config.add_variable("stateEstimate.y", "float")

        cf.log.add_config(log_config)
        log_config.data_received_cb.add_callback(self._sensor_log_callback)
        log_config.start()
        self._log_config = log_config

    def _sensor_log_callback(self, timestamp, data, logconf):
        x = data["stateEstimate.x"]
        y = data["stateEstimate.y"]
        self._sensor_positions.append((x, y))

    def stop_sensor_logging(self):
        if self._log_config is not None:
            self._log_config.stop()
            self._log_config = None
            print("[PositionController] Sensor position logging stopped")

    def plot_sensor_path(self, save_path: str = "./sensor_trajectory.png", show: bool = True):
        """Plot the recorded actual sensor x/y trajectory and save it as an image file.

        Args:
            save_path: Image save path, defaults to "sensor_trajectory.png".
            show:      Whether to also display in a window, defaults to False.
        """
        if not self._sensor_positions:
            print("[PositionController] No sensor position data to plot")
            return

        xs = [p[0] for p in self._sensor_positions]
        ys = [p[1] for p in self._sensor_positions]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(xs, ys, "b-", linewidth=1.0, label="trajectory")
        ax.scatter(xs[0], ys[0], color="green", s=80, zorder=5, label="start")
        ax.scatter(xs[-1], ys[-1], color="red", s=80, zorder=5, label="end")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Sensor Trajectory (stateEstimate.x/y)")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"[PositionController] Trajectory plot saved to {save_path}")
        if show:
            plt.show()
        plt.close(fig)


    def _log(self, msg: str):
        print(f"[PositionController] {msg}  |  pos={self.position}  yaw={self._yaw:.1f}°")

    def __repr__(self):
        return (
            f"PositionController("
            f"pos=({self._x:.3f}, {self._y:.3f}, {self._z:.3f}), "
            f"yaw={self._yaw:.1f}°)"
        )



URI = "radio://0/80/2M/E7E7E7E7E7"


def main():
    cflib.crtp.init_drivers()

    print("Connecting to", URI)
    with SyncCrazyflie(URI) as scf:
        print("Connected.")

        with MotionCommander(scf, default_height=0.5) as mc:
            time.sleep(1)

            pc = PositionController(mc, initial_z=0.5, scf=scf)
            print(pc)

            pc.goto(1.0, 0.0, 0.5)
            time.sleep(0.5)

            pc.goto(1.0, 1.0, 0.5)
            time.sleep(0.5)

            pc.turn_to(90)
            time.sleep(0.5)

            pc.move_to(z=1.0)
            time.sleep(0.5)

            pc.goto(0.0, 0.0, 0.5)
            time.sleep(0.5)

            pc.stop_sensor_logging()
            print("Landing...")


if __name__ == "__main__":
    main()
