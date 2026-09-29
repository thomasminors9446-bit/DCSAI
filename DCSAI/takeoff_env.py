import math
import csv
import datetime as dt
from pathlib import Path
from typing import Optional, Tuple, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Optional utility module for waypoint geometry
try:
    from waypoint_utils import distance_and_bearing
except ImportError:
    # Fallback: define inline if the utility module is missing
    def distance_and_bearing(position: Tuple[float, float], waypoint: Tuple[float, float]) -> Tuple[float, float]:
        dx = waypoint[0] - position[0]
        dy = waypoint[1] - position[1]
        distance = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        return distance, bearing


class TakeoffEnv(gym.Env):
    """Gym environment that simulates aircraft take‑off, climb, and optional waypoint navigation.

    The original environment models take‑off dynamics with three continuous actions:
    ``throttle`` (0‑1), ``elevator`` (‑1‑1), and ``steering`` (‑1‑1).  This extended version adds:

    * Optional waypoint support (coordinates supplied via the environment constructor or via ``reset`` options).
    * Two additional observation dimensions: *distance to waypoint* and *bearing to waypoint*.
    * Reward shaping to encourage progress toward the waypoint.
    * CSV logging of per‑step data and optional Matplotlib trajectory plotting.
    * Backward‑compatible defaults – when no waypoint is provided the environment behaves exactly as before.
    """

    metadata = {"render_modes": []}

    # ----- constants for waypoint handling -----
    WAYPOINT_RADIUS_M = 10.0  # success tolerance around waypoint
    MAX_WAYPOINT_DISTANCE_M = 5000.0  # observation space cap
    WAYPOINT_REWARD_COEFF = 2.0
    WAYPOINT_PENALTY_COEFF = 0.005
    # -------------------------------------------------

    def __init__(
        self,
        waypoint: Optional[Tuple[float, float]] = None,
        log_dir: Optional[str] = None,
        plot: bool = False,
        use_autonomous_steering: bool = False,
    ):
        """Create a new ``TakeoffEnv`` instance.

        Args:
            waypoint: (x, y) coordinates in meters. ``None`` disables waypoint logic.
            log_dir: Directory where CSV logs (and optional plots) will be written. ``None`` disables logging.
            plot: Whether to generate a trajectory plot after the episode ends (requires ``log_dir``).
            use_autonomous_steering: If ``True`` a simple proportional controller will blend
                a heading correction toward the waypoint.  The user requested *pure
                agent‑controlled steering*, so the default is ``False``.
        """
        super().__init__()

        # ----- action space (unchanged) -----
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # ----- observation space (extended) -----
        # Original six dims + distance_to_wp + bearing_to_wp
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    -100.0,
                    -math.pi,
                    0.0,
                    0.0,
                    -50.0,
                    -math.pi / 2,
                    0.0,  # distance to waypoint (>=0)
                    -math.pi,  # bearing relative to world x‑axis
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    100.0,
                    math.pi,
                    150.0,
                    500.0,
                    50.0,
                    math.pi / 2,
                    self.MAX_WAYPOINT_DISTANCE_M,
                    math.pi,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # ----- static simulation parameters -----
        self.dt = 0.05
        self.max_steps = 1000

        self.runway_half_width_m = 20.0
        self.rotation_speed_mps = 50.0
        self.success_altitude_m = 100.0
        self.training_ceiling_m = 150.0

        # ----- waypoint / logging state -----
        self.waypoint = waypoint  # (x, y) or None
        self.use_autonomous_steering = use_autonomous_steering
        self.log_dir: Optional[Path] = Path(log_dir) if log_dir else None
        self.plot_enabled = plot

        self._log_file = None  # type: Optional[object]
        self._csv_writer = None  # type: Optional[csv.DictWriter]
        self._trajectory: List[dict] = []
        self._episode_timestamp = ""

        # dynamic state (reset in ``reset``)
        self.steps = 0
        self.previous_altitude_m = 0.0
        self.prev_wp_distance = 0.0
        self.forward_m = 0.0

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------
    def _compute_wp_metrics(self) -> Tuple[float, float]:
        """Return (distance, bearing) to the current waypoint.

        When ``self.waypoint`` is ``None`` both values are ``0.0``.
        Bearing is normalized to ``[-π, π]``.
        """
        if not self.waypoint:
            return 0.0, 0.0
        # Position uses forward distance (X) and lateral offset (Y)
        distance, bearing = distance_and_bearing((self.forward_m, self.lateral_m), self.waypoint)
        # Normalize bearing to [-π, π]
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        return distance, bearing

    def _open_log(self):
        if not self.log_dir:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._episode_timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f"takeoff_{self._episode_timestamp}.csv"
        self._log_file = open(log_path, "w", newline="")
        fieldnames = [
            "step",
            "time",
            "altitude_m",
            "speed_mps",
            "forward_m",
            "vertical_speed_mps",
            "lateral_m",
            "heading_rad",
            "pitch_rad",
            "distance_to_wp",
            "bearing_to_wp",
            "reward",
            "terminated",
            "crashed",
        ]
        self._csv_writer = csv.DictWriter(self._log_file, fieldnames=fieldnames)
        self._csv_writer.writeheader()

    def _close_log(self):
        if self._log_file:
            self._log_file.close()
            self._log_file = None
            self._csv_writer = None

    def _write_log_row(
        self,
        step: int,
        reward: float,
        terminated: bool,
        crashed: bool,
        distance: float,
        bearing: float,
    ):
        if not self._csv_writer:
            return
        row = {
            "step": step,
            "time": step * self.dt,
            "altitude_m": self.altitude_m,
            "speed_mps": self.speed_mps,
            "forward_m": self.forward_m,
            "vertical_speed_mps": self.vertical_speed_mps,
            "lateral_m": self.lateral_m,
            "heading_rad": self.heading_rad,
            "pitch_rad": self.pitch_rad,
            "distance_to_wp": distance,
            "bearing_to_wp": bearing,
            "reward": reward,
            "terminated": terminated,
            "crashed": crashed,
        }
        self._csv_writer.writerow(row)
        self._log_file.flush()
        # Store for optional plotting later
        self._trajectory.append(row)

    def _generate_plot(self):
        if not self.plot_enabled or not self.log_dir:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            # Plotting is optional; swallow any import errors.
            print(f"[TakeoffEnv] Plotting disabled (import error): {e}")
            return

        # Prepare data series
        forward = [t["forward_m"] if "forward_m" in t else t["step"] * self.dt for t in self._trajectory]
        lateral = [t["lateral_m"] for t in self._trajectory]
        altitude = [t["altitude_m"] for t in self._trajectory]

        fig = plt.figure(figsize=(8, 10))
        ax1 = fig.add_axes([0.1, 0.55, 0.85, 0.4])
        ax2 = fig.add_axes([0.1, 0.1, 0.85, 0.4])

        # Ground‑track plot
        ax1.plot(forward, lateral, label="Aircraft track", marker=None)
        if self.waypoint:
            # ax1.scatter([self.waypoint[0]], [self.waypoint[1]], color="red", marker="X", s=100, label="Waypoint")
            pass
        ax1.set_ylabel("Lateral offset (m)")
        # ax1.legend()
        ax1.grid(True)

        # Altitude profile
        ax2.plot(forward, altitude, label="Altitude", marker=None)
        ax2.set_xlabel("Forward distance (m)")
        ax2.set_ylabel("Altitude (m)")
        # ax2.legend()
        ax2.grid(True)

        plot_path = self.log_dir / f"trajectory_{self._episode_timestamp}.png"
        # fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)

    # ---------------------------------------------------------------------
    # Gym API methods
    # ---------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        distance, bearing = self._compute_wp_metrics()
        return np.array(
            [
                self.lateral_m,
                self.heading_rad,
                self.speed_mps,
                self.altitude_m,
                self.vertical_speed_mps,
                self.pitch_rad,
                distance,
                bearing,
            ],
            dtype=np.float32,
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Allow waypoint to be overridden via options dict (e.g., from a test script)
        if options and "waypoint" in options:
            self.waypoint = tuple(options["waypoint"])  # expects (x, y)
        # Reset dynamic state
        self.lateral_m = float(self.np_random.uniform(-1.0, 1.0))
        self.heading_rad = float(self.np_random.uniform(-0.02, 0.02))

        self.speed_mps = 0.0
        self.altitude_m = 0.0
        self.vertical_speed_mps = 0.0
        self.pitch_rad = 0.0

        self.forward_m = 0.0
        self.steps = 0
        self.previous_altitude_m = 0.0
        self.prev_wp_distance, _ = self._compute_wp_metrics()
        self._trajectory.clear()

        # Open CSV logger if requested
        self._open_log()

        return self._get_obs(), {}

    def step(self, action):
        throttle, elevator, steering = np.asarray(action, dtype=np.float32)

        throttle = float(np.clip(throttle, 0.0, 1.0))
        elevator = float(np.clip(elevator, -1.0, 1.0))
        steering = float(np.clip(steering, -1.0, 1.0))

        self.steps += 1

        # ----- longitudinal dynamics -----
        drag = 0.0025 * self.speed_mps * self.speed_mps
        acceleration = 9.0 * throttle - drag
        self.speed_mps = max(0.0, self.speed_mps + acceleration * self.dt)

        # ----- heading / lateral dynamics -----
        steering_effect = 0.025 * steering * max(self.speed_mps / 20.0, 0.2)
        # Optional autonomous steering towards waypoint (disabled per user request)
        if self.use_autonomous_steering and self.waypoint:
            # Simple proportional law using bearing error
            _, bearing = self._compute_wp_metrics()
            auto_steer = max(min(bearing, 1.0), -1.0)  # clamp to action range
            steering = (steering + auto_steer) / 2.0

        self.heading_rad += steering_effect * self.dt
        self.heading_rad *= 0.995

        lateral_rate = self.speed_mps * math.sin(self.heading_rad)
        self.lateral_m += lateral_rate * self.dt

        # Update forward ground‑track distance (X coordinate)
        self.forward_m += self.speed_mps * math.cos(self.heading_rad) * self.dt

        # ----- pitch & lift -----
        if self.speed_mps >= self.rotation_speed_mps:
            lift_factor = min((self.speed_mps - self.rotation_speed_mps) / 10.0, 1.0)
            pitch_target = 0.35 * elevator
            self.pitch_rad += (pitch_target - self.pitch_rad) * 3.0 * self.dt
            lift_accel = lift_factor * max(0.0, self.pitch_rad) * 80.0
            self.vertical_speed_mps += (lift_accel - 9.81) * self.dt
        else:
            self.pitch_rad *= 0.95
            self.vertical_speed_mps = 0.0

        # Ground collision guard
        if self.altitude_m <= 0.0 and self.vertical_speed_mps < 0.0:
            self.vertical_speed_mps = 0.0

        self.altitude_m = max(0.0, self.altitude_m + self.vertical_speed_mps * self.dt)

        # ----- penalties -----
        runway_penalty = abs(self.lateral_m) / self.runway_half_width_m
        heading_penalty = abs(self.heading_rad)

        altitude_gain_m = self.altitude_m - self.previous_altitude_m
        self.previous_altitude_m = self.altitude_m

        # ----- reward -----
        reward = 0.0

        # Ground phase
        if self.altitude_m < 1.0:
            if self.speed_mps < self.rotation_speed_mps:
                reward += 0.08 * max(acceleration, 0.0)
            reward -= 0.20 * runway_penalty
            reward -= 0.40 * heading_penalty
            if self.speed_mps < self.rotation_speed_mps:
                reward -= 0.10 * max(elevator, 0.0)
        else:
            # Airborne phase
            reward += 8.0 * max(altitude_gain_m, 0.0)
            reward += 1.5 * max(self.vertical_speed_mps, 0.0)
            reward -= 0.30 * runway_penalty
            reward -= 0.60 * heading_penalty
            reward -= 0.10 * abs(self.pitch_rad)

        # Rotation hint
        if self.altitude_m < 1.0 and self.speed_mps >= self.rotation_speed_mps:
            reward += 0.30 * max(elevator, 0.0)

        # Action penalties (as in original)
        reward -= 0.002 * elevator * elevator
        reward -= 0.002 * steering * steering

        # ----- waypoint reward shaping -----
        if self.waypoint:
            distance, _ = self._compute_wp_metrics()
            # encourage getting closer
            distance_delta = self.prev_wp_distance - distance
            if distance_delta > 0:
                reward += self.WAYPOINT_REWARD_COEFF * distance_delta
            # penalty for remaining far away
            reward -= self.WAYPOINT_PENALTY_COEFF * distance
            self.prev_wp_distance = distance

        # ----- success / crash criteria -----
        wp_success = (
            distance <= self.WAYPOINT_RADIUS_M if self.waypoint else True
        )
        success = (
            self.altitude_m >= self.success_altitude_m
            and self.vertical_speed_mps > 1.0
            and self.speed_mps >= self.rotation_speed_mps
            and abs(self.lateral_m) < self.runway_half_width_m
            and abs(self.heading_rad) < math.radians(15.0)
            and wp_success
        )

        crashed = (
            abs(self.lateral_m) > self.runway_half_width_m
            or (
                self.altitude_m > 1.0 and self.pitch_rad < math.radians(-10.0)
            )
            or self.altitude_m > self.training_ceiling_m
        )

        if success:
            reward += 200.0
        if crashed:
            reward -= 200.0

        terminated = success or crashed
        truncated = self.steps >= self.max_steps

        # ----- info dict -----
        info = {
            "success": success,
            "crashed": crashed,
            "speed_mps": self.speed_mps,
            "altitude_m": self.altitude_m,
            "vertical_speed_mps": self.vertical_speed_mps,
            "lateral_m": self.lateral_m,
            "heading_rad": self.heading_rad,
            "pitch_rad": self.pitch_rad,
            "distance_to_wp": distance if self.waypoint else 0.0,
            "bearing_to_wp": _ if self.waypoint else 0.0,
        }

        # ----- logging -----
        self._write_log_row(
            step=self.steps,
            reward=reward,
            terminated=terminated,
            crashed=crashed,
            distance=info["distance_to_wp"],
            bearing=info["bearing_to_wp"],
        )

        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        """Close any resources – CSV file and optional plot generation."""
        self._close_log()
        self._generate_plot()
        # Reset internal references to avoid accidental reuse
        self._trajectory.clear()

    # Support ``with env:`` syntax (gym wraps this automatically)
    __del__ = close
