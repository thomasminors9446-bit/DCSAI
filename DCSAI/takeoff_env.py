import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TakeoffEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # Actions: [throttle, elevator, steering]
        # throttle: 0.0 to 1.0
        # elevator: -1.0 (nose-down) to +1.0 (nose-up)
        # steering: -1.0 to +1.0
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observations:
        # [lateral position, heading error, speed,
        #  altitude, vertical speed, pitch]
        self.observation_space = spaces.Box(
            low=np.array(
                [-100.0, -math.pi, 0.0, 0.0, -50.0, -math.pi / 2],
                dtype=np.float32,
            ),
            high=np.array(
                [100.0, math.pi, 150.0, 500.0, 50.0, math.pi / 2],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.dt = 0.05
        self.max_steps = 1000

        self.runway_half_width_m = 20.0
        self.rotation_speed_mps = 50.0
        self.success_altitude_m = 100.0
        self.training_ceiling_m = 150.0

        self.steps = 0
        self.previous_altitude_m = 0.0

    def _get_obs(self):
        return np.array(
            [
                self.lateral_m,
                self.heading_rad,
                self.speed_mps,
                self.altitude_m,
                self.vertical_speed_mps,
                self.pitch_rad,
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.lateral_m = float(self.np_random.uniform(-1.0, 1.0))
        self.heading_rad = float(self.np_random.uniform(-0.02, 0.02))

        self.speed_mps = 0.0
        self.altitude_m = 0.0
        self.vertical_speed_mps = 0.0
        self.pitch_rad = 0.0

        self.steps = 0
        self.previous_altitude_m = 0.0

        return self._get_obs(), {}

    def step(self, action):
        throttle, elevator, steering = np.asarray(action, dtype=np.float32)

        throttle = float(np.clip(throttle, 0.0, 1.0))
        elevator = float(np.clip(elevator, -1.0, 1.0))
        steering = float(np.clip(steering, -1.0, 1.0))

        self.steps += 1

        # Longitudinal ground/air speed model.
        drag = 0.0025 * self.speed_mps * self.speed_mps
        acceleration = 9.0 * throttle - drag
        self.speed_mps = max(
            0.0,
            self.speed_mps + acceleration * self.dt,
        )

        # Runway/departure-heading model.
        steering_effect = 0.025 * steering * max(
            self.speed_mps / 20.0,
            0.2,
        )
        self.heading_rad += steering_effect * self.dt
        self.heading_rad *= 0.995

        lateral_rate = self.speed_mps * math.sin(self.heading_rad)
        self.lateral_m += lateral_rate * self.dt

        # Pitch and simplified lift model.
        if self.speed_mps >= self.rotation_speed_mps:
            lift_factor = min(
                (self.speed_mps - self.rotation_speed_mps) / 10.0,
                1.0,
            )

            pitch_target = 0.35 * elevator
            self.pitch_rad += (
                pitch_target - self.pitch_rad
            ) * 3.0 * self.dt

            lift_accel = (
                lift_factor
                * max(0.0, self.pitch_rad)
                * 80.0
            )
            self.vertical_speed_mps += (
                lift_accel - 9.81
            ) * self.dt
        else:
            self.pitch_rad *= 0.95
            self.vertical_speed_mps = 0.0

        # Do not permit downward movement through the runway.
        if self.altitude_m <= 0.0 and self.vertical_speed_mps < 0.0:
            self.vertical_speed_mps = 0.0

        self.altitude_m = max(
            0.0,
            self.altitude_m + self.vertical_speed_mps * self.dt,
        )

        runway_penalty = (
            abs(self.lateral_m) / self.runway_half_width_m
        )
        heading_penalty = abs(self.heading_rad)

        altitude_gain_m = (
            self.altitude_m - self.previous_altitude_m
        )
        self.previous_altitude_m = self.altitude_m

        # Reward design.
        reward = 0.0

        # Ground phase: accelerate and remain aligned.
        if self.altitude_m < 1.0:
            if self.speed_mps < self.rotation_speed_mps:
                reward += 0.08 * max(acceleration, 0.0)

            reward -= 0.20 * runway_penalty
            reward -= 0.40 * heading_penalty

            # Avoid rotating aggressively before takeoff speed.
            if self.speed_mps < self.rotation_speed_mps:
                reward -= 0.10 * max(elevator, 0.0)

        # Airborne phase: reward actual climb progress and departure control.
        else:
            reward += 8.0 * max(altitude_gain_m, 0.0)
            reward += 1.5 * max(self.vertical_speed_mps, 0.0)
            reward -= 0.30 * runway_penalty
            reward -= 0.60 * heading_penalty

            # Avoid extreme uncontrolled pitch attitudes.
            reward -= 0.10 * abs(self.pitch_rad)

        # Small rotation hint once airspeed is adequate.
        if (
            self.altitude_m < 1.0
            and self.speed_mps >= self.rotation_speed_mps
        ):
            reward += 0.30 * max(elevator, 0.0)

        reward -= 0.002 * elevator * elevator
        reward -= 0.002 * steering * steering

        success = (
            self.altitude_m >= self.success_altitude_m
            and self.vertical_speed_mps > 1.0
            and self.speed_mps >= self.rotation_speed_mps
            and abs(self.lateral_m) < self.runway_half_width_m
            and abs(self.heading_rad) < math.radians(15.0)
        )

        crashed = (
            abs(self.lateral_m) > self.runway_half_width_m
            or (
                self.altitude_m > 1.0
                and self.pitch_rad < math.radians(-10.0)
            )
            or self.altitude_m > self.training_ceiling_m
        )

        if success:
            reward += 200.0

        if crashed:
            reward -= 200.0

        terminated = success or crashed
        truncated = self.steps >= self.max_steps

        info = {
            "success": success,
            "crashed": crashed,
            "speed_mps": self.speed_mps,
            "altitude_m": self.altitude_m,
            "vertical_speed_mps": self.vertical_speed_mps,
            "lateral_m": self.lateral_m,
            "heading_rad": self.heading_rad,
            "pitch_rad": self.pitch_rad,
        }

        return self._get_obs(), reward, terminated, truncated, info
