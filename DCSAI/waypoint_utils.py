"""Utility functions for waypoint geometry.

Provides a simple helper to compute Euclidean distance and bearing (in radians) between
two planar points. This is used by the `TakeoffEnv` to expose waypoint information to the
agent and for reward shaping.
"""

import math
from typing import Tuple


def distance_and_bearing(
    position: Tuple[float, float], waypoint: Tuple[float, float]
) -> Tuple[float, float]:
    """Return the distance (meters) and bearing (radians) from `position` to `waypoint`.

    Args:
        position: (forward_m, lateral_m) representing aircraft ground‑track position.
        waypoint: (x, y) target coordinates in the same frame.
    Returns:
        distance: Euclidean distance.
        bearing: Angle measured from the world x‑axis to the line pointing at the waypoint,
                normalized to the range ``[-π, π]``.
    """
    dx = waypoint[0] - position[0]
    dy = waypoint[1] - position[1]
    distance = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx)
    return distance, bearing
