"""Small, testable geometry helpers for camera-based planar reaching."""

from __future__ import annotations

import math

import numpy as np


def pixel_to_table_xy(
    pixel_xy: tuple[float, float],
    image_size: tuple[int, int],
    *,
    camera_height: float,
    target_height: float,
    vertical_fov_degrees: float,
) -> tuple[float, float]:
    """Project a top-down camera pixel onto the table plane.

    ``image_size`` is ``(width, height)``. This pinhole-camera mapping assumes the
    camera points straight down and has no lens distortion. The demo uses the
    target's known height so the estimate is calibrated to the target center.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if camera_height <= target_height:
        raise ValueError("camera must be above the target plane")
    if not 0.0 < vertical_fov_degrees < 180.0:
        raise ValueError("vertical field of view must be between 0 and 180 degrees")

    half_height = (camera_height - target_height) * math.tan(
        math.radians(vertical_fov_degrees) / 2.0
    )
    half_width = half_height * width / height
    pixel_x, pixel_y = pixel_xy
    world_x = ((pixel_x + 0.5) / width * 2.0 - 1.0) * half_width
    world_y = (1.0 - (pixel_y + 0.5) / height * 2.0) * half_height
    return world_x, world_y


def inverse_kinematics(
    x: float,
    y: float,
    *,
    link_lengths: tuple[float, float] = (0.42, 0.34),
) -> tuple[float, float]:
    """Return elbow-up joint angles for a planar two-link arm."""
    link1, link2 = link_lengths
    radius_squared = x * x + y * y
    minimum_reach = abs(link1 - link2)
    maximum_reach = link1 + link2
    radius = math.sqrt(radius_squared)
    if radius < minimum_reach - 1e-9 or radius > maximum_reach + 1e-9:
        raise ValueError(
            f"target radius {radius:.3f} m is outside arm reach "
            f"[{minimum_reach:.3f}, {maximum_reach:.3f}] m"
        )

    cosine_elbow = (radius_squared - link1**2 - link2**2) / (2.0 * link1 * link2)
    elbow = math.acos(float(np.clip(cosine_elbow, -1.0, 1.0)))
    shoulder = math.atan2(y, x) - math.atan2(
        link2 * math.sin(elbow), link1 + link2 * math.cos(elbow)
    )
    return shoulder, elbow


def forward_kinematics(
    shoulder: float,
    elbow: float,
    *,
    link_lengths: tuple[float, float] = (0.42, 0.34),
) -> tuple[float, float]:
    """Return the planar end-effector position for the two-link arm."""
    link1, link2 = link_lengths
    total = shoulder + elbow
    x = link1 * math.cos(shoulder) + link2 * math.cos(total)
    y = link1 * math.sin(shoulder) + link2 * math.sin(total)
    return x, y
