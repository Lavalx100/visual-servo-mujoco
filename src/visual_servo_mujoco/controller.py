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


def table_xy_to_pixel(
    world_xy: tuple[float, float],
    image_size: tuple[int, int],
    *,
    camera_height: float,
    point_height: float,
    vertical_fov_degrees: float,
) -> tuple[float, float]:
    """Project a point on the table into the top-down camera image."""
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if camera_height <= point_height:
        raise ValueError("camera must be above the projected point")
    if not 0.0 < vertical_fov_degrees < 180.0:
        raise ValueError("vertical field of view must be between 0 and 180 degrees")

    half_height = (camera_height - point_height) * math.tan(
        math.radians(vertical_fov_degrees) / 2.0
    )
    half_width = half_height * width / height
    world_x, world_y = world_xy
    pixel_x = ((world_x / half_width + 1.0) * width / 2.0) - 0.5
    pixel_y = ((1.0 - world_y / half_height) * height / 2.0) - 0.5
    return pixel_x, pixel_y


def predict_pixel_constant_velocity(
    observations: list[tuple[float, tuple[float, float]]],
    *,
    horizon_s: float,
    window: int = 4,
) -> tuple[float, float]:
    """Linearly extrapolate recent timestamped pixel observations.

    Repeated delayed frames should be omitted by the caller so timestamps are
    strictly increasing. A least-squares line over recent observations reduces
    measurement noise while estimating image-plane velocity.
    """
    if not observations:
        raise ValueError("at least one pixel observation is required")
    if not math.isfinite(horizon_s) or horizon_s < 0.0:
        raise ValueError("prediction horizon must be finite and non-negative")
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer")

    recent = observations[-window:]
    samples = np.asarray(
        [(timestamp, pixel[0], pixel[1]) for timestamp, pixel in recent],
        dtype=float,
    )
    if samples.ndim != 2 or samples.shape[1] != 3 or not np.isfinite(samples).all():
        raise ValueError("observations must contain finite timestamps and pixel pairs")
    if np.any(np.diff(samples[:, 0]) <= 0.0):
        raise ValueError("observation timestamps must be strictly increasing")

    latest_time = samples[-1, 0]
    if len(samples) == 1:
        return float(samples[-1, 1]), float(samples[-1, 2])

    relative_times = samples[:, 0] - latest_time
    centered_times = relative_times - float(np.mean(relative_times))
    time_variance = float(np.dot(centered_times, centered_times))
    centered_pixels = samples[:, 1:3] - np.mean(samples[:, 1:3], axis=0)
    velocity = np.sum(centered_times[:, None] * centered_pixels, axis=0) / time_variance
    fitted_at_latest = np.mean(samples[:, 1:3], axis=0) + velocity * (
        latest_time - float(np.mean(samples[:, 0]))
    )
    predicted = fitted_at_latest + velocity * horizon_s
    return float(predicted[0]), float(predicted[1])


def predict_pixel_constant_acceleration(
    observations: list[tuple[float, tuple[float, float]]],
    *,
    horizon_s: float,
    window: int = 4,
) -> tuple[float, float]:
    """Quadratically extrapolate recent timestamped pixel observations.

    With fewer than three distinct observations, this falls back to the
    constant-velocity estimate. The quadratic fit models local acceleration;
    it does not assume that the target follows a sinusoid.
    """
    if not observations:
        raise ValueError("at least one pixel observation is required")
    if not math.isfinite(horizon_s) or horizon_s < 0.0:
        raise ValueError("prediction horizon must be finite and non-negative")
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer")

    recent = observations[-window:]
    samples = np.asarray(
        [(timestamp, pixel[0], pixel[1]) for timestamp, pixel in recent],
        dtype=float,
    )
    if samples.ndim != 2 or samples.shape[1] != 3 or not np.isfinite(samples).all():
        raise ValueError("observations must contain finite timestamps and pixel pairs")
    if np.any(np.diff(samples[:, 0]) <= 0.0):
        raise ValueError("observation timestamps must be strictly increasing")
    if len(samples) < 3:
        return predict_pixel_constant_velocity(
            recent,
            horizon_s=horizon_s,
            window=window,
        )

    relative_times = samples[:, 0] - samples[-1, 0]
    design = np.column_stack(
        (relative_times**2, relative_times, np.ones_like(relative_times))
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, samples[:, 1:3], rcond=None)
    future = np.array((horizon_s**2, horizon_s, 1.0), dtype=float)
    predicted = future @ coefficients
    return float(predicted[0]), float(predicted[1])


def median_pixel_estimate(
    pixel_samples: list[tuple[float, float]],
) -> tuple[float, float]:
    """Return the coordinate-wise median of recent image-space measurements."""
    if not pixel_samples:
        raise ValueError("at least one pixel sample is required")
    samples = np.asarray(pixel_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("pixel samples must be pairs of coordinates")
    if not np.isfinite(samples).all():
        raise ValueError("pixel samples must be finite")
    median = np.median(samples, axis=0)
    return float(median[0]), float(median[1])


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


def camera_pixel_jacobian(
    shoulder: float,
    elbow: float,
    image_size: tuple[int, int],
    *,
    camera_height: float,
    end_effector_height: float,
    vertical_fov_degrees: float,
    link_lengths: tuple[float, float] = (0.42, 0.34),
) -> np.ndarray:
    """Map small joint changes to end-effector pixel changes for a top-down camera.

    The matrix is the local derivative ``d(pixel_xy) / d(joint_angles)``. Its
    scale uses the supplied camera calibration, while image feedback measures
    the target and end effector directly in pixels.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if camera_height <= end_effector_height:
        raise ValueError("camera must be above the end-effector plane")
    if not 0.0 < vertical_fov_degrees < 180.0:
        raise ValueError("vertical field of view must be between 0 and 180 degrees")

    link1, link2 = link_lengths
    total = shoulder + elbow
    world_jacobian = np.array(
        [
            [-link1 * math.sin(shoulder) - link2 * math.sin(total), -link2 * math.sin(total)],
            [link1 * math.cos(shoulder) + link2 * math.cos(total), link2 * math.cos(total)],
        ],
        dtype=float,
    )
    half_height = (camera_height - end_effector_height) * math.tan(
        math.radians(vertical_fov_degrees) / 2.0
    )
    half_width = half_height * width / height
    pixel_jacobian = np.diag((width / (2.0 * half_width), -height / (2.0 * half_height)))
    return pixel_jacobian @ world_jacobian


def image_servo_joint_delta(
    pixel_jacobian: np.ndarray,
    pixel_error_xy: tuple[float, float],
    *,
    gain: float = 0.5,
    damping: float = 1.0,
    max_step_radians: float = 0.2,
) -> np.ndarray:
    """Compute a bounded damped-least-squares step toward an image target."""
    if pixel_jacobian.shape != (2, 2):
        raise ValueError("pixel_jacobian must have shape (2, 2)")
    if gain <= 0.0 or damping < 0.0 or max_step_radians <= 0.0:
        raise ValueError("gain and max_step_radians must be positive; damping non-negative")

    error = np.asarray(pixel_error_xy, dtype=float)
    regularized = pixel_jacobian @ pixel_jacobian.T + damping**2 * np.eye(2)
    delta = gain * pixel_jacobian.T @ np.linalg.solve(regularized, error)
    magnitude = float(np.linalg.norm(delta))
    if magnitude > max_step_radians:
        delta *= max_step_radians / magnitude
    return delta
