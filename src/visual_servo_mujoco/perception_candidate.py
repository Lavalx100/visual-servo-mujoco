"""Opt-in circle-prior red-marker detector used by appearance experiments.

The project default remains :func:`run.detect_red_target`. This experimental
detector searches for red image circles near the known projected target size and
abstains when multiple size-compatible objects are visible.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


TARGET_RADIUS_M = 0.045
TARGET_TOP_HEIGHT_M = 0.190
CAMERA_HEIGHT_M = 2.0
CAMERA_FOVY_DEG = 40.0
RED_HUE_INTERVALS = ((0, 12), (168, 180))
RED_SATURATION_FLOOR = 80
RED_BRIGHTNESS_FLOOR = 18
HOUGH_BLUR_KERNEL = (9, 9)
HOUGH_BLUR_SIGMA = 2.0
HOUGH_DP = 1.0
HOUGH_MIN_DISTANCE_RADIUS_FRACTION = 1.25
HOUGH_PARAM1 = 80.0
HOUGH_PARAM2 = 12.0
MIN_TARGET_RADIUS_FRACTION = 0.60
MAX_TARGET_RADIUS_FRACTION = 1.55
ACCEPTED_RADIUS_ERROR_FRACTION = 0.19
AMBIGUITY_CENTER_DISTANCE_RADIUS_FRACTION = 1.25
AMBIGUITY_COMPETITOR_RADIUS_ERROR_FRACTION = 0.35
AMBIGUITY_RADIUS_ERROR_DIFFERENCE_PX = 3.0


@dataclass(frozen=True)
class TargetCircleDetection:
    """An opt-in target circle estimate and its geometric fit diagnostics."""

    center_x_px: float
    center_y_px: float
    radius_px: float
    radius_error_px: float
    confidence: float


def expected_target_radius_px(
    image_height: int,
    *,
    camera_height_m: float = CAMERA_HEIGHT_M,
    target_top_height_m: float = TARGET_TOP_HEIGHT_M,
    vertical_fov_degrees: float = CAMERA_FOVY_DEG,
    target_radius_m: float = TARGET_RADIUS_M,
) -> float:
    """Project the known target radius into pixels for the fixed camera model."""
    if image_height <= 0:
        raise ValueError("image_height must be positive")
    if camera_height_m <= target_top_height_m:
        raise ValueError("camera_height_m must be above target_top_height_m")
    if not 0.0 < vertical_fov_degrees < 180.0:
        raise ValueError("vertical_fov_degrees must be between 0 and 180")
    if target_radius_m <= 0.0:
        raise ValueError("target_radius_m must be positive")
    return target_radius_m * image_height / (
        2.0
        * (camera_height_m - target_top_height_m)
        * math.tan(math.radians(vertical_fov_degrees) / 2.0)
    )


def detect_red_target_circle(
    rgb_image: np.ndarray,
    *,
    expected_radius_px: float,
    min_radius_fraction: float = MIN_TARGET_RADIUS_FRACTION,
    max_radius_fraction: float = MAX_TARGET_RADIUS_FRACTION,
    accepted_radius_error_fraction: float = ACCEPTED_RADIUS_ERROR_FRACTION,
    ambiguity_radius_error_px: float = AMBIGUITY_RADIUS_ERROR_DIFFERENCE_PX,
    check_partial_arc_competitors: bool = True,
) -> TargetCircleDetection | None:
    """Find a red circle consistent with the target's known projected size.

    The detector lowers the brightness floor relative to the default HSV
    contour detector, then uses a Hough circle transform to recover centers
    from partial arcs. It only accepts a candidate within the configured size
    band. If two spatially separate red circles have similar geometric
    support, it abstains instead of guessing which one is the target. The
    ambiguity check includes partial target arcs outside the stricter accepted
    radius band.

    The size prior is specific to the project's fixed overhead camera and
    9 cm diameter target. ``confidence`` is a normalized radius-fit score; it
    is not a calibrated probability.
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("rgb_image must have shape (height, width, 3)")
    if expected_radius_px <= 0.0 or not math.isfinite(expected_radius_px):
        raise ValueError("expected_radius_px must be finite and positive")
    if not 0.0 < min_radius_fraction < max_radius_fraction:
        raise ValueError("invalid Hough radius range")
    if not 0.0 < accepted_radius_error_fraction < 0.5:
        raise ValueError("accepted_radius_error_fraction must be between 0 and 0.5")
    if ambiguity_radius_error_px < 0.0:
        raise ValueError("ambiguity_radius_error_px must be non-negative")
    if not isinstance(check_partial_arc_competitors, bool):
        raise ValueError("check_partial_arc_competitors must be a boolean")

    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    low_hue, high_hue = RED_HUE_INTERVALS
    red_low = cv2.inRange(
        hsv,
        np.array([low_hue[0], RED_SATURATION_FLOOR, RED_BRIGHTNESS_FLOOR]),
        np.array([low_hue[1], 255, 255]),
    )
    red_high = cv2.inRange(
        hsv,
        np.array([high_hue[0], RED_SATURATION_FLOOR, RED_BRIGHTNESS_FLOOR]),
        np.array([high_hue[1], 255, 255]),
    )
    red_mask = cv2.bitwise_or(red_low, red_high)
    blurred = cv2.GaussianBlur(red_mask, HOUGH_BLUR_KERNEL, HOUGH_BLUR_SIGMA)
    min_radius = max(1, int(round(expected_radius_px * min_radius_fraction)))
    max_radius = max(min_radius + 1, int(round(expected_radius_px * max_radius_fraction)))
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=max(1.0, expected_radius_px * HOUGH_MIN_DISTANCE_RADIUS_FRACTION),
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return None

    max_error = expected_radius_px * accepted_radius_error_fraction
    raw_candidates: list[TargetCircleDetection] = []
    for center_x, center_y, radius in circles[0]:
        radius_error = abs(float(radius) - expected_radius_px)
        raw_candidates.append(
            TargetCircleDetection(
                center_x_px=float(center_x),
                center_y_px=float(center_y),
                radius_px=float(radius),
                radius_error_px=radius_error,
                confidence=max(0.0, 1.0 - radius_error / max_error),
            )
        )
    candidates = [item for item in raw_candidates if item.radius_error_px <= max_error]
    if not candidates:
        return None

    candidates.sort(key=lambda item: item.radius_error_px)
    best = candidates[0]
    ambiguity_candidates = raw_candidates if check_partial_arc_competitors else candidates
    for other in ambiguity_candidates:
        if other is best:
            continue
        center_separation = math.hypot(
            other.center_x_px - best.center_x_px,
            other.center_y_px - best.center_y_px,
        )
        if (
            center_separation
            >= expected_radius_px * AMBIGUITY_CENTER_DISTANCE_RADIUS_FRACTION
            and other.radius_error_px
            <= expected_radius_px * AMBIGUITY_COMPETITOR_RADIUS_ERROR_FRACTION
            and other.radius_error_px - best.radius_error_px
            <= ambiguity_radius_error_px
        ):
            return None
    return best
