"""Image-only raised-payload confirmation for the cooperative-pick scene."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


VERIFICATION_CAMERA_NAME = "verification_side"
VERIFICATION_CAPTURE_TIMES_S = (8.4, 8.8)
MIN_MARKER_AREA_PX = 24
MIN_UPWARD_DISPLACEMENT_PX = 35.0
ORIENTATION_AWARE_MIN_UPWARD_DISPLACEMENT_PX = 50.0
MAX_INTERFRAME_DRIFT_PX = 12.0
DEFAULT_MAX_MARKER_ROTATION_FROM_START_DEG = 12.0
V3_MAX_MARKER_ROTATION_FROM_START_DEG = 4.0
MARKER_HSV_LOWER = np.asarray((130, 80, 60), dtype=np.uint8)
MARKER_HSV_UPPER = np.asarray((175, 255, 255), dtype=np.uint8)


def detect_verification_marker(
    rgb_image: np.ndarray,
    *,
    vertical_gap_tolerance_px: int = 0,
) -> dict[str, Any] | None:
    """Return the largest magenta fiducial component, optionally joining fragments.

    A positive vertical gap tolerance joins components whose horizontal extents
    substantially overlap and whose vertical separation is within the bound.
    The returned geometry is computed from observed pixels; the gap is never
    filled or treated as observed evidence. A zero tolerance preserves the V3
    single-component detector exactly.
    """
    image = np.asarray(rgb_image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb_image must be an HxWx3 uint8 RGB array")
    if (
        not isinstance(vertical_gap_tolerance_px, int)
        or isinstance(vertical_gap_tolerance_px, bool)
        or not 0 <= vertical_gap_tolerance_px <= 100
    ):
        raise ValueError("vertical marker gap tolerance must be an integer from 0 to 100")

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, MARKER_HSV_LOWER, MARKER_HSV_UPPER)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    components = [
        index for index in range(1, component_count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= MIN_MARKER_AREA_PX
    ]
    if not components:
        return None

    groups: list[list[int]] = [[index] for index in components]
    if vertical_gap_tolerance_px > 0:
        parents = {index: index for index in components}

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        for offset, first in enumerate(components):
            x1, y1, width1, height1, _ = (int(value) for value in stats[first])
            for second in components[offset + 1 :]:
                x2, y2, width2, height2, _ = (int(value) for value in stats[second])
                if y2 < y1:
                    first_x, first_width, first_y, first_height = x2, width2, y2, height2
                    second_x, second_width, second_y, second_height = x1, width1, y1, height1
                else:
                    first_x, first_width, first_y, first_height = x1, width1, y1, height1
                    second_x, second_width, second_y, second_height = x2, width2, y2, height2
                gap = second_y - (first_y + first_height)
                overlap = max(
                    0,
                    min(first_x + first_width, second_x + second_width)
                    - max(first_x, second_x),
                )
                overlap_ratio = overlap / max(1, min(first_width, second_width))
                if 0 <= gap <= vertical_gap_tolerance_px and overlap_ratio >= 0.5:
                    root1, root2 = find(first), find(second)
                    if root1 != root2:
                        parents[root2] = root1

        grouped: dict[int, list[int]] = {}
        for index in components:
            grouped.setdefault(find(index), []).append(index)
        groups = list(grouped.values())

    selected = max(
        groups,
        key=lambda group: sum(int(stats[index, cv2.CC_STAT_AREA]) for index in group),
    )
    selected_mask = np.isin(labels, selected)
    component_yx = np.column_stack(np.where(selected_mask)).astype(np.float64)
    y_min, x_min = component_yx.min(axis=0)
    y_max, x_max = component_yx.max(axis=0)
    x, y = int(x_min), int(y_min)
    width, height = int(x_max - x_min + 1), int(y_max - y_min + 1)
    area = int(len(component_yx))
    center_y, center_x = (float(value) for value in component_yx.mean(axis=0))
    centered_yx = component_yx - component_yx.mean(axis=0, keepdims=True)
    covariance = centered_yx.T @ centered_yx / max(len(component_yx), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_axis_yx = eigenvectors[:, int(np.argmax(eigenvalues))]
    axis_angle_deg = math.degrees(math.atan2(major_axis_yx[0], major_axis_yx[1]))
    # A fiducial's long axis has no arrow direction, so normalize its angle
    # over 180 degrees rather than 360.
    axis_angle_deg = (axis_angle_deg + 90.0) % 180.0 - 90.0
    result = {
        "center_px": [round(center_x, 3), round(center_y, 3)],
        "bbox_xywh_px": [x, y, width, height],
        "area_px": area,
        "major_axis_angle_deg": round(axis_angle_deg, 3),
    }
    if vertical_gap_tolerance_px > 0:
        result["component_count"] = len(selected)
        result["vertical_gap_tolerance_px"] = vertical_gap_tolerance_px
    return result


def verify_raised_payload_from_images(
    initial_rgb: np.ndarray | None,
    post_lift_frames: list[tuple[float, np.ndarray]],
    *,
    min_upward_displacement_px: float = MIN_UPWARD_DISPLACEMENT_PX,
    max_interframe_drift_px: float = MAX_INTERFRAME_DRIFT_PX,
    max_marker_rotation_from_start_deg: float | None = None,
    marker_vertical_gap_tolerance_px: int = 0,
) -> dict[str, Any]:
    """Classify a raised, persistent fiducial using images and capture times only.

    A missing fiducial causes abstention. A visible fiducial must move upward
    by the frozen pixel threshold in both fresh frames and remain within the
    inter-frame drift limit. No simulator state is accepted by this function.
    """
    if len(post_lift_frames) != 2:
        raise ValueError("exactly two post-lift frames are required")
    if (
        not np.isfinite(min_upward_displacement_px)
        or not np.isfinite(max_interframe_drift_px)
        or min_upward_displacement_px < 0
        or max_interframe_drift_px < 0
        or (
            max_marker_rotation_from_start_deg is not None
            and (
                not np.isfinite(max_marker_rotation_from_start_deg)
                or max_marker_rotation_from_start_deg < 0
                or max_marker_rotation_from_start_deg > 90
            )
        )
    ):
        raise ValueError("pixel and angle thresholds must be finite and nonnegative")
    if (
        not isinstance(marker_vertical_gap_tolerance_px, int)
        or isinstance(marker_vertical_gap_tolerance_px, bool)
        or not 0 <= marker_vertical_gap_tolerance_px <= 100
    ):
        raise ValueError("vertical marker gap tolerance must be an integer from 0 to 100")
    times = [float(time_s) for time_s, _ in post_lift_frames]
    if not all(np.isfinite(times)) or times[0] >= times[1]:
        raise ValueError("post-lift capture times must be finite and increasing")

    initial = (
        detect_verification_marker(
            initial_rgb, vertical_gap_tolerance_px=marker_vertical_gap_tolerance_px
        )
        if initial_rgb is not None
        else None
    )
    observations = [
        detect_verification_marker(
            image, vertical_gap_tolerance_px=marker_vertical_gap_tolerance_px
        )
        for _, image in post_lift_frames
    ]
    result: dict[str, Any] = {
        "decision": "abstain",
        "reason": "marker_not_visible",
        "runtime_inputs": "initial and two fresh side-camera RGB frames only",
        "capture_times_s": [round(time_s, 4) for time_s in times],
        "initial_marker": initial,
        "post_lift_markers": observations,
        "upward_displacement_px": None,
        "interframe_drift_px": None,
        "marker_rotation_from_start_deg": None,
        "thresholds_px": {
            "minimum_upward_displacement": min_upward_displacement_px,
            "maximum_interframe_drift": max_interframe_drift_px,
        },
        "thresholds_deg": {
            "maximum_marker_rotation_from_start": max_marker_rotation_from_start_deg,
        },
        "marker_vertical_gap_tolerance_px": marker_vertical_gap_tolerance_px,
    }
    if initial is None:
        result["reason"] = "initial_marker_not_visible"
        return result
    if any(observation is None for observation in observations):
        return result

    initial_center = initial["center_px"]
    post_centers = [observation["center_px"] for observation in observations]
    displacements = [
        round(initial_center[1] - center[1], 3) for center in post_centers
    ]
    drift = round(
        float(np.linalg.norm(np.asarray(post_centers[1]) - np.asarray(post_centers[0]))),
        3,
    )
    initial_angle = float(initial["major_axis_angle_deg"])
    angle_differences = [
        round(abs((float(observation["major_axis_angle_deg"]) - initial_angle + 90.0) % 180.0 - 90.0), 3)
        for observation in observations
    ]
    result["upward_displacement_px"] = displacements
    result["interframe_drift_px"] = drift
    result["marker_rotation_from_start_deg"] = angle_differences
    if min(displacements) < min_upward_displacement_px:
        result["decision"] = "not_confirmed"
        result["reason"] = "insufficient_upward_displacement"
    elif drift > max_interframe_drift_px:
        result["decision"] = "not_confirmed"
        result["reason"] = "marker_not_stable_between_frames"
    elif (
        max_marker_rotation_from_start_deg is not None
        and max(angle_differences) > max_marker_rotation_from_start_deg
    ):
        result["decision"] = "not_confirmed"
        result["reason"] = "marker_orientation_changed_too_much"
    else:
        result["decision"] = "confirmed"
        result["reason"] = "raised_and_stable_in_two_fresh_frames"
    return result


def verify_raised_payload_from_orthogonal_images(
    initial_side_rgb: np.ndarray | None,
    initial_orthogonal_rgb: np.ndarray | None,
    post_lift_frames: list[
        tuple[float, np.ndarray, np.ndarray]
    ],
    *,
    min_upward_displacement_px: float = MIN_UPWARD_DISPLACEMENT_PX,
    max_interframe_drift_px: float = MAX_INTERFRAME_DRIFT_PX,
    max_marker_rotation_from_start_deg: float | None = V3_MAX_MARKER_ROTATION_FROM_START_DEG,
    marker_vertical_gap_tolerance_px: int = 0,
) -> dict[str, Any]:
    """Confirm raised/stable evidence in two perpendicular side-camera views.

    The original and orthogonal views each use their own visible payload marker.
    Both must independently satisfy the same image-only translation and angle
    gates. This makes rotation toward the original camera visible in the second
    projection. Simulator state is deliberately not accepted by this function.
    """
    if len(post_lift_frames) != 2:
        raise ValueError("exactly two post-lift orthogonal frame pairs are required")
    times = [float(time_s) for time_s, _, _ in post_lift_frames]
    if not all(np.isfinite(times)) or times[0] >= times[1]:
        raise ValueError("post-lift capture times must be finite and increasing")

    side = verify_raised_payload_from_images(
        initial_side_rgb,
        [(time_s, side_rgb) for time_s, side_rgb, _ in post_lift_frames],
        min_upward_displacement_px=min_upward_displacement_px,
        max_interframe_drift_px=max_interframe_drift_px,
        max_marker_rotation_from_start_deg=max_marker_rotation_from_start_deg,
        marker_vertical_gap_tolerance_px=marker_vertical_gap_tolerance_px,
    )
    orthogonal = verify_raised_payload_from_images(
        initial_orthogonal_rgb,
        [(time_s, orthogonal_rgb) for time_s, _, orthogonal_rgb in post_lift_frames],
        min_upward_displacement_px=min_upward_displacement_px,
        max_interframe_drift_px=max_interframe_drift_px,
        max_marker_rotation_from_start_deg=max_marker_rotation_from_start_deg,
        marker_vertical_gap_tolerance_px=marker_vertical_gap_tolerance_px,
    )

    if side["decision"] == "abstain" or orthogonal["decision"] == "abstain":
        decision = "abstain"
        failed_view = (
            "side" if side["decision"] == "abstain" else "orthogonal"
        )
        reason = f"{failed_view}_view_insufficient_evidence"
    elif side["decision"] != "confirmed" or orthogonal["decision"] != "confirmed":
        decision = "not_confirmed"
        failed_view = (
            "side" if side["decision"] != "confirmed" else "orthogonal"
        )
        if (
            side["reason"] == "marker_orientation_changed_too_much"
            or orthogonal["reason"] == "marker_orientation_changed_too_much"
        ):
            reason = f"{failed_view}_view_orientation_changed_too_much"
        else:
            reason = f"{failed_view}_view_not_confirmed"
    else:
        decision = "confirmed"
        reason = "raised_and_stable_in_both_orthogonal_views"

    return {
        "decision": decision,
        "reason": reason,
        "runtime_inputs": "initial and two fresh RGB frames from each of two perpendicular side cameras only",
        "camera_views": ["verification_side", "verification_orthogonal"],
        "marker_vertical_gap_tolerance_px": marker_vertical_gap_tolerance_px,
        "capture_times_s": [round(time_s, 4) for time_s in times],
        "view_records": {"side": side, "orthogonal": orthogonal},
        "all_views_confirmed": decision == "confirmed",
        "simulator_truth_is_input": False,
    }
