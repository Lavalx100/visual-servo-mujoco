"""Run deterministic camera-based reaching trials and save a small report/demo."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .camera_distortion import distort_pixel, distort_rgb_image, undistort_pixel
from .controller import (
    camera_pixel_jacobian,
    forward_kinematics,
    image_servo_joint_delta,
    inverse_kinematics,
    median_pixel_estimate,
    predict_pixel_constant_acceleration,
    predict_pixel_constant_velocity,
    predict_pixel_reversal_aware,
    pixel_to_table_xy,
    table_xy_to_pixel,
)
from .model import MODEL_XML
from .terminal_verification import (
    VERIFICATION_VIEWS,
    apply_verification_image_effects,
    model_with_verification_views,
    pixel_to_xy_at_height,
    terminal_verification_condition_label,
)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAMERA_HEIGHT = 2.0
TARGET_HEIGHT = 0.165
END_EFFECTOR_HEIGHT = 0.21
LINK_LENGTHS = (0.42, 0.34)
SIMULATION_STEPS = 1200
VIDEO_FPS = 30.0
SUCCESS_THRESHOLD_METERS = 0.045
HOME_JOINT_ANGLES = (0.0, 0.7)
FEEDBACK_MAX_ITERATIONS = 20
FEEDBACK_MAX_JOINT_STEP_RADIANS = 0.30
FEEDBACK_STEPS_PER_UPDATE = 200
FEEDBACK_PIXEL_TOLERANCE = 8.0
FEEDBACK_TARGET_OCCLUSION_GRACE = 5
FEEDBACK_TARGET_FILTER_WINDOW = 3
FEEDBACK_REQUIRED_TOLERANCE_CHECKS = 3


def detect_red_target(rgb_image: np.ndarray) -> tuple[float, float]:
    """Find the largest red object and estimate its circle center in the image."""
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    red_low = cv2.inRange(hsv, np.array([0, 110, 70]), np.array([12, 255, 255]))
    red_high = cv2.inRange(hsv, np.array([168, 110, 70]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(red_low, red_high)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("camera image contains no red target")
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 20:
        raise RuntimeError("red target detection was smaller than 20 pixels")
    (x, y), _ = cv2.minEnclosingCircle(largest)
    return float(x), float(y)


def detect_green_end_effector(rgb_image: np.ndarray) -> tuple[float, float]:
    """Find the green end-effector marker and estimate its circle center."""
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([42, 90, 55]), np.array([92, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("camera image contains no green end-effector marker")
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 8:
        raise RuntimeError("green end-effector detection was smaller than 8 pixels")
    (x, y), _ = cv2.minEnclosingCircle(largest)
    return float(x), float(y)


def _detect_marker_with_quality(
    rgb_image: np.ndarray,
    *,
    marker: str,
) -> tuple[tuple[float, float], dict]:
    """Detect a terminal marker and report a unitless image-quality score.

    The score is the minimum of normalized contour area, circularity,
    saturation, and brightness. It is a conservative heuristic, not a
    calibrated probability of correct detection.
    """
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    if marker == "target":
        low = cv2.inRange(hsv, np.array([0, 110, 70]), np.array([12, 255, 255]))
        high = cv2.inRange(hsv, np.array([168, 110, 70]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(low, high)
        minimum_area, saturation_floor, brightness_floor = 20.0, 110.0, 70.0
    elif marker == "end_effector":
        mask = cv2.inRange(hsv, np.array([42, 90, 55]), np.array([92, 255, 255]))
        minimum_area, saturation_floor, brightness_floor = 8.0, 90.0, 55.0
    else:
        raise ValueError("marker must be 'target' or 'end_effector'")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError(f"camera image contains no {marker.replace('_', ' ')} marker")
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < minimum_area:
        raise RuntimeError(f"{marker.replace('_', ' ')} detection was too small")

    (x, y), _ = cv2.minEnclosingCircle(contour)
    perimeter = float(cv2.arcLength(contour, True))
    circularity = (
        min(1.0, 4.0 * math.pi * area / (perimeter * perimeter))
        if perimeter > 0.0
        else 0.0
    )
    contour_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
    marker_pixels = contour_mask != 0
    median_saturation = float(np.median(hsv[:, :, 1][marker_pixels]))
    median_brightness = float(np.median(hsv[:, :, 2][marker_pixels]))
    scores = {
        "area": min(1.0, area / (minimum_area * 8.0)),
        "circularity": min(1.0, circularity / 0.8),
        "saturation": min(
            1.0,
            max(0.0, (median_saturation - saturation_floor) / (255.0 - saturation_floor)),
        ),
        "brightness": min(
            1.0,
            max(0.0, (median_brightness - brightness_floor) / (255.0 - brightness_floor)),
        ),
    }
    quality = min(scores.values())
    return (float(x), float(y)), {
        "score": round(float(quality), 4),
        "area_px": round(area, 2),
        "circularity": round(circularity, 4),
        "median_saturation": round(median_saturation, 2),
        "median_brightness": round(median_brightness, 2),
        "component_scores": {name: round(value, 4) for name, value in scores.items()},
    }


def render_rgb(renderer: mujoco.Renderer, data: mujoco.MjData) -> np.ndarray:
    renderer.update_scene(data, camera="overhead")
    return renderer.render()


def _measure_terminal_verification(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_id: int,
    raw_rgb: np.ndarray,
    *,
    fovy_error_deg: float,
    camera_position_error_m: tuple[float, float, float],
    camera_rotation_error_deg: tuple[float, float, float],
    brightness_scale: float,
    occluded_markers: tuple[str, ...],
    minimum_quality_score: float,
) -> tuple[dict, np.ndarray]:
    """Score one verifier condition from a frame captured after control stops."""
    verification_rgb = apply_verification_image_effects(
        raw_rgb,
        lighting_scale=brightness_scale,
        occluded_markers=occluded_markers,
    )
    target_pixel = None
    end_effector_pixel = None
    target_detection_quality = None
    end_effector_detection_quality = None
    try:
        target_pixel, target_detection_quality = _detect_marker_with_quality(
            verification_rgb,
            marker="target",
        )
    except RuntimeError:
        pass
    try:
        end_effector_pixel, end_effector_detection_quality = _detect_marker_with_quality(
            verification_rgb,
            marker="end_effector",
        )
    except RuntimeError:
        pass

    estimated_target_xy = None
    estimated_end_effector_xy = None
    estimated_distance = None
    candidate_distance = None
    visual_success_estimate = None
    quality_score = (
        min(
            target_detection_quality["score"],
            end_effector_detection_quality["score"],
        )
        if target_detection_quality is not None
        and end_effector_detection_quality is not None
        else None
    )
    quality_gate_passed = (
        quality_score is not None and quality_score >= minimum_quality_score
    )
    estimate_status = None
    if target_pixel is not None and end_effector_pixel is not None:
        try:
            candidate_target_xy = pixel_to_xy_at_height(
                model,
                data,
                camera_id,
                target_pixel,
                TARGET_HEIGHT,
                fovy_error_deg=fovy_error_deg,
                camera_position_error_m=camera_position_error_m,
                camera_rotation_error_deg=camera_rotation_error_deg,
            )
            candidate_end_effector_xy = pixel_to_xy_at_height(
                model,
                data,
                camera_id,
                end_effector_pixel,
                END_EFFECTOR_HEIGHT,
                fovy_error_deg=fovy_error_deg,
                camera_position_error_m=camera_position_error_m,
                camera_rotation_error_deg=camera_rotation_error_deg,
            )
        except ValueError:
            estimate_status = "unavailable_camera_geometry"
        else:
            candidate_distance = float(
                np.linalg.norm(candidate_target_xy - candidate_end_effector_xy)
            )
            if quality_gate_passed:
                estimated_target_xy = candidate_target_xy
                estimated_end_effector_xy = candidate_end_effector_xy
                estimated_distance = candidate_distance
                visual_success_estimate = (
                    estimated_distance <= SUCCESS_THRESHOLD_METERS
                )
            else:
                estimate_status = "unavailable_detection_quality_below_threshold"
    if estimated_distance is not None:
        estimate_status = (
            "within_threshold" if visual_success_estimate else "outside_threshold"
        )
    elif estimate_status == "unavailable_camera_geometry":
        pass
    elif target_pixel is None and end_effector_pixel is None:
        estimate_status = "unavailable_both_markers_not_visible"
    elif target_pixel is None:
        estimate_status = "unavailable_target_not_visible"
    elif end_effector_pixel is None:
        estimate_status = "unavailable_end_effector_not_visible"
    elif not quality_gate_passed:
        estimate_status = "unavailable_detection_quality_below_threshold"
    else:
        estimate_status = "unavailable_camera_geometry"

    record = {
        "camera": "verification_east",
        "fresh_observation": True,
        "observation_time_simulated_s": round(float(data.time), 6),
        "camera_observation_age_s": 0.0,
        "measurement_noise_std_px": 0.0,
        "calibration": "simulator camera with estimator-only perturbations",
        "condition_label": terminal_verification_condition_label(
            fovy_error_deg=fovy_error_deg,
            camera_position_error_m=camera_position_error_m,
            camera_rotation_error_deg=camera_rotation_error_deg,
            lighting_scale=brightness_scale,
            occluded_markers=occluded_markers,
        ),
        "estimator_fovy_error_deg": fovy_error_deg,
        "estimator_camera_position_error_m": list(camera_position_error_m),
        "estimator_camera_rotation_error_deg": list(camera_rotation_error_deg),
        "image_brightness_scale": brightness_scale,
        "occluded_markers": list(occluded_markers),
        "minimum_detection_quality_score": minimum_quality_score,
        "target_detection_quality": target_detection_quality,
        "end_effector_detection_quality": end_effector_detection_quality,
        "combined_detection_quality_score": quality_score,
        "detection_quality_gate_passed": quality_gate_passed,
        "target_visible": target_pixel is not None,
        "end_effector_visible": end_effector_pixel is not None,
        "both_markers_visible": target_pixel is not None and end_effector_pixel is not None,
        "target_pixel_xy": (
            [round(value, 2) for value in target_pixel]
            if target_pixel is not None
            else None
        ),
        "end_effector_pixel_xy": (
            [round(value, 2) for value in end_effector_pixel]
            if end_effector_pixel is not None
            else None
        ),
        "estimated_target_xy_m": (
            [round(float(value), 4) for value in estimated_target_xy]
            if estimated_target_xy is not None
            else None
        ),
        "estimated_end_effector_xy_m": (
            [round(float(value), 4) for value in estimated_end_effector_xy]
            if estimated_end_effector_xy is not None
            else None
        ),
        "estimated_distance_m": (
            round(estimated_distance, 5) if estimated_distance is not None else None
        ),
        "candidate_estimated_distance_m_for_evaluation_only": (
            round(candidate_distance, 5) if candidate_distance is not None else None
        ),
        "visual_success_estimate": visual_success_estimate,
        "visual_estimate_status": estimate_status,
        "threshold_m": SUCCESS_THRESHOLD_METERS,
    }
    return record, verification_rgb


def _step_with_target_motion(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_mocap_id: int,
    target_start_xy_m: tuple[float, float],
    target_motion_amplitude_m: float,
    target_motion_frequency_hz: float,
    target_motion_profile: str = "sinusoidal",
    target_motion_phase_rad: float = 0.0,
) -> None:
    """Advance one physics step while moving the target along table x."""
    if target_motion_amplitude_m:
        simulation_time = float(data.time + model.opt.timestep)
        motion_fraction = _target_motion_fraction(
            simulation_time,
            target_motion_frequency_hz,
            target_motion_profile,
            target_motion_phase_rad,
        )
        target_x = target_start_xy_m[0] + target_motion_amplitude_m * motion_fraction
        data.mocap_pos[target_mocap_id] = (
            target_x,
            target_start_xy_m[1],
            TARGET_HEIGHT,
        )
    mujoco.mj_step(model, data)


def _target_motion_fraction(
    simulation_time_s: float,
    frequency_hz: float,
    profile: str,
    phase_rad: float = 0.0,
) -> float:
    phase = 2.0 * math.pi * frequency_hz * simulation_time_s + phase_rad
    if profile == "sinusoidal":
        return math.sin(phase)
    if profile == "piecewise_linear":
        return 2.0 / math.pi * math.asin(math.sin(phase))
    raise ValueError("target_motion_profile must be 'sinusoidal' or 'piecewise_linear'")


def annotated_frame(rgb: np.ndarray, pixel_xy: tuple[float, float]) -> np.ndarray:
    """Return an RGB frame with the pixel used by the controller marked."""
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    center = tuple(int(round(value)) for value in pixel_xy)
    cv2.drawMarker(frame, center, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
    cv2.putText(frame, "target pixel used by controller", (14, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (20, 25, 30), 2, cv2.LINE_AA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def demo_video_frame(
    rgb: np.ndarray,
    pixel_xy: tuple[float, float],
    *,
    controller: str,
    pixel_noise_std_px: float,
    camera_fovy_error_deg: float,
    status: str,
    target_detection_missing: bool = False,
) -> np.ndarray:
    """Add a readable controller, sensor-condition, and status overlay."""
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    center = tuple(int(round(value)) for value in pixel_xy)
    if target_detection_missing:
        # The target remains in the simulated scene; this overlay makes the
        # injected perception blackout visible in the recorded demonstration.
        cv2.circle(frame, center, 18, (48, 48, 48), thickness=-1)
        cv2.circle(frame, center, 18, (190, 190, 190), thickness=2)
        cv2.putText(
            frame,
            "TARGET LOST",
            (max(8, center[0] - 48), max(98, center[1] - 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    cv2.drawMarker(frame, center, (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (IMAGE_WIDTH, 78), (22, 27, 34), thickness=-1)
    frame = cv2.addWeighted(overlay, 0.84, frame, 0.16, 0.0)
    title = (
        "OPEN LOOP | pinhole projection + IK"
        if controller == "open_loop"
        else "IMAGE FEEDBACK | damped Jacobian"
    )
    sensors = (
        f"Target noise std: {pixel_noise_std_px:.1f} px  |  "
        f"FOV model error: {camera_fovy_error_deg:+.1f} deg"
    )
    status_color = (
        (70, 105, 245) if status.startswith("NOT REACHED") else (110, 235, 175)
    )
    for text, y, scale, color in (
        (title, 24, 0.55, (255, 255, 255)),
        (sensors, 49, 0.46, (212, 222, 232)),
        (status, 71, 0.46, status_color),
    ):
        cv2.putText(
            frame,
            text,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame


def terminal_verification_demo_frame(
    overhead_rgb: np.ndarray,
    verification_rgb: np.ndarray,
    verification: dict,
    *,
    physical_success: bool,
    reaching_error_m: float,
) -> np.ndarray:
    """Compose a terminal two-view frame with separate physical and visual results."""
    header_height = 74
    canvas = np.zeros(
        (IMAGE_HEIGHT + header_height, IMAGE_WIDTH * 2, 3), dtype=np.uint8
    )
    physical_text = "YES" if physical_success else "NO"
    estimate = verification["estimated_distance_m"]
    if estimate is None:
        visual_text = "image estimate unavailable"
    else:
        visual_text = (
            f"estimated distance {estimate * 100:.1f} cm | "
            f"within 4.5 cm: {'YES' if verification['visual_success_estimate'] else 'NO'}"
        )
    condition_label = verification["condition_label"]
    quality_score = verification["combined_detection_quality_score"]
    quality_text = "n/a" if quality_score is None else f"{quality_score:.2f}"
    quality_threshold = verification["minimum_detection_quality_score"]
    cv2.putText(
        canvas,
        f"TERMINAL EVIDENCE | Physical reach (MuJoCo): {physical_text} "
        f"({reaching_error_m * 100:.1f} cm)",
        (14, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Fresh east view ({condition_label}): quality {quality_text} / "
        f"gate {quality_threshold:.2f} | {visual_text} | does not change controller result",
        (14, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (145, 225, 255),
        1,
        cv2.LINE_AA,
    )
    panel_data = (
        (
            "OVERHEAD CONTROL VIEW",
            overhead_rgb,
            "overhead_target_pixel_xy",
            "overhead_tip_pixel_xy",
        ),
        (
            "EAST TERMINAL VERIFICATION VIEW",
            verification_rgb,
            "target_pixel_xy",
            "end_effector_pixel_xy",
        ),
    )
    for index, (label, rgb, target_key, tip_key) in enumerate(panel_data):
        x_offset = index * IMAGE_WIDTH
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for key, color in ((target_key, (20, 20, 245)), (tip_key, (60, 230, 110))):
            pixel = verification.get(key)
            if pixel is not None:
                center = tuple(int(round(value)) for value in pixel)
                cv2.circle(frame, center, 11, color, 2, cv2.LINE_AA)
        canvas[header_height:, x_offset : x_offset + IMAGE_WIDTH] = frame
        cv2.putText(
            canvas,
            label,
            (x_offset + 14, IMAGE_HEIGHT + header_height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (250, 250, 250),
            1,
            cv2.LINE_AA,
        )
    cv2.line(
        canvas,
        (IMAGE_WIDTH, 0),
        (IMAGE_WIDTH, IMAGE_HEIGHT + header_height),
        (105, 105, 105),
        1,
    )
    return canvas


def run_trials(
    episodes: int,
    seed: int,
    output_dir: Path,
    *,
    pixel_noise_std_px: float = 0.0,
    camera_fovy_error_deg: float = 0.0,
    camera_radial_distortion_k1: float = 0.0,
    camera_radial_distortion_k2: float = 0.0,
    controller_radial_distortion_k1: float = 0.0,
    controller_radial_distortion_k2: float = 0.0,
    controller: str = "open_loop",
    video_episode_index: int = 0,
    feedback_target_filter_window: int | None = None,
    feedback_required_tolerance_checks: int | None = None,
    feedback_max_joint_step_radians: float | None = None,
    feedback_allow_stale_target_confirmation: bool = False,
    target_dropout_start_observation: int | None = None,
    target_dropout_duration_observations: int = 0,
    target_motion_amplitude_m: float = 0.0,
    target_motion_frequency_hz: float = 0.25,
    target_motion_profile: str = "sinusoidal",
    target_motion_phase_rad: float = 0.0,
    camera_observation_delay_observations: int = 0,
    target_motion_prediction: str = "none",
    use_current_end_effector_state: bool = False,
    verify_terminal: bool = False,
    terminal_verification_fovy_error_deg: float = 0.0,
    terminal_verification_camera_position_error_m: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    ),
    terminal_verification_camera_rotation_error_deg: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    ),
    terminal_verification_brightness_scale: float = 1.0,
    terminal_verification_occluded_markers: tuple[str, ...] = (),
    terminal_verification_min_quality_score: float = 0.0,
    terminal_verification_stress_candidates: dict[str, dict] | None = None,
    save_media: bool = True,
) -> dict:
    """Run reaching trials with optional perception and camera-calibration errors.

    ``pixel_noise_std_px`` adds independent Gaussian noise to each detected
    target centroid. ``camera_fovy_error_deg`` offsets the field of view used by
    the controller while leaving the simulated camera unchanged. The actual
    camera radial coefficients distort rendered overhead frames; the controller
    coefficients undistort detected marker pixels before control. The feedback
    filter window controls how many recent target detections feed its median;
    the tolerance-check count controls the stop confirmation; and the joint
    step value caps each commanded update. A target dropout skips detection for
    a specified range of feedback observations. These knobs do not change the
    physics. The target can move along table x on a sinusoidal or piecewise
    linear path, and image feedback can consume delayed camera frames. Optional
    prediction extrapolates timestamped target pixels; current end-effector
    pixels can come from the measured joint state instead of the delayed image.
    Terminal verification runs after control stops. Its optional quality gate
    can abstain from a visual estimate without affecting physical scoring.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not math.isfinite(pixel_noise_std_px) or pixel_noise_std_px < 0.0:
        raise ValueError("pixel_noise_std_px must be finite and non-negative")
    if not math.isfinite(camera_fovy_error_deg):
        raise ValueError("camera_fovy_error_deg must be finite")
    if not all(
        math.isfinite(value)
        for value in (
            camera_radial_distortion_k1,
            camera_radial_distortion_k2,
            controller_radial_distortion_k1,
            controller_radial_distortion_k2,
        )
    ):
        raise ValueError("radial distortion coefficients must be finite")
    if controller not in {"open_loop", "image_feedback"}:
        raise ValueError("controller must be 'open_loop' or 'image_feedback'")
    if not isinstance(feedback_allow_stale_target_confirmation, bool):
        raise ValueError("feedback_allow_stale_target_confirmation must be a boolean")
    if (
        not math.isfinite(target_motion_amplitude_m)
        or not 0.0 <= target_motion_amplitude_m <= 0.16
    ):
        raise ValueError("target_motion_amplitude_m must be finite and between 0 and 0.16")
    if not math.isfinite(target_motion_frequency_hz) or target_motion_frequency_hz < 0.0:
        raise ValueError("target_motion_frequency_hz must be finite and non-negative")
    if not math.isfinite(target_motion_phase_rad):
        raise ValueError("target_motion_phase_rad must be finite")
    if target_motion_amplitude_m and target_motion_frequency_hz <= 0.0:
        raise ValueError("moving targets require a positive target_motion_frequency_hz")
    if target_motion_profile not in {"sinusoidal", "piecewise_linear"}:
        raise ValueError(
            "target_motion_profile must be 'sinusoidal' or 'piecewise_linear'"
        )
    if (
        isinstance(camera_observation_delay_observations, bool)
        or not isinstance(camera_observation_delay_observations, int)
        or camera_observation_delay_observations < 0
    ):
        raise ValueError("camera_observation_delay_observations must be a non-negative integer")
    if camera_observation_delay_observations and controller != "image_feedback":
        raise ValueError("camera observation delay can only be used with image_feedback")
    if target_motion_prediction not in {
        "none",
        "constant_velocity",
        "constant_acceleration",
        "reversal_aware",
    }:
        raise ValueError(
            "target_motion_prediction must be 'none', 'constant_velocity', "
            "'constant_acceleration', or 'reversal_aware'"
        )
    if target_motion_prediction != "none" and controller != "image_feedback":
        raise ValueError("target motion prediction can only be used with image_feedback")
    if not isinstance(use_current_end_effector_state, bool):
        raise ValueError("use_current_end_effector_state must be a boolean")
    if not isinstance(verify_terminal, bool):
        raise ValueError("verify_terminal must be a boolean")
    if not math.isfinite(terminal_verification_fovy_error_deg):
        raise ValueError("terminal_verification_fovy_error_deg must be finite")
    estimated_terminal_fovy_deg = (
        VERIFICATION_VIEWS["east"]["fovy_deg"] + terminal_verification_fovy_error_deg
    )
    if not 0.0 < estimated_terminal_fovy_deg < 180.0:
        raise ValueError(
            "estimated terminal camera field of view must be between 0 and 180 degrees"
        )
    verification_position_error = np.asarray(
        terminal_verification_camera_position_error_m, dtype=float
    )
    verification_rotation_error = np.asarray(
        terminal_verification_camera_rotation_error_deg, dtype=float
    )
    if (
        verification_position_error.shape != (3,)
        or not np.isfinite(verification_position_error).all()
    ):
        raise ValueError(
            "terminal_verification_camera_position_error_m must contain three finite values"
        )
    if (
        verification_rotation_error.shape != (3,)
        or not np.isfinite(verification_rotation_error).all()
    ):
        raise ValueError(
            "terminal_verification_camera_rotation_error_deg must contain three finite values"
        )
    if (
        not math.isfinite(terminal_verification_brightness_scale)
        or terminal_verification_brightness_scale <= 0.0
    ):
        raise ValueError("terminal_verification_brightness_scale must be positive and finite")
    if (
        not math.isfinite(terminal_verification_min_quality_score)
        or not 0.0 <= terminal_verification_min_quality_score <= 1.0
    ):
        raise ValueError("terminal_verification_min_quality_score must be between 0 and 1")
    unknown_occlusions = set(terminal_verification_occluded_markers) - {
        "target",
        "end_effector",
    }
    if unknown_occlusions:
        raise ValueError(
            "terminal_verification_occluded_markers may contain only target and end_effector"
        )
    terminal_verification_stressed = (
        terminal_verification_fovy_error_deg != 0.0
        or bool(np.any(verification_position_error))
        or bool(np.any(verification_rotation_error))
        or terminal_verification_brightness_scale != 1.0
        or bool(terminal_verification_occluded_markers)
        or terminal_verification_min_quality_score > 0.0
    )
    stress_candidate_overrides = terminal_verification_stress_candidates
    if stress_candidate_overrides is None:
        stress_candidate_overrides = {}
    if not isinstance(stress_candidate_overrides, dict):
        raise ValueError("terminal_verification_stress_candidates must be a mapping")
    allowed_stress_keys = {
        "terminal_verification_fovy_error_deg",
        "terminal_verification_camera_position_error_m",
        "terminal_verification_camera_rotation_error_deg",
        "terminal_verification_brightness_scale",
        "terminal_verification_occluded_markers",
        "terminal_verification_min_quality_score",
    }
    normalized_stress_candidates = {}
    for name, overrides in stress_candidate_overrides.items():
        if not isinstance(name, str) or not name:
            raise ValueError("terminal verification stress candidate names must be non-empty")
        if not isinstance(overrides, dict) or set(overrides) - allowed_stress_keys:
            raise ValueError(f"invalid stress settings for terminal candidate {name!r}")
        candidate_fovy_error = float(
            overrides.get(
                "terminal_verification_fovy_error_deg",
                terminal_verification_fovy_error_deg,
            )
        )
        candidate_position_error = np.asarray(
            overrides.get(
                "terminal_verification_camera_position_error_m",
                terminal_verification_camera_position_error_m,
            ),
            dtype=float,
        )
        candidate_rotation_error = np.asarray(
            overrides.get(
                "terminal_verification_camera_rotation_error_deg",
                terminal_verification_camera_rotation_error_deg,
            ),
            dtype=float,
        )
        candidate_brightness = float(
            overrides.get(
                "terminal_verification_brightness_scale",
                terminal_verification_brightness_scale,
            )
        )
        candidate_occluded_markers = tuple(
            overrides.get(
                "terminal_verification_occluded_markers",
                terminal_verification_occluded_markers,
            )
        )
        candidate_minimum_quality = float(
            overrides.get(
                "terminal_verification_min_quality_score",
                terminal_verification_min_quality_score,
            )
        )
        if (
            not math.isfinite(candidate_fovy_error)
            or not -39.0 < candidate_fovy_error < 140.0
            or candidate_position_error.shape != (3,)
            or not np.isfinite(candidate_position_error).all()
            or candidate_rotation_error.shape != (3,)
            or not np.isfinite(candidate_rotation_error).all()
            or not math.isfinite(candidate_brightness)
            or candidate_brightness <= 0.0
            or set(candidate_occluded_markers) - {"target", "end_effector"}
            or not math.isfinite(candidate_minimum_quality)
            or not 0.0 <= candidate_minimum_quality <= 1.0
        ):
            raise ValueError(f"invalid stress values for terminal candidate {name!r}")
        normalized_stress_candidates[name] = {
            "fovy_error_deg": candidate_fovy_error,
            "camera_position_error_m": tuple(candidate_position_error.tolist()),
            "camera_rotation_error_deg": tuple(candidate_rotation_error.tolist()),
            "brightness_scale": candidate_brightness,
            "occluded_markers": candidate_occluded_markers,
            "minimum_quality_score": candidate_minimum_quality,
        }
    if (terminal_verification_stressed or normalized_stress_candidates) and not verify_terminal:
        raise ValueError("terminal verifier stress settings require verify_terminal=True")
    if use_current_end_effector_state and controller != "image_feedback":
        raise ValueError("current end-effector state requires image_feedback")
    if target_dropout_start_observation is not None and (
        isinstance(target_dropout_start_observation, bool)
        or not isinstance(target_dropout_start_observation, int)
        or target_dropout_start_observation < 1
    ):
        raise ValueError("target_dropout_start_observation must be a positive integer")
    if (
        isinstance(target_dropout_duration_observations, bool)
        or not isinstance(target_dropout_duration_observations, int)
        or target_dropout_duration_observations < 0
    ):
        raise ValueError("target_dropout_duration_observations must be a non-negative integer")
    if target_dropout_duration_observations and target_dropout_start_observation is None:
        raise ValueError("a dropout start observation is required for a non-zero duration")
    if target_dropout_duration_observations and controller != "image_feedback":
        raise ValueError("target dropout can only be used with image_feedback")
    if feedback_target_filter_window is None:
        feedback_target_filter_window = FEEDBACK_TARGET_FILTER_WINDOW
    if (
        isinstance(feedback_target_filter_window, bool)
        or not isinstance(feedback_target_filter_window, int)
        or feedback_target_filter_window <= 0
    ):
        raise ValueError("feedback_target_filter_window must be a positive integer")
    if feedback_required_tolerance_checks is None:
        feedback_required_tolerance_checks = FEEDBACK_REQUIRED_TOLERANCE_CHECKS
    if (
        isinstance(feedback_required_tolerance_checks, bool)
        or not isinstance(feedback_required_tolerance_checks, int)
        or feedback_required_tolerance_checks <= 0
    ):
        raise ValueError("feedback_required_tolerance_checks must be a positive integer")
    if feedback_max_joint_step_radians is None:
        feedback_max_joint_step_radians = FEEDBACK_MAX_JOINT_STEP_RADIANS
    if (
        isinstance(feedback_max_joint_step_radians, bool)
        or not isinstance(feedback_max_joint_step_radians, (int, float))
        or not math.isfinite(feedback_max_joint_step_radians)
        or feedback_max_joint_step_radians <= 0.0
    ):
        raise ValueError("feedback_max_joint_step_radians must be a positive finite number")
    max_observations = (
        FEEDBACK_MAX_ITERATIONS * feedback_required_tolerance_checks
        + feedback_required_tolerance_checks
    )
    if not 0 <= video_episode_index < episodes:
        raise ValueError("video_episode_index must be within the requested episodes")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = (
        model_with_verification_views(("east",))
        if verify_terminal
        else mujoco.MjModel.from_xml_string(MODEL_XML)
    )
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    target_mocap_id = int(model.body_mocapid[target_id])
    if target_mocap_id < 0:
        raise RuntimeError("target body must be a MuJoCo mocap body")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    shoulder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_joint")
    elbow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "elbow_joint")
    shoulder_qpos = model.jnt_qposadr[shoulder_id]
    elbow_qpos = model.jnt_qposadr[elbow_id]
    shoulder_dof = model.jnt_dofadr[shoulder_id]
    elbow_dof = model.jnt_dofadr[elbow_id]
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
    verification_camera_id = (
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "verification_east"
        )
        if verify_terminal
        else None
    )
    camera_fovy = float(model.cam_fovy[camera_id])
    target_rng = np.random.default_rng(seed)
    measurement_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5EED]))
    controller_camera_fovy = camera_fovy + camera_fovy_error_deg
    if not 0.0 < controller_camera_fovy < 180.0:
        raise ValueError("calibrated camera field of view must be between 0 and 180 degrees")

    def render_camera_rgb() -> np.ndarray:
        return distort_rgb_image(
            render_rgb(renderer, data),
            vertical_fov_degrees=camera_fovy,
            k1=camera_radial_distortion_k1,
            k2=camera_radial_distortion_k2,
        )

    def correct_camera_pixel(pixel_xy: tuple[float, float]) -> tuple[float, float]:
        return undistort_pixel(
            pixel_xy,
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            vertical_fov_degrees=controller_camera_fovy,
            k1=controller_radial_distortion_k1,
            k2=controller_radial_distortion_k2,
        )

    def camera_overlay_pixel(pixel_xy: tuple[float, float]) -> tuple[float, float]:
        return distort_pixel(
            pixel_xy,
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            vertical_fov_degrees=camera_fovy,
            k1=camera_radial_distortion_k1,
            k2=camera_radial_distortion_k2,
        )

    # Keep the full target path inside the nominal x span and arm workspace.
    targets = []
    while len(targets) < episodes:
        candidate = (
            float(
                target_rng.uniform(
                    0.35 + target_motion_amplitude_m,
                    0.67 - target_motion_amplitude_m,
                )
            ),
            float(target_rng.uniform(-0.32, 0.32)),
        )
        radii = [
            math.hypot(candidate[0] + offset, candidate[1])
            for offset in (-target_motion_amplitude_m, target_motion_amplitude_m)
        ]
        if all(
            abs(LINK_LENGTHS[0] - LINK_LENGTHS[1]) + 0.02
            < radius
            < sum(LINK_LENGTHS) - 0.04
            for radius in radii
        ):
            targets.append(candidate)

    results = []
    scene_path = output_dir / "camera_view.png"
    video_path = output_dir / "reaching_demo.mp4"
    video_writer = None
    if save_media:
        video_writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            VIDEO_FPS,
            (IMAGE_WIDTH, IMAGE_HEIGHT),
        )
        if not video_writer.isOpened():
            video_writer.release()
            video_writer = None

    for index, (target_x, target_y) in enumerate(targets):
        mujoco.mj_resetData(model, data)
        initial_motion_offset = target_motion_amplitude_m * _target_motion_fraction(
            0.0,
            target_motion_frequency_hz,
            target_motion_profile,
            target_motion_phase_rad,
        )
        data.mocap_pos[target_mocap_id] = (
            target_x + initial_motion_offset,
            target_y,
            TARGET_HEIGHT,
        )
        initial_target_xy = (target_x + initial_motion_offset, target_y)
        data.qpos[shoulder_qpos] = HOME_JOINT_ANGLES[0]
        data.qpos[elbow_qpos] = HOME_JOINT_ANGLES[1]
        data.ctrl[0] = HOME_JOINT_ANGLES[0]
        data.ctrl[1] = HOME_JOINT_ANGLES[1]
        mujoco.mj_forward(model, data)

        # Target localization comes from the camera. Ground-truth target state
        # is reserved for evaluation; an optional current arm-pose projection
        # uses joint state rather than target ground truth.
        rgb = render_camera_rgb()
        camera_frame_history = [rgb]
        camera_target_position_history = [initial_target_xy]
        camera_capture_time_history = [float(data.time)]
        detected_pixel_xy = detect_red_target(rgb)
        noise = measurement_rng.normal(0.0, pixel_noise_std_px, size=2)
        raw_controller_pixel_xy = (
            float(detected_pixel_xy[0] + noise[0]),
            float(detected_pixel_xy[1] + noise[1]),
        )
        pixel_xy = correct_camera_pixel(raw_controller_pixel_xy)
        camera_target_pixel_history: list[tuple[float, float] | None] = [pixel_xy]
        camera_target_raw_pixel_history: list[
            tuple[float, float] | None
        ] = [raw_controller_pixel_xy]
        target_motion_observation_history = [(float(data.time), pixel_xy)]
        estimated_xy = pixel_to_table_xy(
            pixel_xy,
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            camera_height=CAMERA_HEIGHT,
            target_height=TARGET_HEIGHT,
            vertical_fov_degrees=controller_camera_fovy,
        )
        failure_reason = None
        controller_error = None
        controller_stop_reason = None
        target_reacquisitions = 0
        natural_target_visibility_loss_observations = 0
        natural_target_visibility_loss_outside_dropout_observations = 0
        natural_target_reacquisitions = 0
        natural_visibility_pending = False
        injected_dropout_pending = False
        injected_dropout_reacquired = False
        injected_dropout_visible_observations = 0
        injected_dropout_occlusion_overlap_observations = 0
        target_visibility_stop_cause = None
        target_visibility_trace = [
            {
                "observation_index": 0,
                "target_visible_in_render": True,
                "target_visible_in_observation": True,
                "injected_dropout": False,
                "target_measurement_used": True,
                "camera_observation_age_observations": 0,
                "camera_observation_age_simulated_s": 0.0,
                "camera_capture_time_simulated_s": float(data.time),
                "target_prediction_horizon_s": 0.0,
                "target_pixel_xy_raw": [
                    round(value, 2) for value in raw_controller_pixel_xy
                ],
                "target_pixel_xy_undistorted": [
                    round(value, 2) for value in pixel_xy
                ],
                "target_position_xy_m_for_evaluation_only": list(initial_target_xy),
                "observed_target_position_xy_m_for_evaluation_only": [
                    initial_target_xy[0],
                    initial_target_xy[1],
                ],
                "consecutive_target_misses": 0,
                "completed_motion_updates": 0,
                "joint_angles_rad": [
                    round(float(data.qpos[shoulder_qpos]), 5),
                    round(float(data.qpos[elbow_qpos]), 5),
                ],
            }
        ]
        save_episode_media = index == video_episode_index and video_writer is not None
        if controller == "open_loop":
            try:
                shoulder_angle, elbow_angle = inverse_kinematics(
                    *estimated_xy, link_lengths=LINK_LENGTHS
                )
            except ValueError as error:
                # Keep unreachable estimates in the benchmark as safe rejections.
                failure_reason = "estimated_target_unreachable"
                controller_error = str(error)

        if save_episode_media:
            annotated = cv2.cvtColor(
                annotated_frame(rgb, camera_overlay_pixel(pixel_xy)), cv2.COLOR_RGB2BGR
            )
            cv2.imwrite(str(scene_path), annotated)
            initial_frame = demo_video_frame(
                rgb,
                camera_overlay_pixel(pixel_xy),
                controller=controller,
                pixel_noise_std_px=pixel_noise_std_px,
                camera_fovy_error_deg=camera_fovy_error_deg,
                status="initial camera observation",
            )
            for _ in range(int(VIDEO_FPS * 0.5)):
                video_writer.write(initial_frame)

        control_iterations = 0
        if failure_reason is None:
            if controller == "open_loop":
                control_iterations = 1
                data.ctrl[0] = shoulder_angle
                data.ctrl[1] = elbow_angle
                video_step_interval = max(
                    1, round(1.0 / (float(model.opt.timestep) * VIDEO_FPS))
                )
                for step in range(SIMULATION_STEPS):
                    _step_with_target_motion(
                        model,
                        data,
                        target_mocap_id,
                        (target_x, target_y),
                        target_motion_amplitude_m,
                        target_motion_frequency_hz,
                        target_motion_profile,
                        target_motion_phase_rad,
                    )
                    if save_episode_media and step % video_step_interval == 0:
                        video_writer.write(
                            demo_video_frame(
                                render_camera_rgb(),
                                camera_overlay_pixel(pixel_xy),
                                controller=controller,
                                pixel_noise_std_px=pixel_noise_std_px,
                                camera_fovy_error_deg=camera_fovy_error_deg,
                                status="single IK target command",
                            )
                        )
            else:
                target_pixel_history = [pixel_xy]
                consecutive_target_misses = 0
                consecutive_tolerance_checks = 0
                consecutive_stale_tolerance_checks = 0
                observation_rgb = rgb
                observed_target_pixel_xy = pixel_xy
                observed_raw_target_pixel_xy = raw_controller_pixel_xy
                target_visible_in_observation = True
                camera_observation_age = 0
                prediction_horizon_s = 0.0
                for iteration in range(max_observations):
                    visibility_event = target_visibility_trace[0] if iteration == 0 else None
                    if iteration > 0:
                        rgb = render_camera_rgb()
                        camera_frame_history.append(rgb)
                        camera_target_position_history.append(
                            tuple(float(value) for value in data.xpos[target_id][:2])
                        )
                        camera_capture_time_history.append(float(data.time))
                        dropout_active = (
                            target_dropout_start_observation is not None
                            and target_dropout_start_observation
                            <= iteration
                            < target_dropout_start_observation
                            + target_dropout_duration_observations
                        )
                        try:
                            current_detected_pixel_xy = detect_red_target(rgb)
                        except RuntimeError as error:
                            target_visible_in_render = False
                            current_detection_error = error
                            captured_target_pixel_xy = None
                        else:
                            target_visible_in_render = True
                            current_detection_error = None
                            noise = measurement_rng.normal(0.0, pixel_noise_std_px, size=2)
                            captured_raw_target_pixel_xy = (
                                float(current_detected_pixel_xy[0] + noise[0]),
                                float(current_detected_pixel_xy[1] + noise[1]),
                            )
                            captured_target_pixel_xy = correct_camera_pixel(
                                captured_raw_target_pixel_xy
                            )
                        if not target_visible_in_render:
                            captured_raw_target_pixel_xy = None
                        camera_target_pixel_history.append(captured_target_pixel_xy)
                        camera_target_raw_pixel_history.append(
                            captured_raw_target_pixel_xy
                        )
                        if len(camera_frame_history) > camera_observation_delay_observations + 1:
                            camera_frame_history.pop(0)
                            camera_target_position_history.pop(0)
                            camera_capture_time_history.pop(0)
                            camera_target_pixel_history.pop(0)
                            camera_target_raw_pixel_history.pop(0)
                        camera_observation_age = min(
                            camera_observation_delay_observations, iteration
                        )
                        observation_rgb = camera_frame_history[
                            -1 - camera_observation_age
                        ]
                        observed_target_position_xy = camera_target_position_history[
                            -1 - camera_observation_age
                        ]
                        observed_frame_time_s = camera_capture_time_history[
                            -1 - camera_observation_age
                        ]
                        observed_target_pixel_xy = camera_target_pixel_history[
                            -1 - camera_observation_age
                        ]
                        observed_raw_target_pixel_xy = camera_target_raw_pixel_history[
                            -1 - camera_observation_age
                        ]
                        target_visible_in_observation = observed_target_pixel_xy is not None
                        if target_visible_in_observation:
                            detected_pixel_xy = observed_raw_target_pixel_xy
                            raw_controller_pixel_xy = observed_raw_target_pixel_xy
                        target_detection_error = (
                            current_detection_error
                            if camera_observation_age == 0
                            else RuntimeError("delayed camera frame contains no red target")
                        )

                        if dropout_active:
                            injected_dropout_pending = True
                            consecutive_target_misses += 1
                            if target_visible_in_observation:
                                injected_dropout_visible_observations += 1
                            else:
                                injected_dropout_occlusion_overlap_observations += 1
                                natural_target_visibility_loss_observations += 1
                                natural_visibility_pending = True
                        elif not target_visible_in_observation:
                            consecutive_target_misses += 1
                            natural_target_visibility_loss_observations += 1
                            natural_target_visibility_loss_outside_dropout_observations += 1
                            natural_visibility_pending = True
                        else:
                            consecutive_stale_tolerance_checks = 0
                            if injected_dropout_pending:
                                injected_dropout_reacquired = True
                                injected_dropout_pending = False
                            if natural_visibility_pending:
                                natural_target_reacquisitions += 1
                                natural_visibility_pending = False
                            if consecutive_target_misses > 0:
                                target_reacquisitions += 1
                            consecutive_target_misses = 0
                            target_pixel_history.append(observed_target_pixel_xy)
                            target_pixel_history = target_pixel_history[
                                -feedback_target_filter_window:
                            ]
                            pixel_xy = median_pixel_estimate(target_pixel_history)
                        if target_motion_prediction != "none":
                            if target_visible_in_observation and not dropout_active:
                                if (
                                    observed_frame_time_s
                                    > target_motion_observation_history[-1][0] + 1e-9
                                ):
                                    target_motion_observation_history.append(
                                        (
                                            observed_frame_time_s,
                                            observed_target_pixel_xy,
                                        )
                                    )
                                    target_motion_observation_history = (
                                        target_motion_observation_history[-4:]
                                    )
                            prediction_horizon_s = max(
                                0.0,
                                float(data.time) - target_motion_observation_history[-1][0],
                            )
                            model_ready = len(target_motion_observation_history) >= 2
                            if prediction_horizon_s > 1e-9 and model_ready:
                                if target_motion_prediction == "constant_acceleration":
                                    pixel_xy = predict_pixel_constant_acceleration(
                                        target_motion_observation_history,
                                        horizon_s=prediction_horizon_s,
                                    )
                                elif target_motion_prediction == "reversal_aware":
                                    pixel_xy = predict_pixel_reversal_aware(
                                        target_motion_observation_history,
                                        horizon_s=prediction_horizon_s,
                                    )
                                else:
                                    pixel_xy = predict_pixel_constant_velocity(
                                        target_motion_observation_history,
                                        horizon_s=prediction_horizon_s,
                                    )
                        else:
                            prediction_horizon_s = 0.0
                        visibility_event = {
                            "observation_index": iteration,
                            "target_visible_in_render": target_visible_in_render,
                            "target_visible_in_observation": target_visible_in_observation,
                            "injected_dropout": dropout_active,
                            "camera_observation_age_observations": camera_observation_age,
                            "camera_observation_age_simulated_s": round(
                                max(0.0, float(data.time) - observed_frame_time_s),
                                4,
                            ),
                            "camera_capture_time_simulated_s": round(
                                observed_frame_time_s, 4
                            ),
                            "target_prediction_horizon_s": round(
                                prediction_horizon_s, 4
                            ),
                            "target_position_xy_m_for_evaluation_only": [
                                round(float(value), 5)
                                for value in data.xpos[target_id][:2]
                            ],
                            "observed_target_position_xy_m_for_evaluation_only": [
                                round(float(value), 5)
                                for value in observed_target_position_xy
                            ],
                            "target_measurement_used": (
                                target_visible_in_observation and not dropout_active
                            ),
                            "consecutive_target_misses": consecutive_target_misses,
                            "stale_tolerance_checks": consecutive_stale_tolerance_checks,
                            "completed_motion_updates": control_iterations,
                            "joint_angles_rad": [
                                round(float(data.qpos[shoulder_qpos]), 5),
                                round(float(data.qpos[elbow_qpos]), 5),
                            ],
                        }
                        target_visibility_trace.append(visibility_event)
                        if consecutive_target_misses > FEEDBACK_TARGET_OCCLUSION_GRACE:
                            controller_stop_reason = "target_not_visible"
                            if dropout_active and target_visible_in_observation:
                                target_visibility_stop_cause = "injected_detector_dropout"
                                controller_error = "injected target detector dropout"
                            elif dropout_active:
                                target_visibility_stop_cause = (
                                    "injected_dropout_with_natural_visibility_loss"
                                )
                                controller_error = str(target_detection_error)
                            else:
                                target_visibility_stop_cause = "natural_target_visibility_loss"
                                controller_error = str(target_detection_error)
                            break
                    try:
                        if use_current_end_effector_state:
                            current_angles = (
                                float(data.qpos[shoulder_qpos]),
                                float(data.qpos[elbow_qpos]),
                            )
                            current_end_effector_xy = forward_kinematics(
                                *current_angles,
                                link_lengths=LINK_LENGTHS,
                            )
                            end_effector_pixel_xy = table_xy_to_pixel(
                                current_end_effector_xy,
                                (IMAGE_WIDTH, IMAGE_HEIGHT),
                                camera_height=CAMERA_HEIGHT,
                                point_height=END_EFFECTOR_HEIGHT,
                                vertical_fov_degrees=controller_camera_fovy,
                            )
                        else:
                            end_effector_pixel_xy = correct_camera_pixel(
                                detect_green_end_effector(observation_rgb)
                            )
                    except RuntimeError as error:
                        controller_stop_reason = "end_effector_not_visible"
                        controller_error = str(error)
                        break
                    pixel_error = (
                        pixel_xy[0] - end_effector_pixel_xy[0],
                        pixel_xy[1] - end_effector_pixel_xy[1],
                    )
                    visibility_event["image_error_px"] = round(math.hypot(*pixel_error), 2)
                    visibility_event["target_pixel_xy_used"] = [
                        round(value, 2) for value in pixel_xy
                    ]
                    visibility_event["target_pixel_xy_raw_used"] = (
                        [round(value, 2) for value in observed_raw_target_pixel_xy]
                        if observed_raw_target_pixel_xy is not None
                        and target_visible_in_observation
                        and not visibility_event["injected_dropout"]
                        else None
                    )
                    visibility_event["target_motion_prediction_applied"] = (
                        target_motion_prediction != "none"
                        and len(target_motion_observation_history) >= 2
                        and prediction_horizon_s > 1e-9
                    )
                    visibility_event["target_prediction_model_used"] = (
                        "reversal_aware"
                        if (
                            target_motion_prediction == "reversal_aware"
                            and len(target_motion_observation_history) >= 3
                            and prediction_horizon_s > 1e-9
                        )
                        else "constant_acceleration"
                        if (
                            target_motion_prediction == "constant_acceleration"
                            and len(target_motion_observation_history) >= 3
                            and prediction_horizon_s > 1e-9
                        )
                        else "constant_velocity"
                        if (
                            target_motion_prediction != "none"
                            and len(target_motion_observation_history) >= 2
                            and prediction_horizon_s > 1e-9
                        )
                        else "none"
                    )
                    visibility_event["end_effector_pixel_xy"] = [
                        round(value, 2) for value in end_effector_pixel_xy
                    ]
                    visibility_event["end_effector_pixel_source"] = (
                        "current_joint_state_projection"
                        if use_current_end_effector_state
                        else "camera_observation"
                    )
                    visibility_event["joint_angles_rad"] = [
                        round(float(data.qpos[shoulder_qpos]), 5),
                        round(float(data.qpos[elbow_qpos]), 5),
                    ]
                    if consecutive_target_misses > 0:
                        # The default policy requires a fresh target. The
                        # experiment option tests whether recent target history
                        # can safely confirm arrival during the grace window.
                        consecutive_tolerance_checks = 0
                        if math.hypot(*pixel_error) <= FEEDBACK_PIXEL_TOLERANCE:
                            if feedback_allow_stale_target_confirmation:
                                consecutive_stale_tolerance_checks += 1
                                visibility_event["stale_tolerance_checks"] = (
                                    consecutive_stale_tolerance_checks
                                )
                                if (
                                    consecutive_stale_tolerance_checks
                                    >= feedback_required_tolerance_checks
                                ):
                                    controller_stop_reason = (
                                        "stale_target_tolerance_reached"
                                    )
                                    break
                            else:
                                consecutive_stale_tolerance_checks = 0
                            continue
                        consecutive_stale_tolerance_checks = 0
                    elif math.hypot(*pixel_error) <= FEEDBACK_PIXEL_TOLERANCE:
                        consecutive_stale_tolerance_checks = 0
                        consecutive_tolerance_checks += 1
                        if consecutive_tolerance_checks >= feedback_required_tolerance_checks:
                            controller_stop_reason = "pixel_tolerance_reached"
                            break
                        continue
                    else:
                        consecutive_stale_tolerance_checks = 0
                        consecutive_tolerance_checks = 0
                    if control_iterations >= FEEDBACK_MAX_ITERATIONS:
                        controller_stop_reason = "iteration_limit"
                        break

                    current_angles = np.array(
                        [data.qpos[shoulder_qpos], data.qpos[elbow_qpos]], dtype=float
                    )
                    jacobian = camera_pixel_jacobian(
                        *current_angles,
                        (IMAGE_WIDTH, IMAGE_HEIGHT),
                        camera_height=CAMERA_HEIGHT,
                        end_effector_height=0.21,
                        vertical_fov_degrees=controller_camera_fovy,
                        link_lengths=LINK_LENGTHS,
                    )
                    delta = image_servo_joint_delta(
                        jacobian,
                        pixel_error,
                        max_step_radians=feedback_max_joint_step_radians,
                    )
                    desired_angles = current_angles + delta
                    desired_angles[0] = np.clip(desired_angles[0], -2.8, 2.8)
                    desired_angles[1] = np.clip(desired_angles[1], -2.6, 2.6)
                    data.ctrl[0], data.ctrl[1] = desired_angles
                    video_step_interval = max(
                        1, round(1.0 / (float(model.opt.timestep) * VIDEO_FPS))
                    )
                    for step in range(FEEDBACK_STEPS_PER_UPDATE):
                        _step_with_target_motion(
                            model,
                            data,
                            target_mocap_id,
                            (target_x, target_y),
                            target_motion_amplitude_m,
                            target_motion_frequency_hz,
                            target_motion_profile,
                            target_motion_phase_rad,
                        )
                        if save_episode_media and step % video_step_interval == 0:
                            dropout_status = (
                                f"TARGET LOST {consecutive_target_misses}/"
                                f"{FEEDBACK_TARGET_OCCLUSION_GRACE} | reusing last target  |  "
                                if consecutive_target_misses > 0
                                else ""
                            )
                            video_writer.write(
                                demo_video_frame(
                                    render_camera_rgb(),
                                    camera_overlay_pixel(pixel_xy),
                                    controller=controller,
                                    pixel_noise_std_px=pixel_noise_std_px,
                                    camera_fovy_error_deg=camera_fovy_error_deg,
                                    status=(
                                        f"{dropout_status}"
                                        f"camera age {camera_observation_age} obs  |  "
                                        f"image error {math.hypot(*pixel_error):.1f} px  |  "
                                        f"motion update {control_iterations + 1}"
                                    ),
                                    target_detection_missing=consecutive_target_misses > 0,
                                )
                            )
                    control_iterations += 1
                if controller_stop_reason is None:
                    controller_stop_reason = "iteration_limit"

        end_xy = data.site_xpos[site_id][:2].copy()
        true_xy = data.xpos[target_id][:2].copy()
        final_error = float(np.linalg.norm(end_xy - true_xy))
        initial_true_xy = np.asarray((target_x, target_y), dtype=float)
        estimate_error = float(np.linalg.norm(np.asarray(estimated_xy) - initial_true_xy))
        simulated_steps = (
            SIMULATION_STEPS
            if controller == "open_loop" and controller_error is None
            else control_iterations * FEEDBACK_STEPS_PER_UPDATE
        )
        final_pixel_error = None
        overhead_target_pixel = None
        overhead_tip_pixel = None
        final_rgb = render_camera_rgb()
        try:
            final_target_pixel = detect_red_target(final_rgb)
            final_end_effector_pixel = detect_green_end_effector(final_rgb)
            overhead_target_pixel = final_target_pixel
            overhead_tip_pixel = final_end_effector_pixel
            final_pixel_error = float(
                np.linalg.norm(np.asarray(final_target_pixel) - final_end_effector_pixel)
            )
        except RuntimeError:
            # A final occlusion is still scored using simulator ground truth;
            # leave the image-space diagnostic unavailable for that trial.
            pass
        success = final_error <= SUCCESS_THRESHOLD_METERS
        if failure_reason is None and not success:
            failure_reason = controller_stop_reason or "final_error_exceeded_threshold"
        terminal_verification = None
        terminal_verification_candidates = None
        verification_rgb = None
        if verify_terminal:
            renderer.update_scene(data, camera="verification_east")
            raw_verification_rgb = renderer.render()
            terminal_verification, verification_rgb = _measure_terminal_verification(
                model,
                data,
                verification_camera_id,
                raw_verification_rgb,
                fovy_error_deg=terminal_verification_fovy_error_deg,
                camera_position_error_m=tuple(verification_position_error),
                camera_rotation_error_deg=tuple(verification_rotation_error),
                brightness_scale=terminal_verification_brightness_scale,
                occluded_markers=terminal_verification_occluded_markers,
                minimum_quality_score=terminal_verification_min_quality_score,
            )
            if normalized_stress_candidates:
                terminal_verification_candidates = {}
                for name, settings in normalized_stress_candidates.items():
                    candidate_record, _ = _measure_terminal_verification(
                        model,
                        data,
                        verification_camera_id,
                        raw_verification_rgb,
                        **settings,
                    )
                    terminal_verification_candidates[name] = candidate_record

        if save_episode_media:
            displayed_final_error = round(final_error, 4)
            final_status = (
                f"SUCCESS | final error {displayed_final_error * 100.0:.1f} cm"
                if success
                else f"NOT REACHED | final error {displayed_final_error * 100.0:.1f} cm"
            )
            final_frame = demo_video_frame(
                final_rgb,
                camera_overlay_pixel(pixel_xy),
                controller=controller,
                pixel_noise_std_px=pixel_noise_std_px,
                camera_fovy_error_deg=camera_fovy_error_deg,
                status=final_status,
            )
            for _ in range(int(VIDEO_FPS * 0.8)):
                video_writer.write(final_frame)
        terminal_verification_video = None
        if save_episode_media and terminal_verification is not None:
            terminal_verification_video_path = output_dir / "terminal_verification.mp4"
            verification_writer = cv2.VideoWriter(
                str(terminal_verification_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                VIDEO_FPS,
                (IMAGE_WIDTH * 2, IMAGE_HEIGHT + 74),
            )
            if not verification_writer.isOpened():
                verification_writer.release()
                raise RuntimeError(
                    f"could not open video writer for {terminal_verification_video_path}"
                )
            verification_record = {
                **terminal_verification,
                "overhead_target_pixel_xy": (
                    overhead_target_pixel if overhead_target_pixel is not None else None
                ),
                "overhead_tip_pixel_xy": (
                    overhead_tip_pixel if overhead_tip_pixel is not None else None
                ),
            }
            verification_frame = terminal_verification_demo_frame(
                final_rgb,
                verification_rgb,
                verification_record,
                physical_success=success,
                reaching_error_m=final_error,
            )
            for _ in range(int(VIDEO_FPS * 2.0)):
                verification_writer.write(verification_frame)
            verification_writer.release()
            terminal_verification_video = terminal_verification_video_path.name
        results.append({
            "episode": index,
            "controller": controller,
            "camera_radial_distortion_k1": camera_radial_distortion_k1,
            "camera_radial_distortion_k2": camera_radial_distortion_k2,
            "controller_radial_distortion_k1": controller_radial_distortion_k1,
            "controller_radial_distortion_k2": controller_radial_distortion_k2,
            "controller_iterations": control_iterations,
            "simulated_motion_time_s": round(
                simulated_steps * float(model.opt.timestep), 4
            ),
            "target_pixel_xy": [round(value, 2) for value in detected_pixel_xy],
            "controller_pixel_xy": [round(value, 2) for value in pixel_xy],
            "controller_pixel_raw_xy": [
                round(value, 2) for value in raw_controller_pixel_xy
            ],
            "estimated_target_xy_m": [round(value, 4) for value in estimated_xy],
            "true_target_xy_m_for_evaluation_only": [round(float(value), 4) for value in true_xy],
            "target_start_xy_m_for_evaluation_only": [target_x, target_y],
            "controller_stop_target_xy_m_for_evaluation_only": [
                round(float(value), 5) for value in true_xy
            ],
            "final_joint_angles_rad_for_evaluation_only": [
                float(data.qpos[shoulder_qpos]),
                float(data.qpos[elbow_qpos]),
            ],
            "final_joint_velocities_rad_s_for_evaluation_only": [
                float(data.qvel[shoulder_dof]),
                float(data.qvel[elbow_dof]),
            ],
            "controller_stop_time_simulated_s": round(float(data.time), 6),
            "end_effector_xy_m": [round(float(value), 4) for value in end_xy],
            "perception_error_m": round(estimate_error, 4),
            "reaching_error_m": round(final_error, 4),
            "final_pixel_error_px": (
                round(final_pixel_error, 2) if final_pixel_error is not None else None
            ),
            "success": success,
            "terminal_visual_verification": terminal_verification,
            "terminal_verification_stress_candidates": terminal_verification_candidates,
            "terminal_verification_video": terminal_verification_video,
            "failure_reason": failure_reason,
            "controller_stop_reason": controller_stop_reason,
            "controller_error": controller_error,
            "target_reacquisitions": target_reacquisitions,
            "injected_dropout_reacquired": injected_dropout_reacquired,
            "natural_target_visibility_loss_observations": (
                natural_target_visibility_loss_observations
            ),
            "natural_target_visibility_loss_outside_dropout_observations": (
                natural_target_visibility_loss_outside_dropout_observations
            ),
            "natural_target_reacquisitions": natural_target_reacquisitions,
            "injected_dropout_visible_observations": injected_dropout_visible_observations,
            "injected_dropout_occlusion_overlap_observations": (
                injected_dropout_occlusion_overlap_observations
            ),
            "target_visibility_stop_cause": target_visibility_stop_cause,
            "target_visibility_trace": target_visibility_trace,
            "target_dropout_start_observation": target_dropout_start_observation,
            "target_dropout_duration_observations": target_dropout_duration_observations,
            "target_motion_amplitude_m": target_motion_amplitude_m,
            "target_motion_frequency_hz": target_motion_frequency_hz,
            "target_motion_profile": target_motion_profile,
            "target_motion_phase_rad": target_motion_phase_rad,
            "camera_observation_delay_observations": (
                camera_observation_delay_observations
            ),
            "camera_observation_delay_simulated_s": round(
                camera_observation_delay_observations
                * FEEDBACK_STEPS_PER_UPDATE
                * float(model.opt.timestep),
                4,
            ),
            "target_motion_prediction": target_motion_prediction,
            "use_current_end_effector_state": use_current_end_effector_state,
            "feedback_allow_stale_target_confirmation": (
                feedback_allow_stale_target_confirmation
            ),
        })

    renderer.close()
    if video_writer is not None:
        video_writer.release()
    successes = sum(result["success"] for result in results)
    errors = [result["reaching_error_m"] for result in results]
    report = {
        "seed": seed,
        "trials": episodes,
        "controller": controller,
        "pixel_noise_std_px": pixel_noise_std_px,
        "camera_fovy_error_deg": camera_fovy_error_deg,
        "camera_radial_distortion": {
            "model": "Brown-Conrady radial k1,k2",
            "actual_k1": camera_radial_distortion_k1,
            "actual_k2": camera_radial_distortion_k2,
            "controller_estimated_k1": controller_radial_distortion_k1,
            "controller_estimated_k2": controller_radial_distortion_k2,
        },
        "feedback_target_filter": {
            "method": "rolling_coordinate_median",
            "window": feedback_target_filter_window,
        },
        "feedback_required_tolerance_checks": feedback_required_tolerance_checks,
        "feedback_max_joint_step_radians": feedback_max_joint_step_radians,
        "feedback_allow_stale_target_confirmation": (
            feedback_allow_stale_target_confirmation
        ),
        "target_dropout_start_observation": target_dropout_start_observation,
        "target_dropout_duration_observations": target_dropout_duration_observations,
        "target_motion_amplitude_m": target_motion_amplitude_m,
        "target_motion_frequency_hz": target_motion_frequency_hz,
        "target_motion_profile": target_motion_profile,
        "target_motion_phase_rad": target_motion_phase_rad,
        "camera_observation_delay_observations": camera_observation_delay_observations,
        "camera_observation_delay_simulated_s": round(
            camera_observation_delay_observations
            * FEEDBACK_STEPS_PER_UPDATE
            * float(model.opt.timestep),
            4,
        ),
        "target_motion_prediction": target_motion_prediction,
        "use_current_end_effector_state": use_current_end_effector_state,
        "terminal_verification_enabled": verify_terminal,
        "terminal_verification_min_quality_score": (
            terminal_verification_min_quality_score if verify_terminal else None
        ),
        "terminal_verification_stress_settings": {
            "fovy_error_deg": terminal_verification_fovy_error_deg,
            "camera_position_error_m": verification_position_error.tolist(),
            "camera_rotation_error_deg": verification_rotation_error.tolist(),
            "image_brightness_scale": terminal_verification_brightness_scale,
            "occluded_markers": list(terminal_verification_occluded_markers),
            "minimum_quality_score": terminal_verification_min_quality_score,
        },
        "terminal_verification_summary": {
            "camera": "verification_east" if verify_terminal else None,
            "fresh_observation_after_control_stopped": verify_terminal,
            "visual_estimates_available": sum(
                result["terminal_visual_verification"] is not None
                and result["terminal_visual_verification"]["estimated_distance_m"]
                is not None
                for result in results
            ),
            "visual_success_estimates_within_threshold": sum(
                result["terminal_visual_verification"] is not None
                and result["terminal_visual_verification"]["visual_success_estimate"]
                is True
                for result in results
            ),
            "visual_success_estimates_outside_threshold": sum(
                result["terminal_visual_verification"] is not None
                and result["terminal_visual_verification"]["visual_success_estimate"]
                is False
                for result in results
            ),
            "visually_unavailable_trials": sum(
                result["terminal_visual_verification"] is not None
                and result["terminal_visual_verification"]["visual_success_estimate"]
                is None
                for result in results
            ),
            "quality_threshold_abstentions": sum(
                result["terminal_visual_verification"] is not None
                and result["terminal_visual_verification"]["visual_estimate_status"]
                == "unavailable_detection_quality_below_threshold"
                for result in results
            ),
        },
        "success_threshold_m": SUCCESS_THRESHOLD_METERS,
        "successes": successes,
        "success_rate": successes / episodes,
        "median_reaching_error_m": float(np.median(errors)),
        "max_reaching_error_m": max(errors),
        "camera_view": scene_path.name if save_media and scene_path.is_file() else None,
        "demo_video": video_path.name if save_media and video_path.is_file() else None,
        "video_episode_index": video_episode_index if save_media else None,
        "results": results,
    }
    (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes", type=int, default=20, help="number of randomized target trials"
    )
    parser.add_argument("--seed", type=int, default=7, help="deterministic random seed")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--video-episode-index",
        type=int,
        default=0,
        help="episode index to record in the demo video",
    )
    parser.add_argument(
        "--feedback-target-filter-window",
        type=int,
        default=FEEDBACK_TARGET_FILTER_WINDOW,
        help="number of recent target detections used by the feedback median",
    )
    parser.add_argument(
        "--feedback-required-tolerance-checks",
        type=int,
        default=FEEDBACK_REQUIRED_TOLERANCE_CHECKS,
        help="consecutive image-space checks required before stopping",
    )
    parser.add_argument(
        "--feedback-max-joint-step-radians",
        type=float,
        default=FEEDBACK_MAX_JOINT_STEP_RADIANS,
        help="maximum joint-angle change for one feedback motion update",
    )
    parser.add_argument(
        "--feedback-allow-stale-target-confirmation",
        action="store_true",
        help=(
            "experimental: allow arrival confirmation from the last target estimate "
            "during brief loss"
        ),
    )
    parser.add_argument(
        "--target-dropout-start-observation",
        type=int,
        help="feedback observation index at which injected target detection loss begins",
    )
    parser.add_argument(
        "--target-dropout-duration-observations",
        type=int,
        default=0,
        help="number of feedback observations with injected target detection loss",
    )
    parser.add_argument(
        "--target-motion-amplitude-m",
        type=float,
        default=0.0,
        help="target-motion amplitude along table x, in metres",
    )
    parser.add_argument(
        "--target-motion-frequency-hz",
        type=float,
        default=0.25,
        help="target-motion frequency in hertz",
    )
    parser.add_argument(
        "--target-motion-profile",
        choices=("sinusoidal", "piecewise_linear"),
        default="sinusoidal",
        help="target path used by moving-target experiments",
    )
    parser.add_argument(
        "--target-motion-phase-rad",
        type=float,
        default=0.0,
        help="starting phase of the moving-target path, in radians",
    )
    parser.add_argument(
        "--camera-observation-delay-observations",
        type=int,
        default=0,
        help="number of feedback observations by which the camera frame lags",
    )
    parser.add_argument(
        "--target-motion-prediction",
        choices=(
            "none",
            "constant_velocity",
            "constant_acceleration",
            "reversal_aware",
        ),
        default="none",
        help="experimental image-plane target prediction method",
    )
    parser.add_argument(
        "--use-current-end-effector-state",
        action="store_true",
        help=(
            "project current joint state for end-effector pixels instead of "
            "using the delayed camera image"
        ),
    )
    parser.add_argument(
        "--verify-terminal",
        action="store_true",
        help=(
            "after control stops, render a fresh calibrated east view and report "
            "its visual distance separately from physical reach"
        ),
    )
    parser.add_argument(
        "--terminal-verification-fovy-error-deg",
        type=float,
        default=0.0,
        help="estimator-only vertical field-of-view error for the terminal camera",
    )
    parser.add_argument(
        "--terminal-verification-camera-position-error-m",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="estimator-only world-frame camera position offset in metres",
    )
    parser.add_argument(
        "--terminal-verification-camera-rotation-error-deg",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("ROLL", "PITCH", "YAW"),
        help="estimator-only camera-frame roll, pitch, and yaw error in degrees",
    )
    parser.add_argument(
        "--terminal-verification-brightness-scale",
        type=float,
        default=1.0,
        help="uniform exposure scale applied only to the terminal verification image",
    )
    parser.add_argument(
        "--terminal-verification-min-quality-score",
        type=float,
        default=0.0,
        help=(
            "experimental abstention threshold from 0 to 1; the image-quality "
            "score is a heuristic, not a probability"
        ),
    )
    parser.add_argument(
        "--terminal-verification-occlude",
        choices=("target", "end_effector"),
        action="append",
        default=[],
        help="hide one marker in the terminal verification image; may be repeated",
    )
    parser.add_argument(
        "--controller",
        choices=("open_loop", "image_feedback"),
        default="open_loop",
        help="reaching controller to run",
    )
    parser.add_argument(
        "--pixel-noise-std-px",
        type=float,
        default=0.0,
        help="Gaussian standard deviation applied to detected target pixels",
    )
    parser.add_argument(
        "--camera-fovy-error-deg",
        type=float,
        default=0.0,
        help="field-of-view calibration error used by the controller, in degrees",
    )
    parser.add_argument(
        "--camera-radial-k1",
        type=float,
        default=0.0,
        help="actual simulated camera radial distortion coefficient k1",
    )
    parser.add_argument(
        "--camera-radial-k2",
        type=float,
        default=0.0,
        help="actual simulated camera radial distortion coefficient k2",
    )
    parser.add_argument(
        "--controller-radial-k1",
        type=float,
        default=0.0,
        help="estimated radial k1 used to correct measured pixels",
    )
    parser.add_argument(
        "--controller-radial-k2",
        type=float,
        default=0.0,
        help="estimated radial k2 used to correct measured pixels",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if not math.isfinite(args.pixel_noise_std_px) or args.pixel_noise_std_px < 0.0:
        parser.error("--pixel-noise-std-px must be finite and non-negative")
    if not math.isfinite(args.camera_fovy_error_deg):
        parser.error("--camera-fovy-error-deg must be finite")
    if not all(
        math.isfinite(value)
        for value in (
            args.camera_radial_k1,
            args.camera_radial_k2,
            args.controller_radial_k1,
            args.controller_radial_k2,
        )
    ):
        parser.error("radial distortion coefficients must be finite")
    if args.feedback_target_filter_window <= 0:
        parser.error("--feedback-target-filter-window must be a positive integer")
    if args.feedback_required_tolerance_checks <= 0:
        parser.error("--feedback-required-tolerance-checks must be a positive integer")
    if (
        not math.isfinite(args.feedback_max_joint_step_radians)
        or args.feedback_max_joint_step_radians <= 0.0
    ):
        parser.error("--feedback-max-joint-step-radians must be positive and finite")
    if (
        args.target_dropout_start_observation is not None
        and args.target_dropout_start_observation < 1
    ):
        parser.error("--target-dropout-start-observation must be a positive integer")
    if args.target_dropout_duration_observations < 0:
        parser.error("--target-dropout-duration-observations must be non-negative")
    if args.target_dropout_duration_observations and args.target_dropout_start_observation is None:
        parser.error("--target-dropout-start-observation is required for a non-zero dropout")
    if args.target_dropout_duration_observations and args.controller != "image_feedback":
        parser.error("target dropout requires --controller image_feedback")
    if (
        not math.isfinite(args.target_motion_amplitude_m)
        or not 0.0 <= args.target_motion_amplitude_m <= 0.16
    ):
        parser.error("--target-motion-amplitude-m must be between 0 and 0.16")
    if (
        not math.isfinite(args.target_motion_frequency_hz)
        or args.target_motion_frequency_hz < 0.0
    ):
        parser.error("--target-motion-frequency-hz must be finite and non-negative")
    if args.target_motion_amplitude_m and args.target_motion_frequency_hz <= 0.0:
        parser.error("moving targets require a positive --target-motion-frequency-hz")
    if not math.isfinite(args.target_motion_phase_rad):
        parser.error("--target-motion-phase-rad must be finite")
    if args.camera_observation_delay_observations < 0:
        parser.error("--camera-observation-delay-observations must be non-negative")
    if args.camera_observation_delay_observations and args.controller != "image_feedback":
        parser.error("camera observation delay requires --controller image_feedback")
    if args.target_motion_prediction != "none" and args.controller != "image_feedback":
        parser.error("target motion prediction requires --controller image_feedback")
    if args.use_current_end_effector_state and args.controller != "image_feedback":
        parser.error("current end-effector state requires --controller image_feedback")
    if (
        args.feedback_allow_stale_target_confirmation
        and args.controller != "image_feedback"
    ):
        parser.error("stale-target confirmation requires --controller image_feedback")
    if not math.isfinite(args.terminal_verification_fovy_error_deg):
        parser.error("--terminal-verification-fovy-error-deg must be finite")
    estimated_terminal_fovy_deg = (
        VERIFICATION_VIEWS["east"]["fovy_deg"]
        + args.terminal_verification_fovy_error_deg
    )
    if not 0.0 < estimated_terminal_fovy_deg < 180.0:
        parser.error(
            "estimated terminal camera field of view must be between 0 and 180 degrees"
        )
    if (
        not all(
            math.isfinite(value)
            for value in args.terminal_verification_camera_position_error_m
        )
        or not all(
            math.isfinite(value)
            for value in args.terminal_verification_camera_rotation_error_deg
        )
    ):
        parser.error("terminal verification camera errors must be finite")
    if (
        not math.isfinite(args.terminal_verification_brightness_scale)
        or args.terminal_verification_brightness_scale <= 0.0
    ):
        parser.error("--terminal-verification-brightness-scale must be positive and finite")
    if (
        not math.isfinite(args.terminal_verification_min_quality_score)
        or not 0.0 <= args.terminal_verification_min_quality_score <= 1.0
    ):
        parser.error("--terminal-verification-min-quality-score must be between 0 and 1")
    terminal_verification_has_stress = (
        args.terminal_verification_fovy_error_deg != 0.0
        or any(args.terminal_verification_camera_position_error_m)
        or any(args.terminal_verification_camera_rotation_error_deg)
        or args.terminal_verification_brightness_scale != 1.0
        or args.terminal_verification_min_quality_score > 0.0
        or bool(args.terminal_verification_occlude)
    )
    if terminal_verification_has_stress and not args.verify_terminal:
        parser.error("terminal verification stress settings require --verify-terminal")

    report = run_trials(
        args.episodes,
        args.seed,
        args.output_dir,
        pixel_noise_std_px=args.pixel_noise_std_px,
        camera_fovy_error_deg=args.camera_fovy_error_deg,
        camera_radial_distortion_k1=args.camera_radial_k1,
        camera_radial_distortion_k2=args.camera_radial_k2,
        controller_radial_distortion_k1=args.controller_radial_k1,
        controller_radial_distortion_k2=args.controller_radial_k2,
        controller=args.controller,
        video_episode_index=args.video_episode_index,
        feedback_target_filter_window=args.feedback_target_filter_window,
        feedback_required_tolerance_checks=args.feedback_required_tolerance_checks,
        feedback_max_joint_step_radians=args.feedback_max_joint_step_radians,
        feedback_allow_stale_target_confirmation=(
            args.feedback_allow_stale_target_confirmation
        ),
        target_dropout_start_observation=args.target_dropout_start_observation,
        target_dropout_duration_observations=args.target_dropout_duration_observations,
        target_motion_amplitude_m=args.target_motion_amplitude_m,
        target_motion_frequency_hz=args.target_motion_frequency_hz,
        target_motion_profile=args.target_motion_profile,
        target_motion_phase_rad=args.target_motion_phase_rad,
        camera_observation_delay_observations=(
            args.camera_observation_delay_observations
        ),
        target_motion_prediction=args.target_motion_prediction,
        use_current_end_effector_state=args.use_current_end_effector_state,
        verify_terminal=args.verify_terminal,
        terminal_verification_fovy_error_deg=args.terminal_verification_fovy_error_deg,
        terminal_verification_camera_position_error_m=tuple(
            args.terminal_verification_camera_position_error_m
        ),
        terminal_verification_camera_rotation_error_deg=tuple(
            args.terminal_verification_camera_rotation_error_deg
        ),
        terminal_verification_brightness_scale=args.terminal_verification_brightness_scale,
        terminal_verification_occluded_markers=tuple(args.terminal_verification_occlude),
        terminal_verification_min_quality_score=args.terminal_verification_min_quality_score,
    )
    print(
        f"Physical reaching (MuJoCo): {report['successes']}/{report['trials']} successful "
        f"({report['success_rate']:.0%}); median error "
        f"{report['median_reaching_error_m']:.3f} m"
    )
    if report["terminal_verification_enabled"]:
        summary = report["terminal_verification_summary"]
        print(
            "Fresh east-view visual estimates: "
            f"{summary['visual_estimates_available']}/{report['trials']} available; "
            f"within threshold {summary['visual_success_estimates_within_threshold']}, "
            f"outside {summary['visual_success_estimates_outside_threshold']}, "
            f"unavailable {summary['visually_unavailable_trials']}; "
            f"quality abstentions {summary['quality_threshold_abstentions']}. "
            "These estimates do not change the controller stop or physical score."
        )
    print(f"Saved report and demo artifacts in {args.output_dir.expanduser()}")
    return 0 if report["successes"] == report["trials"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
