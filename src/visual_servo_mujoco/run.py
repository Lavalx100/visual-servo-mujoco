"""Run deterministic camera-based reaching trials and save a small report/demo."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .controller import (
    camera_pixel_jacobian,
    image_servo_joint_delta,
    inverse_kinematics,
    median_pixel_estimate,
    pixel_to_table_xy,
)
from .model import MODEL_XML

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAMERA_HEIGHT = 2.0
TARGET_HEIGHT = 0.165
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


def render_rgb(renderer: mujoco.Renderer, data: mujoco.MjData) -> np.ndarray:
    renderer.update_scene(data, camera="overhead")
    return renderer.render()


def _step_with_target_motion(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_mocap_id: int,
    target_start_xy_m: tuple[float, float],
    target_motion_amplitude_m: float,
    target_motion_frequency_hz: float,
) -> None:
    """Advance one physics step while moving the target sinusoidally on x."""
    if target_motion_amplitude_m:
        simulation_time = float(data.time + model.opt.timestep)
        target_x = target_start_xy_m[0] + target_motion_amplitude_m * math.sin(
            2.0 * math.pi * target_motion_frequency_hz * simulation_time
        )
        data.mocap_pos[target_mocap_id] = (
            target_x,
            target_start_xy_m[1],
            TARGET_HEIGHT,
        )
    mujoco.mj_step(model, data)


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


def run_trials(
    episodes: int,
    seed: int,
    output_dir: Path,
    *,
    pixel_noise_std_px: float = 0.0,
    camera_fovy_error_deg: float = 0.0,
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
    camera_observation_delay_observations: int = 0,
    save_media: bool = True,
) -> dict:
    """Run reaching trials with optional perception and camera-calibration errors.

    ``pixel_noise_std_px`` adds independent Gaussian noise to each detected
    target centroid. ``camera_fovy_error_deg`` offsets the field of view used by
    the controller while leaving the simulated camera unchanged. The feedback
    filter window controls how many recent target detections feed its median;
    the tolerance-check count controls the stop confirmation; and the joint
    step value caps each commanded update. A target dropout skips detection for
    a specified range of feedback observations. These knobs do not change the
    physics. The target can also move sinusoidally along the table's x axis,
    and image feedback can consume a camera frame from earlier feedback
    observations. Stale-target confirmation is an experimental option for the
    stationary-target scene.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not math.isfinite(pixel_noise_std_px) or pixel_noise_std_px < 0.0:
        raise ValueError("pixel_noise_std_px must be finite and non-negative")
    if not math.isfinite(camera_fovy_error_deg):
        raise ValueError("camera_fovy_error_deg must be finite")
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
    if target_motion_amplitude_m and target_motion_frequency_hz <= 0.0:
        raise ValueError("moving targets require a positive target_motion_frequency_hz")
    if (
        isinstance(camera_observation_delay_observations, bool)
        or not isinstance(camera_observation_delay_observations, int)
        or camera_observation_delay_observations < 0
    ):
        raise ValueError("camera_observation_delay_observations must be a non-negative integer")
    if camera_observation_delay_observations and controller != "image_feedback":
        raise ValueError("camera observation delay can only be used with image_feedback")
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
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
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
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
    camera_fovy = float(model.cam_fovy[camera_id])
    target_rng = np.random.default_rng(seed)
    measurement_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5EED]))
    controller_camera_fovy = camera_fovy + camera_fovy_error_deg
    if not 0.0 < controller_camera_fovy < 180.0:
        raise ValueError("calibrated camera field of view must be between 0 and 180 degrees")

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
        data.mocap_pos[target_mocap_id] = (target_x, target_y, TARGET_HEIGHT)
        data.qpos[shoulder_qpos] = HOME_JOINT_ANGLES[0]
        data.qpos[elbow_qpos] = HOME_JOINT_ANGLES[1]
        data.ctrl[0] = HOME_JOINT_ANGLES[0]
        data.ctrl[1] = HOME_JOINT_ANGLES[1]
        mujoco.mj_forward(model, data)

        # The controller gets only the camera image. Ground truth is used below
        # for evaluation, never for target localization or control.
        rgb = render_rgb(renderer, data)
        camera_frame_history = [rgb]
        camera_target_position_history = [(target_x, target_y)]
        detected_pixel_xy = detect_red_target(rgb)
        noise = measurement_rng.normal(0.0, pixel_noise_std_px, size=2)
        pixel_xy = (
            float(detected_pixel_xy[0] + noise[0]),
            float(detected_pixel_xy[1] + noise[1]),
        )
        camera_target_pixel_history: list[tuple[float, float] | None] = [pixel_xy]
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
                "target_position_xy_m_for_evaluation_only": [target_x, target_y],
                "observed_target_position_xy_m_for_evaluation_only": [
                    target_x,
                    target_y,
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
            annotated = cv2.cvtColor(annotated_frame(rgb, pixel_xy), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(scene_path), annotated)
            initial_frame = demo_video_frame(
                rgb,
                pixel_xy,
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
                    )
                    if save_episode_media and step % video_step_interval == 0:
                        video_writer.write(
                            demo_video_frame(
                                render_rgb(renderer, data),
                                pixel_xy,
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
                camera_observation_age = 0
                for iteration in range(max_observations):
                    visibility_event = target_visibility_trace[0] if iteration == 0 else None
                    if iteration > 0:
                        rgb = render_rgb(renderer, data)
                        camera_frame_history.append(rgb)
                        camera_target_position_history.append(
                            tuple(float(value) for value in data.xpos[target_id][:2])
                        )
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
                            captured_target_pixel_xy = (
                                float(current_detected_pixel_xy[0] + noise[0]),
                                float(current_detected_pixel_xy[1] + noise[1]),
                            )
                        camera_target_pixel_history.append(captured_target_pixel_xy)
                        if len(camera_frame_history) > camera_observation_delay_observations + 1:
                            camera_frame_history.pop(0)
                            camera_target_position_history.pop(0)
                            camera_target_pixel_history.pop(0)
                        camera_observation_age = min(
                            camera_observation_delay_observations, iteration
                        )
                        observation_rgb = camera_frame_history[
                            -1 - camera_observation_age
                        ]
                        observed_target_position_xy = camera_target_position_history[
                            -1 - camera_observation_age
                        ]
                        observed_target_pixel_xy = camera_target_pixel_history[
                            -1 - camera_observation_age
                        ]
                        target_visible_in_observation = observed_target_pixel_xy is not None
                        if target_visible_in_observation:
                            detected_pixel_xy = observed_target_pixel_xy
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
                            target_pixel_history.append(detected_pixel_xy)
                            target_pixel_history = target_pixel_history[
                                -feedback_target_filter_window:
                            ]
                            pixel_xy = median_pixel_estimate(target_pixel_history)
                        visibility_event = {
                            "observation_index": iteration,
                            "target_visible_in_render": target_visible_in_render,
                            "target_visible_in_observation": target_visible_in_observation,
                            "injected_dropout": dropout_active,
                            "camera_observation_age_observations": camera_observation_age,
                            "camera_observation_age_simulated_s": round(
                                camera_observation_age
                                * FEEDBACK_STEPS_PER_UPDATE
                                * float(model.opt.timestep),
                                4,
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
                        end_effector_pixel_xy = detect_green_end_effector(observation_rgb)
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
                    visibility_event["end_effector_pixel_xy"] = [
                        round(value, 2) for value in end_effector_pixel_xy
                    ]
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
                                    render_rgb(renderer, data),
                                    pixel_xy,
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
        try:
            final_rgb = render_rgb(renderer, data)
            final_target_pixel = detect_red_target(final_rgb)
            final_end_effector_pixel = detect_green_end_effector(final_rgb)
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
        if save_episode_media:
            displayed_final_error = round(final_error, 4)
            final_status = (
                f"SUCCESS | final error {displayed_final_error * 100.0:.1f} cm"
                if success
                else f"NOT REACHED | final error {displayed_final_error * 100.0:.1f} cm"
            )
            final_frame = demo_video_frame(
                final_rgb,
                pixel_xy,
                controller=controller,
                pixel_noise_std_px=pixel_noise_std_px,
                camera_fovy_error_deg=camera_fovy_error_deg,
                status=final_status,
            )
            for _ in range(int(VIDEO_FPS * 0.8)):
                video_writer.write(final_frame)
        results.append({
            "episode": index,
            "controller": controller,
            "controller_iterations": control_iterations,
            "simulated_motion_time_s": round(
                simulated_steps * float(model.opt.timestep), 4
            ),
            "target_pixel_xy": [round(value, 2) for value in detected_pixel_xy],
            "controller_pixel_xy": [round(value, 2) for value in pixel_xy],
            "estimated_target_xy_m": [round(value, 4) for value in estimated_xy],
            "true_target_xy_m_for_evaluation_only": [round(float(value), 4) for value in true_xy],
            "target_start_xy_m_for_evaluation_only": [target_x, target_y],
            "end_effector_xy_m": [round(float(value), 4) for value in end_xy],
            "perception_error_m": round(estimate_error, 4),
            "reaching_error_m": round(final_error, 4),
            "final_pixel_error_px": (
                round(final_pixel_error, 2) if final_pixel_error is not None else None
            ),
            "success": success,
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
            "camera_observation_delay_observations": (
                camera_observation_delay_observations
            ),
            "camera_observation_delay_simulated_s": round(
                camera_observation_delay_observations
                * FEEDBACK_STEPS_PER_UPDATE
                * float(model.opt.timestep),
                4,
            ),
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
        "camera_observation_delay_observations": camera_observation_delay_observations,
        "camera_observation_delay_simulated_s": round(
            camera_observation_delay_observations
            * FEEDBACK_STEPS_PER_UPDATE
            * float(model.opt.timestep),
            4,
        ),
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
        help="sinusoidal target-motion amplitude along table x, in metres",
    )
    parser.add_argument(
        "--target-motion-frequency-hz",
        type=float,
        default=0.25,
        help="sinusoidal target-motion frequency in hertz",
    )
    parser.add_argument(
        "--camera-observation-delay-observations",
        type=int,
        default=0,
        help="number of feedback observations by which the camera frame lags",
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
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if not math.isfinite(args.pixel_noise_std_px) or args.pixel_noise_std_px < 0.0:
        parser.error("--pixel-noise-std-px must be finite and non-negative")
    if not math.isfinite(args.camera_fovy_error_deg):
        parser.error("--camera-fovy-error-deg must be finite")
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
    if args.camera_observation_delay_observations < 0:
        parser.error("--camera-observation-delay-observations must be non-negative")
    if args.camera_observation_delay_observations and args.controller != "image_feedback":
        parser.error("camera observation delay requires --controller image_feedback")
    if (
        args.feedback_allow_stale_target_confirmation
        and args.controller != "image_feedback"
    ):
        parser.error("stale-target confirmation requires --controller image_feedback")

    report = run_trials(
        args.episodes,
        args.seed,
        args.output_dir,
        pixel_noise_std_px=args.pixel_noise_std_px,
        camera_fovy_error_deg=args.camera_fovy_error_deg,
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
        camera_observation_delay_observations=(
            args.camera_observation_delay_observations
        ),
    )
    print(
        f"Camera-based reaching: {report['successes']}/{report['trials']} successful "
        f"({report['success_rate']:.0%}); median error "
        f"{report['median_reaching_error_m']:.3f} m"
    )
    print(f"Saved report and demo artifacts in {args.output_dir.expanduser()}")
    return 0 if report["successes"] == report["trials"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
