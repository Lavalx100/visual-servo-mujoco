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
SUCCESS_THRESHOLD_METERS = 0.045
HOME_JOINT_ANGLES = (0.0, 0.7)
FEEDBACK_MAX_ITERATIONS = 20
FEEDBACK_STEPS_PER_UPDATE = 200
FEEDBACK_PIXEL_TOLERANCE = 8.0
FEEDBACK_TARGET_OCCLUSION_GRACE = 5
FEEDBACK_TARGET_FILTER_WINDOW = 3


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


def annotated_frame(rgb: np.ndarray, pixel_xy: tuple[float, float]) -> np.ndarray:
    """Return an RGB frame with the pixel used by the controller marked."""
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    center = tuple(int(round(value)) for value in pixel_xy)
    cv2.drawMarker(frame, center, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
    cv2.putText(frame, "target pixel used by controller", (14, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (20, 25, 30), 2, cv2.LINE_AA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def run_trials(
    episodes: int,
    seed: int,
    output_dir: Path,
    *,
    pixel_noise_std_px: float = 0.0,
    camera_fovy_error_deg: float = 0.0,
    controller: str = "open_loop",
    save_media: bool = True,
) -> dict:
    """Run reaching trials with optional perception and camera-calibration errors.

    ``pixel_noise_std_px`` adds independent Gaussian noise to each detected
    target centroid. ``camera_fovy_error_deg`` offsets the field of view used by
    the controller while leaving the simulated camera unchanged. These knobs
    model measurement and calibration error; they do not change the physics.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not math.isfinite(pixel_noise_std_px) or pixel_noise_std_px < 0.0:
        raise ValueError("pixel_noise_std_px must be finite and non-negative")
    if not math.isfinite(camera_fovy_error_deg):
        raise ValueError("camera_fovy_error_deg must be finite")
    if controller not in {"open_loop", "image_feedback"}:
        raise ValueError("controller must be 'open_loop' or 'image_feedback'")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
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

    # Targets are kept inside both the table and the arm's reachable annulus.
    targets = []
    while len(targets) < episodes:
        candidate = (
            float(target_rng.uniform(0.35, 0.67)),
            float(target_rng.uniform(-0.32, 0.32)),
        )
        radius = math.hypot(*candidate)
        if abs(LINK_LENGTHS[0] - LINK_LENGTHS[1]) + 0.02 < radius < sum(LINK_LENGTHS) - 0.04:
            targets.append(candidate)

    results = []
    scene_path = output_dir / "camera_view.png"
    video_path = output_dir / "reaching_demo.mp4"
    video_writer = None
    if save_media:
        video_writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (IMAGE_WIDTH, IMAGE_HEIGHT),
        )
        if not video_writer.isOpened():
            video_writer.release()
            video_writer = None

    for index, (target_x, target_y) in enumerate(targets):
        mujoco.mj_resetData(model, data)
        model.body_pos[target_id] = (target_x, target_y, TARGET_HEIGHT)
        data.qpos[shoulder_qpos] = HOME_JOINT_ANGLES[0]
        data.qpos[elbow_qpos] = HOME_JOINT_ANGLES[1]
        data.ctrl[0] = HOME_JOINT_ANGLES[0]
        data.ctrl[1] = HOME_JOINT_ANGLES[1]
        mujoco.mj_forward(model, data)

        # The controller gets only the camera image. Ground truth is used below
        # for evaluation, never for target localization or control.
        rgb = render_rgb(renderer, data)
        detected_pixel_xy = detect_red_target(rgb)
        noise = measurement_rng.normal(0.0, pixel_noise_std_px, size=2)
        pixel_xy = (
            float(detected_pixel_xy[0] + noise[0]),
            float(detected_pixel_xy[1] + noise[1]),
        )
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
        if controller == "open_loop":
            try:
                shoulder_angle, elbow_angle = inverse_kinematics(
                    *estimated_xy, link_lengths=LINK_LENGTHS
                )
            except ValueError as error:
                # Keep unreachable estimates in the benchmark as safe rejections.
                failure_reason = "estimated_target_unreachable"
                controller_error = str(error)

        if index == 0 and save_media:
            annotated = cv2.cvtColor(annotated_frame(rgb, pixel_xy), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(scene_path), annotated)

        control_iterations = 0
        if failure_reason is None:
            if controller == "open_loop":
                control_iterations = 1
                data.ctrl[0] = shoulder_angle
                data.ctrl[1] = elbow_angle
                for step in range(SIMULATION_STEPS):
                    mujoco.mj_step(model, data)
                    if index == 0 and video_writer is not None and step % 20 == 0:
                        frame = cv2.cvtColor(render_rgb(renderer, data), cv2.COLOR_RGB2BGR)
                        cv2.putText(
                            frame,
                            "Camera detection -> geometric projection -> IK",
                            (14, 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.62,
                            (20, 25, 30),
                            2,
                            cv2.LINE_AA,
                        )
                        video_writer.write(frame)
            else:
                target_pixel_history = [pixel_xy]
                consecutive_target_misses = 0
                for iteration in range(FEEDBACK_MAX_ITERATIONS):
                    if iteration > 0:
                        rgb = render_rgb(renderer, data)
                        try:
                            detected_pixel_xy = detect_red_target(rgb)
                        except RuntimeError as error:
                            consecutive_target_misses += 1
                            if consecutive_target_misses > FEEDBACK_TARGET_OCCLUSION_GRACE:
                                controller_stop_reason = "target_not_visible"
                                controller_error = str(error)
                                break
                        else:
                            consecutive_target_misses = 0
                            noise = measurement_rng.normal(0.0, pixel_noise_std_px, size=2)
                            measured_pixel_xy = (
                                float(detected_pixel_xy[0] + noise[0]),
                                float(detected_pixel_xy[1] + noise[1]),
                            )
                            target_pixel_history.append(measured_pixel_xy)
                            target_pixel_history = target_pixel_history[
                                -FEEDBACK_TARGET_FILTER_WINDOW:
                            ]
                            pixel_xy = median_pixel_estimate(target_pixel_history)
                    try:
                        end_effector_pixel_xy = detect_green_end_effector(rgb)
                    except RuntimeError as error:
                        controller_stop_reason = "end_effector_not_visible"
                        controller_error = str(error)
                        break
                    pixel_error = (
                        pixel_xy[0] - end_effector_pixel_xy[0],
                        pixel_xy[1] - end_effector_pixel_xy[1],
                    )
                    if math.hypot(*pixel_error) <= FEEDBACK_PIXEL_TOLERANCE:
                        controller_stop_reason = "pixel_tolerance_reached"
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
                    delta = image_servo_joint_delta(jacobian, pixel_error)
                    desired_angles = current_angles + delta
                    desired_angles[0] = np.clip(desired_angles[0], -2.8, 2.8)
                    desired_angles[1] = np.clip(desired_angles[1], -2.6, 2.6)
                    data.ctrl[0], data.ctrl[1] = desired_angles
                    for _ in range(FEEDBACK_STEPS_PER_UPDATE):
                        mujoco.mj_step(model, data)
                    control_iterations += 1

                    if index == 0 and video_writer is not None and iteration % 2 == 0:
                        frame = cv2.cvtColor(render_rgb(renderer, data), cv2.COLOR_RGB2BGR)
                        cv2.putText(
                            frame,
                            "Image-space target error -> damped Jacobian step",
                            (14, 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.58,
                            (20, 25, 30),
                            2,
                            cv2.LINE_AA,
                        )
                        video_writer.write(frame)
                if controller_stop_reason is None:
                    controller_stop_reason = "iteration_limit"

        end_xy = data.site_xpos[site_id][:2].copy()
        true_xy = model.body_pos[target_id][:2].copy()
        final_error = float(np.linalg.norm(end_xy - true_xy))
        estimate_error = float(np.linalg.norm(np.asarray(estimated_xy) - true_xy))
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
        "success_threshold_m": SUCCESS_THRESHOLD_METERS,
        "successes": successes,
        "success_rate": successes / episodes,
        "median_reaching_error_m": float(np.median(errors)),
        "max_reaching_error_m": max(errors),
        "camera_view": scene_path.name if save_media and scene_path.is_file() else None,
        "demo_video": video_path.name if save_media and video_path.is_file() else None,
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

    report = run_trials(
        args.episodes,
        args.seed,
        args.output_dir,
        pixel_noise_std_px=args.pixel_noise_std_px,
        camera_fovy_error_deg=args.camera_fovy_error_deg,
        controller=args.controller,
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
