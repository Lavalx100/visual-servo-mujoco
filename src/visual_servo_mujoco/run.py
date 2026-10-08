"""Run deterministic camera-based reaching trials and save a small report/demo."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .controller import inverse_kinematics, pixel_to_table_xy
from .model import MODEL_XML

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAMERA_HEIGHT = 2.0
TARGET_HEIGHT = 0.165
LINK_LENGTHS = (0.42, 0.34)
SIMULATION_STEPS = 1200
SUCCESS_THRESHOLD_METERS = 0.045


def detect_red_target(rgb_image: np.ndarray) -> tuple[float, float]:
    """Find the largest red object in an RGB render and return its pixel centroid."""
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    red_low = cv2.inRange(hsv, np.array([0, 110, 70]), np.array([12, 255, 255]))
    red_high = cv2.inRange(hsv, np.array([168, 110, 70]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(red_low, red_high)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        raise RuntimeError("camera image contains no red target")
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest_label, cv2.CC_STAT_AREA] < 20:
        raise RuntimeError("red target detection was smaller than 20 pixels")
    x, y = centroids[largest_label]
    return float(x), float(y)


def render_rgb(renderer: mujoco.Renderer, data: mujoco.MjData) -> np.ndarray:
    renderer.update_scene(data, camera="overhead")
    return renderer.render()


def annotated_frame(rgb: np.ndarray, pixel_xy: tuple[float, float]) -> np.ndarray:
    """Return an RGB frame with the detector's centroid marked."""
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    center = tuple(int(round(value)) for value in pixel_xy)
    cv2.drawMarker(frame, center, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
    cv2.putText(frame, "camera-detected target", (14, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (20, 25, 30), 2, cv2.LINE_AA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def run_trials(episodes: int, seed: int, output_dir: Path) -> dict:
    """Run trials, estimate targets from rendered pixels, and save artifacts."""
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
    rng = np.random.default_rng(seed)

    # Targets are kept inside both the table and the arm's reachable annulus.
    targets = []
    while len(targets) < episodes:
        candidate = (float(rng.uniform(0.35, 0.67)), float(rng.uniform(-0.32, 0.32)))
        radius = math.hypot(*candidate)
        if abs(LINK_LENGTHS[0] - LINK_LENGTHS[1]) + 0.02 < radius < sum(LINK_LENGTHS) - 0.04:
            targets.append(candidate)

    results = []
    scene_path = output_dir / "camera_view.png"
    video_path = output_dir / "reaching_demo.mp4"
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
        mujoco.mj_forward(model, data)

        # The controller gets only the camera image. Ground truth is used below
        # for evaluation, never for target localization or control.
        rgb = render_rgb(renderer, data)
        pixel_xy = detect_red_target(rgb)
        estimated_xy = pixel_to_table_xy(
            pixel_xy,
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            camera_height=CAMERA_HEIGHT,
            target_height=TARGET_HEIGHT,
            vertical_fov_degrees=camera_fovy,
        )
        shoulder_angle, elbow_angle = inverse_kinematics(
            *estimated_xy, link_lengths=LINK_LENGTHS
        )

        if index == 0:
            cv2.imwrite(str(scene_path), cv2.cvtColor(annotated_frame(rgb, pixel_xy), cv2.COLOR_RGB2BGR))

        data.ctrl[0] = shoulder_angle
        data.ctrl[1] = elbow_angle
        for step in range(SIMULATION_STEPS):
            mujoco.mj_step(model, data)
            if index == 0 and video_writer is not None and step % 20 == 0:
                frame = cv2.cvtColor(render_rgb(renderer, data), cv2.COLOR_RGB2BGR)
                cv2.putText(
                    frame,
                    "OpenCV target detection -> IK joint targets",
                    (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (20, 25, 30),
                    2,
                    cv2.LINE_AA,
                )
                video_writer.write(frame)

        end_xy = data.site_xpos[site_id][:2].copy()
        true_xy = model.body_pos[target_id][:2].copy()
        final_error = float(np.linalg.norm(end_xy - true_xy))
        estimate_error = float(np.linalg.norm(np.asarray(estimated_xy) - true_xy))
        results.append({
            "episode": index,
            "target_pixel_xy": [round(value, 2) for value in pixel_xy],
            "estimated_target_xy_m": [round(value, 4) for value in estimated_xy],
            "true_target_xy_m_for_evaluation_only": [round(float(value), 4) for value in true_xy],
            "end_effector_xy_m": [round(float(value), 4) for value in end_xy],
            "perception_error_m": round(estimate_error, 4),
            "reaching_error_m": round(final_error, 4),
            "success": final_error <= SUCCESS_THRESHOLD_METERS,
        })

    renderer.close()
    if video_writer is not None:
        video_writer.release()
    successes = sum(result["success"] for result in results)
    errors = [result["reaching_error_m"] for result in results]
    report = {
        "seed": seed,
        "trials": episodes,
        "success_threshold_m": SUCCESS_THRESHOLD_METERS,
        "successes": successes,
        "success_rate": successes / episodes,
        "median_reaching_error_m": float(np.median(errors)),
        "max_reaching_error_m": max(errors),
        "camera_view": scene_path.name,
        "demo_video": video_path.name if video_path.is_file() else None,
        "results": results,
    }
    (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20, help="number of randomized target trials")
    parser.add_argument("--seed", type=int, default=7, help="deterministic random seed")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    report = run_trials(args.episodes, args.seed, args.output_dir)
    print(
        f"Camera-based reaching: {report['successes']}/{report['trials']} successful "
        f"({report['success_rate']:.0%}); median error "
        f"{report['median_reaching_error_m']:.3f} m"
    )
    print(f"Saved report and demo artifacts in {args.output_dir.expanduser()}")
    return 0 if report["successes"] == report["trials"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
