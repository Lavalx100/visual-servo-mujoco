"""Render synchronized overhead and oblique views for terminal verification."""

from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np

from visual_servo_mujoco.run import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    VIDEO_FPS,
    detect_green_end_effector,
    detect_red_target,
    run_trials,
)
from visual_servo_mujoco.verification_benchmark import (
    VERIFICATION_VIEWS,
    _model_with_verification_views,
)


DEMO_CASES = ((0.0, 0, "clean"), (20.0, 1, "20px-noise"))
HEADER_HEIGHT = 64
FRAMES_PER_OBSERVATION = 12
VERIFICATION_VIEW = "east"


def _render_state_video(
    *,
    repository: Path,
    noise: float,
    seed: int,
    name: str,
) -> Path:
    output_dir = repository / "assets" / "demos"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _model_with_verification_views()
    renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
    data = mujoco.MjData(model)
    result = run_trials(
        episodes=1,
        seed=seed,
        output_dir=repository / "artifacts" / f"verification-demo-{name}",
        controller="image_feedback",
        pixel_noise_std_px=noise,
        save_media=False,
    )["results"][0]
    trace = result["target_visibility_trace"]
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    target_mocap_id = int(model.body_mocapid[target_body_id])
    shoulder_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_joint"
    )
    elbow_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "elbow_joint")
    verification_camera = f"verification_{VERIFICATION_VIEW}"
    destination = output_dir / f"verification_view_{name}.mp4"
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        VIDEO_FPS,
        (IMAGE_WIDTH * 2, IMAGE_HEIGHT + HEADER_HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {destination}")

    for event_index, event in enumerate(trace):
        target_xy = event["target_position_xy_m_for_evaluation_only"]
        data.mocap_pos[target_mocap_id] = (*target_xy, 0.165)
        data.qpos[model.jnt_qposadr[shoulder_joint_id]] = event["joint_angles_rad"][0]
        data.qpos[model.jnt_qposadr[elbow_joint_id]] = event["joint_angles_rad"][1]
        mujoco.mj_forward(model, data)

        frames = []
        visibility = []
        for camera_name in ("overhead", verification_camera):
            renderer.update_scene(data, camera=camera_name)
            rgb = renderer.render()
            try:
                detect_red_target(rgb)
                target_visible = True
            except RuntimeError:
                target_visible = False
            try:
                detect_green_end_effector(rgb)
                tip_visible = True
            except RuntimeError:
                tip_visible = False
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            frames.append(frame)
            visibility.append((target_visible, tip_visible))

        terminal = event_index == len(trace) - 1
        if terminal:
            outcome = "PHYSICAL REACH: YES" if result["success"] else "PHYSICAL REACH: NO"
            stop = result["controller_stop_reason"]
            secondary = visibility[1]
            secondary_text = "both markers visible" if all(secondary) else "marker missing"
            header = (
                f"TERMINAL STATE | {outcome} | error {result['reaching_error_m'] * 100:.1f} cm | "
                f"overhead stop: {stop} | oblique view: {secondary_text}"
            )
        else:
            header = (
                f"{name} | seed {seed} | observation {event['observation_index']} | "
                "fresh current-time renders at control updates"
            )
        canvas = np.zeros(
            (IMAGE_HEIGHT + HEADER_HEIGHT, IMAGE_WIDTH * 2, 3), dtype=np.uint8
        )
        cv2.putText(
            canvas,
            header,
            (14, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        for index, label in enumerate(("CONTROL: OVERHEAD", "VERIFICATION: EAST OBLIQUE")):
            x = index * IMAGE_WIDTH
            state = visibility[index]
            marker_text = (
                f"target {'visible' if state[0] else 'hidden'} | "
                f"tip {'visible' if state[1] else 'hidden'}"
            )
            color = (105, 230, 170) if all(state) else (90, 150, 250)
            cv2.putText(
                canvas,
                f"{label} | {marker_text}",
                (x + 14, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                color,
                1,
                cv2.LINE_AA,
            )
            canvas[HEADER_HEIGHT:, x : x + IMAGE_WIDTH] = frames[index]
        cv2.line(
            canvas,
            (IMAGE_WIDTH, 0),
            (IMAGE_WIDTH, IMAGE_HEIGHT + HEADER_HEIGHT),
            (100, 100, 100),
            1,
        )
        repeat = FRAMES_PER_OBSERVATION
        if terminal:
            repeat *= 2
        for _ in range(repeat):
            writer.write(canvas)

    writer.release()
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"verification demo was not created: {destination}")
    print(
        f"Wrote {destination.relative_to(repository)} "
        f"({destination.stat().st_size / 1024:.0f} KiB)"
    )
    return destination


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    if VERIFICATION_VIEW not in VERIFICATION_VIEWS:
        raise RuntimeError(f"unknown verification view {VERIFICATION_VIEW}")
    for noise, seed, name in DEMO_CASES:
        _render_state_video(
            repository=repository,
            noise=noise,
            seed=seed,
            name=name,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
