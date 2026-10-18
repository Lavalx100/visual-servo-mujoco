"""Render the prior global and conditional uncertainty policies on one replay."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from visual_servo_mujoco.completion_speed_benchmark import MOTION_CONDITIONS
from visual_servo_mujoco.conditional_uncertainty_benchmark import (
    PRIOR_GLOBAL_Q99_MARGIN_M,
    SPEED_BOUND_M_S,
    evaluate_conditional_envelope_policy,
)
from visual_servo_mujoco.run import IMAGE_HEIGHT, IMAGE_WIDTH, run_trials
from visual_servo_mujoco.timestamped_verification_benchmark import (
    CONTROL_MODES,
    DELIVERY_PROFILES,
    PostStopCameraReplay,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "artifacts" / "conditional-uncertainty-benchmark.json"
VIDEO_PATH = ROOT / "assets" / "demos" / "conditional_uncertainty_comparison.mp4"
GIF_PATH = ROOT / "assets" / "demos" / "previews" / "conditional_uncertainty_comparison.gif"
HEADER_H = 94
FOOTER_H = 112
PANEL_SIZE = (IMAGE_WIDTH, HEADER_H + IMAGE_HEIGHT + FOOTER_H)
VIDEO_SIZE = (IMAGE_WIDTH * 2, PANEL_SIZE[1])


def _identity(record: dict) -> tuple:
    return (
        record["motion_condition"],
        int(record["seed"]),
        int(record["episode"]),
        float(record["target_motion_phase_rad"]),
        record["control_mode"],
        int(record["control_delay_observations"]),
        record["delivery_profile"],
    )


def _select_example(report: dict) -> tuple[dict, dict]:
    pairs: dict[tuple, dict[str, dict]] = {}
    for row in report["validation_trial_results"]:
        if row["control_mode"] != "fully_delayed_rgb" or row["control_delay_observations"] != 2:
            continue
        if row["policy"] not in (
            "two_post_settle_prior_global_q99",
            "two_post_settle_conditional_q99",
        ):
            continue
        pairs.setdefault(_identity(row), {})[row["policy"]] = row
    candidates = []
    for identity, policy_rows in pairs.items():
        prior = policy_rows.get("two_post_settle_prior_global_q99")
        conditional = policy_rows.get("two_post_settle_conditional_q99")
        if prior is None or conditional is None:
            continue
        if prior["false_confirmation_at_delivery"] and not conditional["false_confirmation_at_delivery"]:
            priority = 0
        elif (
            conditional["visual_confirmed"]
            and not prior["visual_confirmed"]
            and not conditional["false_confirmation_at_delivery"]
        ):
            priority = 1
        elif prior["visual_confirmed"] != conditional["visual_confirmed"]:
            priority = 2
        else:
            priority = 3
        delivery_priority = {"high_latency": 0, "moderate_latency": 1, "near_real_time": 2}[
            identity[-1]
        ]
        candidates.append(((priority, delivery_priority, identity), prior, conditional))
    if not candidates:
        raise ValueError("benchmark report has no paired fully-delayed RGB outcomes")
    _, prior, conditional = min(candidates, key=lambda item: item[0])
    return prior, conditional


def _rerun_trial(record: dict, temp_dir: Path) -> tuple[dict, dict]:
    condition = next(item for item in MOTION_CONDITIONS if item.name == record["motion_condition"])
    mode_name = record["control_mode"]
    delay = int(record["control_delay_observations"])
    use_current_tip = mode_name == "delayed_target_current_joint_tip"
    episode = int(record["episode"])
    run = run_trials(
        episode + 1,
        int(record["seed"]),
        temp_dir / condition.name / mode_name / str(delay),
        controller="image_feedback",
        pixel_noise_std_px=2.0,
        target_motion_amplitude_m=condition.amplitude_m,
        target_motion_frequency_hz=condition.frequency_hz,
        target_motion_profile=condition.profile,
        target_motion_phase_rad=float(record["target_motion_phase_rad"]),
        camera_observation_delay_observations=delay,
        target_motion_prediction="none",
        use_current_end_effector_state=use_current_tip,
        feedback_allow_stale_target_confirmation=False,
        verify_terminal=False,
        save_media=False,
    )
    trial = next(row for row in run["results"] if int(row["episode"]) == episode)
    trial = {"seed": int(record["seed"]), **trial}
    profile = next(item for item in DELIVERY_PROFILES if item.name == record["delivery_profile"])
    replay = PostStopCameraReplay(save_images=True)
    try:
        stream = replay.replay(
            trial,
            delivery_profiles=(profile,),
            seed=int(record["seed"]),
        )
    finally:
        replay.close()
    if stream["settled_time_s"] is None:
        raise ValueError("selected demo trial did not settle")
    return trial, stream


def _panel(
    *,
    title: str,
    color: tuple[int, int, int],
    trial: dict,
    stream: dict,
    now_s: float,
    margin_model: dict,
    fixed_margin: float | None,
) -> np.ndarray:
    panel = np.zeros((PANEL_SIZE[1], IMAGE_WIDTH, 3), dtype=np.uint8)
    panel[:] = (28, 38, 50)
    events = stream["events_by_profile"]["near_real_time"] if "near_real_time" in stream["events_by_profile"] else next(iter(stream["events_by_profile"].values()))
    available = [
        event
        for event in events
        if event["delivery_time_s"] is not None
        and float(event["delivery_time_s"]) <= now_s + 1e-9
    ]
    latest_event = (
        max(available, key=lambda item: (item["delivery_time_s"], item["frame_index"]))
        if available
        else None
    )
    result = evaluate_conditional_envelope_policy(
        trial,
        available,
        settled_time_s=stream["settled_time_s"],
        margin_model=margin_model,
        max_target_speed_m_s=SPEED_BOUND_M_S,
        fixed_margin_m=fixed_margin,
    )
    decision_index = result.get("decision_frame_index")
    event = (
        next(
            item
            for item in available
            if int(item["frame_index"]) == int(decision_index)
        )
        if decision_index is not None
        else latest_event
    )
    if event is not None:
        image = stream["images_by_frame"].get(int(event["frame_index"]))
        if image is not None:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            panel[HEADER_H : HEADER_H + IMAGE_HEIGHT] = bgr
    cv2.rectangle(panel, (0, 0), (IMAGE_WIDTH, HEADER_H - 1), (20, 29, 39), -1)
    cv2.putText(panel, title, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)
    status = "VISUAL CONFIRMATION" if result["visual_confirmed"] else "ABSTAIN"
    cv2.putText(
        panel,
        f"{status} | {result['usable_frames']} usable post-settle frame(s) | t={now_s - float(trial['controller_stop_time_simulated_s']):.2f}s after stop",
        (14, 63),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (108, 225, 173) if result["visual_confirmed"] else (225, 194, 128),
        1,
        cv2.LINE_AA,
    )
    footer_y = HEADER_H + IMAGE_HEIGHT
    cv2.rectangle(panel, (0, footer_y), (IMAGE_WIDTH, PANEL_SIZE[1]), (20, 29, 39), -1)
    if event is None:
        lines = ["Waiting for a delivered camera frame", "", ""]
    elif event.get("estimated_distance_m") is None:
        lines = [
            f"Frame {event['frame_index']} | capture age {float(event['camera_age_at_delivery_s']):.2f}s",
            "Image distance unavailable; no margin test can pass",
            "MuJoCo distance is evaluation-only and not used by the policy",
        ]
    else:
        estimate = float(event["estimated_distance_m"])
        age = float(event["camera_age_at_delivery_s"])
        if fixed_margin is None:
            margin = float(event["conditional_margin_m"])
            band = int(event["conditional_bin"])
            label = f"conditional bin {band}"
        else:
            margin = fixed_margin
            label = "locked prior global q99"
        upper = estimate + margin + SPEED_BOUND_M_S * age
        actual = event.get("physical_error_at_delivery_m")
        lines = [
            f"Frame {event['frame_index']} | capture age {age:.2f}s | image estimate {estimate * 100:.2f} cm",
            f"{label} margin {margin * 100:.2f} cm + motion allowance {SPEED_BOUND_M_S * age * 100:.2f} cm = upper {upper * 100:.2f} cm",
            f"MuJoCo delivery distance {actual * 100:.2f} cm (evaluation label only)" if actual is not None else "Delivery label unavailable",
        ]
    for index, line in enumerate(lines):
        cv2.putText(
            panel,
            line[:106],
            (14, footer_y + 27 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (221, 231, 239),
            1,
            cv2.LINE_AA,
        )
    return panel


def generate_demo(
    *,
    report_path: Path = REPORT_PATH,
    video_path: Path = VIDEO_PATH,
    gif_path: Path = GIF_PATH,
) -> tuple[Path, Path]:
    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))
    prior, conditional = _select_example(report)
    if _identity(prior) != _identity(conditional):
        raise RuntimeError("paired demo selection changed trial identity")
    with tempfile.TemporaryDirectory(prefix="visual-servo-conditional-demo-") as temp_dir:
        trial, stream = _rerun_trial(prior, Path(temp_dir))
    profile = prior["delivery_profile"]
    events = stream["events_by_profile"][profile]
    for event in events:
        event["conditional_margin_m"], event["conditional_bin"] = (
            _margin_for_event(event, report["calibration"])
        )
    times = [
        float(event["delivery_time_s"])
        for event in events
        if event["delivery_time_s"] is not None
    ]
    start = float(trial["controller_stop_time_simulated_s"])
    end = max(times, default=start + 2.0)
    fps = 12
    step = 1.0 / fps
    video_path.parent.mkdir(parents=True, exist_ok=True)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, VIDEO_SIZE
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {video_path}")
    gif_frames = []
    try:
        now_s = start
        while now_s <= end + 0.5:
            global_panel = _panel(
                title="Prior global q99",
                color=(85, 149, 229),
                trial=trial,
                stream=stream,
                now_s=now_s,
                margin_model=report["calibration"],
                fixed_margin=PRIOR_GLOBAL_Q99_MARGIN_M,
            )
            conditional_panel = _panel(
                title="Pixel and marker-quality conditional q99",
                color=(92, 205, 151),
                trial=trial,
                stream=stream,
                now_s=now_s,
                margin_model=report["calibration"],
                fixed_margin=None,
            )
            frame = np.concatenate((global_panel, conditional_panel), axis=1)
            writer.write(frame)
            if len(gif_frames) < 42:
                gif_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            now_s += step
    finally:
        writer.release()
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=1000 // fps,
        loop=0,
        optimize=True,
    )
    return video_path, gif_path


def _margin_for_event(event: dict, model: dict) -> tuple[float, int]:
    feature = event.get("quality_adjusted_pixel_noise_scale_m")
    if feature is None:
        index = 0
    else:
        index = int(np.searchsorted(model["feature_edges_m"], float(feature), side="right"))
        index = min(index, len(model["bins"]) - 1)
    return float(model["bins"][index]["margin_m"]), index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--video", type=Path, default=VIDEO_PATH)
    parser.add_argument("--gif", type=Path, default=GIF_PATH)
    args = parser.parse_args()
    video, gif = generate_demo(report_path=args.report, video_path=args.video, gif_path=args.gif)
    print(f"Wrote {video} and {gif}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
