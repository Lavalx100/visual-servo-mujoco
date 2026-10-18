"""Render a held-out replay showing fixed-age and motion-envelope decisions."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from visual_servo_mujoco.completion_speed_benchmark import MOTION_CONDITIONS
from visual_servo_mujoco.motion_envelope_benchmark import evaluate_motion_envelope_policy
from visual_servo_mujoco.run import IMAGE_HEIGHT, IMAGE_WIDTH, run_trials
from visual_servo_mujoco.timestamped_verification_benchmark import (
    CONTROL_MODES,
    DELIVERY_PROFILES,
    PostStopCameraReplay,
    evaluate_timestamp_guarded_policy,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "artifacts" / "motion-envelope-benchmark.json"
PANEL_HEADER = 105
PANEL_FOOTER = 104
VIDEO_SIZE = (IMAGE_WIDTH * 2, PANEL_HEADER + IMAGE_HEIGHT + PANEL_FOOTER)
DEFAULT_SPEED_BOUND_M_S = 0.126


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


def _select_example(report: dict, speed_bound_m_s: float) -> tuple[dict, dict]:
    by_identity: dict[tuple, dict] = {}
    for record in report["validation_trial_results"]:
        if record["policy"] == "two_post_settle_frames" and record["max_frame_age_s"] == 1.0:
            by_identity[(*_identity(record), "age")] = record
        elif (
            record["policy"] == "two_post_settle_motion_envelope"
            and record["max_target_speed_bound_m_s"] == speed_bound_m_s
        ):
            by_identity[(*_identity(record), "envelope")] = record

    pairs = []
    identities = {_identity(row) for row in report["validation_trial_results"]}
    for identity in identities:
        if identity[4] != "fully_delayed_rgb" or identity[5] != 2:
            continue
        age = by_identity.get((*identity, "age"))
        envelope = by_identity.get((*identity, "envelope"))
        if age is None or envelope is None:
            continue
        if age["false_confirmation_at_delivery"] and not envelope["visual_confirmed"]:
            priority = 0
        elif age["false_confirmation_at_delivery"] and not envelope["false_confirmation_at_delivery"]:
            priority = 1
        elif age["visual_confirmed"] and not envelope["visual_confirmed"]:
            priority = 2
        else:
            priority = 3
        delivery_priority = {"high_latency": 0, "moderate_latency": 1, "near_real_time": 2}[
            identity[-1]
        ]
        pairs.append(((priority, delivery_priority, identity), age, envelope))
    if not pairs:
        raise ValueError("report has no paired age-guard and motion-envelope validation outcomes")
    _, age, envelope = min(pairs, key=lambda row: row[0])
    return age, envelope


def _rerun_trial(age_record: dict, temp_dir: Path) -> tuple[dict, dict]:
    condition = next(item for item in MOTION_CONDITIONS if item.name == age_record["motion_condition"])
    mode = next(
        item
        for item in CONTROL_MODES
        if item[0]
        == (
            "delayed_target_current_joint_tip"
            if age_record["control_mode"] == "delayed_target_current_joint_tip"
            else "fully_delayed_rgb"
            if age_record["control_mode"] == "fully_delayed_rgb"
            else "fresh_rgb"
        )
        and item[1] == age_record["control_delay_observations"]
    )
    mode_name, delay, use_current_tip = mode
    episode = int(age_record["episode"])
    run = run_trials(
        episode + 1,
        int(age_record["seed"]),
        temp_dir / condition.name / mode_name / str(delay),
        controller="image_feedback",
        pixel_noise_std_px=2.0,
        target_motion_amplitude_m=condition.amplitude_m,
        target_motion_frequency_hz=condition.frequency_hz,
        target_motion_profile=condition.profile,
        target_motion_phase_rad=float(age_record["target_motion_phase_rad"]),
        camera_observation_delay_observations=delay,
        target_motion_prediction="none",
        use_current_end_effector_state=use_current_tip,
        feedback_allow_stale_target_confirmation=False,
        verify_terminal=False,
        save_media=False,
    )
    trial = next(row for row in run["results"] if int(row["episode"]) == episode)
    trial = {"seed": int(age_record["seed"]), **trial}
    profile = next(item for item in DELIVERY_PROFILES if item.name == age_record["delivery_profile"])
    replay = PostStopCameraReplay(save_images=True)
    try:
        stream = replay.replay(trial, delivery_profiles=(profile,), seed=int(age_record["seed"]))
    finally:
        replay.close()
    if stream["settled_time_s"] is None:
        raise ValueError("selected demo trial did not settle; no bounded policy can confirm it")
    return trial, stream


def _status(result: dict) -> str:
    return "CONFIRMED" if result["visual_confirmed"] else "ABSTAINED"


def _panel(
    scenario: dict,
    now_s: float,
    *,
    policy: str,
    margin_m: float,
    speed_bound_m_s: float,
) -> np.ndarray:
    panel = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    panel[:] = (30, 40, 51)
    arrived = [
        event
        for event in scenario["events"]
        if not event["transport_dropped"]
        and event["delivery_time_s"] is not None
        and float(event["delivery_time_s"]) <= now_s + 1e-9
    ]
    if arrived:
        event = max(arrived, key=lambda item: (item["delivery_time_s"], item["frame_index"]))
        image = scenario["images"].get(event["frame_index"])
        if image is not None:
            panel[:] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        estimate = event.get("estimated_distance_m")
        age = float(event["camera_age_at_delivery_s"])
        if estimate is None:
            estimate_line = "Image distance unavailable"
        elif policy == "age":
            estimate_line = f"Image distance {estimate * 100:.1f} cm | age {age:.2f} s | age-only cap 1.00 s"
        else:
            upper = estimate + margin_m + speed_bound_m_s * age
            estimate_line = (
                f"Image {estimate * 100:.1f} cm + margin {margin_m * 100:.1f} cm + "
                f"motion {speed_bound_m_s * age * 100:.1f} cm = {upper * 100:.1f} cm"
            )
        actual = event.get("physical_error_at_delivery_m")
        actual_line = (
            f"MuJoCo delivery distance {actual * 100:.1f} cm (evaluation only)"
            if actual is not None
            else "Delivery distance unavailable"
        )
        lines = [
            f"Frame {event['frame_index']:02d} captured at {event['capture_time_s']:.2f}s | delivered age {age:.2f}s",
            estimate_line,
            actual_line,
        ]
    else:
        event = None
        lines = ["Waiting for the first delivered frame", "", ""]

    available = [
        item
        for item in scenario["events"]
        if item["delivery_time_s"] is not None
        and float(item["delivery_time_s"]) <= now_s + 1e-9
    ]
    if policy == "age":
        result = evaluate_timestamp_guarded_policy(
            scenario["trial"],
            available,
            settled_time_s=scenario["settled_time_s"],
            max_frame_age_s=1.0,
        )
    else:
        result = evaluate_motion_envelope_policy(
            scenario["trial"],
            available,
            settled_time_s=scenario["settled_time_s"],
            calibrated_margin_m=margin_m,
            max_target_speed_m_s=speed_bound_m_s,
        )
    overlay_height = 94
    overlay = np.zeros((overlay_height, IMAGE_WIDTH, 3), dtype=np.uint8)
    overlay[:] = (24, 34, 44)
    status_color = (87, 214, 163) if result["visual_confirmed"] else (205, 176, 112)
    cv2.putText(
        overlay,
        f"{_status(result)} | {result['usable_frames']} usable frame(s) | since stop {now_s - scenario['stop_time']:.2f}s",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        status_color,
        1,
        cv2.LINE_AA,
    )
    for index, line in enumerate(lines):
        cv2.putText(
            overlay,
            line[:100],
            (12, 49 + 20 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            (226, 235, 242),
            1,
            cv2.LINE_AA,
        )
    panel[:overlay_height] = overlay
    return panel


def _caption(
    frame: np.ndarray,
    age_result: dict,
    envelope_result: dict,
    trial: dict,
    margin: float,
) -> None:
    y0 = PANEL_HEADER + IMAGE_HEIGHT
    frame[y0:] = (22, 31, 42)
    age_delivery = age_result["physical_error_at_decision_delivery_m"]
    envelope_delivery = envelope_result["physical_error_at_decision_delivery_m"]
    age_label = "confirmation delivery" if age_result["visual_confirmed"] else "latest frame"
    envelope_label = (
        "confirmation delivery" if envelope_result["visual_confirmed"] else "latest frame"
    )
    age_score = "no usable frame" if age_delivery is None else f"MuJoCo {age_label} {age_delivery * 100:.1f} cm"
    env_score = "no usable frame" if envelope_delivery is None else f"MuJoCo {envelope_label} {envelope_delivery * 100:.1f} cm"
    lines = [
        f"Age <=1.0 s: {_status(age_result)} | {age_score}",
        f"Envelope: {_status(envelope_result)} | {env_score} | training q99 margin {margin * 100:.2f} cm",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (18, y0 + 31 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (226, 235, 242),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        f"Physical reach at controller stop: {trial['reaching_error_m'] * 100:.1f} cm | score threshold 4.5 cm",
        (18, y0 + 91),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (87, 214, 163),
        1,
        cv2.LINE_AA,
    )


def generate_demo(
    *,
    report_path: Path = REPORT_PATH,
    output_path: Path,
    preview_path: Path,
    speed_bound_m_s: float = DEFAULT_SPEED_BOUND_M_S,
) -> tuple[Path, dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("benchmark") != "held_out_motion_envelope_visual_completion":
        raise ValueError(f"not a motion-envelope benchmark report: {report_path}")
    age_record, envelope_record = _select_example(report, speed_bound_m_s)
    with tempfile.TemporaryDirectory(prefix="visual-servo-envelope-demo-") as temp_dir:
        trial, stream = _rerun_trial(age_record, Path(temp_dir))
    delivery_name = age_record["delivery_profile"]
    events = stream["events_by_profile"][delivery_name]
    scenario = {
        "trial": trial,
        "events": events,
        "images": stream["images_by_frame"],
        "stop_time": stream["controller_stop_time_s"],
        "settled_time_s": stream["settled_time_s"],
    }
    margin = float(report["calibration"]["margin_m"])
    end_s = scenario["stop_time"] + 2.0
    times = np.arange(scenario["stop_time"], end_s + 1e-9, 0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, VIDEO_SIZE
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"could not open demo video writer for {output_path}")
    previews = []
    age_result = envelope_result = None
    try:
        for frame_index, now_s in enumerate(times):
            frame = np.zeros((VIDEO_SIZE[1], VIDEO_SIZE[0], 3), dtype=np.uint8)
            frame[:] = (22, 31, 42)
            cv2.putText(
                frame,
                "Same held-out camera stream | fixed-age decision versus motion envelope",
                (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.57,
                (245, 248, 250),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{age_record['motion_condition']} | {age_record['delivery_profile']} | {age_record['control_mode']} delay {age_record['control_delay_s']:.1f}s | held-out seed {age_record['seed']}",
                (18, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (205, 218, 228),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Two post-settle frames, max age 1.0 s",
                (18, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (92, 132, 214),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Two post-settle frames, speed cap {speed_bound_m_s:.3f} m/s",
                (IMAGE_WIDTH + 18, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (87, 214, 163),
                1,
                cv2.LINE_AA,
            )
            frame[PANEL_HEADER : PANEL_HEADER + IMAGE_HEIGHT, :IMAGE_WIDTH] = _panel(
                scenario,
                float(now_s),
                policy="age",
                margin_m=margin,
                speed_bound_m_s=speed_bound_m_s,
            )
            frame[PANEL_HEADER : PANEL_HEADER + IMAGE_HEIGHT, IMAGE_WIDTH:] = _panel(
                scenario,
                float(now_s),
                policy="envelope",
                margin_m=margin,
                speed_bound_m_s=speed_bound_m_s,
            )
            available = [
                event
                for event in events
                if event["delivery_time_s"] is not None
                and float(event["delivery_time_s"]) <= now_s + 1e-9
            ]
            age_result = evaluate_timestamp_guarded_policy(
                trial,
                available,
                settled_time_s=stream["settled_time_s"],
                max_frame_age_s=1.0,
            )
            envelope_result = evaluate_motion_envelope_policy(
                trial,
                available,
                settled_time_s=stream["settled_time_s"],
                calibrated_margin_m=margin,
                max_target_speed_m_s=speed_bound_m_s,
            )
            _caption(frame, age_result, envelope_result, trial, margin)
            writer.write(frame)
            if frame_index % 2 == 0:
                scaled = cv2.resize(frame, (640, VIDEO_SIZE[1] // 2), interpolation=cv2.INTER_AREA)
                previews.append(Image.fromarray(cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)))
    finally:
        writer.release()
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    previews[0].save(
        preview_path,
        save_all=True,
        append_images=previews[1:],
        duration=400,
        loop=0,
        optimize=True,
    )
    return output_path, {
        "age_guard_outcome": age_record,
        "motion_envelope_outcome": envelope_record,
        "calibration_margin_m": margin,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--output", type=Path, default=Path("assets/demos/motion_envelope_comparison.mp4"))
    parser.add_argument("--preview", type=Path, default=Path("assets/demos/previews/motion_envelope_comparison.gif"))
    parser.add_argument("--speed-bound-m-s", type=float, default=DEFAULT_SPEED_BOUND_M_S)
    args = parser.parse_args()
    try:
        path, selected = generate_demo(
            report_path=args.report,
            output_path=args.output,
            preview_path=args.preview,
            speed_bound_m_s=args.speed_bound_m_s,
        )
    except (OSError, ValueError, KeyError, StopIteration) as error:
        parser.error(str(error))
    age = selected["age_guard_outcome"]
    envelope = selected["motion_envelope_outcome"]
    print(
        f"Saved {path} and {args.preview}; selected {age['motion_condition']} seed {age['seed']} "
        f"{age['delivery_profile']}: age guard {age['status']}, envelope {envelope['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
