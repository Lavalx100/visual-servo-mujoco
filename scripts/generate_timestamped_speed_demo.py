"""Render a paired low-speed versus high-speed post-stop camera demo."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from visual_servo_mujoco.run import IMAGE_HEIGHT, IMAGE_WIDTH, run_trials
from visual_servo_mujoco.timestamped_verification_benchmark import (
    CONTROL_MODES,
    DELIVERY_PROFILES,
    PostStopCameraReplay,
    _evaluate_trial_policies,
)
from visual_servo_mujoco.completion_speed_benchmark import MOTION_CONDITIONS

SEED = 950
DELIVERY = next(profile for profile in DELIVERY_PROFILES if profile.name == "high_latency")
MODE = next(
    mode for mode in CONTROL_MODES if mode[0] == "fully_delayed_rgb" and mode[1] == 2
)
PANEL_HEIGHT = 480
HEADER_HEIGHT = 154
FOOTER_HEIGHT = 72
VIDEO_SIZE = (IMAGE_WIDTH * 2, HEADER_HEIGHT + PANEL_HEIGHT + FOOTER_HEIGHT)


def _render_scenario(condition_name: str, temp_dir: Path) -> dict:
    condition = next(item for item in MOTION_CONDITIONS if item.name == condition_name)
    mode, delay, use_current_tip = MODE
    report = run_trials(
        1,
        SEED,
        temp_dir / condition.name,
        controller="image_feedback",
        pixel_noise_std_px=2.0,
        target_motion_amplitude_m=condition.amplitude_m,
        target_motion_frequency_hz=condition.frequency_hz,
        target_motion_profile=condition.profile,
        target_motion_phase_rad=0.0,
        camera_observation_delay_observations=delay,
        target_motion_prediction="none",
        use_current_end_effector_state=use_current_tip,
        feedback_allow_stale_target_confirmation=False,
        verify_terminal=False,
        save_media=False,
    )
    trial = {"seed": SEED, **report["results"][0]}
    replay = PostStopCameraReplay(save_images=True)
    try:
        stream = replay.replay(
            trial,
            delivery_profiles=(DELIVERY,),
            seed=SEED,
        )
    finally:
        replay.close()
    events = stream["events_by_profile"][DELIVERY.name]
    outcomes = _evaluate_trial_policies(trial, stream, (DELIVERY,))
    outcomes = [item for item in outcomes if item["delivery_profile"] == DELIVERY.name]
    return {
        "condition": condition,
        "trial": trial,
        "events": events,
        "images": stream["images_by_frame"],
        "stop_time": stream["controller_stop_time_s"],
        "outcomes": outcomes,
    }


def _latest_delivered(scenario: dict, now_s: float) -> dict | None:
    delivered = [
        event
        for event in scenario["events"]
        if not event["transport_dropped"]
        and event["delivery_time_s"] is not None
        and float(event["delivery_time_s"]) <= now_s + 1e-9
    ]
    if not delivered:
        return None
    return max(delivered, key=lambda event: (float(event["delivery_time_s"]), event["frame_index"]))


def _panel_image(scenario: dict, now_s: float) -> np.ndarray:
    panel = np.zeros((PANEL_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    panel[:] = (29, 40, 51)
    event = _latest_delivered(scenario, now_s)
    if event is None:
        cv2.putText(panel, "Waiting for first frame", (178, 244), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 238, 244), 2, cv2.LINE_AA)
        return panel
    rgb = scenario["images"].get(event["frame_index"])
    if rgb is not None:
        panel[:] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    age = float(event["camera_age_at_delivery_s"])
    actual = event["physical_error_at_delivery_m"]
    estimated = event["estimated_distance_m"]
    overlay = np.zeros((86, IMAGE_WIDTH, 3), dtype=np.uint8)
    overlay[:] = (25, 35, 45)
    lines = [
        f"Latest delivered frame {event['frame_index']:02d} | captured t={event['capture_time_s']:.2f}s",
        f"age at delivery {age:.2f}s | image estimate {estimated * 100:.1f} cm" if estimated is not None else f"age at delivery {age:.2f}s | visual estimate unavailable",
        f"MuJoCo distance at that delivery {actual * 100:.1f} cm (evaluation only)" if actual is not None else "MuJoCo delivery distance unavailable",
    ]
    for index, line in enumerate(lines):
        cv2.putText(overlay, line, (12, 20 + index * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (226, 235, 242), 1, cv2.LINE_AA)
    panel[:86] = overlay
    return panel


def _draw_header(frame: np.ndarray, scenarios: list[dict], now_s: float) -> None:
    frame[:] = (22, 31, 42)
    cv2.putText(frame, "Does camera transport age matter more as the target moves faster?", (20, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (245, 248, 250), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Post-stop simulation clock t={now_s:.2f}s | high-latency camera: 0.80 +/- 0.10s, 10% frame loss", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (205, 218, 228), 1, cv2.LINE_AA)
    for index, scenario in enumerate(scenarios):
        x = index * IMAGE_WIDTH + 18
        condition = scenario["condition"]
        label = "Stationary target" if condition.amplitude_m == 0 else f"{condition.name.replace('_', ' ')} | peak {condition.peak_speed_m_s:.3f} m/s"
        cv2.putText(frame, label, (x, 101), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (87, 214, 163), 1, cv2.LINE_AA)
        trial = scenario["trial"]
        cv2.putText(frame, f"seed {SEED} | fully delayed RGB {trial['camera_observation_delay_simulated_s']:.1f}s | +{now_s - scenario['stop_time']:.2f}s since stop | reach {trial['reaching_error_m'] * 100:.1f} cm", (x, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (205, 218, 228), 1, cv2.LINE_AA)


def _draw_footer(frame: np.ndarray, scenarios: list[dict]) -> None:
    y0 = HEADER_HEIGHT + PANEL_HEIGHT
    frame[y0:] = (22, 31, 42)
    for index, scenario in enumerate(scenarios):
        x = index * IMAGE_WIDTH + 18
        by_policy = {
            item["max_frame_age_s"]: item
            for item in scenario["outcomes"]
            if item["policy"] == "two_post_settle_frames"
        }
        half = by_policy[0.5]
        one = by_policy[1.0]
        cv2.putText(frame, f"Two post-settle frames: age <= 0.5s: {half['status'].replace('_', ' ')} | age <= 1.0s: {one['status'].replace('_', ' ')}", (x, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (226, 235, 242), 1, cv2.LINE_AA)
        cv2.putText(frame, "Physical success at controller stop is reported above; this panel reports camera completion separately.", (x, y0 + 51), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (205, 218, 228), 1, cv2.LINE_AA)


def generate_demo(*, output_path: Path, preview_path: Path, seed: int = SEED) -> Path:
    if seed != SEED:
        raise ValueError("the paired demo uses fixed seed 950 to match the benchmark seed block")
    names = ("stationary", "sinusoidal_0p50_hz")
    with tempfile.TemporaryDirectory(prefix="visual-servo-speed-demo-") as temp_dir:
        scenarios = [_render_scenario(name, Path(temp_dir)) for name in names]
    start = max(scenario["stop_time"] for scenario in scenarios)
    end = start + 3.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, VIDEO_SIZE
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"could not open demo video writer for {output_path}")
    gif_frames = []
    try:
        for frame_index, now_s in enumerate(np.arange(start, end + 1e-9, 0.2)):
            frame = np.zeros((VIDEO_SIZE[1], VIDEO_SIZE[0], 3), dtype=np.uint8)
            _draw_header(frame, scenarios, float(now_s))
            for index, scenario in enumerate(scenarios):
                x0 = index * IMAGE_WIDTH
                frame[HEADER_HEIGHT : HEADER_HEIGHT + PANEL_HEIGHT, x0 : x0 + IMAGE_WIDTH] = _panel_image(scenario, float(now_s))
            _draw_footer(frame, scenarios)
            writer.write(frame)
            if frame_index % 2 == 0:
                small = cv2.resize(frame, (640, 345), interpolation=cv2.INTER_AREA)
                gif_frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
    finally:
        writer.release()
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    gif_frames[0].save(
        preview_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=400,
        loop=0,
        optimize=True,
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("assets/demos/timestamped_completion_speed_comparison.mp4"))
    parser.add_argument("--preview", type=Path, default=Path("assets/demos/previews/timestamped_completion_speed_comparison.gif"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    try:
        path = generate_demo(output_path=args.output, preview_path=args.preview, seed=args.seed)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"Saved {path} and {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
