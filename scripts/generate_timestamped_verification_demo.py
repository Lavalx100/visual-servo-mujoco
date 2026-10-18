"""Render a paired timestamp-aware post-stop verification episode."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np

from visual_servo_mujoco.run import IMAGE_HEIGHT, IMAGE_WIDTH, run_trials
from visual_servo_mujoco.timestamped_verification_benchmark import (
    DELIVERY_PROFILES,
    AGE_BUDGETS_S,
    MAX_CONFIRMATION_WAIT_S,
    PostStopCameraReplay,
    evaluate_arrival_only_policy,
    evaluate_timestamp_guarded_policy,
)


def _status_lines(
    events: list[dict],
    trial: dict,
    settled_time_s: float | None,
) -> list[str]:
    naive = evaluate_arrival_only_policy(trial, events)
    guarded = {
        age: evaluate_timestamp_guarded_policy(
            trial,
            events,
            settled_time_s=settled_time_s,
            max_frame_age_s=age,
        )
        for age in AGE_BUDGETS_S
    }

    def format_result(result: dict) -> str:
        status = result["status"].replace("_", " ")
        if result["false_confirmation_at_delivery"]:
            return "confirmed; out at delivery"
        if result["visual_confirmed"]:
            return f"confirmed {result['decision_latency_after_stop_s']:.2f} s after stop"
        return status

    return [
        (
            f"stop physical: {'within' if trial['success'] else 'outside'} "
            f"4.5 cm; error {trial['reaching_error_m'] * 100:.1f} cm"
        ),
        f"arrival-only final: {format_result(naive)}",
        f"two-frame age <= 0.5 s final: {format_result(guarded[0.5])}",
        f"two-frame age <= 1.0 s final: {format_result(guarded[1.0])}",
    ]


def _annotate(rgb: np.ndarray, *, title: str, lines: list[str]) -> np.ndarray:
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    band_h = 184
    band = np.zeros((band_h, image.shape[1], 3), dtype=np.uint8)
    band[:] = (26, 36, 46)
    cv2.putText(
        band,
        title,
        (14, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 248, 250),
        1,
        cv2.LINE_AA,
    )
    for index, line in enumerate(lines):
        cv2.putText(
            band,
            line,
            (14, 56 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (220, 231, 238),
            1,
            cv2.LINE_AA,
        )
    return np.vstack((band, image))


def generate_demo(*, output_path: Path, seed: int = 902) -> Path:
    episode = 0
    with tempfile.TemporaryDirectory(prefix="visual-servo-timestamped-demo-") as temp_dir:
        report = run_trials(
            episode + 1,
            seed,
            Path(temp_dir) / "reaching",
            controller="image_feedback",
            pixel_noise_std_px=2.0,
            target_motion_amplitude_m=0.04,
            target_motion_frequency_hz=0.25,
            target_motion_profile="piecewise_linear",
            target_motion_phase_rad=0.0,
            camera_observation_delay_observations=2,
            target_motion_prediction="none",
            use_current_end_effector_state=False,
            feedback_allow_stale_target_confirmation=False,
            save_media=False,
        )
    trial = {"seed": seed, **report["results"][0]}
    replay = PostStopCameraReplay(save_images=True)
    try:
        stream = replay.replay(trial, seed=seed)
    finally:
        replay.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (IMAGE_WIDTH, IMAGE_HEIGHT + 184),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"could not open demo video writer for {output_path}")
    try:
        for profile in DELIVERY_PROFILES:
            events = stream["events_by_profile"][profile.name]
            lines = _status_lines(events, trial, stream["settled_time_s"])
            section = np.zeros((IMAGE_HEIGHT + 184, IMAGE_WIDTH, 3), dtype=np.uint8)
            section[:] = (32, 43, 54)
            cv2.putText(
                section,
                f"Camera delivery profile: {profile.name.replace('_', ' ')}",
                (22, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (245, 248, 250),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                section,
                "controller: fully delayed RGB | control delay: 0.8 s",
                (22, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 231, 238),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                section,
                f"delivery latency {profile.latency_s:.2f} s +/- {profile.jitter_s:.2f} s; "
                f"drop {profile.frame_drop_probability:.0%}",
                (22, 315),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 231, 238),
                1,
                cv2.LINE_AA,
            )
            for _ in range(10):
                writer.write(section)
            ordered_events = sorted(
                events,
                key=lambda event: (
                    event["scheduled_arrival_time_s"], event["frame_index"]
                ),
            )
            for event in ordered_events:
                if event["scheduled_arrival_time_s"] > (
                    float(trial["controller_stop_time_simulated_s"])
                    + MAX_CONFIRMATION_WAIT_S
                ):
                    continue
                rgb = stream["images_by_frame"].get(event["frame_index"])
                if rgb is None:
                    continue
                if event["transport_dropped"]:
                    arrival_label = "FRAME DROPPED in transport"
                else:
                    arrival_label = (
                        f"delivery t={event['delivery_time_s']:.2f}s  "
                        f"frame age={event['camera_age_at_delivery_s']:.2f}s"
                    )
                settle_label = (
                    "captured after settle"
                    if event["capture_after_settle"]
                    else "captured before arm settled"
                )
                distance = event["estimated_distance_m"]
                distance_label = (
                    f"estimated distance {distance * 100:.1f} cm"
                    if distance is not None
                    else "visual estimate unavailable"
                )
                title = (
                    f"{profile.name} | frame {event['frame_index']:02d} | "
                    f"capture t={event['capture_time_s']:.2f}s"
                )
                frame = _annotate(
                    rgb,
                    title=title,
                    lines=[
                        arrival_label,
                        settle_label + " | " + distance_label,
                        *lines,
                    ],
                )
                for _ in range(5):
                    writer.write(frame)
    finally:
        writer.release()
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/demos/timestamped_post_stop_verification.mp4"),
    )
    parser.add_argument("--seed", type=int, default=902)
    args = parser.parse_args()
    try:
        path = generate_demo(output_path=args.output, seed=args.seed)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
