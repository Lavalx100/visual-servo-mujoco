"""Render a paired no-delay versus delayed-camera moving-target demo."""

from __future__ import annotations

import tempfile
from pathlib import Path

from generate_filter_comparison import compose_paired_videos
from visual_servo_mujoco.dynamic_benchmark import (
    DEFAULT_MOTION_AMPLITUDE_M,
    DEFAULT_MOTION_FREQUENCY_HZ,
)
from visual_servo_mujoco.run import run_trials


SEED = 7
CAMERA_DELAYS = (0, 1)


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    destination = repository / "assets" / "demos" / "feedback_moving_target_latency.mp4"

    with tempfile.TemporaryDirectory(prefix="visual-servo-latency-ab-") as temporary:
        temporary_root = Path(temporary)
        sources = []
        for delay in CAMERA_DELAYS:
            output_dir = temporary_root / f"delay-{delay}"
            report = run_trials(
                episodes=1,
                seed=SEED,
                output_dir=output_dir,
                controller="image_feedback",
                target_motion_amplitude_m=DEFAULT_MOTION_AMPLITUDE_M,
                target_motion_frequency_hz=DEFAULT_MOTION_FREQUENCY_HZ,
                camera_observation_delay_observations=delay,
                save_media=True,
            )
            trial = report["results"][0]
            if delay == 0:
                label = "No camera delay"
            else:
                label = (
                    f"{delay} observation delay | "
                    f"{report['camera_observation_delay_simulated_s']:.1f} s"
                )
            status = "REACHED" if trial["success"] else "MISSED"
            label = f"{label} | {status} | {trial['reaching_error_m'] * 100:.1f} cm"
            color = (80, 220, 110) if trial["success"] else (90, 130, 255)
            sources.append((output_dir / "reaching_demo.mp4", label, color))

        compose_paired_videos(
            sources,
            destination,
            title=(
                f"Moving target ±{DEFAULT_MOTION_AMPLITUDE_M * 100:.0f} cm at "
                f"{DEFAULT_MOTION_FREQUENCY_HZ:.2f} Hz | seed {SEED}"
            ),
        )

    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("moving-target latency video was not created")
    print(
        f"Wrote {destination.relative_to(repository)} "
        f"({destination.stat().st_size / 1024:.0f} KiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
