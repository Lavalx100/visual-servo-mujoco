"""Render delayed-target demos with baseline, velocity, and acceleration fits."""

from __future__ import annotations

import tempfile
from pathlib import Path

from generate_filter_comparison import compose_paired_videos
from visual_servo_mujoco.prediction_benchmark import (
    DEFAULT_MOTION_AMPLITUDE_M,
    DEFAULT_MOTION_FREQUENCY_HZ,
)
from visual_servo_mujoco.run import run_trials


SEED = 7
CAMERA_DELAY_OBSERVATIONS = 1


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    destination = repository / "assets" / "demos" / "feedback_target_prediction.mp4"

    with tempfile.TemporaryDirectory(prefix="visual-servo-prediction-ab-") as temporary:
        temporary_root = Path(temporary)
        sources = []
        for mode in ("none", "constant_velocity", "constant_acceleration"):
            output_dir = temporary_root / mode
            report = run_trials(
                episodes=1,
                seed=SEED,
                output_dir=output_dir,
                controller="image_feedback",
                target_motion_amplitude_m=DEFAULT_MOTION_AMPLITUDE_M,
                target_motion_frequency_hz=DEFAULT_MOTION_FREQUENCY_HZ,
                camera_observation_delay_observations=CAMERA_DELAY_OBSERVATIONS,
                target_motion_prediction=mode,
                use_current_end_effector_state=True,
                save_media=True,
            )
            trial = report["results"][0]
            label = {
                "none": "Median measurement",
                "constant_velocity": "Constant-velocity fit",
                "constant_acceleration": "Constant-acceleration fit",
            }[mode]
            outcome = "REACHED" if trial["success"] else "MISSED"
            label = f"{label} | {outcome} | {trial['reaching_error_m'] * 100:.1f} cm"
            color = (80, 220, 110) if trial["success"] else (90, 130, 255)
            sources.append((output_dir / "reaching_demo.mp4", label, color))

        compose_paired_videos(
            sources,
            destination,
            title=(
                f"Target prediction with {CAMERA_DELAY_OBSERVATIONS} update of delay | "
                f"seed {SEED}"
            ),
        )

    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("target-prediction video was not created")
    print(
        f"Wrote {destination.relative_to(repository)} "
        f"({destination.stat().st_size / 1024:.0f} KiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
