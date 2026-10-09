"""Render baseline and reversal-aware tracking for two moving-target paths."""

from __future__ import annotations

import tempfile
from pathlib import Path

from generate_filter_comparison import compose_paired_videos
from visual_servo_mujoco.run import run_trials


SEED = 7
CAMERA_DELAY_OBSERVATIONS = 1
PIXEL_NOISE_STD_PX = 2.0
PREDICTION_MODES = ("none", "constant_velocity", "reversal_aware")
TARGET_PROFILES = ("sinusoidal", "piecewise_linear")


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="visual-servo-reversal-demo-") as temporary:
        temporary_root = Path(temporary)
        for profile in TARGET_PROFILES:
            sources = []
            for mode in PREDICTION_MODES:
                output_dir = temporary_root / profile / mode
                report = run_trials(
                    episodes=1,
                    seed=SEED,
                    output_dir=output_dir,
                    controller="image_feedback",
                    pixel_noise_std_px=PIXEL_NOISE_STD_PX,
                    target_motion_amplitude_m=0.04,
                    target_motion_frequency_hz=0.25,
                    target_motion_profile=profile,
                    camera_observation_delay_observations=CAMERA_DELAY_OBSERVATIONS,
                    target_motion_prediction=mode,
                    use_current_end_effector_state=True,
                    save_media=True,
                )
                trial = report["results"][0]
                label = {
                    "none": "Measured target",
                    "constant_velocity": "Constant velocity",
                    "reversal_aware": "Reversal aware",
                }[mode]
                outcome = "REACHED" if trial["success"] else "MISSED"
                label = f"{label} | {outcome} | {trial['reaching_error_m'] * 100:.1f} cm"
                color = (80, 220, 110) if trial["success"] else (90, 130, 255)
                sources.append((output_dir / "reaching_demo.mp4", label, color))

            destination = (
                repository
                / "assets"
                / "demos"
                / f"feedback_prediction_{profile}_reversal.mp4"
            )
            compose_paired_videos(
                sources,
                destination,
                title=(
                    f"{profile.replace('_', ' ')} target | 0.4 s delay | "
                    f"{PIXEL_NOISE_STD_PX:.0f} px noise | seed {SEED}"
                ),
            )
            if not destination.is_file() or destination.stat().st_size == 0:
                raise RuntimeError(f"comparison video was not created: {destination}")
            print(
                f"Wrote {destination.relative_to(repository)} "
                f"({destination.stat().st_size / 1024:.0f} KiB)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
