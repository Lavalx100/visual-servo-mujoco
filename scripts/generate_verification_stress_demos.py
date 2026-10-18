"""Generate a terminal-verifier demo for every robustness stress condition."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from visual_servo_mujoco.run import run_trials
from visual_servo_mujoco.verification_robustness_benchmark import (
    VERIFIER_STRESS_CONDITIONS,
)


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    destination_dir = repository / "assets" / "demos"
    destination_dir.mkdir(parents=True, exist_ok=True)

    for condition_name, stress_settings in VERIFIER_STRESS_CONDITIONS.items():
        if condition_name == "nominal":
            continue
        with tempfile.TemporaryDirectory(prefix="visual-servo-verifier-stress-demo-") as temp:
            output_dir = Path(temp)
            report = run_trials(
                episodes=1,
                seed=7,
                output_dir=output_dir,
                controller="image_feedback",
                pixel_noise_std_px=20.0,
                verify_terminal=True,
                save_media=True,
                **stress_settings,
            )
            trial = report["results"][0]
            source = output_dir / trial["terminal_verification_video"]
            destination = (
                destination_dir / f"terminal_verification_stress_{condition_name}.mp4"
            )
            shutil.copyfile(source, destination)
            evidence = trial["terminal_visual_verification"]
            distance = evidence["estimated_distance_m"]
            distance_text = "unavailable" if distance is None else f"{distance * 100:.2f} cm"
            print(
                f"{destination.relative_to(repository)}: "
                f"{evidence['condition_label']}, {distance_text}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
