"""Record a paired verifier false-positive and quality-abstention example."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from visual_servo_mujoco.run import run_trials


SEED = 101
EPISODE_INDEX = 16
EPISODE_COUNT = EPISODE_INDEX + 1
FOVY_ERROR_DEG = -5.0
QUALITY_GATES = {
    "terminal_verification_quality_false_positive": 0.0,
    "terminal_verification_quality_abstention": 0.5,
}


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    destination_dir = repository / "assets" / "demos"
    destination_dir.mkdir(parents=True, exist_ok=True)
    paired_results = []

    for file_stem, quality_gate in QUALITY_GATES.items():
        with tempfile.TemporaryDirectory(prefix="visual-servo-quality-demo-") as temp:
            output_dir = Path(temp)
            report = run_trials(
                episodes=EPISODE_COUNT,
                seed=SEED,
                output_dir=output_dir,
                controller="image_feedback",
                pixel_noise_std_px=20.0,
                verify_terminal=True,
                terminal_verification_fovy_error_deg=FOVY_ERROR_DEG,
                terminal_verification_min_quality_score=quality_gate,
                video_episode_index=EPISODE_INDEX,
                save_media=True,
            )
            trial = report["results"][EPISODE_INDEX]
            evidence = trial["terminal_visual_verification"]
            source = output_dir / trial["terminal_verification_video"]
            destination = destination_dir / f"{file_stem}.mp4"
            shutil.copyfile(source, destination)
            paired_results.append(trial)
            print(
                f"{destination.relative_to(repository)}: "
                f"quality={evidence['combined_detection_quality_score']}, "
                f"gate={quality_gate:.2f}, "
                f"estimate={evidence['estimated_distance_m']}, "
                f"physical_success={trial['success']}"
            )

    baseline, selective = paired_results
    for key in ("success", "reaching_error_m", "controller_stop_reason", "controller_iterations"):
        if baseline[key] != selective[key]:
            raise RuntimeError(f"quality gate changed reaching outcome field {key}")
    if baseline["terminal_visual_verification"]["visual_success_estimate"] is not True:
        raise RuntimeError("the paired baseline clip no longer shows the selected false positive")
    if selective["terminal_visual_verification"]["visual_success_estimate"] is not None:
        raise RuntimeError("the strict quality gate should abstain in the paired clip")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
