"""Generate clean and noisy demos of the opt-in terminal verifier."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from visual_servo_mujoco.run import run_trials


DEMO_CASES = ((0.0, "clean"), (20.0, "20px-noise"))


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    destination_dir = repository / "assets" / "demos"
    destination_dir.mkdir(parents=True, exist_ok=True)

    for noise_std_px, label in DEMO_CASES:
        with tempfile.TemporaryDirectory(prefix="visual-servo-terminal-demo-") as temp:
            output_dir = Path(temp)
            report = run_trials(
                episodes=1,
                seed=7,
                output_dir=output_dir,
                controller="image_feedback",
                pixel_noise_std_px=noise_std_px,
                verify_terminal=True,
                save_media=True,
            )
            trial = report["results"][0]
            source = output_dir / trial["terminal_verification_video"]
            destination = destination_dir / f"terminal_verification_{label}.mp4"
            shutil.copyfile(source, destination)
            evidence = trial["terminal_visual_verification"]
            estimate = evidence["estimated_distance_m"]
            estimate_text = "unavailable" if estimate is None else f"{estimate * 100:.2f} cm"
            print(
                f"{destination.relative_to(repository)}: physical reach "
                f"{'yes' if trial['success'] else 'no'}, visual estimate {estimate_text}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
