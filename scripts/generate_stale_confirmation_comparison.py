"""Render strict and stale-target arrival rules on the same simulated trial."""

from __future__ import annotations

import tempfile
from pathlib import Path

from generate_filter_comparison import compose_paired_videos
from visual_servo_mujoco.run import run_trials


SEED = 7


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    destination = repository / "assets" / "demos" / "feedback_stale_confirmation_comparison.mp4"

    with tempfile.TemporaryDirectory(prefix="visual-servo-stale-stop-ab-") as temporary:
        temporary_root = Path(temporary)
        sources = []
        for label, allow_stale in (("Fresh vision required", False), ("Stale estimate allowed", True)):
            output_dir = temporary_root / label.lower().replace(" ", "-")
            report = run_trials(
                episodes=1,
                seed=SEED,
                output_dir=output_dir,
                controller="image_feedback",
                feedback_allow_stale_target_confirmation=allow_stale,
                save_media=True,
            )
            trial = report["results"][0]
            status = trial["controller_stop_reason"] or "unknown stop"
            text = f"{label} | {status} | {trial['reaching_error_m'] * 100:.1f} cm"
            color = (80, 220, 110) if trial["success"] else (90, 130, 255)
            sources.append((output_dir / "reaching_demo.mp4", text, color))

        compose_paired_videos(
            sources,
            destination,
            title=f"Same camera and target sequence | seed {SEED} | stop-rule comparison",
        )

    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("stale-confirmation comparison video was not created")
    print(
        f"Wrote {destination.relative_to(repository)} "
        f"({destination.stat().st_size / 1024:.0f} KiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
