"""Regenerate the small, deterministic video gallery in ``assets/demos``."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from visual_servo_mujoco.run import run_trials


DEMOS = (
    {
        "name": "open_loop_clean",
        "controller": "open_loop",
        "seed": 7,
        "episodes": 1,
    },
    {
        "name": "feedback_clean",
        "controller": "image_feedback",
        "seed": 7,
        "episodes": 1,
    },
    {
        "name": "feedback_noise_20px",
        "controller": "image_feedback",
        "seed": 0,
        "episodes": 1,
        "pixel_noise_std_px": 20.0,
    },
    {
        "name": "feedback_noise_and_fov_error",
        "controller": "image_feedback",
        "seed": 2,
        "episodes": 1,
        "pixel_noise_std_px": 8.0,
        "camera_fovy_error_deg": 5.0,
    },
    {
        "name": "feedback_high_noise_failure",
        "controller": "image_feedback",
        "seed": 3,
        "episodes": 6,
        "pixel_noise_std_px": 20.0,
        "video_episode_index": 5,
    },
)


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    gallery = repository / "assets" / "demos"
    gallery.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="visual-servo-gallery-") as temporary:
        temporary_root = Path(temporary)
        for demo in DEMOS:
            parameters = {
                "pixel_noise_std_px": 0.0,
                "camera_fovy_error_deg": 0.0,
                "video_episode_index": 0,
                **demo,
            }
            name = parameters.pop("name")
            output_dir = temporary_root / name
            report = run_trials(
                output_dir=output_dir,
                save_media=True,
                **parameters,
            )
            source_video = output_dir / "reaching_demo.mp4"
            if not source_video.is_file() or source_video.stat().st_size == 0:
                raise RuntimeError(f"video writer did not produce a clip for {name}")
            destination = gallery / f"{name}.mp4"
            shutil.copyfile(source_video, destination)
            if name == "open_loop_clean":
                shutil.copyfile(
                    output_dir / "camera_view.png",
                    repository / "assets" / "camera_view.png",
                )
                shutil.copyfile(destination, repository / "assets" / "reaching_demo.mp4")
            trial = report["results"][parameters["video_episode_index"]]
            print(
                f"{destination.relative_to(repository)}: "
                f"{'success' if trial['success'] else 'not reached'}, "
                f"{trial['reaching_error_m'] * 100.0:.1f} cm, "
                f"{destination.stat().st_size / 1024:.0f} KiB"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
