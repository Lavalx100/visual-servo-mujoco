"""Create a paired video showing optional radial correction in the controller."""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np

from visual_servo_mujoco.run import IMAGE_HEIGHT, IMAGE_WIDTH, VIDEO_FPS, run_trials

SEED = 617
EPISODES = 6
EPISODE_INDEX = 5
ACTUAL_K1 = 0.10
PIXEL_NOISE_STD_PX = 2.0
HEADER_HEIGHT = 56


def compose_paired_videos(
    sources: list[tuple[Path, str, tuple[int, int, int]]],
    destination: Path,
    *,
    title: str,
) -> None:
    captures = [cv2.VideoCapture(str(path)) for path, _, _ in sources]
    if not all(capture.isOpened() for capture in captures):
        for capture in captures:
            capture.release()
        raise RuntimeError("could not open both source videos")

    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        VIDEO_FPS,
        (IMAGE_WIDTH * len(sources), IMAGE_HEIGHT + HEADER_HEIGHT),
    )
    if not writer.isOpened():
        for capture in captures:
            capture.release()
        raise RuntimeError("could not open the comparison video writer")

    last_frames: list[np.ndarray | None] = [None] * len(sources)
    active = [True] * len(sources)
    try:
        while any(active):
            for index, capture in enumerate(captures):
                if active[index]:
                    ok, frame = capture.read()
                    if ok:
                        last_frames[index] = frame
                    else:
                        active[index] = False
            if all(frame is None for frame in last_frames):
                break

            canvas = np.zeros(
                (IMAGE_HEIGHT + HEADER_HEIGHT, IMAGE_WIDTH * len(sources), 3),
                dtype=np.uint8,
            )
            cv2.putText(
                canvas,
                title,
                (18, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            for index, (_, label, color) in enumerate(sources):
                x = index * IMAGE_WIDTH
                cv2.putText(
                    canvas,
                    label,
                    (18 + x, 47),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                frame = last_frames[index]
                if frame is not None:
                    canvas[HEADER_HEIGHT:, x : x + IMAGE_WIDTH] = frame
            cv2.line(
                canvas,
                (IMAGE_WIDTH, 0),
                (IMAGE_WIDTH, IMAGE_HEIGHT + HEADER_HEIGHT),
                (90, 90, 90),
                1,
            )
            writer.write(canvas)
    finally:
        for capture in captures:
            capture.release()
        writer.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixel-noise-std-px", type=float, default=PIXEL_NOISE_STD_PX)
    parser.add_argument("--actual-k1", type=float, default=ACTUAL_K1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/demos/radial_distortion_correction.mp4"),
    )
    args = parser.parse_args()
    if not math.isfinite(args.pixel_noise_std_px) or args.pixel_noise_std_px < 0:
        parser.error("--pixel-noise-std-px must be finite and non-negative")
    if not math.isfinite(args.actual_k1):
        parser.error("--actual-k1 must be finite")

    repository = Path(__file__).resolve().parents[1]
    destination = args.output
    if not destination.is_absolute():
        destination = repository / destination
    modes = (("No pixel correction", 0.0), ("Matched k1 correction", args.actual_k1))

    with tempfile.TemporaryDirectory(prefix="visual-servo-radial-correction-") as root:
        temporary_root = Path(root)
        sources = []
        for label, estimated_k1 in modes:
            output_dir = temporary_root / label.lower().replace(" ", "-")
            report = run_trials(
                episodes=EPISODES,
                seed=SEED,
                output_dir=output_dir,
                controller="image_feedback",
                pixel_noise_std_px=args.pixel_noise_std_px,
                camera_radial_distortion_k1=args.actual_k1,
                controller_radial_distortion_k1=estimated_k1,
                video_episode_index=EPISODE_INDEX,
                camera_observation_delay_observations=0,
                target_motion_amplitude_m=0.0,
                save_media=True,
            )
            trial = report["results"][EPISODE_INDEX]
            outcome = "REACHED" if trial["success"] else "MISS"
            label_text = (
                f"{label} | {outcome} | "
                f"{trial['reaching_error_m'] * 100:.1f} cm physical error"
            )
            color = (80, 220, 110) if trial["success"] else (90, 130, 255)
            sources.append((output_dir / "reaching_demo.mp4", label_text, color))

        compose_paired_videos(
            sources,
            destination,
            title=(
                f"One episode | k1 {args.actual_k1:+.2f} | "
                f"noise {args.pixel_noise_std_px:g} px | "
                f"seed {SEED}, ep {EPISODE_INDEX}"
            ),
        )

    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("comparison video was not created")
    print(
        f"Wrote {destination.relative_to(repository)} "
        f"({destination.stat().st_size / 1024:.0f} KiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
