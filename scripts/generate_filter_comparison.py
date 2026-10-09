"""Render a paired high-noise comparison of the 3- and 5-sample filters."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from visual_servo_mujoco.run import IMAGE_HEIGHT, IMAGE_WIDTH, VIDEO_FPS, run_trials


SEED = 4
EPISODES = 13
EPISODE_INDEX = 12
PIXEL_NOISE_STD_PX = 20.0
FILTER_WINDOWS = (3, 5)
HEADER_HEIGHT = 56


def _compose_videos(
    sources: list[tuple[Path, str, tuple[int, int, int]]], destination: Path
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
                "Same target and noise sequence | seed 4, episode 12 | 20 px noise",
                (18, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
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
                    0.62,
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
    repository = Path(__file__).resolve().parents[1]
    destination = repository / "assets" / "demos" / "feedback_filter_window_comparison.mp4"

    with tempfile.TemporaryDirectory(prefix="visual-servo-filter-ab-") as temporary:
        temporary_root = Path(temporary)
        sources = []
        for window in FILTER_WINDOWS:
            output_dir = temporary_root / f"window-{window}"
            report = run_trials(
                episodes=EPISODES,
                seed=SEED,
                output_dir=output_dir,
                pixel_noise_std_px=PIXEL_NOISE_STD_PX,
                controller="image_feedback",
                video_episode_index=EPISODE_INDEX,
                feedback_target_filter_window=window,
                save_media=True,
            )
            trial = report["results"][EPISODE_INDEX]
            status = "SUCCESS" if trial["success"] else "NOT REACHED"
            label = (
                f"{window}-sample median | {status} | "
                f"{trial['reaching_error_m'] * 100.0:.1f} cm error"
            )
            color = (80, 220, 110) if trial["success"] else (90, 130, 255)
            sources.append((output_dir / "reaching_demo.mp4", label, color))

        _compose_videos(sources, destination)

    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("comparison video was not created")
    print(
        f"Wrote {destination.relative_to(repository)} "
        f"({destination.stat().st_size / 1024:.0f} KiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
