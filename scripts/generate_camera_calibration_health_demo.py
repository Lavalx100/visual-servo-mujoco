"""Generate the fixed-landmark calibration-health report and comparison video."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from visual_servo_mujoco.camera_calibration_health import (
    render_calibration_condition,
    run_camera_calibration_health_benchmark,
)

VIDEO_FPS = 24
SECONDS_PER_CONDITION = 4
PANEL_WIDTH = 640
PANEL_HEIGHT = 480
HEADER_HEIGHT = 64


def _annotate_panel(rgb_image: np.ndarray, row: dict) -> np.ndarray:
    panel = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    landmarks = row["landmarks"] or {}
    for item in landmarks.values():
        expected = tuple(round(value) for value in item["expected_pixel_xy"])
        observed = tuple(round(value) for value in item["observed_pixel_xy"])
        cv2.drawMarker(panel, expected, (255, 255, 255), cv2.MARKER_CROSS, 13, 2)
        cv2.circle(panel, observed, 5, (40, 235, 60), 2)
        cv2.line(panel, expected, observed, (20, 170, 255), 1, cv2.LINE_AA)

    header = np.zeros((HEADER_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    status = row["diagnostic_status"].replace("_", " ").upper()
    color = (60, 220, 60) if status == "HEALTHY" else (40, 120, 255)
    cv2.putText(
        header,
        row["condition"].replace("_", " "),
        (18, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    rms = row["reprojection_rms_px"]
    rms_text = f"RMS {rms:.2f} px" if rms is not None else "RMS unavailable"
    cv2.putText(
        header,
        f"{status}  |  {rms_text}  |  {row['matched_landmark_count']}/4 landmarks",
        (18, 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )
    return np.vstack((header, panel))


def _make_comparison_video(destination: Path) -> list[dict]:
    conditions = ("nominal", "render_combined_fovy_x_yaw_shift")
    frames = []
    rows = []
    for condition in conditions:
        rgb_image, row = render_calibration_condition(condition)
        frames.append(_annotate_panel(rgb_image, row))
        rows.append(row)
    canvas_height = HEADER_HEIGHT + PANEL_HEIGHT
    title_bar = np.zeros((canvas_height, 32, 3), dtype=np.uint8)
    frame = np.hstack((frames[0], title_bar, frames[1]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    preview_path = destination.with_suffix(".png")
    if not cv2.imwrite(str(preview_path), frame):
        raise RuntimeError(f"cannot create preview image at {preview_path}")
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        VIDEO_FPS,
        (frame.shape[1], frame.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video at {destination}")
    for _ in range(VIDEO_FPS * SECONDS_PER_CONDITION):
        writer.write(frame)
    writer.release()
    return rows


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    report_path = repository / "artifacts" / "camera-calibration-health.json"
    video_path = repository / "assets" / "demos" / "camera_calibration_health.mp4"
    report = run_camera_calibration_health_benchmark(output_path=report_path)
    demo_rows = _make_comparison_video(video_path)
    print(
        json.dumps(
            {
                "report": str(report_path.relative_to(repository)),
                "video": str(video_path.relative_to(repository)),
                "preview": str(video_path.with_suffix(".png").relative_to(repository)),
                "conditions": [
                    {
                        "condition": row["condition"],
                        "rms_px": row["reprojection_rms_px"],
                        "status": row["diagnostic_status"],
                    }
                    for row in report["results"]
                ],
                "demo_conditions": [row["condition"] for row in demo_rows],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
