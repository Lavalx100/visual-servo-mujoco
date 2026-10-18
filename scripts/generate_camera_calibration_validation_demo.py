"""Create paired videos for the frozen camera-health threshold experiment."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from visual_servo_mujoco.camera_calibration_health import CalibrationScene

FPS = 24
SECONDS_PER_COMPARISON = 3
PANEL_WIDTH = 640
PANEL_HEIGHT = 480
HEADER_HEIGHT = 68
GAP_WIDTH = 32


def _recreate_record(scene: CalibrationScene, row: dict) -> np.ndarray:
    image, _ = scene.render(
        condition_name=f"demo_{row['label']}",
        fovy_error_deg=(row["render_camera_fovy_deg"] - scene.nominal_fovy),
        camera_position_error_m=tuple(row["renderer_camera_position_offset_m"]),
        rotation_error_deg=tuple(row["render_camera_rotation_offset_deg"]),
        brightness_scale=row["brightness_scale"],
        image_noise_std_intensity=row["image_noise_std_intensity"],
        blur_kernel_px=row["blur_kernel_px"],
        image_noise_seed=row["image_noise_seed"],
    )
    return image


def _panel(
    image: np.ndarray,
    row: dict,
    *,
    threshold: float,
    title: str,
) -> np.ndarray:
    panel = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for marker in (row["landmarks"] or {}).values():
        expected = tuple(round(value) for value in marker["expected_pixel_xy"])
        observed = tuple(round(value) for value in marker["observed_pixel_xy"])
        cv2.drawMarker(panel, expected, (255, 255, 255), cv2.MARKER_CROSS, 13, 2)
        cv2.circle(panel, observed, 5, (40, 235, 60), 2)
        cv2.line(panel, expected, observed, (20, 170, 255), 1, cv2.LINE_AA)

    score = row["reprojection_rms_px"]
    if score is None:
        status = "UNAVAILABLE"
        color = (40, 120, 255)
    elif score > threshold:
        status = "DRIFT FLAG"
        color = (40, 170, 255)
    else:
        status = "NO DRIFT FLAG"
        color = (60, 220, 60)
    header = np.zeros((HEADER_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (16, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    score_text = f"RMS {score:.2f} px" if score is not None else "RMS unavailable"
    cv2.putText(
        header,
        f"{status} | threshold {threshold:.2f} px | {score_text}",
        (16, 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        color,
        1,
        cv2.LINE_AA,
    )
    return np.vstack((header, panel))


def _pick(rows: list[dict], predicate, key=None) -> dict:
    matches = [row for row in rows if predicate(row)]
    if not matches:
        raise RuntimeError("validation report lacks a representative demo case")
    return sorted(matches, key=key)[len(matches) // 2] if key else matches[0]


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    report_path = repository / "artifacts" / "camera-calibration-health-validation.json"
    video_path = repository / "assets" / "demos" / "camera_calibration_validation.mp4"
    report = json.loads(report_path.read_text())
    rows = report["trial_results"]
    nominal_quantile_threshold = report["threshold_comparisons"][
        "nominal_only_p95"
    ]["threshold_rms_px"]
    supervised_threshold = report["threshold_comparisons"][
        "supervised_fpr_constrained"
    ]["threshold_rms_px"]
    fixed_threshold = report["threshold_comparisons"][
        "fixed_3px_exploratory"
    ]["threshold_rms_px"]

    nominal_false_alarm = _pick(
        [row for row in rows if row["split"] == "validation" and row["label"] == "nominal"],
        lambda row: (
            nominal_quantile_threshold < row["reprojection_rms_px"]
            <= supervised_threshold
        ),
        key=lambda row: row["reprojection_rms_px"],
    )
    subtle_below_three = _pick(
        [row for row in rows if row["label"] == "subtle_drift"],
        lambda row: supervised_threshold < row["reprojection_rms_px"] <= fixed_threshold,
        key=lambda row: row["reprojection_rms_px"],
    )
    severe = _pick(
        [row for row in rows if row["label"] == "severe_drift"],
        lambda row: row["reprojection_rms_px"] is not None,
        key=lambda row: row["reprojection_rms_px"],
    )
    dim = _pick(rows, lambda row: row["label"] == "nominal_dim_probe")

    slides = [
        (
            (nominal_false_alarm, nominal_quantile_threshold, "Nominal-only p95 rule"),
            (nominal_false_alarm, supervised_threshold, "FPR-constrained frozen rule"),
        ),
        (
            (subtle_below_three, supervised_threshold, "Frozen rule on subtle drift"),
            (subtle_below_three, fixed_threshold, "Initial fixed 3 px rule"),
        ),
        (
            (severe, supervised_threshold, "Frozen rule on severe drift"),
            (dim, supervised_threshold, "Dim image with no landmark decision"),
        ),
    ]

    video_path.parent.mkdir(parents=True, exist_ok=True)
    canvas_height = HEADER_HEIGHT + PANEL_HEIGHT
    separator = np.zeros((canvas_height, GAP_WIDTH, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (PANEL_WIDTH * 2 + GAP_WIDTH, canvas_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video at {video_path}")

    with CalibrationScene() as scene:
        first_frame = None
        for slide_index, pair in enumerate(slides):
            panels = []
            for row, threshold, title in pair:
                image = _recreate_record(scene, row)
                panels.append(_panel(image, row, threshold=threshold, title=title))
            frame = np.hstack((panels[0], separator, panels[1]))
            if slide_index == 0:
                first_frame = frame
            for _ in range(FPS * SECONDS_PER_COMPARISON):
                writer.write(frame)
    writer.release()

    preview_path = video_path.with_suffix(".png")
    if first_frame is None or not cv2.imwrite(str(preview_path), first_frame):
        raise RuntimeError(f"cannot create preview image at {preview_path}")
    print(
        json.dumps(
            {
                "video": str(video_path.relative_to(repository)),
                "preview": str(preview_path.relative_to(repository)),
                "frozen_threshold_px": supervised_threshold,
                "nominal_false_alarm_example_rms_px": nominal_false_alarm[
                    "reprojection_rms_px"
                ],
                "subtle_drift_example_rms_px": subtle_below_three[
                    "reprojection_rms_px"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
