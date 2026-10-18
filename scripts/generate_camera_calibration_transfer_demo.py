"""Visualize the frozen threshold's camera-pose and distortion transfer limits."""

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
HEADER_HEIGHT = 74
GAP_WIDTH = 32


def _recreate_record(scene: CalibrationScene, row: dict) -> np.ndarray:
    replay_image, replay_row = scene.render(
        condition_name=f"demo_{row['label']}",
        fovy_error_deg=row["render_camera_fovy_deg"] - scene.nominal_fovy,
        camera_position_error_m=tuple(row["renderer_camera_position_offset_m"]),
        rotation_error_deg=tuple(row["render_camera_rotation_offset_deg"]),
        brightness_scale=row["brightness_scale"],
        image_noise_std_intensity=row["image_noise_std_intensity"],
        blur_kernel_px=row["blur_kernel_px"],
        image_noise_seed=row["image_noise_seed"],
        radial_distortion_k1=row["radial_distortion_k1"],
        radial_distortion_k2=row["radial_distortion_k2"],
    )
    # Recompute the residual using the replayed image so displayed dots, RMS,
    # and the frozen decision all refer to the exact frame being shown.
    if replay_row["diagnostic_status"] != row["diagnostic_status"]:
        raise RuntimeError("replayed image changed the diagnostic availability status")
    original_candidate = row["candidate_reprojection_rms_px"]
    replay_candidate = replay_row["candidate_reprojection_rms_px"]
    if original_candidate is not None and not np.isclose(
        original_candidate, replay_candidate, atol=1e-9
    ):
        raise RuntimeError("replayed image changed the candidate reprojection score")
    return replay_image


def _panel(image: np.ndarray, row: dict, *, title: str, threshold: float) -> np.ndarray:
    panel = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for marker in (row["landmarks"] or {}).values():
        expected = tuple(round(value) for value in marker["expected_pixel_xy"])
        observed = tuple(round(value) for value in marker["observed_pixel_xy"])
        cv2.drawMarker(panel, expected, (255, 255, 255), cv2.MARKER_CROSS, 13, 2)
        cv2.circle(panel, observed, 5, (40, 235, 60), 2)
        cv2.line(panel, expected, observed, (20, 170, 255), 1, cv2.LINE_AA)

    status = row["frozen_diagnostic_status"].replace("_", " ").upper()
    color = (60, 220, 60) if status == "HEALTHY" else (40, 120, 255)
    header = np.zeros((HEADER_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    cv2.putText(
        header, title, (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
        (245, 245, 245), 1, cv2.LINE_AA,
    )
    rms = row["reprojection_rms_px"]
    if rms is not None:
        score_text = f"RMS {rms:.2f} px"
    elif row["candidate_reprojection_rms_px"] is not None:
        score_text = f"candidate RMS {row['candidate_reprojection_rms_px']:.2f} px"
    else:
        score_text = "RMS unavailable"
    k1 = row["radial_distortion_k1"]
    distortion_text = f" | radial k1 {k1:+.3f}" if k1 else ""
    cv2.putText(
        header,
        f"{status} | frozen {threshold:.3f} px | {score_text}{distortion_text}",
        (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.39, color, 1, cv2.LINE_AA,
    )
    return np.vstack((header, panel))


def _pick(rows: list[dict], *, view: str, label: str, predicate=None) -> dict:
    matches = [
        row for row in rows
        if row["camera_view"] == view and row["label"] == label
        and (predicate is None or predicate(row))
    ]
    if not matches:
        raise RuntimeError(f"report lacks demo sample: {view} / {label}")
    return sorted(
        matches,
        key=lambda row: (
            row["reprojection_rms_px"] is None,
            row["reprojection_rms_px"] if row["reprojection_rms_px"] is not None else 0.0,
        ),
    )[len(matches) // 2]


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    report_path = repository / "artifacts" / "camera-calibration-transfer-validation.json"
    video_path = repository / "assets" / "demos" / "camera_calibration_transfer.mp4"
    report = json.loads(report_path.read_text())
    rows = report["trial_results"]
    threshold = report["protocol"]["threshold_px"]

    pairs = [
        (
            (_pick(rows, view="east", label="nominal"), "East reference | nominal"),
            (_pick(rows, view="southwest_diagonal", label="nominal"), "Diagonal view | nominal"),
        ),
        (
            (_pick(rows, view="west_lower", label="nominal"), "Lower west view | nominal"),
            (_pick(rows, view="east_grazing", label="nominal"), "Low east view | clipped markers"),
        ),
        (
            (_pick(rows, view="east", label="k1_+0.0100"), "East view | radial k1 +0.010"),
            (_pick(rows, view="north_higher", label="k1_+0.0100"), "High view | same k1 +0.010"),
        ),
        (
            (_pick(rows, view="east", label="k1_-0.0050"), "East view | radial k1 -0.005"),
            (_pick(rows, view="east", label="k1_+0.0200"), "East view | radial k1 +0.020"),
        ),
        (
            (_pick(rows, view="east", label="severe_drift"), "East view | severe shift"),
            (_pick(
                rows, view="west_lower", label="severe_drift",
                predicate=lambda row: row["reprojection_rms_px"] is None,
            ), "Lower west view | landmark loss"),
        ),
    ]

    video_path.parent.mkdir(parents=True, exist_ok=True)
    canvas_height = HEADER_HEIGHT + PANEL_HEIGHT
    separator = np.zeros((canvas_height, GAP_WIDTH, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
        (PANEL_WIDTH * 2 + GAP_WIDTH, canvas_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video at {video_path}")

    first_frame = None
    for slide_index, pair in enumerate(pairs):
        panels = []
        for row, title in pair:
            with CalibrationScene(row["camera_view"]) as scene:
                image = _recreate_record(scene, row)
            panels.append(_panel(image, row, title=title, threshold=threshold))
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
                "frozen_threshold_px": threshold,
                "comparisons": [
                    [row["camera_view"], row["label"]] for pair in pairs for row, _ in pair
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
