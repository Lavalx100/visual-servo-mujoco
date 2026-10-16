"""Render the radius-resolved checkerboard diagnostic as a labeled video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from visual_servo_mujoco.camera_calibration_health import (
    CalibrationScene,
    project_world_point_to_pixel,
)
from visual_servo_mujoco.camera_calibration_planar import (
    LEGACY_FOUR_MARKER_THRESHOLD_PX,
    PLANAR_TARGET_VISIBILITY_MARGIN_PX,
    PlanarCalibrationScene,
    analyze_planar_frame,
    planar_checkerboard_world_points,
)
from visual_servo_mujoco.terminal_verification import IMAGE_HEIGHT, IMAGE_WIDTH

FPS = 24
SECONDS_PER_CASE = 2
PANEL_WIDTH = 320
HEADER_HEIGHT = 60
FOOTER_HEIGHT = 60
CANVAS_SIZE = (IMAGE_WIDTH + PANEL_WIDTH, HEADER_HEIGHT + IMAGE_HEIGHT + FOOTER_HEIGHT)
RADIUS_COLORS = (
    (230, 180, 70),
    (90, 210, 130),
    (40, 205, 235),
    (40, 125, 250),
    (170, 90, 235),
)

DEMO_CASES = (
    ("east", "nominal", {}),
    ("west_lower", "nominal", {}),
    ("north_higher", "nominal", {}),
    ("southwest_diagonal", "nominal", {}),
    ("east_grazing", "nominal", {}),
    ("east", "radial k1 +0.005", {"radial_distortion_k1": 0.005}),
    ("east", "radial k1 +0.010", {"radial_distortion_k1": 0.01}),
    ("east", "radial k1 +0.020", {"radial_distortion_k1": 0.02}),
    ("east", "radial k1 +0.040", {"radial_distortion_k1": 0.04}),
    ("east", "radial k1 -0.005", {"radial_distortion_k1": -0.005}),
    ("east", "radial k1 -0.020", {"radial_distortion_k1": -0.02}),
    ("west_lower", "radial k1 +0.010", {"radial_distortion_k1": 0.01}),
    ("west_lower", "radial k1 +0.040", {"radial_distortion_k1": 0.04}),
    ("north_higher", "radial k1 +0.010", {"radial_distortion_k1": 0.01}),
    ("southwest_diagonal", "radial k1 +0.040", {"radial_distortion_k1": 0.04}),
    ("east", "rendered FOV +5°", {"fovy_error_deg": 5.0}),
    ("east", "rendered FOV -5°", {"fovy_error_deg": -5.0}),
    ("east", "camera world X +2 cm", {"camera_position_error_m": (0.02, 0.0, 0.0)}),
    ("east", "camera yaw +3°", {"rotation_error_deg": (0.0, 0.0, 3.0)}),
)


def _radius_color(radius_px: float) -> tuple[int, int, int]:
    if radius_px < 80.0:
        return RADIUS_COLORS[0]
    if radius_px < 160.0:
        return RADIUS_COLORS[1]
    if radius_px < 240.0:
        return RADIUS_COLORS[2]
    if radius_px < 320.0:
        return RADIUS_COLORS[3]
    return RADIUS_COLORS[4]


def _draw_feature_overlay(frame: np.ndarray, diagnostic: dict) -> None:
    for feature in diagnostic["radius_bins"]:
        expected = tuple(round(value) for value in feature["expected_pixel_xy"])
        color = _radius_color(feature["expected_radius_px"])
        cv2.circle(frame, expected, 4, color, 1, cv2.LINE_AA)
        observed = feature["observed_pixel_xy"]
        if observed is None:
            continue
        observed_xy = np.asarray(observed, dtype=float)
        expected_xy = np.asarray(feature["expected_pixel_xy"], dtype=float)
        exaggerated_endpoint = expected_xy + 3.0 * (observed_xy - expected_xy)
        end = tuple(round(value) for value in exaggerated_endpoint)
        cv2.line(frame, expected, end, color, 1, cv2.LINE_AA)
        cv2.circle(frame, tuple(round(value) for value in observed), 2, (245, 245, 245), -1)


def _draw_profile_panel(panel: np.ndarray, diagnostic: dict) -> None:
    panel[:] = (18, 27, 43)
    cv2.putText(
        panel,
        "Reprojection error by feature radius",
        (14, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.49,
        (244, 246, 250),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "49 known planar corners | vectors shown 3x",
        (14, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (170, 184, 204),
        1,
        cv2.LINE_AA,
    )

    left, right, top, bottom = 44, PANEL_WIDTH - 18, 95, 365
    cv2.rectangle(panel, (left, top), (right, bottom), (66, 79, 98), 1)
    for radius in range(0, 401, 100):
        x = left + round(radius / 400.0 * (right - left))
        cv2.line(panel, (x, top), (x, bottom), (41, 54, 72), 1)
        cv2.putText(panel, str(radius), (x - 9, bottom + 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.31, (150, 165, 185), 1, cv2.LINE_AA)
    for error in range(0, 11, 2):
        y = bottom - round(error / 10.0 * (bottom - top))
        cv2.line(panel, (left, y), (right, y), (41, 54, 72), 1)
        cv2.putText(panel, str(error), (18, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.31, (150, 165, 185), 1, cv2.LINE_AA)
    cv2.putText(panel, "error px", (10, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.32, (180, 193, 210), 1, cv2.LINE_AA)
    cv2.putText(panel, "radius px", (right - 52, bottom + 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.32, (180, 193, 210), 1, cv2.LINE_AA)

    visible_points = [
        point
        for point in diagnostic["radius_bins"]
        if point["reprojection_error_px"] is not None
    ]
    for point in visible_points:
        radius = min(point["expected_radius_px"], 400.0)
        error = min(point["reprojection_error_px"], 10.0)
        x = left + round(radius / 400.0 * (right - left))
        y = bottom - round(error / 10.0 * (bottom - top))
        cv2.circle(panel, (x, y), 3, _radius_color(radius), -1, cv2.LINE_AA)

    if diagnostic["status"] != "available_profile_only":
        message = diagnostic["unavailable_reason"].replace("_", " ")
        cv2.putText(panel, "PROFILE UNAVAILABLE", (18, 422), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (80, 150, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, message[:36], (18, 445), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (195, 205, 216), 1, cv2.LINE_AA)
    else:
        cv2.putText(
            panel,
            f"Profile RMS {diagnostic['profile_reprojection_rms_px']:.2f} px",
            (18, 422),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (95, 220, 155),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(panel, "dot color follows radius band", (18, 445),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (195, 205, 216), 1, cv2.LINE_AA)

    legend_y = 472
    labels = ("0-80", "80-160", "160-240", "240-320", "320+")
    x = 12
    for label, color in zip(labels, RADIUS_COLORS):
        cv2.circle(panel, (x + 4, legend_y - 4), 4, color, -1)
        cv2.putText(panel, label, (x + 12, legend_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.25, (184, 196, 212), 1, cv2.LINE_AA)
        x += 61


def _legacy_line(legacy: dict) -> str:
    score = legacy["reprojection_rms_px"]
    if score is None:
        candidate = legacy["candidate_reprojection_rms_px"]
        if candidate is None:
            return "four markers: unavailable"
        return f"four markers: unavailable (candidate {candidate:.2f} px)"
    decision = "alarm" if score > LEGACY_FOUR_MARKER_THRESHOLD_PX else "below threshold"
    return f"four markers: {score:.2f} px, {decision}"


def _render_case(view_name: str, condition_name: str, settings: dict) -> tuple[np.ndarray, dict, dict]:
    world_points = planar_checkerboard_world_points()
    with PlanarCalibrationScene(view_name) as scene:
        expected = np.asarray(
            [
                project_world_point_to_pixel(
                    point,
                    scene.nominal_position,
                    scene.nominal_rotation,
                    scene.nominal_fovy,
                )
                for point in world_points
            ],
            dtype=float,
        )
        image = scene.render(condition_name=condition_name, **settings)
    with CalibrationScene(view_name) as legacy_scene:
        _, legacy = legacy_scene.render(
            condition_name=condition_name, **settings
        )
    return image, analyze_planar_frame(image, expected), legacy


def _compose_frame(
    image_rgb: np.ndarray,
    diagnostic: dict,
    legacy: dict,
    *,
    view_name: str,
    condition_name: str,
) -> np.ndarray:
    width, height = CANVAS_SIZE
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (11, 18, 32)
    header = frame[:HEADER_HEIGHT]
    cv2.putText(header, "PLANAR CAMERA DIAGNOSTIC", (18, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.57, (95, 220, 155), 1, cv2.LINE_AA)
    cv2.putText(header, f"{view_name.replace('_', ' ')} | {condition_name}",
                (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (238, 242, 248), 1, cv2.LINE_AA)

    view = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _draw_feature_overlay(view, diagnostic)
    frame[HEADER_HEIGHT:HEADER_HEIGHT + IMAGE_HEIGHT, :IMAGE_WIDTH] = view
    panel = frame[HEADER_HEIGHT:HEADER_HEIGHT + IMAGE_HEIGHT,
                  IMAGE_WIDTH:IMAGE_WIDTH + PANEL_WIDTH]
    _draw_profile_panel(panel, diagnostic)

    footer = frame[HEADER_HEIGHT + IMAGE_HEIGHT:]
    if diagnostic["profile_reprojection_rms_px"] is None:
        dense_line = f"49-corner profile: unavailable ({diagnostic['unavailable_reason']})"
    else:
        dense_line = (
            f"49-corner profile: RMS {diagnostic['profile_reprojection_rms_px']:.2f} px"
        )
    cv2.putText(footer, dense_line, (16, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.39, (235, 239, 245), 1, cv2.LINE_AA)
    cv2.putText(
        footer,
        f"{_legacy_line(legacy)} | frozen 0.841 px applies to four markers only",
        (16, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (170, 184, 204),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(footer, "synthetic MuJoCo image | not a physical-camera result",
                (width - 347, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.27,
                (122, 143, 167), 1, cv2.LINE_AA)
    return frame


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    demo_dir = repository / "assets" / "demos"
    video_path = demo_dir / "camera_calibration_planar.mp4"
    preview_path = demo_dir / "camera_calibration_planar.png"
    demo_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        CANVAS_SIZE,
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video at {video_path}")

    first_frame = None
    for view_name, condition_name, settings in DEMO_CASES:
        image, diagnostic, legacy = _render_case(view_name, condition_name, settings)
        frame = _compose_frame(
            image,
            diagnostic,
            legacy,
            view_name=view_name,
            condition_name=condition_name,
        )
        if first_frame is None:
            first_frame = frame
        for _ in range(FPS * SECONDS_PER_CASE):
            writer.write(frame)
    writer.release()

    if first_frame is None or not cv2.imwrite(str(preview_path), first_frame):
        raise RuntimeError(f"cannot create preview image at {preview_path}")
    print(
        {
            "video": str(video_path.relative_to(repository)),
            "preview": str(preview_path.relative_to(repository)),
            "cases": len(DEMO_CASES),
            "duration_s": len(DEMO_CASES) * SECONDS_PER_CASE,
            "visibility_margin_px": PLANAR_TARGET_VISIBILITY_MARGIN_PX,
            "legacy_four_marker_threshold_px": LEGACY_FOUR_MARKER_THRESHOLD_PX,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
