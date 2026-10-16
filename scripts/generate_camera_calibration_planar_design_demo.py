"""Compare the selected checkerboard size with the wider rejected design."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from visual_servo_mujoco.camera_calibration_health import (
    CALIBRATION_CAMERA_VIEWS,
    project_world_point_to_pixel,
)
from visual_servo_mujoco.camera_calibration_planar import (
    DEFAULT_BASE_SEED,
    PlanarCalibrationScene,
    analyze_planar_frame,
    planar_checkerboard_world_points,
)

FPS = 24
SECONDS_PER_VIEW = 2
PANEL_WIDTH = 640
HEADER_HEIGHT = 72
FOOTER_HEIGHT = 48
CANVAS_SIZE = (PANEL_WIDTH * 2, HEADER_HEIGHT + 480 + FOOTER_HEIGHT)
TARGET_SIZES_M = (0.15, 0.17)
RADIUS_COLORS = (
    (230, 180, 70),
    (90, 210, 130),
    (40, 205, 235),
    (40, 125, 250),
    (170, 90, 235),
)


def _appearance(view_index: int, seed: int) -> dict:
    rng = np.random.default_rng(seed * 1_000_003 + view_index * 10_007)
    return {
        "brightness_scale": float(rng.uniform(0.55, 1.45)),
        "image_noise_std_intensity": float(rng.uniform(0.0, 8.0)),
        "blur_kernel_px": int(rng.choice((1, 3, 5))),
        "image_noise_seed": int(rng.integers(0, 2**32 - 1)),
    }


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


def _render(view_name: str, square_size_m: float, settings: dict) -> tuple[np.ndarray, dict]:
    with PlanarCalibrationScene(view_name, square_size_m=square_size_m) as scene:
        expected = np.asarray(
            [
                project_world_point_to_pixel(
                    point,
                    scene.nominal_position,
                    scene.nominal_rotation,
                    scene.nominal_fovy,
                )
                for point in planar_checkerboard_world_points(square_size_m)
            ],
            dtype=float,
        )
        image = scene.render(condition_name="target_size_design_probe", **settings)
    return image, analyze_planar_frame(image, expected)


def _overlay(image_rgb: np.ndarray, diagnostic: dict) -> np.ndarray:
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for feature in diagnostic["radius_bins"]:
        expected = tuple(round(value) for value in feature["expected_pixel_xy"])
        color = _radius_color(feature["expected_radius_px"])
        cv2.circle(image, expected, 4, color, 1, cv2.LINE_AA)
        observed = feature["observed_pixel_xy"]
        if observed is None:
            continue
        observed = np.asarray(observed, dtype=float)
        expected_xy = np.asarray(feature["expected_pixel_xy"], dtype=float)
        amplified = expected_xy + 3.0 * (observed - expected_xy)
        end = tuple(round(value) for value in amplified)
        cv2.line(image, expected, end, color, 1, cv2.LINE_AA)
        cv2.circle(image, tuple(round(value) for value in observed), 2, (245, 245, 245), -1)
    return image


def _compose(
    view_name: str,
    seed: int,
    appearances: dict,
    cases: dict[float, tuple[np.ndarray, dict]],
) -> np.ndarray:
    frame = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8)
    frame[:] = (11, 18, 32)
    cv2.putText(
        frame,
        "PLANAR TARGET SIZE PROBE | SAME CAMERA VIEW AND APPEARANCE SETTINGS",
        (18, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (95, 220, 155),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{view_name.replace('_', ' ')} | seed {seed} | brightness {appearances['brightness_scale']:.2f} "
        f"noise {appearances['image_noise_std_intensity']:.1f}px | blur {appearances['blur_kernel_px']}px",
        (18, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (205, 215, 229),
        1,
        cv2.LINE_AA,
    )
    short_status = {
        "available_profile_only": "available",
        "unavailable_checkerboard_not_detected": "unavailable: detector failed",
        "unavailable_expected_corner_outside_safe_margin": "unavailable: edge clipped",
        "unavailable_observed_corner_outside_safe_margin": "unavailable: edge clipped",
        "unavailable_feature_association_failed": "unavailable: association failed",
    }
    for index, size in enumerate(TARGET_SIZES_M):
        x0 = index * PANEL_WIDTH
        image, diagnostic = cases[size]
        cv2.putText(
            frame,
            f"{size:.2f} m squares | {short_status[diagnostic['status']]}",
            (x0 + 16, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (238, 242, 248),
            1,
            cv2.LINE_AA,
        )
        overlay = _overlay(image, diagnostic)
        frame[HEADER_HEIGHT:HEADER_HEIGHT + 480, x0:x0 + PANEL_WIDTH] = overlay
        radius = max(feature["expected_radius_px"] for feature in diagnostic["radius_bins"])
        rms = diagnostic["profile_reprojection_rms_px"]
        score = "RMS unavailable" if rms is None else f"grid RMS {rms:.2f}px"
        line = (
            f"{diagnostic['detected_feature_count']}/49 corners | "
            f"{diagnostic['safely_visible_expected_feature_count']}/49 expected safe | "
            f"max radius {radius:.0f}px | {score}"
        )
        cv2.putText(
            frame,
            line,
            (x0 + 14, HEADER_HEIGHT + 480 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.31,
            (225, 232, 241),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        "Larger grid covers more radii but is harder to detect at low views; design seeds 700-704 are excluded from evaluation.",
        (18, CANVAS_SIZE[1] - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        (150, 168, 190),
        1,
        cv2.LINE_AA,
    )
    return frame


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    output_dir = repository / "assets" / "demos"
    video_path = output_dir / "camera_calibration_planar_target_design.mp4"
    preview_path = video_path.with_suffix(".png")
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, CANVAS_SIZE
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video at {video_path}")

    first_frame = None
    for view_index, view_name in enumerate(CALIBRATION_CAMERA_VIEWS):
        seed = 701 if view_name == "southwest_diagonal" else DEFAULT_BASE_SEED - 100
        appearances = _appearance(view_index, seed)
        cases = {
            size: _render(view_name, size, appearances)
            for size in TARGET_SIZES_M
        }
        frame = _compose(view_name, seed, appearances, cases)
        if first_frame is None:
            first_frame = frame
        for _ in range(FPS * SECONDS_PER_VIEW):
            writer.write(frame)
    writer.release()

    if first_frame is None or not cv2.imwrite(str(preview_path), first_frame):
        raise RuntimeError(f"cannot create preview at {preview_path}")
    print(
        {
            "video": str(video_path.relative_to(repository)),
            "preview": str(preview_path.relative_to(repository)),
            "views": len(CALIBRATION_CAMERA_VIEWS),
            "duration_s": len(CALIBRATION_CAMERA_VIEWS) * SECONDS_PER_VIEW,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
