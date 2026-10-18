"""Radius-resolved planar-target camera diagnostic for the MuJoCo scene."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .camera_calibration_health import (
    CALIBRATION_CAMERA_VIEWS,
    CalibrationScene,
    CALIBRATION_CAMERA_NAME,
    _freeze_camera_at_nominal_pose,
    _rotation_x,
    _rotation_y,
    _rotation_z,
    project_world_point_to_pixel,
)
from .camera_calibration_transfer import RADIAL_K1_VALUES
from .model import SIMULATION_TIMESTEP_S
from .terminal_verification import IMAGE_HEIGHT, IMAGE_WIDTH

PLANAR_TARGET_CENTER_M = (0.0, 0.0)
PLANAR_TARGET_SQUARES_X = 8
PLANAR_TARGET_SQUARES_Y = 8
PLANAR_TARGET_SQUARE_SIZE_M = 0.15
PLANAR_TARGET_SURFACE_Z_M = 0.1418
PLANAR_TARGET_INNER_COLUMNS = PLANAR_TARGET_SQUARES_X - 1
PLANAR_TARGET_INNER_ROWS = PLANAR_TARGET_SQUARES_Y - 1
PLANAR_TARGET_FEATURE_COUNT = (
    PLANAR_TARGET_INNER_COLUMNS * PLANAR_TARGET_INNER_ROWS
)
PLANAR_TARGET_VISIBILITY_MARGIN_PX = 12.0
LEGACY_FOUR_MARKER_THRESHOLD_PX = 0.8409987385743665
LEGACY_THRESHOLD_SOURCE = "artifacts/camera-calibration-health-validation.json"
DEFAULT_BASE_SEED = 800
DEFAULT_SEEDS = 5
RADIUS_BINS_PX = (0.0, 80.0, 160.0, 240.0, 320.0, math.inf)

CAMERA_PERTURBATIONS = {
    "nominal": {},
    "render_fovy_plus_5deg": {"fovy_error_deg": 5.0},
    "render_fovy_minus_5deg": {"fovy_error_deg": -5.0},
    "render_camera_world_x_plus_2cm": {
        "camera_position_error_m": (0.02, 0.0, 0.0)
    },
    "render_camera_yaw_plus_3deg": {"rotation_error_deg": (0.0, 0.0, 3.0)},
}
EXCLUDED_SEED_RANGES = ((400, 404), (500, 504), (600, 604), (700, 704))


def _square_size(value: float | None) -> float:
    square = PLANAR_TARGET_SQUARE_SIZE_M if value is None else float(value)
    if not math.isfinite(square) or square <= 0.0:
        raise ValueError("square_size_m must be finite and positive")
    return square


def planar_checkerboard_worldbody_xml(square_size_m: float | None = None) -> str:
    """Return thin, non-colliding checker squares for a known planar target."""
    center_x, center_y = PLANAR_TARGET_CENTER_M
    square = _square_size(square_size_m)
    left = center_x - PLANAR_TARGET_SQUARES_X * square / 2.0
    bottom = center_y - PLANAR_TARGET_SQUARES_Y * square / 2.0
    square_geoms = []
    for row in range(PLANAR_TARGET_SQUARES_Y):
        for column in range(PLANAR_TARGET_SQUARES_X):
            color = "0.025 0.03 0.04" if (row + column) % 2 == 0 else "0.94 0.94 0.90"
            square_geoms.append(
                '<geom name="planar_checker_{row}_{column}" type="box" '
                'pos="{x:.5f} {y:.5f} {z:.5f}" '
                'size="{half:.5f} {half:.5f} 0.0008" rgba="{color} 1" '
                'contype="0" conaffinity="0"/>'.format(
                    row=row,
                    column=column,
                    x=left + (column + 0.5) * square,
                    y=bottom + (row + 0.5) * square,
                    z=PLANAR_TARGET_SURFACE_Z_M - 0.0008,
                    half=square / 2.0,
                    color=color,
                )
            )
    return "\n".join(square_geoms)


def planar_checkerboard_world_points(
    square_size_m: float | None = None,
) -> np.ndarray:
    """Return the known world coordinates of the checkerboard's inner corners."""
    center_x, center_y = PLANAR_TARGET_CENTER_M
    square = _square_size(square_size_m)
    left = center_x - PLANAR_TARGET_SQUARES_X * square / 2.0
    bottom = center_y - PLANAR_TARGET_SQUARES_Y * square / 2.0
    return np.asarray(
        [
            (left + column * square, bottom + row * square, PLANAR_TARGET_SURFACE_Z_M)
            for row in range(1, PLANAR_TARGET_SQUARES_Y)
            for column in range(1, PLANAR_TARGET_SQUARES_X)
        ],
        dtype=float,
    )


class PlanarCalibrationScene:
    """Render a broad calibration target without robot-body occlusion."""

    def __init__(
        self, view_name: str = "east", *, square_size_m: float | None = None
    ) -> None:
        if view_name not in CALIBRATION_CAMERA_VIEWS:
            raise ValueError(f"unknown calibration camera view: {view_name}")
        self.view_name = view_name
        self.square_size_m = _square_size(square_size_m)
        camera_position = CALIBRATION_CAMERA_VIEWS[view_name]
        camera = (
            f'<camera name="{CALIBRATION_CAMERA_NAME}" mode="targetbody" '
            f'target="arm_base" pos="{camera_position[0]:.3f} '
            f'{camera_position[1]:.3f} {camera_position[2]:.3f}" fovy="65"/>'
        )
        xml = f"""
        <mujoco model="planar_camera_calibration">
          <compiler angle="radian"/>
          <option timestep="{SIMULATION_TIMESTEP_S}" gravity="0 0 -9.81" integrator="RK4"/>
          <visual>
            <global offwidth="640" offheight="480"/>
            <quality shadowsize="2048"/>
          </visual>
          <worldbody>
            <light pos="0 0 3" directional="false" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
            <geom name="floor" type="plane" size="2 2 0.1" rgba="0.12 0.14 0.17 1"/>
            <geom name="table" type="box" pos="0 0 0.07" size="0.75 0.75 0.07" rgba="0.78 0.73 0.62 1"/>
            <body name="arm_base" pos="0 0 0.21"/>
            {planar_checkerboard_worldbody_xml(self.square_size_m)}
            {camera}
          </worldbody>
        </mujoco>
        """
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, CALIBRATION_CAMERA_NAME
        )
        (
            self.nominal_position,
            self.nominal_rotation,
            self.nominal_fovy,
        ) = _freeze_camera_at_nominal_pose(self.model, self.data, self.camera_id)
        self.renderer = mujoco.Renderer(
            self.model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH
        )
        self._distortion_maps: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}

    def close(self) -> None:
        self.renderer.close()

    def __enter__(self) -> PlanarCalibrationScene:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def render(
        self,
        *,
        condition_name: str,
        fovy_error_deg: float = 0.0,
        camera_position_error_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation_error_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        brightness_scale: float = 1.0,
        image_noise_std_intensity: float = 0.0,
        blur_kernel_px: int = 1,
        image_noise_seed: int | None = None,
        radial_distortion_k1: float = 0.0,
        radial_distortion_k2: float = 0.0,
    ) -> np.ndarray:
        """Render a target-only calibration frame with controlled image changes."""
        values = np.asarray(
            (
                fovy_error_deg,
                brightness_scale,
                image_noise_std_intensity,
                radial_distortion_k1,
                radial_distortion_k2,
            ),
            dtype=float,
        )
        position_error = np.asarray(camera_position_error_m, dtype=float)
        rotation_error = np.asarray(rotation_error_deg, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("camera and image settings must be finite")
        if values[1] <= 0.0 or values[2] < 0.0:
            raise ValueError("brightness must be positive and image noise non-negative")
        if position_error.shape != (3,) or not np.isfinite(position_error).all():
            raise ValueError("camera_position_error_m must contain three finite values")
        if rotation_error.shape != (3,) or not np.isfinite(rotation_error).all():
            raise ValueError("rotation_error_deg must contain three finite values")
        if blur_kernel_px < 1 or blur_kernel_px % 2 == 0:
            raise ValueError("blur_kernel_px must be a positive odd integer")
        actual_fovy = self.nominal_fovy + float(fovy_error_deg)
        if not 0.0 < actual_fovy < 180.0:
            raise ValueError("render camera field of view must be between 0 and 180 degrees")

        actual_position = self.nominal_position + position_error
        roll, pitch, yaw = rotation_error
        world_rotation_error = (
            _rotation_z(float(yaw))
            @ _rotation_y(float(pitch))
            @ _rotation_x(float(roll))
        )
        actual_rotation = world_rotation_error @ self.nominal_rotation
        self.model.cam_pos[self.camera_id] = actual_position
        mujoco.mju_mat2Quat(
            self.model.cam_quat[self.camera_id], actual_rotation.reshape(-1)
        )
        self.model.cam_fovy[self.camera_id] = actual_fovy
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera_id)
        rgb_image = self.renderer.render()

        if radial_distortion_k1 != 0.0 or radial_distortion_k2 != 0.0:
            map_key = (float(radial_distortion_k1), float(radial_distortion_k2))
            if map_key not in self._distortion_maps:
                focal_px = IMAGE_HEIGHT / (
                    2.0 * math.tan(math.radians(self.nominal_fovy) / 2.0)
                )
                camera_matrix = np.asarray(
                    (
                        (focal_px, 0.0, (IMAGE_WIDTH - 1) / 2.0),
                        (0.0, focal_px, (IMAGE_HEIGHT - 1) / 2.0),
                        (0.0, 0.0, 1.0),
                    ),
                    dtype=np.float64,
                )
                grid_x, grid_y = np.meshgrid(
                    np.arange(IMAGE_WIDTH, dtype=np.float32),
                    np.arange(IMAGE_HEIGHT, dtype=np.float32),
                )
                destination_points = np.stack((grid_x, grid_y), axis=-1).reshape(
                    -1, 1, 2
                )
                coefficients = np.asarray(
                    (radial_distortion_k1, radial_distortion_k2, 0.0, 0.0, 0.0),
                    dtype=np.float64,
                )
                undistorted = cv2.undistortPoints(
                    destination_points, camera_matrix, coefficients
                ).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 2)
                map_x = undistorted[:, :, 0] * focal_px + (IMAGE_WIDTH - 1) / 2.0
                map_y = undistorted[:, :, 1] * focal_px + (IMAGE_HEIGHT - 1) / 2.0
                self._distortion_maps[map_key] = (
                    map_x.astype(np.float32),
                    map_y.astype(np.float32),
                )
            map_x, map_y = self._distortion_maps[map_key]
            rgb_image = cv2.remap(
                rgb_image,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        if brightness_scale != 1.0:
            rgb_image = np.clip(
                rgb_image.astype(np.float32) * brightness_scale, 0, 255
            ).astype(np.uint8)
        if blur_kernel_px > 1:
            rgb_image = cv2.GaussianBlur(
                rgb_image, (blur_kernel_px, blur_kernel_px), 0
            )
        if image_noise_std_intensity > 0.0:
            image_rng = np.random.default_rng(image_noise_seed)
            noise = image_rng.normal(
                0.0, image_noise_std_intensity, rgb_image.shape
            )
            rgb_image = np.clip(rgb_image.astype(np.float32) + noise, 0, 255).astype(
                np.uint8
            )
        return rgb_image


def _detect_checkerboard_corners(rgb_image: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3 or rgb_image.dtype != np.uint8:
        raise ValueError("rgb_image must be a uint8 RGB image")
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    pattern_size = (PLANAR_TARGET_INNER_COLUMNS, PLANAR_TARGET_INNER_ROWS)
    sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
    found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=sb_flags)
    if found and corners is not None:
        return corners.reshape(-1, 2).astype(float), "sector_based"

    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=classic_flags)
    if not found or corners is None:
        return None, None
    refined = cv2.cornerSubPix(
        gray,
        corners.astype(np.float32),
        (5, 5),
        (-1, -1),
        (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            40,
            0.001,
        ),
    )
    return refined.reshape(-1, 2).astype(float), "classic_refined"


def _ordered_grid_candidates(detected: np.ndarray) -> list[tuple[str, np.ndarray]]:
    rows = PLANAR_TARGET_INNER_ROWS
    columns = PLANAR_TARGET_INNER_COLUMNS
    if detected.shape != (PLANAR_TARGET_FEATURE_COUNT, 2):
        return []
    grids = [
        ("rows_by_columns", detected.reshape(rows, columns, 2)),
        (
            "transposed_rows_by_columns",
            detected.reshape(columns, rows, 2).transpose(1, 0, 2),
        ),
    ]
    candidates = []
    seen = set()
    for base_name, grid in grids:
        for flip_rows in (False, True):
            for flip_columns in (False, True):
                candidate = grid
                suffix = ""
                if flip_rows:
                    candidate = candidate[::-1, :, :]
                    suffix += "_flip_rows"
                if flip_columns:
                    candidate = candidate[:, ::-1, :]
                    suffix += "_flip_columns"
                candidate = np.ascontiguousarray(candidate)
                identity = candidate.tobytes()
                if identity not in seen:
                    seen.add(identity)
                    candidates.append((base_name + suffix, candidate))
    return candidates


def associate_checkerboard_corners(
    detected_corners_px: np.ndarray, expected_corners_px: np.ndarray
) -> dict | None:
    """Associate OpenCV's grid ordering to known points by its planar symmetries."""
    detected = np.asarray(detected_corners_px, dtype=float)
    expected = np.asarray(expected_corners_px, dtype=float)
    if expected.shape != (PLANAR_TARGET_FEATURE_COUNT, 2):
        raise ValueError(
            f"expected_corners_px must have shape ({PLANAR_TARGET_FEATURE_COUNT}, 2)"
        )
    if not np.isfinite(expected).all():
        raise ValueError("expected_corners_px must be finite")
    if detected.shape != expected.shape or not np.isfinite(detected).all():
        return None

    expected_grid = expected.reshape(
        PLANAR_TARGET_INNER_ROWS, PLANAR_TARGET_INNER_COLUMNS, 2
    )
    choices = []
    for name, candidate_grid in _ordered_grid_candidates(detected):
        residuals = candidate_grid - expected_grid
        rms = math.sqrt(float(np.mean(np.sum(np.square(residuals), axis=2))))
        choices.append((rms, name, candidate_grid.copy(), residuals.copy()))
    if not choices:
        return None
    choices.sort(key=lambda item: (item[0], item[1]))
    rms, orientation, matched_grid, residual_grid = choices[0]
    return {
        "orientation": orientation,
        "rms_px": rms,
        "matched_corners_px": matched_grid.reshape(-1, 2),
        "residual_vectors_px": residual_grid.reshape(-1, 2),
    }


def _safe_point_mask(points: np.ndarray, margin_px: float) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    return (
        (x >= margin_px)
        & (x < IMAGE_WIDTH - margin_px)
        & (y >= margin_px)
        & (y < IMAGE_HEIGHT - margin_px)
    )


def _radius_bin(radius_px: float) -> str:
    for lower, upper in zip(RADIUS_BINS_PX[:-1], RADIUS_BINS_PX[1:]):
        if lower <= radius_px < upper:
            upper_label = f"{upper:.0f}" if math.isfinite(upper) else "inf"
            return f"{lower:.0f}_{upper_label}px"
    return f"{RADIUS_BINS_PX[-2]:.0f}_infpx"


def analyze_planar_frame(
    rgb_image: np.ndarray, expected_corners_px: np.ndarray
) -> dict:
    """Detect, associate, and profile the known planar target in one image."""
    expected = np.asarray(expected_corners_px, dtype=float)
    expected_safe = _safe_point_mask(expected, PLANAR_TARGET_VISIBILITY_MARGIN_PX)
    detected, detector = _detect_checkerboard_corners(rgb_image)
    observed_safe_count = 0
    if detected is None:
        if bool(expected_safe.all()):
            status = "unavailable_checkerboard_not_detected"
            reason = "checkerboard_not_detected"
        else:
            status = "unavailable_expected_corner_outside_safe_margin"
            reason = "expected_corner_outside_safe_image_margin"
        matched = None
    else:
        matched = associate_checkerboard_corners(detected, expected)
        if matched is None:
            status = "unavailable_feature_association_failed"
            reason = "feature_association_failed"
        else:
            observed_safe = _safe_point_mask(
                matched["matched_corners_px"], PLANAR_TARGET_VISIBILITY_MARGIN_PX
            )
            observed_safe_count = int(np.count_nonzero(observed_safe))
            if not bool(expected_safe.all()):
                status = "unavailable_expected_corner_outside_safe_margin"
                reason = "expected_corner_outside_safe_image_margin"
            elif not bool(observed_safe.all()):
                status = "unavailable_observed_corner_outside_safe_margin"
                reason = "observed_corner_outside_safe_image_margin"
            else:
                status = "available_profile_only"
                reason = None

    expected_radii = np.linalg.norm(
        expected - np.asarray(((IMAGE_WIDTH - 1) / 2.0, (IMAGE_HEIGHT - 1) / 2.0)),
        axis=1,
    )
    points = []
    if matched is None:
        residual_vectors = np.full_like(expected, np.nan)
        observed_points = None
    else:
        observed_points = matched["matched_corners_px"]
        residual_vectors = matched["residual_vectors_px"]

    for index, (world, expected_px, radius) in enumerate(
        zip(planar_checkerboard_world_points(), expected, expected_radii)
    ):
        feature = {
            "index": index,
            "grid_column": index % PLANAR_TARGET_INNER_COLUMNS + 1,
            "grid_row": index // PLANAR_TARGET_INNER_COLUMNS + 1,
            "world_xyz_m": [float(value) for value in world],
            "expected_pixel_xy": [float(value) for value in expected_px],
            "expected_radius_px": float(radius),
            "radius_bin": _radius_bin(float(radius)),
            "expected_inside_safe_margin": bool(expected_safe[index]),
            "observed_pixel_xy": None,
            "residual_xy_px": None,
            "reprojection_error_px": None,
            "radial_residual_px": None,
            "tangential_residual_px": None,
        }
        if observed_points is not None:
            observed_px = observed_points[index]
            residual = residual_vectors[index]
            radial_unit = (
                expected_px
                - np.asarray(((IMAGE_WIDTH - 1) / 2.0, (IMAGE_HEIGHT - 1) / 2.0))
            ) / max(float(radius), 1e-12)
            feature.update(
                {
                    "observed_pixel_xy": [float(value) for value in observed_px],
                    "residual_xy_px": [float(value) for value in residual],
                    "reprojection_error_px": float(np.linalg.norm(residual)),
                    "radial_residual_px": float(np.dot(residual, radial_unit)),
                    "tangential_residual_px": float(
                        radial_unit[0] * residual[1] - radial_unit[1] * residual[0]
                    ),
                }
            )
        points.append(feature)

    return {
        "status": status,
        "unavailable_reason": reason,
        "detector": detector,
        "expected_feature_count": PLANAR_TARGET_FEATURE_COUNT,
        "detected_feature_count": 0 if detected is None else int(len(detected)),
        "safely_visible_expected_feature_count": int(np.count_nonzero(expected_safe)),
        "safely_visible_observed_feature_count": observed_safe_count,
        "visibility_margin_px": PLANAR_TARGET_VISIBILITY_MARGIN_PX,
        "association_orientation": None if matched is None else matched["orientation"],
        "candidate_reprojection_rms_px": None if matched is None else matched["rms_px"],
        "profile_reprojection_rms_px": (
            matched["rms_px"] if status == "available_profile_only" else None
        ),
        "profile_reprojection_max_px": (
            float(np.max(np.linalg.norm(residual_vectors, axis=1)))
            if status == "available_profile_only"
            else None
        ),
        "radius_bins": points,
    }


def _appearance_sample(rng: np.random.Generator) -> dict:
    return {
        "brightness_scale": float(rng.uniform(0.55, 1.45)),
        "image_noise_std_intensity": float(rng.uniform(0.0, 8.0)),
        "blur_kernel_px": int(rng.choice((1, 3, 5))),
        "image_noise_seed": int(rng.integers(0, 2**32 - 1)),
    }


def _validate_protocol(*, seeds: int, base_seed: int) -> None:
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    evaluation_seeds = set(range(base_seed, base_seed + seeds))
    for first, last in EXCLUDED_SEED_RANGES:
        if evaluation_seeds & set(range(first, last + 1)):
            raise ValueError("planar evaluation seeds overlap prior calibration splits")


def _conditions() -> list[tuple[str, dict]]:
    conditions = list(CAMERA_PERTURBATIONS.items())
    conditions.extend(
        (f"radial_k1_{k1:+.4f}", {"radial_distortion_k1": k1})
        for k1 in RADIAL_K1_VALUES
    )
    return conditions


def _legacy_four_marker_record(row: dict) -> dict:
    rms = row["reprojection_rms_px"]
    return {
        "frozen_threshold_px": LEGACY_FOUR_MARKER_THRESHOLD_PX,
        "frozen_threshold_source": LEGACY_THRESHOLD_SOURCE,
        "diagnostic_status": row["diagnostic_status"],
        "unavailable_reason": row["diagnostic_unavailable_reason"],
        "available_reprojection_rms_px": rms,
        "candidate_reprojection_rms_px": row["candidate_reprojection_rms_px"],
        "frozen_decision": (
            None if rms is None else rms > LEGACY_FOUR_MARKER_THRESHOLD_PX
        ),
    }


def _profile_summary(rows: list[dict]) -> dict:
    available = [
        row
        for row in rows
        if row["planar_diagnostic"]["profile_reprojection_rms_px"] is not None
    ]
    errors_by_bin: dict[str, list[dict]] = {}
    for row in available:
        for point in row["planar_diagnostic"]["radius_bins"]:
            errors_by_bin.setdefault(point["radius_bin"], []).append(point)
    bins = {}
    for bin_name, features in errors_by_bin.items():
        errors = np.asarray(
            [point["reprojection_error_px"] for point in features], dtype=float
        )
        radial = np.asarray(
            [point["radial_residual_px"] for point in features], dtype=float
        )
        tangential = np.asarray(
            [point["tangential_residual_px"] for point in features], dtype=float
        )
        bins[bin_name] = {
            "feature_observations": len(features),
            "median_error_px": float(np.median(errors)),
            "mean_error_px": float(np.mean(errors)),
            "p95_error_px": float(np.quantile(errors, 0.95)),
            "median_signed_radial_residual_px": float(np.median(radial)),
            "median_absolute_tangential_residual_px": float(
                np.median(np.abs(tangential))
            ),
        }
    rms_values = np.asarray(
        [row["planar_diagnostic"]["profile_reprojection_rms_px"] for row in available],
        dtype=float,
    )
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["planar_diagnostic"]["status"]] = (
            statuses.get(row["planar_diagnostic"]["status"], 0) + 1
        )
    legacy_available = [
        row["legacy_four_marker"]["available_reprojection_rms_px"]
        for row in rows
        if row["legacy_four_marker"]["available_reprojection_rms_px"] is not None
    ]
    legacy_alarms = sum(
        row["legacy_four_marker"]["frozen_decision"] is True for row in rows
    )
    return {
        "samples": len(rows),
        "available_profile_samples": len(available),
        "unavailable_profile_samples": len(rows) - len(available),
        "availability_rate": len(available) / len(rows) if rows else None,
        "status_counts": statuses,
        "median_profile_rms_px": (
            float(np.median(rms_values)) if rms_values.size else None
        ),
        "p95_profile_rms_px": (
            float(np.quantile(rms_values, 0.95)) if rms_values.size else None
        ),
        "legacy_four_marker_available_samples": len(legacy_available),
        "legacy_four_marker_unavailable_samples": len(rows) - len(legacy_available),
        "legacy_four_marker_frozen_alarm_samples": legacy_alarms,
        "legacy_four_marker_median_rms_px": (
            float(np.median(legacy_available)) if legacy_available else None
        ),
        "radius_bins": bins,
    }


def run_camera_calibration_planar_benchmark(
    *,
    seeds: int = DEFAULT_SEEDS,
    base_seed: int = DEFAULT_BASE_SEED,
    output_path: Path | None = None,
) -> dict:
    """Measure dense planar reprojection by feature radius across camera poses.

    The legacy four-marker threshold is reported for reference only. It is never
    applied to the dense checkerboard profile or retuned from these evaluations.
    """
    _validate_protocol(seeds=seeds, base_seed=base_seed)
    rows: list[dict] = []
    view_results = {}
    conditions = _conditions()
    for view_index, view_name in enumerate(CALIBRATION_CAMERA_VIEWS):
        view_rows = []
        with (
            PlanarCalibrationScene(view_name) as planar_scene,
            CalibrationScene(view_name) as legacy_scene,
        ):
            expected = np.asarray(
                [
                    project_world_point_to_pixel(
                        point,
                        planar_scene.nominal_position,
                        planar_scene.nominal_rotation,
                        planar_scene.nominal_fovy,
                    )
                    for point in planar_checkerboard_world_points()
                ],
                dtype=float,
            )
            dense_radii = np.linalg.norm(
                expected
                - np.asarray(((IMAGE_WIDTH - 1) / 2.0, (IMAGE_HEIGHT - 1) / 2.0)),
                axis=1,
            )
            radius_bin_counts = {
                bin_name: int(
                    sum(_radius_bin(float(radius)) == bin_name for radius in dense_radii)
                )
                for bin_name in sorted({_radius_bin(float(value)) for value in dense_radii})
            }
            legacy_radii = {
                name: float(
                    np.linalg.norm(
                        np.asarray(pixel)
                        - np.asarray(((IMAGE_WIDTH - 1) / 2.0, (IMAGE_HEIGHT - 1) / 2.0))
                    )
                )
                for name, pixel in legacy_scene.expected_pixels.items()
            }
            for condition_index, (condition_name, settings) in enumerate(conditions):
                for seed in range(base_seed, base_seed + seeds):
                    sample_index = seed - base_seed
                    rng = np.random.default_rng(
                        seed * 1_000_003 + view_index * 10_007 + condition_index
                    )
                    appearance = _appearance_sample(rng)
                    render_settings = {
                        "fovy_error_deg": float(settings.get("fovy_error_deg", 0.0)),
                        "camera_position_error_m": settings.get(
                            "camera_position_error_m", (0.0, 0.0, 0.0)
                        ),
                        "rotation_error_deg": settings.get(
                            "rotation_error_deg", (0.0, 0.0, 0.0)
                        ),
                        "radial_distortion_k1": float(
                            settings.get("radial_distortion_k1", 0.0)
                        ),
                        **appearance,
                    }
                    image = planar_scene.render(
                        condition_name=f"{condition_name}_seed_{seed}",
                        **render_settings,
                    )
                    _, legacy_row = legacy_scene.render(
                        condition_name=f"{condition_name}_seed_{seed}",
                        **render_settings,
                    )
                    planar = analyze_planar_frame(image, expected)
                    result = {
                        "camera_view": view_name,
                        "condition": condition_name,
                        "seed": seed,
                        "sample_index": sample_index,
                        "appearance": appearance,
                        "render_settings": {
                            key: list(value) if isinstance(value, tuple) else value
                            for key, value in render_settings.items()
                        },
                        "legacy_four_marker": _legacy_four_marker_record(legacy_row),
                        "planar_diagnostic": planar,
                    }
                    rows.append(result)
                    view_rows.append(result)

        condition_summaries = {}
        for condition_name, _ in conditions:
            condition_rows = [row for row in view_rows if row["condition"] == condition_name]
            condition_summaries[condition_name] = _profile_summary(condition_rows)
        view_results[view_name] = {
            "camera_position_relative_to_targetbody_m": list(
                CALIBRATION_CAMERA_VIEWS[view_name]
            ),
            "feature_radius_coverage": {
                "dense_target_min_radius_px": float(np.min(dense_radii)),
                "dense_target_max_radius_px": float(np.max(dense_radii)),
                "dense_target_feature_count_by_radius_bin": radius_bin_counts,
                "legacy_four_marker_radius_px_by_name": legacy_radii,
            },
            "conditions": condition_summaries,
        }

    report = {
        "benchmark": "radius_resolved_planar_camera_calibration_diagnostic",
        "protocol": {
            "scope": (
                "simulator-only camera diagnostic with a target-only tabletop render "
                "and a paired legacy-marker render in the original arm scene; no "
                "reaching controller, physical success, or fresh visual confirmation "
                "is evaluated"
            ),
            "evidence_status": "synthetic MuJoCo images; no recorded camera frames",
            "target": {
                "geometry": "planar black/ivory checkerboard squares rendered as thin non-colliding geoms",
                "square_count_xy": [PLANAR_TARGET_SQUARES_X, PLANAR_TARGET_SQUARES_Y],
                "square_size_m": PLANAR_TARGET_SQUARE_SIZE_M,
                "center_xy_m": list(PLANAR_TARGET_CENTER_M),
                "inner_corner_grid_xy": [
                    PLANAR_TARGET_INNER_COLUMNS,
                    PLANAR_TARGET_INNER_ROWS,
                ],
                "inner_corner_count": PLANAR_TARGET_FEATURE_COUNT,
                "corner_height_m": PLANAR_TARGET_SURFACE_Z_M,
                "design_selection": (
                    "centered 8-by-8-square layout selected from nominal-view "
                    "visibility checks; a wider 0.17 m-square probe had lower "
                    "appearance-robust availability and is retained separately"
                ),
                "design_probe_seed_range_excluded": [700, 704],
                "rejected_wider_target_probe": (
                    "0.17 m squares: 2/5 West nominal profiles, 4/5 diagonal, "
                    "and 0/5 grazing with the exploratory appearance seeds"
                ),
                "rejected_wider_target_report": (
                    "artifacts/camera-calibration-planar-profile-wide-target-probe.json"
                ),
            },
            "image_size_px": [IMAGE_WIDTH, IMAGE_HEIGHT],
            "camera_views": {
                name: list(position) for name, position in CALIBRATION_CAMERA_VIEWS.items()
            },
            "seed_range": [base_seed, base_seed + seeds - 1],
            "samples_per_view_condition": seeds,
            "conditions": [name for name, _ in conditions],
            "appearance_ranges": {
                "brightness_scale": [0.55, 1.45],
                "image_noise_std_intensity": [0.0, 8.0],
                "blur_kernel_px": [1, 3, 5],
            },
            "corner_detection": (
                "OpenCV sector-based checkerboard detector with classic subpixel "
                "fallback; only a complete detected grid is used"
            ),
            "availability_rule": (
                "profile is available only when all expected and associated observed "
                f"corners lie inside the {PLANAR_TARGET_VISIBILITY_MARGIN_PX:.0f} px safe margin"
            ),
            "reprojection_reference": (
                "nominal pinhole projection from the known MuJoCo camera pose, FOV, "
                "and planar target coordinates"
            ),
            "radius_profile": {
                "center_px": [(IMAGE_WIDTH - 1) / 2.0, (IMAGE_HEIGHT - 1) / 2.0],
                "bins_px": [
                    [lower, upper if math.isfinite(upper) else None]
                    for lower, upper in zip(RADIUS_BINS_PX[:-1], RADIUS_BINS_PX[1:])
                ],
                "residual_direction": "observed pixel minus nominally projected pixel",
            },
            "legacy_four_marker_reference": {
                "threshold_px": LEGACY_FOUR_MARKER_THRESHOLD_PX,
                "threshold_source": LEGACY_THRESHOLD_SOURCE,
                "use": (
                "reported from the paired original arm-scene render with the same "
                "view, condition, and appearance seed; never applied to dense-grid "
                "scores or retuned here"
                ),
            },
            "excluded_seed_ranges": [list(pair) for pair in EXCLUDED_SEED_RANGES],
            "limits": (
                "Synthetic lens warps and renderer images do not establish performance "
                "with a physical camera, printed target, changing illumination, or "
                "independent camera calibration. Dense profiles are descriptive and "
                "have no acceptance threshold."
            ),
        },
        "views": view_results,
        "trial_results": rows,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure camera reprojection error across planar feature radii."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/camera-calibration-planar-profile.json"),
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    args = parser.parse_args()
    report = run_camera_calibration_planar_benchmark(
        seeds=args.seeds,
        base_seed=args.base_seed,
        output_path=args.output,
    )
    summary = {
        view_name: {
            condition: {
                "dense_available": condition_report["available_profile_samples"],
                "dense_samples": condition_report["samples"],
                "dense_median_rms_px": condition_report["median_profile_rms_px"],
                "legacy_four_marker_alarms": condition_report[
                    "legacy_four_marker_frozen_alarm_samples"
                ],
                "legacy_four_marker_available": condition_report[
                    "legacy_four_marker_available_samples"
                ],
            }
            for condition, condition_report in data["conditions"].items()
        }
        for view_name, data in report["views"].items()
    }
    print(json.dumps({"report": str(args.output), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
