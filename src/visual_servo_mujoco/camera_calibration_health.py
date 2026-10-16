"""Independent fixed-landmark reprojection diagnostic for camera calibration.

This experiment perturbs the camera used to render the image while keeping the
projection model nominal. It is deliberately separate from reaching control
and from verifier tests that perturb only the estimator's camera parameters.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .model import MODEL_XML
from .terminal_verification import IMAGE_HEIGHT, IMAGE_WIDTH

CALIBRATION_CAMERA_NAME = "calibration_health_camera"
CALIBRATION_CAMERA_VIEWS = {
    "east": (1.0, 0.0, 0.85),
    "west_lower": (-1.0, 0.0, 0.65),
    "north_higher": (0.0, 1.0, 1.25),
    "southwest_diagonal": (-0.85, -0.85, 0.70),
    "east_grazing": (1.0, 0.0, 0.55),
}
LANDMARKS = {
    "south_west": (-0.52, -0.52, 0.143),
    "south_east": (0.52, -0.52, 0.143),
    "north_west": (-0.52, 0.52, 0.143),
    "north_east": (0.52, 0.52, 0.143),
}
REPROJECTION_ALARM_THRESHOLD_PX = 3.0
LANDMARK_VISIBILITY_MARGIN_PX = 12.0

CAMERA_HEALTH_CONDITIONS = {
    "nominal": {},
    "render_fovy_plus_5deg": {"fovy_error_deg": 5.0},
    "render_fovy_minus_5deg": {"fovy_error_deg": -5.0},
    "render_camera_world_x_plus_2cm": {"position_error_m": (0.02, 0.0, 0.0)},
    "render_camera_yaw_plus_3deg": {"yaw_error_deg": 3.0},
    "render_combined_fovy_x_yaw_shift": {
        "fovy_error_deg": 5.0,
        "position_error_m": (0.02, 0.0, 0.0),
        "yaw_error_deg": 3.0,
    },
    "bright_exposure_x1_5": {"brightness_scale": 1.5},
    "dim_exposure_x0_25": {"brightness_scale": 0.25},
}


def model_with_calibration_landmarks(
    view_name: str = "east", *, extra_worldbody_xml: str = ""
) -> mujoco.MjModel:
    """Build a diagnostic scene with fixed landmarks and a chosen oblique view."""
    if view_name not in CALIBRATION_CAMERA_VIEWS:
        raise ValueError(f"unknown calibration camera view: {view_name}")
    landmark_geoms = "\n".join(
        '<geom name="calibration_landmark_{name}" type="cylinder" '
        'pos="{x:.3f} {y:.3f} {z:.3f}" size="0.022 0.003" '
        'rgba="0.90 0.04 0.85 1" contype="0" conaffinity="0"/>'.format(
            name=name, x=position[0], y=position[1], z=position[2]
        )
        for name, position in LANDMARKS.items()
    )
    camera_position = CALIBRATION_CAMERA_VIEWS[view_name]
    camera = (
        f'<camera name="{CALIBRATION_CAMERA_NAME}" mode="targetbody" '
        f'target="arm_base" pos="{camera_position[0]:.3f} '
        f'{camera_position[1]:.3f} {camera_position[2]:.3f}" fovy="65"/>'
    )
    xml = MODEL_XML.replace(
        "</worldbody>",
        f"{landmark_geoms}\n{extra_worldbody_xml}\n{camera}\n</worldbody>",
    )
    return mujoco.MjModel.from_xml_string(xml)


def _freeze_camera_at_nominal_pose(
    model: mujoco.MjModel, data: mujoco.MjData, camera_id: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return nominal world pose/FOV and make the camera independently movable."""
    mujoco.mj_forward(model, data)
    position = data.cam_xpos[camera_id].copy()
    rotation = data.cam_xmat[camera_id].reshape(3, 3).copy()
    quaternion = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    fovy_deg = float(model.cam_fovy[camera_id])
    model.cam_mode[camera_id] = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    model.cam_pos[camera_id] = position
    model.cam_quat[camera_id] = quaternion
    mujoco.mj_forward(model, data)
    return position, rotation, fovy_deg


def _rotation_z(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.asarray(
        ((math.cos(angle), -math.sin(angle), 0.0),
         (math.sin(angle), math.cos(angle), 0.0),
         (0.0, 0.0, 1.0)),
        dtype=float,
    )


def _rotation_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.asarray(
        ((math.cos(angle), 0.0, math.sin(angle)),
         (0.0, 1.0, 0.0),
         (-math.sin(angle), 0.0, math.cos(angle))),
        dtype=float,
    )


def _rotation_x(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.asarray(
        ((1.0, 0.0, 0.0),
         (0.0, math.cos(angle), -math.sin(angle)),
         (0.0, math.sin(angle), math.cos(angle))),
        dtype=float,
    )


def project_world_point_to_pixel(
    point_world: tuple[float, float, float] | np.ndarray,
    camera_position_world: np.ndarray,
    camera_to_world_rotation: np.ndarray,
    fovy_deg: float,
    *,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
) -> tuple[float, float]:
    """Project a world point using a pinhole camera and nominal calibration."""
    point = np.asarray(point_world, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("point_world must contain three finite values")
    position = np.asarray(camera_position_world, dtype=float)
    rotation = np.asarray(camera_to_world_rotation, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("camera_position_world must contain three finite values")
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("camera_to_world_rotation must be a finite 3x3 matrix")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not math.isfinite(fovy_deg) or not 0.0 < fovy_deg < 180.0:
        raise ValueError("fovy_deg must be finite and between 0 and 180 degrees")

    camera_point = rotation.T @ (point - position)
    if camera_point[2] >= -1e-9:
        raise ValueError("point must lie in front of the camera")
    focal_px = image_height / (2.0 * math.tan(math.radians(fovy_deg) / 2.0))
    return (
        (image_width - 1) / 2.0 + focal_px * camera_point[0] / -camera_point[2],
        (image_height - 1) / 2.0 - focal_px * camera_point[1] / -camera_point[2],
    )


def detect_calibration_landmarks(rgb_image: np.ndarray) -> list[tuple[float, float]]:
    """Return image centroids for visible magenta landmark blobs."""
    image = np.asarray(rgb_image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb_image must be a uint8 RGB image")
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([135, 90, 55]), np.array([175, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 8.0 <= area <= 3000.0:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0.0:
            continue
        detections.append(
            (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        )
    return sorted(detections, key=lambda pixel: (pixel[1], pixel[0]))


def match_landmark_detections(
    expected_pixels: dict[str, tuple[float, float]],
    detected_pixels: list[tuple[float, float]],
) -> dict | None:
    """Find the minimum-cost one-to-one assignment for four same-color points."""
    names = tuple(expected_pixels)
    if len(names) != 4 or len(detected_pixels) != len(names):
        return None
    expected = np.asarray([expected_pixels[name] for name in names], dtype=float)
    observed = np.asarray(detected_pixels, dtype=float)
    if observed.shape != (4, 2) or not np.isfinite(observed).all():
        return None
    best = None
    for assignment in itertools.permutations(range(4)):
        matched = observed[list(assignment)]
        residuals = np.linalg.norm(matched - expected, axis=1)
        cost = float(np.square(residuals).sum())
        if best is None or cost < best[0]:
            best = (cost, assignment, residuals)
    assert best is not None
    return {
        name: {
            "expected_pixel_xy": [float(value) for value in expected[index]],
            "observed_pixel_xy": [float(value) for value in observed[best[1][index]]],
            "reprojection_residual_px": float(best[2][index]),
        }
        for index, name in enumerate(names)
    }


class CalibrationScene:
    """Reusable MuJoCo renderer with a fixed nominal projection reference."""

    def __init__(
        self, view_name: str = "east", *, extra_worldbody_xml: str = ""
    ) -> None:
        if view_name not in CALIBRATION_CAMERA_VIEWS:
            raise ValueError(f"unknown calibration camera view: {view_name}")
        self.view_name = view_name
        self.model = model_with_calibration_landmarks(
            view_name, extra_worldbody_xml=extra_worldbody_xml
        )
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
        self.expected_pixels = {
            name: project_world_point_to_pixel(
                position,
                self.nominal_position,
                self.nominal_rotation,
                self.nominal_fovy,
            )
            for name, position in LANDMARKS.items()
        }

    def close(self) -> None:
        self.renderer.close()

    def __enter__(self) -> CalibrationScene:
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
    ) -> tuple[np.ndarray, dict]:
        """Render one sample and withhold RMS when projected landmarks are clipped."""
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
                map_x = (undistorted[:, :, 0] * focal_px + (IMAGE_WIDTH - 1) / 2.0)
                map_y = (undistorted[:, :, 1] * focal_px + (IMAGE_HEIGHT - 1) / 2.0)
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

        detections = detect_calibration_landmarks(rgb_image)
        matched = match_landmark_detections(self.expected_pixels, detections)
        candidate_residuals = (
            [item["reprojection_residual_px"] for item in matched.values()]
            if matched is not None
            else []
        )
        candidate_rms = (
            math.sqrt(float(np.mean(np.square(candidate_residuals))))
            if candidate_residuals
            else None
        )
        candidate_maximum = max(candidate_residuals) if candidate_residuals else None
        margin = LANDMARK_VISIBILITY_MARGIN_PX
        safely_visible_landmarks = [
            name
            for name, (x, y) in self.expected_pixels.items()
            if margin <= x < IMAGE_WIDTH - margin
            and margin <= y < IMAGE_HEIGHT - margin
        ]
        all_expected_landmarks_visible = len(safely_visible_landmarks) == len(LANDMARKS)
        if matched is None:
            status = "unavailable"
            unavailable_reason = "landmark_detection_or_association_failed"
        elif not all_expected_landmarks_visible:
            status = "unavailable"
            unavailable_reason = "expected_landmark_outside_safe_image_margin"
        elif candidate_rms > REPROJECTION_ALARM_THRESHOLD_PX:
            status = "drift_detected"
            unavailable_reason = None
        else:
            status = "healthy"
            unavailable_reason = None
        rms = candidate_rms if status != "unavailable" else None
        maximum = candidate_maximum if status != "unavailable" else None
        row = {
            "condition": condition_name,
            "render_camera_fovy_deg": actual_fovy,
            "render_camera_world_position_m": actual_position.tolist(),
            "renderer_camera_position_offset_m": position_error.tolist(),
            "render_camera_rotation_offset_deg": rotation_error.tolist(),
            "camera_view": self.view_name,
            "brightness_scale": float(brightness_scale),
            "image_noise_std_intensity": float(image_noise_std_intensity),
            "image_noise_seed": image_noise_seed,
            "blur_kernel_px": int(blur_kernel_px),
            "radial_distortion_k1": float(radial_distortion_k1),
            "radial_distortion_k2": float(radial_distortion_k2),
            "nominal_projection_calibration_used": True,
            "renderer_camera_perturbed": any(
                (
                    actual_fovy != self.nominal_fovy,
                    not np.allclose(actual_position, self.nominal_position),
                    not np.allclose(actual_rotation, self.nominal_rotation),
                )
            ),
            "expected_landmark_count": len(LANDMARKS),
            "safely_visible_expected_landmark_count": len(safely_visible_landmarks),
            "safely_visible_expected_landmarks": safely_visible_landmarks,
            "landmark_visibility_margin_px": margin,
            "detected_landmark_count": len(detections),
            "matched_landmark_count": len(matched) if matched is not None else 0,
            "landmarks": matched,
            "reprojection_rms_px": rms,
            "reprojection_max_px": maximum,
            "candidate_reprojection_rms_px": candidate_rms,
            "candidate_reprojection_max_px": candidate_maximum,
            "alarm_threshold_rms_px": REPROJECTION_ALARM_THRESHOLD_PX,
            "diagnostic_status": status,
            "diagnostic_unavailable_reason": unavailable_reason,
        }
        return rgb_image, row


def render_calibration_condition(condition_name: str) -> tuple[np.ndarray, dict]:
    """Render one named camera condition and return its RGB frame and row."""
    if condition_name not in CAMERA_HEALTH_CONDITIONS:
        raise ValueError(f"unknown camera health condition: {condition_name}")
    settings = CAMERA_HEALTH_CONDITIONS[condition_name]
    position_error = settings.get("position_error_m", (0.0, 0.0, 0.0))
    rotation_error = (0.0, 0.0, float(settings.get("yaw_error_deg", 0.0)))
    with CalibrationScene() as scene:
        return scene.render(
            condition_name=condition_name,
            fovy_error_deg=float(settings.get("fovy_error_deg", 0.0)),
            camera_position_error_m=position_error,
            rotation_error_deg=rotation_error,
            brightness_scale=float(settings.get("brightness_scale", 1.0)),
        )


def run_camera_calibration_health_benchmark(
    *, output_path: Path | None = None
) -> dict:
    """Measure fixed-landmark reprojection under renderer-side camera drift."""
    results = []
    for condition_name in CAMERA_HEALTH_CONDITIONS:
        _, row = render_calibration_condition(condition_name)
        results.append(row)
    report = {
        "benchmark": "fixed_landmark_camera_calibration_health",
        "protocol": {
            "scope": "standalone simulator diagnostic; no reaching controller is run",
            "landmarks": {
                name: list(position) for name, position in LANDMARKS.items()
            },
            "landmark_marker": "four fixed magenta tabletop cylinders",
            "condition_count": len(CAMERA_HEALTH_CONDITIONS),
            "expected_landmark_count": len(LANDMARKS),
            "camera_model": "pinhole projection from nominal camera pose and fovy",
            "perturbation_direction": (
                "rendered camera pose/intrinsics; the diagnostic always projects "
                "with nominal calibration"
            ),
            "alarm_rule": (
                "drift_detected when all four landmarks match and reprojection RMS "
                f"exceeds {REPROJECTION_ALARM_THRESHOLD_PX:.1f} px; missing/extra "
                "detections or expected centers outside the 12 px safe image "
                "margin are unavailable"
            ),
            "alarm_threshold_status": (
                "exploratory fixed threshold with margin over the deterministic "
                "nominal residual; not validated for a real camera"
            ),
            "image_size_px": [IMAGE_WIDTH, IMAGE_HEIGHT],
        },
        "results": results,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure fixed-landmark reprojection under camera drift."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/camera-calibration-health.json"),
    )
    args = parser.parse_args()
    report = run_camera_calibration_health_benchmark(output_path=args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
