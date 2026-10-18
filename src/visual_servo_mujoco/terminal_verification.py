"""Calibrated oblique camera geometry for optional terminal visual checks."""

from __future__ import annotations

import math

import cv2
import mujoco
import numpy as np

from .model import MODEL_XML

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
VERIFICATION_VIEWS = {
    "east": {"position_m": (1.0, 0.0, 0.85), "fovy_deg": 65.0},
    "west": {"position_m": (-1.0, 0.0, 0.85), "fovy_deg": 65.0},
    "north": {"position_m": (0.0, 1.0, 0.85), "fovy_deg": 65.0},
    "south": {"position_m": (0.0, -1.0, 0.85), "fovy_deg": 65.0},
}


def model_with_verification_views(
    view_names: tuple[str, ...] = tuple(VERIFICATION_VIEWS),
) -> mujoco.MjModel:
    """Build the project scene with the requested calibrated oblique cameras."""
    unknown_views = set(view_names) - VERIFICATION_VIEWS.keys()
    if unknown_views:
        raise ValueError(f"unknown verification view(s): {', '.join(sorted(unknown_views))}")
    cameras = "\n".join(
        '<camera name="verification_{name}" mode="targetbody" target="arm_base" '
        'pos="{x:.3f} {y:.3f} {z:.3f}" fovy="{fovy:.1f}"/>'.format(
            name=name,
            x=VERIFICATION_VIEWS[name]["position_m"][0],
            y=VERIFICATION_VIEWS[name]["position_m"][1],
            z=VERIFICATION_VIEWS[name]["position_m"][2],
            fovy=VERIFICATION_VIEWS[name]["fovy_deg"],
        )
        for name in view_names
    )
    xml = MODEL_XML.replace("</worldbody>", f"{cameras}\n</worldbody>")
    return mujoco.MjModel.from_xml_string(xml)


def pixel_to_xy_at_height(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_id: int,
    pixel_xy: tuple[float, float],
    plane_height_m: float,
    *,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
    fovy_error_deg: float = 0.0,
    camera_position_error_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    camera_rotation_error_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Back-project a pixel using nominal calibration plus estimator-only offsets.

    Position errors are added in world coordinates. Rotation errors are
    roll/pitch/yaw about the calibrated camera frame and do not alter rendering.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not math.isfinite(plane_height_m):
        raise ValueError("plane_height_m must be finite")
    if not math.isfinite(fovy_error_deg):
        raise ValueError("fovy_error_deg must be finite")
    position_error = np.asarray(camera_position_error_m, dtype=float)
    rotation_error_deg = np.asarray(camera_rotation_error_deg, dtype=float)
    if position_error.shape != (3,) or not np.isfinite(position_error).all():
        raise ValueError("camera_position_error_m must contain three finite values")
    if rotation_error_deg.shape != (3,) or not np.isfinite(rotation_error_deg).all():
        raise ValueError("camera_rotation_error_deg must contain three finite values")
    estimated_fovy_deg = float(model.cam_fovy[camera_id]) + fovy_error_deg
    if not 0.0 < estimated_fovy_deg < 180.0:
        raise ValueError("estimated camera field of view must be between 0 and 180 degrees")
    focal_px = image_height / (
        2.0 * math.tan(math.radians(estimated_fovy_deg) / 2.0)
    )
    ray_camera = np.asarray(
        (
            (pixel_xy[0] - image_width / 2.0) / focal_px,
            -(pixel_xy[1] - image_height / 2.0) / focal_px,
            -1.0,
        ),
        dtype=float,
    )
    rotation_camera_to_world = data.cam_xmat[camera_id].reshape(3, 3)
    roll, pitch, yaw = np.radians(rotation_error_deg)
    rotation_x = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, math.cos(roll), -math.sin(roll)),
         (0.0, math.sin(roll), math.cos(roll)))
    )
    rotation_y = np.asarray(
        ((math.cos(pitch), 0.0, math.sin(pitch)), (0.0, 1.0, 0.0),
         (-math.sin(pitch), 0.0, math.cos(pitch)))
    )
    rotation_z = np.asarray(
        ((math.cos(yaw), -math.sin(yaw), 0.0),
         (math.sin(yaw), math.cos(yaw), 0.0), (0.0, 0.0, 1.0))
    )
    rotation_error = rotation_z @ rotation_y @ rotation_x
    assumed_camera_to_world = rotation_camera_to_world @ rotation_error
    camera_origin = data.cam_xpos[camera_id] + position_error
    ray_world = assumed_camera_to_world @ ray_camera
    if abs(float(ray_world[2])) < 1e-9:
        raise ValueError("verification camera ray is parallel to the target plane")
    ray_scale = (plane_height_m - float(camera_origin[2])) / float(ray_world[2])
    if ray_scale <= 0.0:
        raise ValueError("verification camera ray does not intersect the visible plane")
    return camera_origin[:2] + ray_scale * ray_world[:2]


def apply_verification_image_effects(
    rgb_image: np.ndarray,
    *,
    lighting_scale: float = 1.0,
    occluded_markers: tuple[str, ...] = (),
) -> np.ndarray:
    """Apply verifier-only brightness and marker-occlusion stress to an RGB frame."""
    if not math.isfinite(lighting_scale) or lighting_scale <= 0.0:
        raise ValueError("lighting_scale must be finite and positive")
    unknown_markers = set(occluded_markers) - {"target", "end_effector"}
    if unknown_markers:
        raise ValueError(f"unknown marker(s) to occlude: {', '.join(sorted(unknown_markers))}")
    image = np.asarray(rgb_image, dtype=np.uint8).copy()
    if not occluded_markers and lighting_scale == 1.0:
        return image

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    occlusion_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if "target" in occluded_markers:
        red_low = cv2.inRange(hsv, np.array([0, 110, 70]), np.array([12, 255, 255]))
        red_high = cv2.inRange(hsv, np.array([168, 110, 70]), np.array([180, 255, 255]))
        occlusion_mask = cv2.bitwise_or(occlusion_mask, cv2.bitwise_or(red_low, red_high))
    if "end_effector" in occluded_markers:
        green = cv2.inRange(hsv, np.array([42, 90, 55]), np.array([92, 255, 255]))
        occlusion_mask = cv2.bitwise_or(occlusion_mask, green)
    if np.any(occlusion_mask):
        occlusion_mask = cv2.dilate(
            occlusion_mask, np.ones((9, 9), dtype=np.uint8), iterations=2
        )
        image = cv2.inpaint(image, occlusion_mask, 5, cv2.INPAINT_TELEA)
    if lighting_scale != 1.0:
        image = np.clip(image.astype(np.float32) * lighting_scale, 0, 255).astype(np.uint8)
    return image


def terminal_verification_condition_label(
    *,
    fovy_error_deg: float = 0.0,
    camera_position_error_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    camera_rotation_error_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    lighting_scale: float = 1.0,
    occluded_markers: tuple[str, ...] = (),
) -> str:
    """Return a compact human-readable label for a verifier stress condition."""
    labels = []
    if fovy_error_deg:
        labels.append(f"FOV {fovy_error_deg:+g} deg")
    if any(camera_position_error_m):
        labels.append(
            "camera position error "
            + ",".join(f"{value * 100:+g} cm" for value in camera_position_error_m)
        )
    if any(camera_rotation_error_deg):
        labels.append(
            "camera rotation error "
            + ",".join(f"{value:+g} deg" for value in camera_rotation_error_deg)
        )
    if lighting_scale != 1.0:
        labels.append(f"exposure x{lighting_scale:g}")
    if occluded_markers:
        labels.append("occluded " + "+".join(occluded_markers))
    return "; ".join(labels) if labels else "nominal"
