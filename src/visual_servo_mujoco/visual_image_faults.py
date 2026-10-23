"""Image-only fault injection for fresh side-camera verifier experiments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from .cooperative_pick_visual_verification import detect_verification_marker


def validate_visual_image_fault(
    fault: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate and copy a supported image-only fault description."""
    if fault is None:
        return None
    result = dict(fault)
    if "label" in result and not isinstance(result["label"], str):
        raise ValueError("image fault label must be a string")
    kind = result.get("kind")
    if kind == "exposure_scale":
        scale = result.get("scale")
        if not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("exposure scale must be finite and positive")
        result["scale"] = float(scale)
    elif kind == "marker_occlusion":
        fraction = result.get("occluded_fraction")
        if (
            not isinstance(fraction, (int, float))
            or isinstance(fraction, bool)
            or not np.isfinite(fraction)
            or not 0 < fraction < 1
        ):
            raise ValueError("marker occlusion fraction must be between zero and one")
        result["occluded_fraction"] = float(fraction)
        shape = result.get("shape", "horizontal_band")
        if shape not in {"horizontal_band", "vertical_band", "diagonal_band", "corner_patch"}:
            raise ValueError("marker occlusion shape must be a supported band or corner patch")
        result["shape"] = shape
        for axis in ("offset_x_fraction", "offset_y_fraction"):
            value = result.get(axis, 0.0)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(value)
                or not -0.5 <= value <= 0.5
            ):
                raise ValueError("marker occlusion offsets must be finite fractions from -0.5 to 0.5")
            result[axis] = float(value)
        angle = result.get("angle_deg", 35.0)
        if (
            not isinstance(angle, (int, float))
            or isinstance(angle, bool)
            or not np.isfinite(angle)
            or not -80.0 <= angle <= 80.0
        ):
            raise ValueError("marker occlusion angle must be finite and between -80 and 80 degrees")
        result["angle_deg"] = float(angle)
    elif kind == "horizontal_motion_blur":
        size = result.get("kernel_size")
        if not isinstance(size, int) or size < 3 or size % 2 == 0:
            raise ValueError("motion blur kernel size must be an odd integer of at least 3")
    elif kind == "camera_translation":
        for axis in ("dx_px", "dy_px"):
            value = result.get(axis)
            if not isinstance(value, int) or abs(value) > 100:
                raise ValueError("camera translation must use integer shifts within 100 px")
    else:
        raise ValueError(f"unsupported visual image fault: {kind!r}")
    return result


def apply_visual_image_fault(
    rgb_image: np.ndarray,
    fault: Mapping[str, Any] | None,
    *,
    is_initial_frame: bool,
) -> np.ndarray:
    """Return a faulted RGB observation without changing the source image.

    Camera translation is applied only to fresh frames to model camera motion
    after the initial planning observation. Other faults affect all verifier
    frames consistently.
    """
    image = np.asarray(rgb_image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb_image must be an HxWx3 uint8 RGB array")
    spec = validate_visual_image_fault(fault)
    if spec is None:
        return image.copy()

    kind = spec["kind"]
    if kind == "exposure_scale":
        return np.clip(image.astype(np.float32) * spec["scale"], 0, 255).astype(np.uint8)
    if kind == "marker_occlusion":
        result = image.copy()
        marker = detect_verification_marker(result)
        if marker is None:
            return result
        x, y, width, height = marker["bbox_xywh_px"]
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        center_x = round((width - 1) / 2 + spec["offset_x_fraction"] * width)
        center_y = round((height - 1) / 2 + spec["offset_y_fraction"] * height)
        fraction = spec["occluded_fraction"]
        shape = spec["shape"]
        if shape == "horizontal_band":
            thickness = max(1, round(height * fraction))
            cv2.rectangle(
                roi_mask,
                (0, center_y - thickness // 2),
                (width - 1, center_y + (thickness - 1) // 2),
                255,
                -1,
            )
        elif shape == "vertical_band":
            thickness = max(1, round(width * fraction))
            cv2.rectangle(
                roi_mask,
                (center_x - thickness // 2, 0),
                (center_x + (thickness - 1) // 2, height - 1),
                255,
                -1,
            )
        elif shape == "diagonal_band":
            thickness = max(1, round(min(width, height) * fraction))
            length = int(np.ceil(np.hypot(width, height)))
            corners = cv2.boxPoints(
                (
                    (float(center_x), float(center_y)),
                    (float(length), float(thickness)),
                    float(spec["angle_deg"]),
                )
            ).round().astype(np.int32)
            cv2.fillPoly(roi_mask, [corners], 255)
        else:
            patch_width = max(1, round(width * fraction))
            patch_height = max(1, round(height * fraction))
            cv2.rectangle(
                roi_mask,
                (center_x - patch_width // 2, center_y - patch_height // 2),
                (
                    center_x + (patch_width - 1) // 2,
                    center_y + (patch_height - 1) // 2,
                ),
                255,
                -1,
            )
        region = result[y : y + height, x : x + width]
        region[roi_mask != 0] = (96, 96, 96)
        return result
    if kind == "horizontal_motion_blur":
        size = spec["kernel_size"]
        kernel = np.zeros((1, size), dtype=np.float32)
        kernel[0, :] = 1.0 / size
        return cv2.filter2D(image, -1, kernel)
    if kind == "camera_translation":
        result = image.copy()
        if is_initial_frame:
            return result
        dx_px = spec["dx_px"]
        dy_px = spec["dy_px"]
        matrix = np.asarray(((1.0, 0.0, dx_px), (0.0, 1.0, dy_px)), dtype=np.float32)
        return cv2.warpAffine(
            result,
            matrix,
            (result.shape[1], result.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(32, 32, 32),
        )
    raise AssertionError("validated fault kind was not handled")
