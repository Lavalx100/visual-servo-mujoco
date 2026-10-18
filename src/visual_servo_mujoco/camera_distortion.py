"""Brown-Conrady radial lens distortion helpers for rendered camera frames."""

from __future__ import annotations

from functools import lru_cache
import math

import cv2
import numpy as np


def _camera_matrix(
    image_size: tuple[int, int], vertical_fov_degrees: float
) -> np.ndarray:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not math.isfinite(vertical_fov_degrees) or not 0.0 < vertical_fov_degrees < 180.0:
        raise ValueError("vertical_fov_degrees must be finite and between 0 and 180")
    focal_px = height / (2.0 * math.tan(math.radians(vertical_fov_degrees) / 2.0))
    return np.asarray(
        (
            (focal_px, 0.0, (width - 1) / 2.0),
            (0.0, focal_px, (height - 1) / 2.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _validate_coefficients(k1: float, k2: float) -> tuple[float, float]:
    if not math.isfinite(k1) or not math.isfinite(k2):
        raise ValueError("radial distortion coefficients must be finite")
    return float(k1), float(k2)


@lru_cache(maxsize=32)
def _distortion_maps(
    width: int,
    height: int,
    vertical_fov_degrees: float,
    k1: float,
    k2: float,
) -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = _camera_matrix((width, height), vertical_fov_degrees)
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    distorted_pixels = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 1, 2)
    coefficients = np.asarray((k1, k2, 0.0, 0.0, 0.0), dtype=np.float64)
    ideal_pixels = cv2.undistortPoints(
        distorted_pixels, camera_matrix, coefficients, P=camera_matrix
    ).reshape(height, width, 2)
    map_x = ideal_pixels[:, :, 0].astype(np.float32)
    map_y = ideal_pixels[:, :, 1].astype(np.float32)
    map_x.setflags(write=False)
    map_y.setflags(write=False)
    return map_x, map_y


def distort_rgb_image(
    rgb_image: np.ndarray,
    *,
    vertical_fov_degrees: float,
    k1: float,
    k2: float = 0.0,
) -> np.ndarray:
    """Apply radial lens distortion to an ideal rendered RGB frame.

    The coefficients follow OpenCV's normalized Brown-Conrady convention.
    Passing zero coefficients returns the original image unchanged.
    """
    if (
        rgb_image.ndim != 3
        or rgb_image.shape[2] != 3
        or rgb_image.dtype != np.uint8
    ):
        raise ValueError("rgb_image must be a uint8 RGB image")
    k1, k2 = _validate_coefficients(k1, k2)
    if k1 == 0.0 and k2 == 0.0:
        return rgb_image
    height, width = rgb_image.shape[:2]
    map_x, map_y = _distortion_maps(
        width, height, float(vertical_fov_degrees), k1, k2
    )
    return cv2.remap(
        rgb_image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def undistort_pixel(
    pixel_xy: tuple[float, float],
    image_size: tuple[int, int],
    *,
    vertical_fov_degrees: float,
    k1: float,
    k2: float = 0.0,
) -> tuple[float, float]:
    """Map a measured distorted pixel back to ideal pinhole image coordinates."""
    k1, k2 = _validate_coefficients(k1, k2)
    pixel = np.asarray(pixel_xy, dtype=np.float64)
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        raise ValueError("pixel_xy must contain two finite coordinates")
    if k1 == 0.0 and k2 == 0.0:
        return float(pixel[0]), float(pixel[1])
    camera_matrix = _camera_matrix(image_size, vertical_fov_degrees)
    coefficients = np.asarray((k1, k2, 0.0, 0.0, 0.0), dtype=np.float64)
    ideal = cv2.undistortPoints(
        pixel.reshape(1, 1, 2), camera_matrix, coefficients, P=camera_matrix
    ).reshape(2)
    return float(ideal[0]), float(ideal[1])


def distort_pixel(
    pixel_xy: tuple[float, float],
    image_size: tuple[int, int],
    *,
    vertical_fov_degrees: float,
    k1: float,
    k2: float = 0.0,
) -> tuple[float, float]:
    """Map an ideal pinhole pixel to its radially distorted image location."""
    k1, k2 = _validate_coefficients(k1, k2)
    pixel = np.asarray(pixel_xy, dtype=np.float64)
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        raise ValueError("pixel_xy must contain two finite coordinates")
    if k1 == 0.0 and k2 == 0.0:
        return float(pixel[0]), float(pixel[1])
    camera_matrix = _camera_matrix(image_size, vertical_fov_degrees)
    focal_x, focal_y = camera_matrix[0, 0], camera_matrix[1, 1]
    center_x, center_y = camera_matrix[0, 2], camera_matrix[1, 2]
    normalized_x = (pixel[0] - center_x) / focal_x
    normalized_y = (pixel[1] - center_y) / focal_y
    radius_squared = normalized_x**2 + normalized_y**2
    radial_scale = 1.0 + k1 * radius_squared + k2 * radius_squared**2
    return (
        float(center_x + focal_x * normalized_x * radial_scale),
        float(center_y + focal_y * normalized_y * radial_scale),
    )
