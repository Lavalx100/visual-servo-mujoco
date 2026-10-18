import cv2
import numpy as np
import pytest

from visual_servo_mujoco.camera_distortion import (
    distort_pixel,
    distort_rgb_image,
    undistort_pixel,
)
from visual_servo_mujoco.run import detect_red_target


def test_distorted_pixel_round_trips_through_calibration_coefficients():
    ideal = (520.0, 112.0)
    observed = distort_pixel(
        ideal,
        (640, 480),
        vertical_fov_degrees=40.0,
        k1=0.10,
        k2=-0.01,
    )

    corrected = undistort_pixel(
        observed,
        (640, 480),
        vertical_fov_degrees=40.0,
        k1=0.10,
        k2=-0.01,
    )

    assert corrected == pytest.approx(ideal, abs=0.01)
    assert observed[0] > ideal[0]
    assert observed[1] < ideal[1]


def test_rendered_radial_warp_moves_off_center_target_and_correction_recovers_it():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(image, (520, 120), 12, (255, 0, 0), thickness=-1)

    distorted = distort_rgb_image(
        image,
        vertical_fov_degrees=40.0,
        k1=0.10,
    )
    measured = detect_red_target(distorted)
    corrected = undistort_pixel(
        measured,
        (640, 480),
        vertical_fov_degrees=40.0,
        k1=0.10,
    )

    assert measured[0] > 520
    assert measured[1] < 120
    assert corrected == pytest.approx((520, 120), abs=1.0)


def test_zero_distortion_leaves_image_and_pixel_coordinates_unchanged():
    image = np.zeros((16, 24, 3), dtype=np.uint8)

    assert distort_rgb_image(
        image,
        vertical_fov_degrees=40.0,
        k1=0.0,
        k2=0.0,
    ) is image
    assert undistort_pixel(
        (4.25, 8.5),
        (24, 16),
        vertical_fov_degrees=40.0,
        k1=0.0,
    ) == (4.25, 8.5)


def test_radial_distortion_helpers_reject_non_finite_inputs():
    with pytest.raises(ValueError, match="coefficients must be finite"):
        distort_pixel(
            (10.0, 10.0),
            (24, 16),
            vertical_fov_degrees=40.0,
            k1=float("nan"),
        )
    with pytest.raises(ValueError, match="pixel_xy must contain two finite"):
        undistort_pixel(
            (float("inf"), 0.0),
            (24, 16),
            vertical_fov_degrees=40.0,
            k1=0.1,
        )
