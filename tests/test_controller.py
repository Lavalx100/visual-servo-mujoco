import cv2
import numpy as np
import pytest

from visual_servo_mujoco.controller import forward_kinematics, inverse_kinematics, pixel_to_table_xy
from visual_servo_mujoco.model import MODEL_XML
from visual_servo_mujoco.run import detect_red_target


@pytest.mark.parametrize("target", [(0.5, 0.1), (0.35, -0.2), (0.62, 0.0)])
def test_inverse_kinematics_reaches_requested_xy(target):
    angles = inverse_kinematics(*target)
    actual = forward_kinematics(*angles)
    assert actual == pytest.approx(target, abs=1e-9)


def test_inverse_kinematics_rejects_unreachable_target():
    with pytest.raises(ValueError, match="outside arm reach"):
        inverse_kinematics(1.0, 0.0)


def test_center_pixel_maps_to_center_of_camera_view():
    point = pixel_to_table_xy(
        (319.5, 239.5),
        (640, 480),
        camera_height=2.0,
        target_height=0.165,
        vertical_fov_degrees=40.0,
    )
    assert point == pytest.approx((0.0, 0.0), abs=1e-12)


def test_pixel_projection_has_expected_scale_and_vertical_direction():
    center = pixel_to_table_xy(
        (319.5, 239.5), (640, 480), camera_height=2.0, target_height=0.165,
        vertical_fov_degrees=40.0,
    )
    right = pixel_to_table_xy(
        (419.5, 239.5), (640, 480), camera_height=2.0, target_height=0.165,
        vertical_fov_degrees=40.0,
    )
    above = pixel_to_table_xy(
        (319.5, 139.5), (640, 480), camera_height=2.0, target_height=0.165,
        vertical_fov_degrees=40.0,
    )
    assert right[0] > center[0]
    assert above[1] > center[1]


def test_red_target_detector_returns_component_centroid():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    cv2.circle(image, (72, 31), 9, (255, 0, 0), thickness=-1)

    assert detect_red_target(image) == pytest.approx((72, 31), abs=0.5)


def test_red_target_detector_reports_missing_target():
    with pytest.raises(RuntimeError, match="contains no red target"):
        detect_red_target(np.zeros((100, 120, 3), dtype=np.uint8))


def test_mujoco_scene_compiles_with_camera_and_arm():
    import mujoco

    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector") >= 0
