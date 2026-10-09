import cv2
import numpy as np
import pytest

from visual_servo_mujoco.benchmark import CONDITIONS, CONTROLLERS, run_benchmark
from visual_servo_mujoco.controller import (
    camera_pixel_jacobian,
    forward_kinematics,
    image_servo_joint_delta,
    inverse_kinematics,
    pixel_to_table_xy,
)
from visual_servo_mujoco.model import MODEL_XML
from visual_servo_mujoco.run import detect_green_end_effector, detect_red_target, run_trials


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


def test_red_target_detector_estimates_center_of_circular_component():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    cv2.circle(image, (72, 31), 9, (255, 0, 0), thickness=-1)

    assert detect_red_target(image) == pytest.approx((72, 31), abs=0.5)


def test_red_target_detector_estimates_center_from_partially_occluded_circle():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    cv2.circle(image, (72, 31), 9, (255, 0, 0), thickness=-1)
    image[:, :72] = 0

    assert detect_red_target(image) == pytest.approx((72, 31), abs=0.5)


def test_red_target_detector_reports_missing_target():
    with pytest.raises(RuntimeError, match="contains no red target"):
        detect_red_target(np.zeros((100, 120, 3), dtype=np.uint8))


def test_green_end_effector_detector_estimates_center_of_visible_marker():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    cv2.circle(image, (48, 63), 7, (0, 255, 0), thickness=-1)
    image[:, :48] = 0

    assert detect_green_end_effector(image) == pytest.approx((48, 63), abs=0.5)


def test_image_servo_step_moves_in_the_requested_pixel_direction():
    jacobian = camera_pixel_jacobian(
        0.2,
        0.7,
        (640, 480),
        camera_height=2.0,
        end_effector_height=0.21,
        vertical_fov_degrees=40.0,
    )
    pixel_error = np.array((24.0, -12.0))
    joint_delta = image_servo_joint_delta(jacobian, tuple(pixel_error))

    assert np.dot(jacobian @ joint_delta, pixel_error) > 0.0
    assert np.linalg.norm(joint_delta) <= 0.2


def test_mujoco_scene_compiles_with_camera_and_arm():
    import mujoco

    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector") >= 0


def test_robustness_benchmark_is_reproducible_and_pairs_trials():
    report = run_benchmark(seeds=1, episodes_per_seed=2, base_seed=13)
    replay = run_benchmark(seeds=1, episodes_per_seed=2, base_seed=13)
    assert report == replay
    assert len(report["conditions"]) == len(CONDITIONS) * len(CONTROLLERS)
    assert report["target_sequences_are_paired_across_controllers_and_conditions"] is True

    clean = next(
        condition
        for condition in report["conditions"]
        if condition["controller"] == "open_loop"
        and condition["condition"]["name"] == "clean"
    )
    noisy = next(
        condition
        for condition in report["conditions"]
        if condition["controller"] == "open_loop"
        and condition["condition"]["name"] == "pixel_noise_8px"
    )
    assert [trial["true_target_xy_m_for_evaluation_only"] for trial in clean["trial_results"]] == [
        trial["true_target_xy_m_for_evaluation_only"] for trial in noisy["trial_results"]
    ]
    assert clean["trial_results"][0]["controller_pixel_xy"] != noisy["trial_results"][0][
        "controller_pixel_xy"
    ]

    feedback = next(
        condition
        for condition in report["conditions"]
        if condition["controller"] == "image_feedback"
        and condition["condition"]["name"] == "clean"
    )
    assert [trial["true_target_xy_m_for_evaluation_only"] for trial in clean["trial_results"]] == [
        trial["true_target_xy_m_for_evaluation_only"] for trial in feedback["trial_results"]
    ]


def test_unreachable_camera_estimate_is_recorded_as_trial_failure(tmp_path):
    report = run_trials(
        episodes=2,
        seed=13,
        output_dir=tmp_path,
        camera_fovy_error_deg=10.0,
        save_media=False,
    )

    assert report["successes"] == 0
    assert all(
        trial["failure_reason"] == "estimated_target_unreachable"
        for trial in report["results"]
    )
