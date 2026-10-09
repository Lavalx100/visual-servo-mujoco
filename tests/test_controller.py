import cv2
import numpy as np
import pytest

from visual_servo_mujoco.benchmark import (
    CONDITIONS,
    CONTROLLERS,
    TARGET_DROPOUT_CONDITIONS,
    run_benchmark,
)
from visual_servo_mujoco.controller import (
    camera_pixel_jacobian,
    forward_kinematics,
    image_servo_joint_delta,
    inverse_kinematics,
    median_pixel_estimate,
    pixel_to_table_xy,
)
from visual_servo_mujoco.dynamic_benchmark import run_dynamic_benchmark
from visual_servo_mujoco.model import MODEL_XML
from visual_servo_mujoco.run import (
    FEEDBACK_MAX_ITERATIONS,
    detect_green_end_effector,
    detect_red_target,
    run_trials,
)


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


def test_median_pixel_estimate_rejects_a_single_noisy_outlier():
    estimate = median_pixel_estimate([(100.0, 200.0), (102.0, 199.0), (900.0, -500.0)])

    assert estimate == pytest.approx((102.0, 199.0))


def test_median_pixel_estimate_requires_finite_coordinate_pairs():
    with pytest.raises(ValueError, match="at least one"):
        median_pixel_estimate([])
    with pytest.raises(ValueError, match="pairs"):
        median_pixel_estimate([(1.0, 2.0, 3.0)])
    with pytest.raises(ValueError, match="finite"):
        median_pixel_estimate([(float("nan"), 0.0)])


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
    assert report["feedback_target_filter"] == {
        "method": "rolling_coordinate_median",
        "window": 3,
    }
    assert report["feedback_required_tolerance_checks"] == 3
    assert report["feedback_max_joint_step_radians"] == 0.3
    assert len(report["conditions"]) == (
        len(CONDITIONS) * len(CONTROLLERS) + len(TARGET_DROPOUT_CONDITIONS)
    )
    assert report["target_sequences_are_paired_across_controllers_and_conditions"] is True

    dropout_timeout = next(
        condition
        for condition in report["conditions"]
        if condition["controller"] == "image_feedback"
        and condition["condition"]["name"] == "target_dropout_6_observations"
    )
    assert dropout_timeout["injected_dropout_trials"] == 2
    assert dropout_timeout["injected_dropout_recovery_trials"] == 0
    assert dropout_timeout["target_visibility_timeout_trials"] == 2

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


def test_benchmark_records_selected_feedback_filter_window():
    report = run_benchmark(
        seeds=1,
        episodes_per_seed=1,
        feedback_target_filter_window=5,
        feedback_required_tolerance_checks=2,
        feedback_max_joint_step_radians=0.25,
        feedback_allow_stale_target_confirmation=True,
    )

    assert report["feedback_target_filter"]["window"] == 5
    assert report["feedback_required_tolerance_checks"] == 2
    assert report["feedback_max_joint_step_radians"] == 0.25
    assert report["feedback_allow_stale_target_confirmation"] is True


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


def test_feedback_filter_window_can_be_selected_and_is_recorded(tmp_path):
    report = run_trials(
        episodes=1,
        seed=13,
        output_dir=tmp_path,
        controller="image_feedback",
        feedback_target_filter_window=5,
        feedback_required_tolerance_checks=2,
        feedback_max_joint_step_radians=0.25,
        save_media=False,
    )

    assert report["feedback_target_filter"] == {
        "method": "rolling_coordinate_median",
        "window": 5,
    }
    assert report["feedback_required_tolerance_checks"] == 2
    assert report["feedback_max_joint_step_radians"] == 0.25


def test_feedback_filter_window_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        run_trials(
            episodes=1,
            seed=13,
            output_dir=tmp_path,
            feedback_target_filter_window=0,
            save_media=False,
        )


def test_feedback_tolerance_confirmation_count_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        run_trials(
            episodes=1,
            seed=13,
            output_dir=tmp_path,
            feedback_required_tolerance_checks=0,
            save_media=False,
        )


def test_feedback_joint_step_cap_must_be_positive_and_finite(tmp_path):
    with pytest.raises(ValueError, match="positive finite"):
        run_trials(
            episodes=1,
            seed=13,
            output_dir=tmp_path,
            feedback_max_joint_step_radians=float("inf"),
            save_media=False,
        )


def test_feedback_reacquires_target_after_temporary_dropout(tmp_path):
    report = run_trials(
        episodes=1,
        seed=7,
        output_dir=tmp_path,
        controller="image_feedback",
        target_dropout_start_observation=1,
        target_dropout_duration_observations=3,
        save_media=False,
    )

    trial = report["results"][0]
    assert trial["target_reacquisitions"] >= 1
    assert trial["injected_dropout_reacquired"] is True
    injected_events = [
        event for event in trial["target_visibility_trace"] if event["injected_dropout"]
    ]
    assert len(injected_events) == 3
    assert all(event["target_visible_in_render"] for event in injected_events)
    assert all(not event["target_measurement_used"] for event in injected_events)
    assert trial["injected_dropout_visible_observations"] == 3
    assert trial["natural_target_visibility_loss_outside_dropout_observations"] > 0
    assert trial["target_visibility_stop_cause"] == "natural_target_visibility_loss"


def test_feedback_stops_after_dropout_exceeds_visibility_grace(tmp_path):
    report = run_trials(
        episodes=1,
        seed=7,
        output_dir=tmp_path,
        controller="image_feedback",
        target_dropout_start_observation=1,
        target_dropout_duration_observations=6,
        save_media=False,
    )

    trial = report["results"][0]
    assert trial["target_reacquisitions"] == 0
    assert trial["injected_dropout_reacquired"] is False
    assert trial["controller_stop_reason"] == "target_not_visible"
    assert trial["target_visibility_stop_cause"] == "injected_detector_dropout"
    assert len(trial["target_visibility_trace"]) == 7


def test_stale_target_confirmation_is_opt_in_for_natural_visibility_loss(tmp_path):
    strict_report = run_trials(
        episodes=1,
        seed=7,
        output_dir=tmp_path / "strict",
        controller="image_feedback",
        save_media=False,
    )
    experimental_report = run_trials(
        episodes=1,
        seed=7,
        output_dir=tmp_path / "experimental",
        controller="image_feedback",
        feedback_allow_stale_target_confirmation=True,
        save_media=False,
    )

    strict_trial = strict_report["results"][0]
    experimental_trial = experimental_report["results"][0]
    assert strict_trial["target_visibility_stop_cause"] == "natural_target_visibility_loss"
    assert experimental_trial["controller_stop_reason"] == "stale_target_tolerance_reached"
    assert experimental_trial["feedback_allow_stale_target_confirmation"] is True
    assert experimental_trial["success"] is True
    assert max(
        event.get("stale_tolerance_checks", 0)
        for event in experimental_trial["target_visibility_trace"]
    ) == 3


def test_moving_target_and_delayed_camera_trace_are_recorded(tmp_path):
    report = run_trials(
        episodes=1,
        seed=7,
        output_dir=tmp_path,
        controller="image_feedback",
        pixel_noise_std_px=2.0,
        target_motion_amplitude_m=0.04,
        target_motion_frequency_hz=0.25,
        camera_observation_delay_observations=2,
        save_media=False,
    )

    trial = report["results"][0]
    trace = trial["target_visibility_trace"]
    assert report["target_motion_amplitude_m"] == 0.04
    assert report["camera_observation_delay_observations"] == 2
    assert report["camera_observation_delay_simulated_s"] == 0.8
    assert max(event["camera_observation_age_observations"] for event in trace) == 2
    assert max(event["camera_observation_age_simulated_s"] for event in trace) == 0.8
    assert any(
        abs(
            event["target_position_xy_m_for_evaluation_only"][0]
            - trial["target_start_xy_m_for_evaluation_only"][0]
        )
        > 0.001
        for event in trace
    )
    assert any(
        event["target_position_xy_m_for_evaluation_only"]
        != event["observed_target_position_xy_m_for_evaluation_only"]
        for event in trace
    )
    assert trace[0]["target_pixel_xy_used"] == trace[1]["target_pixel_xy_used"]
    assert trace[1]["target_pixel_xy_used"] == trace[2]["target_pixel_xy_used"]


def test_camera_observation_delay_requires_image_feedback(tmp_path):
    with pytest.raises(ValueError, match="only be used with image_feedback"):
        run_trials(
            episodes=1,
            seed=7,
            output_dir=tmp_path,
            controller="open_loop",
            camera_observation_delay_observations=1,
            save_media=False,
        )


def test_dynamic_benchmark_pairs_policy_runs_and_reports_conditions(tmp_path):
    report = run_dynamic_benchmark(
        seeds=1,
        episodes_per_seed=1,
        output_path=tmp_path / "dynamic.json",
    )

    assert report["benchmark"] == "dynamic_target_camera_latency"
    assert len(report["conditions"]) == 10
    moving_delay_zero = next(
        condition
        for condition in report["conditions"]
        if condition["policy"] == "strict_fresh_vision"
        and condition["condition"]["name"] == "moving_delay_0"
    )
    moving_delay_two = next(
        condition
        for condition in report["conditions"]
        if condition["policy"] == "strict_fresh_vision"
        and condition["condition"]["name"] == "moving_delay_2"
    )
    assert moving_delay_zero["trial_results"][0]["target_start_xy_m_for_evaluation_only"] == (
        moving_delay_two["trial_results"][0]["target_start_xy_m_for_evaluation_only"]
    )
    assert (tmp_path / "dynamic.json").is_file()


def test_target_dropout_requires_valid_start_and_feedback_controller(tmp_path):
    with pytest.raises(ValueError, match="start observation is required"):
        run_trials(
            episodes=1,
            seed=7,
            output_dir=tmp_path,
            controller="image_feedback",
            target_dropout_duration_observations=1,
            save_media=False,
        )
    with pytest.raises(ValueError, match="only be used with image_feedback"):
        run_trials(
            episodes=1,
            seed=7,
            output_dir=tmp_path,
            controller="open_loop",
            target_dropout_start_observation=1,
            target_dropout_duration_observations=1,
            save_media=False,
        )


def test_feedback_confirmation_keeps_motion_update_budget_bounded(tmp_path):
    report = run_trials(
        episodes=2,
        seed=13,
        output_dir=tmp_path,
        pixel_noise_std_px=20.0,
        controller="image_feedback",
        save_media=False,
    )

    assert all(
        trial["controller_iterations"] <= FEEDBACK_MAX_ITERATIONS
        for trial in report["results"]
    )


def test_video_episode_index_must_select_a_requested_episode(tmp_path):
    with pytest.raises(ValueError, match="video_episode_index"):
        run_trials(
            episodes=1,
            seed=13,
            output_dir=tmp_path,
            video_episode_index=1,
            save_media=False,
        )
