import csv
import math

import pytest

from visual_servo_mujoco.run import run_trials
from visual_servo_mujoco.timestamped_verification_benchmark import (
    DeliveryProfile,
    MAX_SETTLE_S,
    PostStopCameraReplay,
    _write_summary_csv,
    evaluate_arrival_only_policy,
    evaluate_timestamp_guarded_policy,
    _motion_fraction,
)


def _trial(*, success=True):
    return {
        "seed": 1,
        "episode": 0,
        "controller_stop_time_simulated_s": 10.0,
        "success": success,
        "reaching_error_m": 0.02 if success else 0.08,
        "target_motion_profile": "sinusoidal",
        "target_motion_phase_rad": 0.0,
        "use_current_end_effector_state": False,
        "camera_observation_delay_observations": 1,
        "camera_observation_delay_simulated_s": 0.4,
    }


def _event(index, *, capture, arrival, estimate=0.02, physical_arrival=0.02, dropped=False):
    return {
        "frame_index": index,
        "capture_time_s": capture,
        "delivery_time_s": arrival,
        "scheduled_arrival_time_s": arrival,
        "camera_age_at_delivery_s": arrival - capture,
        "physical_error_at_capture_m": 0.02,
        "physical_error_at_delivery_m": physical_arrival,
        "estimated_distance_m": estimate,
        "visual_success_estimate": (
            estimate <= 0.045 if estimate is not None else None
        ),
        "both_markers_visible": estimate is not None,
        "transport_dropped": dropped,
    }


def test_arrival_only_ablation_can_accept_a_frame_captured_before_settling():
    trial = _trial()
    event = _event(0, capture=10.0, arrival=10.4)

    result = evaluate_arrival_only_policy(trial, [event])

    assert result["status"] == "visually_confirmed"
    assert result["decision_latency_after_stop_s"] == pytest.approx(0.4)


def test_timestamp_guard_requires_two_consecutive_post_settle_frames():
    trial = _trial()
    events = [
        _event(0, capture=10.0, arrival=10.4),  # pre-settle
        _event(1, capture=10.2, arrival=10.6),
        _event(2, capture=10.4, arrival=10.8),
    ]

    result = evaluate_timestamp_guarded_policy(
        trial,
        events,
        settled_time_s=10.1,
        max_frame_age_s=0.5,
    )

    assert result["status"] == "visually_confirmed"
    assert result["pre_settle_rejections"] == 1
    assert result["usable_frames"] == 2
    assert result["decision_latency_after_stop_s"] == pytest.approx(0.8)


def test_timestamp_guard_rejects_over_age_frames_and_returns_unverified():
    trial = _trial()
    events = [
        _event(0, capture=10.2, arrival=11.0),
        _event(1, capture=10.4, arrival=11.2),
    ]

    result = evaluate_timestamp_guarded_policy(
        trial,
        events,
        settled_time_s=10.1,
        max_frame_age_s=0.5,
    )

    assert result["status"] == "unverified"
    assert result["visual_confirmed"] is False
    assert result["stale_rejections"] == 2


def test_timestamp_guard_abstains_when_arm_never_settles():
    trial = _trial()
    events = [
        _event(0, capture=10.2, arrival=10.4),
        _event(1, capture=10.4, arrival=10.6),
    ]

    result = evaluate_timestamp_guarded_policy(
        trial,
        events,
        settled_time_s=None,
        max_frame_age_s=0.5,
    )

    assert result["status"] == "unverified"
    assert result["visual_confirmed"] is False
    assert result["pre_settle_rejections"] == 2


def test_dropped_capture_breaks_consecutive_frame_confirmation():
    trial = _trial()
    events = [
        _event(0, capture=10.2, arrival=10.4),
        _event(1, capture=10.4, arrival=10.6, dropped=True),
        _event(2, capture=10.6, arrival=10.8),
        _event(3, capture=10.8, arrival=11.0),
    ]

    result = evaluate_timestamp_guarded_policy(
        trial,
        events,
        settled_time_s=10.1,
        max_frame_age_s=0.5,
    )

    assert result["status"] == "visually_confirmed"
    assert result["dropped_frames"] == 1
    assert result["usable_frames"] == 3
    assert result["decision_latency_after_stop_s"] == pytest.approx(1.0)


def test_frames_lost_after_decision_deadline_do_not_count_as_drops():
    trial = _trial()
    event = _event(0, capture=12.5, arrival=13.0, dropped=True)

    result = evaluate_arrival_only_policy(trial, [event])

    assert result["status"] == "unverified"
    assert result["dropped_frames"] == 0


def test_out_of_order_delivery_is_rejected_before_confirmation():
    trial = _trial()
    events = [
        _event(1, capture=10.2, arrival=10.3),
        _event(0, capture=10.0, arrival=10.4),
        _event(2, capture=10.4, arrival=10.5),
        _event(3, capture=10.6, arrival=10.7),
    ]

    result = evaluate_timestamp_guarded_policy(
        trial,
        events,
        settled_time_s=10.1,
        max_frame_age_s=0.5,
    )

    assert result["status"] == "visually_confirmed"
    assert result["out_of_order_rejections"] == 1
    assert result["decision_latency_after_stop_s"] == pytest.approx(0.7)


def test_no_positive_estimate_does_not_become_visual_confirmation():
    trial = _trial()
    events = [
        _event(0, capture=10.2, arrival=10.4, estimate=0.08),
        _event(1, capture=10.4, arrival=10.6, estimate=0.08),
    ]

    result = evaluate_timestamp_guarded_policy(
        trial,
        events,
        settled_time_s=10.1,
        max_frame_age_s=0.5,
    )

    assert result["status"] == "not_confirmed"
    assert result["visual_confirmed"] is False
    assert result["decision_latency_after_stop_s"] == pytest.approx(0.6)
    assert result["physical_success_at_decision_capture"] is True


def test_trial_records_phase_and_initial_target_matches_it(tmp_path):
    phase = math.pi / 2.0
    report = run_trials(
        1,
        33,
        tmp_path,
        controller="image_feedback",
        target_motion_amplitude_m=0.04,
        target_motion_frequency_hz=0.25,
        target_motion_profile="sinusoidal",
        target_motion_phase_rad=phase,
        save_media=False,
    )
    trial = report["results"][0]
    initial_event = trial["target_visibility_trace"][0]
    target_start = trial["target_start_xy_m_for_evaluation_only"]

    assert trial["target_motion_phase_rad"] == pytest.approx(phase)
    assert initial_event["target_position_xy_m_for_evaluation_only"][0] == pytest.approx(
        target_start[0] + 0.04,
        abs=1e-5,
    )
    assert "final_joint_velocities_rad_s_for_evaluation_only" in trial
    assert "controller_stop_time_simulated_s" in trial


def test_trial_rejects_non_finite_motion_phase(tmp_path):
    with pytest.raises(ValueError, match="target_motion_phase_rad must be finite"):
        run_trials(
            1,
            1,
            tmp_path,
            controller="image_feedback",
            target_motion_phase_rad=float("nan"),
            save_media=False,
        )


def test_summary_csv_keeps_aggregate_rows_compact_and_serializes_phases(tmp_path):
    output = tmp_path / "summary.csv"
    _write_summary_csv(
        [
            {
                "target_motion_profile": "sinusoidal",
                "target_motion_phases_rad": [0.0, math.pi],
                "delivery_profile": "near_real_time",
                "policy": "two_post_settle_frames",
                "trials": 20,
                "per_seed": [{"seed": 1, "trials": 4}],
            }
        ],
        output,
    )

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 1
    assert rows[0]["target_motion_phases_rad"] == "[0.0,3.141592653589793]"
    assert rows[0]["trials"] == "20"
    assert "per_seed" not in rows[0]


def test_post_stop_replay_records_settling_and_capture_to_delivery_ages(tmp_path):
    report = run_trials(
        1,
        47,
        tmp_path / "reaching",
        controller="image_feedback",
        camera_observation_delay_observations=1,
        target_motion_amplitude_m=0.04,
        target_motion_profile="piecewise_linear",
        target_motion_phase_rad=math.pi,
        save_media=False,
    )
    trial = {"seed": 47, **report["results"][0]}
    replay = PostStopCameraReplay()
    try:
        stream = replay.replay(
            trial,
            delivery_profiles=(DeliveryProfile("test", 0.2, 0.0, 0.0),),
            seed=47,
        )
    finally:
        replay.close()

    events = stream["events_by_profile"]["test"]
    delivered = [event for event in events if not event["transport_dropped"]]

    assert stream["settled_time_s"] is not None
    assert stream["settle_duration_s"] <= MAX_SETTLE_S
    assert delivered
    assert all(event["delivery_time_s"] >= event["capture_time_s"] for event in delivered)
    assert all(event["physical_error_at_capture_m"] is not None for event in delivered)
    assert all(event["physical_error_at_delivery_m"] is not None for event in delivered)
    assert any(not event["capture_after_settle"] for event in delivered)
    assert any(event["capture_after_settle"] for event in delivered)


def test_motion_fraction_uses_requested_frequency():
    assert _motion_fraction(1.0, "sinusoidal", 0.0, 0.25) == pytest.approx(1.0)
    assert _motion_fraction(1.0, "sinusoidal", 0.0, 0.5) == pytest.approx(0.0)


def test_post_stop_replay_starts_at_the_recorded_half_hz_target_position(tmp_path):
    phase = math.pi / 2.0
    report = run_trials(
        1,
        950,
        tmp_path / "half-hz-reaching",
        controller="image_feedback",
        target_motion_amplitude_m=0.04,
        target_motion_frequency_hz=0.5,
        target_motion_profile="sinusoidal",
        target_motion_phase_rad=phase,
        save_media=False,
    )
    trial = {"seed": 950, **report["results"][0]}
    replay = PostStopCameraReplay()
    try:
        stream = replay.replay(
            trial,
            delivery_profiles=(DeliveryProfile("test", 0.0, 0.0, 0.0),),
            seed=950,
        )
    finally:
        replay.close()

    first_capture = stream["events_by_profile"]["test"][0]
    expected_error = math.dist(
        trial["controller_stop_target_xy_m_for_evaluation_only"],
        trial["end_effector_xy_m"],
    )
    assert first_capture["physical_error_at_capture_m"] == pytest.approx(
        expected_error, abs=0.001
    )
