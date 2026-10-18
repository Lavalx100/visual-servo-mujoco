import math

import pytest

from visual_servo_mujoco.motion_envelope_benchmark import (
    _calibrate_margin,
    _envelope_diagnostics,
    _write_figure,
    evaluate_motion_envelope_policy,
)
from visual_servo_mujoco.timestamped_verification_benchmark import (
    SUCCESS_THRESHOLD_METERS,
)


def _trial():
    return {
        "seed": 963,
        "episode": 0,
        "controller_stop_time_simulated_s": 10.0,
        "success": True,
        "reaching_error_m": 0.02,
    }


def _event(index, *, age=0.2, estimate=0.02, capture_error=0.02, delivery_error=0.02):
    capture = 10.2 + 0.2 * index
    arrival = capture + age
    return {
        "frame_index": index,
        "capture_time_s": capture,
        "delivery_time_s": arrival,
        "scheduled_arrival_time_s": arrival,
        "camera_age_at_delivery_s": age,
        "transport_dropped": False,
        "both_markers_visible": estimate is not None,
        "estimated_distance_m": estimate,
        "visual_success_estimate": (
            estimate <= SUCCESS_THRESHOLD_METERS if estimate is not None else None
        ),
        "physical_error_at_capture_m": capture_error,
        "physical_error_at_delivery_m": delivery_error,
    }


def test_envelope_uses_image_distance_margin_and_age_bound_only_for_decision():
    trial = _trial()
    events = [
        _event(0, capture_error=0.05, delivery_error=0.06),
        _event(1, capture_error=0.05, delivery_error=0.06),
    ]

    result = evaluate_motion_envelope_policy(
        trial,
        events,
        settled_time_s=10.1,
        calibrated_margin_m=0.01,
        max_target_speed_m_s=0.05,
    )

    assert result["status"] == "visually_confirmed"
    assert result["false_confirmation_at_delivery"] is True
    assert result["estimated_upper_distance_at_decision_m"] == pytest.approx(0.04)
    assert result["image_margin_rejections"] == 0
    assert result["motion_margin_rejections"] == 0

    # Changing simulator-only labels changes the scored outcome, not the rule.
    close_events = [
        {**event, "physical_error_at_capture_m": 0.01, "physical_error_at_delivery_m": 0.01}
        for event in events
    ]
    close_result = evaluate_motion_envelope_policy(
        trial,
        close_events,
        settled_time_s=10.1,
        calibrated_margin_m=0.01,
        max_target_speed_m_s=0.05,
    )
    assert close_result["status"] == result["status"]
    assert close_result["estimated_upper_distance_at_decision_m"] == pytest.approx(
        result["estimated_upper_distance_at_decision_m"]
    )
    assert close_result["false_confirmation_at_delivery"] is False


def test_motion_margin_rejects_two_frames_that_exceed_remaining_radius():
    result = evaluate_motion_envelope_policy(
        _trial(),
        [_event(0), _event(1)],
        settled_time_s=10.1,
        calibrated_margin_m=0.01,
        max_target_speed_m_s=0.10,
    )

    assert result["status"] == "not_confirmed"
    assert result["visual_confirmed"] is False
    assert result["image_margin_rejections"] == 0
    assert result["motion_margin_rejections"] == 2


def test_image_margin_rejects_even_fresh_close_estimates_when_margin_is_too_large():
    result = evaluate_motion_envelope_policy(
        _trial(),
        [
            _event(0, age=0.0, estimate=0.04),
            _event(1, age=0.0, estimate=0.04),
        ],
        settled_time_s=10.1,
        calibrated_margin_m=0.01,
        max_target_speed_m_s=0.0,
    )

    assert result["status"] == "not_confirmed"
    assert result["image_margin_rejections"] == 2
    assert result["motion_margin_rejections"] == 0


def test_motion_envelope_requires_consecutive_post_settle_frames():
    events = [
        _event(0),
        _event(1),
        _event(3),  # skipped frame prevents a two-frame confirmation
    ]

    result = evaluate_motion_envelope_policy(
        _trial(),
        events,
        settled_time_s=10.3,
        calibrated_margin_m=0.005,
        max_target_speed_m_s=0.04,
    )

    assert result["status"] == "not_confirmed"
    assert result["usable_frames"] == 2
    assert result["pre_settle_rejections"] == 1


def test_training_margin_uses_higher_empirical_quantile():
    samples = [
        {"seed": 960, "optimistic_error_m_evaluation_only": residual}
        for residual in (0.0, 0.01, 0.02)
    ]

    result = _calibrate_margin(samples)

    assert result["margin_m"] == pytest.approx(0.02)
    assert result["frame_count"] == 3
    assert result["seed_count"] == 1
    assert "not a statistical or physical guarantee" in result["interpretation"]


def test_envelope_diagnostics_report_profile_delivery_events():
    rows = _envelope_diagnostics(
        [
            {
                "motion_condition": "stationary",
                "peak_target_speed_m_s": 0.0,
                "delivery_profile": "near_real_time",
                "max_target_speed_bound_m_s": 0.126,
                "target_speed_within_configured_bound": True,
                "image_margin_violation": False,
                "delivery_envelope_violation": False,
            },
            {
                "motion_condition": "stationary",
                "peak_target_speed_m_s": 0.0,
                "delivery_profile": "near_real_time",
                "max_target_speed_bound_m_s": 0.126,
                "target_speed_within_configured_bound": True,
                "image_margin_violation": True,
                "delivery_envelope_violation": True,
            },
        ],
        calibrated_margin_m=0.01,
    )

    assert rows[0]["delivered_post_settle_frame_events"] == 2
    assert rows[0]["image_margin_violations"] == 1
    assert rows[0]["delivery_envelope_violations"] == 1


def test_scorecard_keeps_fully_delayed_rgb_separate_from_current_tip_hybrid(tmp_path):
    common = {
        "motion_condition": "stationary",
        "delivery_profile": "near_real_time",
        "max_frame_age_s": None,
        "max_target_speed_bound_m_s": None,
    }
    rows = [
        {
            **common,
            "control_mode": "fully_delayed_rgb",
            "control_delay_observations": 2,
            "policy": "two_post_settle_frames",
            "max_frame_age_s": 1.0,
            "trials": 12,
            "visually_confirmed": 6,
            "false_confirmations_at_delivery": 1,
        },
        {
            **common,
            "control_mode": "fully_delayed_rgb",
            "control_delay_observations": 2,
            "policy": "two_post_settle_motion_envelope",
            "max_target_speed_bound_m_s": 0.126,
            "trials": 12,
            "visually_confirmed": 4,
            "false_confirmations_at_delivery": 0,
        },
        {
            **common,
            "control_mode": "delayed_target_current_joint_tip",
            "control_delay_observations": 2,
            "policy": "two_post_settle_frames",
            "max_frame_age_s": 1.0,
            "trials": 80,
            "visually_confirmed": 80,
            "false_confirmations_at_delivery": 0,
        },
    ]

    figure_path = tmp_path / "motion-envelope.svg"
    _write_figure(rows, figure_path)
    figure = figure_path.read_text(encoding="utf-8")

    assert "delayed target and tip, 2 observation delay" in figure
    assert "50.0% (6/12)" in figure
    assert "33.3% (4/12)" in figure
    assert "100.0% (80/80)" not in figure


@pytest.mark.parametrize(
    ("margin", "speed_bound"),
    ((-0.01, 0.0), (math.nan, 0.0), (0.0, -0.01), (0.0, math.inf)),
)
def test_motion_envelope_rejects_invalid_bounds(margin, speed_bound):
    with pytest.raises(ValueError):
        evaluate_motion_envelope_policy(
            _trial(),
            [],
            settled_time_s=10.1,
            calibrated_margin_m=margin,
            max_target_speed_m_s=speed_bound,
        )
