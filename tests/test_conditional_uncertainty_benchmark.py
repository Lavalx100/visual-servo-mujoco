import math

import pytest

from visual_servo_mujoco.conditional_uncertainty_benchmark import (
    _fit_conditional_margins,
    evaluate_conditional_envelope_policy,
)
from visual_servo_mujoco.timestamped_verification_benchmark import (
    SUCCESS_THRESHOLD_METERS,
)


def _trial():
    return {
        "seed": 977,
        "episode": 0,
        "controller_stop_time_simulated_s": 10.0,
        "success": True,
        "reaching_error_m": 0.02,
    }


def _event(index, *, feature=0.2, estimate=0.02, physical=0.02):
    capture = 10.2 + 0.2 * index
    return {
        "frame_index": index,
        "capture_time_s": capture,
        "delivery_time_s": capture,
        "scheduled_arrival_time_s": capture,
        "camera_age_at_delivery_s": 0.0,
        "transport_dropped": False,
        "both_markers_visible": estimate is not None,
        "estimated_distance_m": estimate,
        "visual_success_estimate": (
            estimate <= SUCCESS_THRESHOLD_METERS if estimate is not None else None
        ),
        "quality_adjusted_pixel_noise_scale_m": feature,
        "physical_error_at_capture_m": physical,
        "physical_error_at_delivery_m": physical,
    }


def _model():
    return {
        "feature_edges_m": [0.5],
        "bins": [{"margin_m": 0.001}, {"margin_m": 0.03}],
    }


def test_conditional_policy_uses_feature_specific_margin_and_two_frames():
    low = evaluate_conditional_envelope_policy(
        _trial(),
        [_event(0, feature=0.2), _event(1, feature=0.2)],
        settled_time_s=10.1,
        margin_model=_model(),
        max_target_speed_m_s=0.0,
    )
    high = evaluate_conditional_envelope_policy(
        _trial(),
        [_event(0, feature=0.8), _event(1, feature=0.8)],
        settled_time_s=10.1,
        margin_model=_model(),
        max_target_speed_m_s=0.0,
    )

    assert low["status"] == "visually_confirmed"
    assert low["decision_margin_m"] == pytest.approx(0.001)
    assert high["status"] == "not_confirmed"
    assert high["image_margin_rejections"] == 2


def test_training_bins_use_training_labels_and_fallback_for_small_bins():
    samples = [
        {
            "seed": seed,
            "quality_adjusted_pixel_noise_scale_m": float(seed) / 1000,
            "optimistic_error_m_evaluation_only": float(seed) / 10000,
        }
        for seed in (1, 2, 3, 4, 5, 6)
    ]
    model = _fit_conditional_margins(
        samples,
        bin_count=3,
        minimum_samples=2,
        minimum_seeds=2,
    )

    assert model["sample_count"] == 6
    assert len(model["bins"]) == 3
    assert [row["index"] for row in model["bins"]] == [0, 1, 2]
    assert all(
        left["margin_m"] <= right["margin_m"]
        for left, right in zip(model["bins"], model["bins"][1:])
    )
    assert all(not row["fallback_to_global"] for row in model["bins"])

    fallback = _fit_conditional_margins(
        samples,
        bin_count=3,
        minimum_samples=10,
        minimum_seeds=1,
    )
    assert all(row["fallback_to_global"] for row in fallback["bins"])
    assert all(
        row["margin_m"] == pytest.approx(fallback["global_q99_margin_m"])
        for row in fallback["bins"]
    )


@pytest.mark.parametrize("value", (-0.01, math.nan, math.inf))
def test_conditional_policy_rejects_invalid_speed_or_margin(value):
    with pytest.raises(ValueError):
        evaluate_conditional_envelope_policy(
            _trial(),
            [],
            settled_time_s=10.1,
            margin_model=_model(),
            max_target_speed_m_s=value,
        )
