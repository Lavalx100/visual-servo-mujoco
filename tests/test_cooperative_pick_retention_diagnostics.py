from visual_servo_mujoco.cooperative_pick_retention_diagnostics import (
    _condition_summary,
    _paired_deltas,
)


def test_retention_summary_keeps_contact_success_and_drift_as_distinct_metrics():
    rows = [
        {
            "physical_stable_lift_success": False,
            "full_sequence_collision_safe": True,
            "hold_contact_sample_fraction_mean_across_pads": 0.0,
            "hold_maximum_downward_payload_drift_in_hand_frame_m": 0.16,
            "mean_tangential_to_normal_force_ratio_when_contacting": 0.5,
            "maximum_slide_tracking_error_lift_or_hold_m": 0.019,
            "maximum_slide_tracking_error_during_hold_m": 0.00002,
        },
        {
            "physical_stable_lift_success": True,
            "full_sequence_collision_safe": True,
            "hold_contact_sample_fraction_mean_across_pads": 0.75,
            "hold_maximum_downward_payload_drift_in_hand_frame_m": 0.01,
            "mean_tangential_to_normal_force_ratio_when_contacting": 0.42,
            "maximum_slide_tracking_error_lift_or_hold_m": 0.003,
            "maximum_slide_tracking_error_during_hold_m": 0.001,
        },
    ]

    summary = _condition_summary(rows)

    assert summary["physical_stable_lift_successes"] == 1
    assert summary["collision_safe_trials"] == 2
    assert summary["hold_contact_sample_fraction_mean"] == 0.375
    assert summary["maximum_downward_payload_drift_in_hand_frame_m_median"] == 0.085
    assert summary["fresh_visual_confirmation"] == "not measured"


def test_paired_friction_deltas_preserve_matched_pose_transition_counts():
    rows = []
    for controller in ("camera_estimate", "oracle_xy"):
        for pair_id, low_success, high_success in (
            ("42000-000", False, True),
            ("42000-001", False, False),
        ):
            for friction, success, drift, contact, slide, ratio in (
                (0.5, low_success, 0.16, 0.0, 0.018, 0.5),
                (2.2, high_success, 0.01, 0.75, 0.003, 0.42),
            ):
                rows.append(
                    {
                        "controller": controller,
                        "pair_id": pair_id,
                        "friction": friction,
                        "physical_stable_lift_success": success,
                        "hold_maximum_downward_payload_drift_in_hand_frame_m": drift,
                        "hold_contact_sample_fraction_mean_across_pads": contact,
                        "maximum_slide_tracking_error_lift_or_hold_m": slide,
                        "mean_tangential_to_normal_force_ratio_when_contacting": ratio,
                        "maximum_slide_tracking_error_during_hold_m": slide,
                    }
                )

    summary = _paired_deltas(rows)

    assert summary["camera_estimate"]["matched_pose_pairs"] == 2
    assert summary["camera_estimate"]["physical_stable_lift_transitions"] == {
        "low_only_success": 0,
        "high_only_success": 1,
        "both_success": 0,
        "neither_success": 1,
    }
    assert (
        summary["oracle_xy"]["median_high_minus_low"]["high_minus_low_hold_downward_drift_m"]
        == -0.15
    )
