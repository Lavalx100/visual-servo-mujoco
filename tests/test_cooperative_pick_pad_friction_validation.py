import csv

import pytest

from visual_servo_mujoco import cooperative_pick_pad_friction_validation as study


def _row(success):
    return {"physical_stable_lift_success": success}


def test_paired_outcomes_keep_improvement_and_regression_counts_separate():
    baseline = {"a": _row(True), "b": _row(False), "c": _row(False)}
    candidate = {"a": _row(False), "b": _row(True), "c": _row(False)}

    assert study._paired_outcomes(baseline, candidate) == {
        "both_success": 0,
        "candidate_only_success": 1,
        "baseline_only_success": 1,
        "both_failure": 1,
    }


def test_paired_outcomes_reject_different_trial_ids():
    with pytest.raises(ValueError, match="same paired trial ids"):
        study._paired_outcomes({"a": _row(False)}, {"b": _row(True)})


def test_heldout_run_writes_all_paired_rows_and_separate_physics_metrics(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        study,
        "_sample_positions",
        lambda **kwargs: [
            (kwargs["base_seed"], 0, (-0.02, 0.01)),
            (kwargs["base_seed"], 1, (0.03, -0.04)),
        ],
    )

    def fake_trial(target_xy, *, controller, finger_pad_half_size_m, finger_pad_sliding_friction, **kwargs):
        improved_low_friction_candidate = (
            finger_pad_sliding_friction == 0.5
            and finger_pad_half_size_m > study.FINGER_PAD_HALF_SIZE_M
        )
        success = improved_low_friction_candidate or finger_pad_sliding_friction == 2.2
        return {
            "payload_body_mass_kg": study.PAYLOAD_MASS_KG,
            "control_target_source": (
                "detected_image_position"
                if controller == "camera_estimate"
                else "oracle_xy"
            ),
            "target_detected": True,
            "physical_stable_lift_success": success,
            "payload_stable_lift_success": success,
            "full_sequence_collision_safe": True,
            "minimum_finger_payload_contact_points_during_hold": 3 if success else 0,
            "minimum_payload_z_during_hold_m": 0.4,
            "maximum_payload_tilt_during_hold_deg": 1.0,
            "physics_steps_with_arm_table_or_ground_contact": 0,
            "physics_steps_with_nonfinger_arm_payload_contact": 0,
            "maximum_inter_arm_penetration_m": 0.0,
            "fresh_visual_confirmation": "not measured",
            "failure_reason": "" if success else "no_grasp_contact",
        }

    monkeypatch.setattr(study, "run_trial", fake_trial)
    result = study.run_validation(
        output_dir=tmp_path / "results",
        media_dir=tmp_path / "media",
        base_seed=51234,
        samples_per_seed=2,
        render_demos=False,
    )

    assert result["total_physics_trials"] == 16
    low_camera = result["conditions"]["friction-0.5"]["controllers"]["camera_estimate"]
    assert low_camera["pad_half_size_m_0.016"]["physical_stable_lift_success"]["successes"] == 0
    assert low_camera["pad_half_size_m_0.018"]["physical_stable_lift_success"]["successes"] == 2
    assert result["paired_success_outcomes"]["friction-0.5"]["camera_estimate"] == {
        "both_success": 0,
        "candidate_only_success": 2,
        "baseline_only_success": 0,
        "both_failure": 0,
    }
    assert low_camera["pad_half_size_m_0.018"]["full_sequence_collision_safe"]["successes"] == 2
    assert low_camera["pad_half_size_m_0.018"]["fresh_visual_confirmation"] == "not measured"

    with (tmp_path / "results" / study.TRIALS_FILE).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16
    assert {(row["pair_id"], row["target_x_m"], row["target_y_m"]) for row in rows} == {
        ("51234-000", "-0.02", "0.01"),
        ("51234-001", "0.03", "-0.04"),
    }
    assert (tmp_path / "results" / study.SCORECARD_FILE).exists()
