import csv

from visual_servo_mujoco import cooperative_pick_contact_diagnostics as study


def _trace(*, hold_contact):
    phases = {"CLOSE SIX FINGERS": 20}
    if hold_contact:
        phases["HOLD UNDER LOAD"] = 40
    return {
        f"arm{arm}_{side}_pad": {
            "contact_steps": sum(phases.values()),
            "first_contact_time_s": 4.2,
            "first_contact_phase": "CLOSE SIX FINGERS",
            "last_contact_time_s": 8.2 if hold_contact else 5.5,
            "maximum_contact_points": 2,
            "maximum_normal_contact_force_n": 1.25,
            "normal_contact_impulse_ns": 0.04,
            "contact_steps_by_phase": phases,
            "maximum_normal_force_by_phase_n": {phase: 1.25 for phase in phases},
            "normal_contact_impulse_by_phase_ns": {phase: 0.02 for phase in phases},
        }
        for arm in range(3)
        for side in ("left", "right")
    }


def _tracking():
    return {
        f"arm{arm}_{side}_pad": {
            "close_phase_samples": 375,
            "commanded_slide_start_m": 0.012,
            "commanded_slide_end_m": 0.0,
            "measured_slide_start_m": 0.012,
            "measured_slide_end_m": 0.001,
            "maximum_absolute_tracking_error_m": 0.001,
            "mean_absolute_tracking_error_m": 0.0002,
        }
        for arm in range(3)
        for side in ("left", "right")
    }


def test_rollup_distinguishes_contact_during_close_from_contact_during_hold():
    close_only = _trace(hold_contact=False)
    close_and_hold = _trace(hold_contact=True)

    assert study._count_contacting_pads(close_only, study.CONTACT_CLOSING_PHASES) == 6
    assert study._count_contacting_pads(close_only, (study.HOLD_PHASE,)) == 0
    assert study._count_contacting_pads(close_and_hold, (study.HOLD_PHASE,)) == 6


def test_diagnostic_validation_writes_trial_and_per_finger_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(
        study,
        "_sample_positions",
        lambda **kwargs: [
            (kwargs["base_seed"], 0, (-0.02, 0.01)),
            (kwargs["base_seed"], 1, (0.03, -0.04)),
        ],
    )

    def fake_trial(target_xy, *, controller, finger_pad_sliding_friction, **kwargs):
        hold_contact = finger_pad_sliding_friction == 2.2
        return {
            "payload_body_mass_kg": study.PAYLOAD_MASS_KG,
            "control_target_source": (
                "detected_image_position"
                if controller == "camera_estimate"
                else "oracle_xy"
            ),
            "target_detected": True,
            "physical_stable_lift_success": hold_contact,
            "payload_stable_lift_success": hold_contact,
            "full_sequence_collision_safe": True,
            "minimum_finger_payload_contact_points_during_hold": 3 if hold_contact else 0,
            "minimum_payload_z_during_hold_m": 0.4,
            "maximum_payload_tilt_during_hold_deg": 1.0,
            "maximum_inter_arm_penetration_m": 0.0,
            "physics_steps_with_arm_table_or_ground_contact": 0,
            "physics_steps_with_nonfinger_arm_payload_contact": 0,
            "finger_pad_payload_contact_trace": _trace(hold_contact=hold_contact),
            "finger_slide_tracking_during_close": _tracking(),
            "fresh_visual_confirmation": "not measured",
        }

    monkeypatch.setattr(study, "run_trial", fake_trial)
    summary = study.run_validation(
        output_dir=tmp_path / "results",
        media_dir=tmp_path / "media",
        base_seed=51234,
        samples_per_seed=2,
        render_demos=False,
    )

    assert summary["total_physics_trials"] == 8
    assert summary["total_finger_trial_rows"] == 48
    low = summary["conditions"]["friction-0.5"]["camera_estimate"]
    high = summary["conditions"]["friction-2.2"]["camera_estimate"]
    assert low["pad_trials_with_contact_during_approach_close_or_secure"] == 12
    assert low["pad_trials_with_contact_during_hold_under_load"] == 0
    assert low["contact_retention_outcomes"] == {
        "contact_not_retained_to_hold": 2
    }
    assert high["pad_trials_with_contact_during_hold_under_load"] == 12
    assert low["failure_reasons"] == {"no_grasp_contact": 2}

    with (tmp_path / "results" / study.FINGERS_FILE).open(newline="") as stream:
        finger_rows = list(csv.DictReader(stream))
    assert len(finger_rows) == 48
    assert finger_rows[0]["contact_steps_by_phase_json"]
    assert finger_rows[0]["maximum_absolute_tracking_error_m"] == "0.001"
