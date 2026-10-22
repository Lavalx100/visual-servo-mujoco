import csv
import json
import shutil
from pathlib import Path

import pytest

from visual_servo_mujoco.evidence_audit import (
    DatasetSpec,
    EvidenceAuditError,
    audit_cooperative_pick_contact_diagnostics,
    audit_cooperative_pick_material_validation,
    audit_cooperative_pick_occlusion_dataset,
    audit_cooperative_pick_occlusion_shape_challenge,
    audit_cooperative_pick_orthogonal_validation,
    audit_cooperative_pick_pad_friction_validation,
    audit_cooperative_pick_retention_diagnostics,
    audit_dataset,
)

SPEC = DatasetSpec(
    "fixture",
    "trials.csv",
    "summary.csv",
    ("group",),
    decision_frame_policies=("margin_policy",),
)
TRIAL_FIELDS = (
    "group",
    "policy",
    "status",
    "visual_confirmed",
    "decision_frame_index",
    "physical_success_at_controller_stop",
    "physical_success_at_decision_delivery",
    "physical_success_at_decision_capture",
    "false_confirmation_at_delivery",
    "missed_controller_success",
    "usable_frames",
    "dropped_frames",
)
SUMMARY_FIELDS = (
    "group",
    "trials",
    "controller_stop_physical_successes",
    "controller_stop_physical_success_rate",
    "visually_confirmed",
    "visual_confirmation_rate",
    "true_confirmations_at_delivery",
    "false_confirmations_at_delivery",
    "delivery_mismatches_from_motion_after_capture",
    "delivery_mismatches_already_out_at_capture",
    "visual_confirmation_precision_at_delivery",
    "missed_controller_successes",
    "not_confirmed",
    "unverified",
    "usable_frames_total",
    "dropped_frames_total",
    "false_confirmation_rate",
)


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path, *, decision_frame="8", false_count="0"):
    results = tmp_path / "results"
    results.mkdir()
    trials = [
        {
            "group": "A",
            "policy": "margin_policy",
            "status": "visually_confirmed",
            "visual_confirmed": "True",
            "decision_frame_index": decision_frame,
            "physical_success_at_controller_stop": "True",
            "physical_success_at_decision_delivery": "True",
            "physical_success_at_decision_capture": "True",
            "false_confirmation_at_delivery": "False",
            "missed_controller_success": "False",
            "usable_frames": "3",
            "dropped_frames": "1",
        },
        {
            "group": "A",
            "policy": "margin_policy",
            "status": "not_confirmed",
            "visual_confirmed": "False",
            "decision_frame_index": "",
            "physical_success_at_controller_stop": "False",
            "physical_success_at_decision_delivery": "False",
            "physical_success_at_decision_capture": "False",
            "false_confirmation_at_delivery": "False",
            "missed_controller_success": "False",
            "usable_frames": "1",
            "dropped_frames": "0",
        },
    ]
    summary = [
        {
            "group": "A",
            "trials": "2",
            "controller_stop_physical_successes": "1",
            "controller_stop_physical_success_rate": "0.5",
            "visually_confirmed": "1",
            "visual_confirmation_rate": "0.5",
            "true_confirmations_at_delivery": "1",
            "false_confirmations_at_delivery": false_count,
            "delivery_mismatches_from_motion_after_capture": "0",
            "delivery_mismatches_already_out_at_capture": "0",
            "visual_confirmation_precision_at_delivery": "1.0",
            "missed_controller_successes": "0",
            "not_confirmed": "1",
            "unverified": "0",
            "usable_frames_total": "4",
            "dropped_frames_total": "1",
            "false_confirmation_rate": "0.0",
        }
    ]
    _write_csv(results / "trials.csv", TRIAL_FIELDS, trials)
    _write_csv(results / "summary.csv", SUMMARY_FIELDS, summary)
    return tmp_path


def test_trial_rows_reconcile_with_summary_counts(tmp_path):
    root = _fixture(tmp_path)

    assert audit_dataset(root, SPEC) == (2, 1)


def test_audit_rejects_a_summary_count_that_disagrees_with_trials(tmp_path):
    root = _fixture(tmp_path, false_count="1")

    with pytest.raises(EvidenceAuditError, match="false_confirmations_at_delivery"):
        audit_dataset(root, SPEC)


def test_audit_requires_the_decision_frame_for_confirmed_margin_policy(tmp_path):
    root = _fixture(tmp_path, decision_frame="")

    with pytest.raises(EvidenceAuditError, match="decision_frame_index"):
        audit_dataset(root, SPEC)


def test_cooperative_occlusion_splits_reconcile_with_saved_trial_data():
    repository = Path(__file__).resolve().parents[1]

    assert audit_cooperative_pick_occlusion_dataset(repository, stage="development") == (60, 2)
    assert audit_cooperative_pick_occlusion_dataset(repository, stage="heldout") == (120, 2)


def test_occlusion_geometry_challenge_reconciles_masks_and_physical_pose_labels():
    repository = Path(__file__).resolve().parents[1]

    assert audit_cooperative_pick_occlusion_shape_challenge(repository) == (1800, 9)


def test_orthogonal_verification_challenge_reconciles_two_view_and_physics_counts():
    repository = Path(__file__).resolve().parents[1]

    assert audit_cooperative_pick_orthogonal_validation(repository) == (240, 2)


def test_material_stress_challenge_reconciles_paired_physics_and_intervals():
    repository = Path(__file__).resolve().parents[1]

    assert audit_cooperative_pick_material_validation(repository) == (216, 9)


def test_material_stress_audit_rejects_a_changed_confidence_interval(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    source = repository / "results" / "cooperative-pick-material-validation"
    destination = tmp_path / "results" / "cooperative-pick-material-validation"
    destination.mkdir(parents=True)
    for filename in (
        "cooperative-pick-material-trials.csv",
        "cooperative-pick-material-summary.json",
    ):
        shutil.copy2(source / filename, destination / filename)
    summary_path = destination / "cooperative-pick-material-summary.json"
    summary = json.loads(summary_path.read_text())
    interval = summary["conditions"]["mass-0.45_friction-0.5"]["controllers"][
        "camera_estimate"
    ]["physical_stable_lift_success"]["wilson_95_interval"]
    interval[0] = 0.01
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(EvidenceAuditError, match="interval does not reconcile"):
        audit_cooperative_pick_material_validation(tmp_path)


def test_pad_friction_validation_reconciles_physics_pairs_and_demo_replays():
    repository = Path(__file__).resolve().parents[1]

    assert audit_cooperative_pick_pad_friction_validation(repository) == (96, 4)


def test_pad_friction_audit_rejects_claimed_visual_confirmation(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    source = repository / "results" / "cooperative-pick-pad-friction-validation"
    destination = tmp_path / "results" / "cooperative-pick-pad-friction-validation"
    destination.mkdir(parents=True)
    for filename in (
        "cooperative-pick-pad-friction-trials.csv",
        "cooperative-pick-pad-friction-summary.json",
    ):
        shutil.copy2(source / filename, destination / filename)
    summary_path = destination / "cooperative-pick-pad-friction-summary.json"
    summary = json.loads(summary_path.read_text())
    summary["fresh_visual_confirmation"] = "confirmed"
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(EvidenceAuditError, match="conflates physical and visual outcomes"):
        audit_cooperative_pick_pad_friction_validation(tmp_path)


def test_contact_diagnostics_reconcile_per_finger_phase_traces_and_replays():
    repository = Path(__file__).resolve().parents[1]

    assert audit_cooperative_pick_contact_diagnostics(repository) == (48, 4)


def test_contact_diagnostic_audit_rejects_changed_hold_contact_count(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    source = repository / "results" / "cooperative-pick-contact-diagnostics"
    destination = tmp_path / "results" / "cooperative-pick-contact-diagnostics"
    destination.mkdir(parents=True)
    for filename in (
        "cooperative-pick-contact-trials.csv",
        "cooperative-pick-contact-by-finger.csv",
        "cooperative-pick-contact-summary.json",
    ):
        shutil.copy2(source / filename, destination / filename)
    summary_path = destination / "cooperative-pick-contact-summary.json"
    summary = json.loads(summary_path.read_text())
    summary["conditions"]["friction-0.5"]["camera_estimate"][
        "pad_trials_with_contact_during_hold_under_load"
    ] = 1
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(EvidenceAuditError, match="hold_under_load does not reconcile"):
        audit_cooperative_pick_contact_diagnostics(tmp_path)


def test_retention_diagnostics_reconcile_matched_trials_force_and_motion_summaries():
    repository = Path(__file__).resolve().parents[1]

    assert audit_cooperative_pick_retention_diagnostics(repository) == (48, 4)


def test_retention_audit_rejects_changed_hand_relative_drift_summary(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    source_results = repository / "results" / "cooperative-pick-retention-diagnostics"
    destination_results = tmp_path / "results" / "cooperative-pick-retention-diagnostics"
    destination_results.mkdir(parents=True)
    for filename in (
        "cooperative-pick-retention-trials.csv",
        "cooperative-pick-retention-by-pad-phase.csv",
        "cooperative-pick-retention-summary.json",
        "cooperative-pick-retention-scorecard.svg",
    ):
        shutil.copy2(source_results / filename, destination_results / filename)
    source_media = repository / "assets" / "demos"
    destination_media = tmp_path / "assets" / "demos"
    destination_media.mkdir(parents=True)
    for filename in (
        "cooperative_pick_retention_diagnostic_demos.json",
        "cooperative_pick_retention_diagnostic_pair.mp4",
        "cooperative_pick_retention_diagnostic_pair.gif",
    ):
        shutil.copy2(source_media / filename, destination_media / filename)
    summary_path = destination_results / "cooperative-pick-retention-summary.json"
    summary = json.loads(summary_path.read_text())
    summary["conditions"]["friction-0.5"]["camera_estimate"][
        "maximum_downward_payload_drift_in_hand_frame_m_median"
    ] = 0.0
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(EvidenceAuditError, match="maximum_downward_payload_drift"):
        audit_cooperative_pick_retention_diagnostics(tmp_path)
