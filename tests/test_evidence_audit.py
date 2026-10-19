import csv

import pytest

from visual_servo_mujoco.evidence_audit import (
    DatasetSpec,
    EvidenceAuditError,
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
