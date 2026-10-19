"""Recompute published post-stop benchmark counts from their trial CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    trial_file: str
    summary_file: str
    groups: tuple[str, ...]
    decision_frame_policies: tuple[str, ...] = ()


DATASETS = (
    DatasetSpec(
        "timestamped verification",
        "timestamped-verification-trials.csv",
        "timestamped-verification-summary.csv",
        (
            "target_motion_profile",
            "control_mode",
            "control_delay_observations",
            "delivery_profile",
            "policy",
            "max_frame_age_s",
        ),
    ),
    DatasetSpec(
        "completion speed",
        "timestamped-completion-speed-trials.csv",
        "timestamped-completion-speed-summary.csv",
        (
            "motion_condition",
            "control_mode",
            "control_delay_observations",
            "delivery_profile",
            "policy",
            "max_frame_age_s",
        ),
    ),
    DatasetSpec(
        "motion envelope",
        "motion-envelope-validation-trials.csv",
        "motion-envelope-validation-summary.csv",
        (
            "motion_condition",
            "control_mode",
            "control_delay_observations",
            "delivery_profile",
            "policy",
            "max_frame_age_s",
            "max_target_speed_bound_m_s",
        ),
    ),
    DatasetSpec(
        "conditional uncertainty",
        "conditional-uncertainty-validation-trials.csv",
        "conditional-uncertainty-validation-summary.csv",
        ("control_mode", "control_delay_observations", "delivery_profile", "policy"),
        (
            "two_post_settle_prior_global_q99",
            "two_post_settle_new_global_q99",
            "two_post_settle_conditional_q99",
        ),
    ),
)

# Summary field -> (trial field, value type). Rejection totals are checked when
# the trial table records their per-outcome source field.
COUNT_SOURCES = {
    "controller_stop_physical_successes": ("physical_success_at_controller_stop", "bool"),
    "visually_confirmed": ("visual_confirmed", "bool"),
    "false_confirmations_at_delivery": ("false_confirmation_at_delivery", "bool"),
    "missed_controller_successes": ("missed_controller_success", "bool"),
    "usable_frames_total": ("usable_frames", "int"),
    "dropped_frames_total": ("dropped_frames", "int"),
    "pre_settle_rejections_total": ("pre_settle_rejections", "int"),
    "stale_rejections_total": ("stale_rejections", "int"),
    "out_of_order_rejections_total": ("out_of_order_rejections", "int"),
    "image_margin_rejections_total": ("image_margin_rejections", "int"),
    "motion_margin_rejections_total": ("motion_margin_rejections", "int"),
}
CORE_COUNTS = (
    "trials",
    "controller_stop_physical_successes",
    "visually_confirmed",
    "true_confirmations_at_delivery",
    "false_confirmations_at_delivery",
    "delivery_mismatches_from_motion_after_capture",
    "delivery_mismatches_already_out_at_capture",
    "missed_controller_successes",
    "not_confirmed",
    "unverified",
    "usable_frames_total",
    "dropped_frames_total",
)
RATES = {
    "controller_stop_physical_success_rate": ("controller_stop_physical_successes", "trials"),
    "visual_confirmation_rate": ("visually_confirmed", "trials"),
    "visual_confirmation_precision_at_delivery": (
        "true_confirmations_at_delivery",
        "visually_confirmed",
    ),
    "false_confirmation_rate": ("false_confirmations_at_delivery", "trials"),
}


class EvidenceAuditError(ValueError):
    """A published trial table and its aggregate summary disagree."""


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise EvidenceAuditError(f"{path} has no CSV header")
        return reader.fieldnames, list(reader)


def _bool(row: dict[str, str], field: str, path: Path, line: int) -> bool:
    value = row.get(field, "").strip().lower()
    if value in {"true", "1"}:
        return True
    if value in {"false", "0", ""}:
        return False
    raise EvidenceAuditError(f"{path}:{line} has invalid {field}: {value!r}")


def _group_key(row: dict[str, str], fields: tuple[str, ...], path: Path) -> tuple[str, ...]:
    missing = set(fields) - row.keys()
    if missing:
        raise EvidenceAuditError(f"{path} is missing fields: {', '.join(sorted(missing))}")
    return tuple(row[field] for field in fields)


def _counts(rows: list[tuple[int, dict[str, str]]], path: Path) -> dict[str, int]:
    result = {"trials": len(rows)}
    for summary_field, (source, value_type) in COUNT_SOURCES.items():
        if source not in rows[0][1]:
            continue
        values = []
        for line, row in rows:
            raw = row[source]
            if value_type == "int":
                try:
                    values.append(int(raw) if raw else 0)
                except ValueError as error:
                    raise EvidenceAuditError(
                        f"{path}:{line} has invalid integer {source}: {raw!r}"
                    ) from error
            else:
                values.append(int(_bool(row, source, path, line)))
        result[summary_field] = sum(values)

    confirmed = [_bool(row, "visual_confirmed", path, line) for line, row in rows]
    delivered = [
        _bool(row, "physical_success_at_decision_delivery", path, line) for line, row in rows
    ]
    captured = [
        _bool(row, "physical_success_at_decision_capture", path, line) for line, row in rows
    ]
    mismatch = [is_confirmed and not success for is_confirmed, success in zip(confirmed, delivered)]
    result.update(
        true_confirmations_at_delivery=sum(a and b for a, b in zip(confirmed, delivered)),
        delivery_mismatches_from_motion_after_capture=sum(
            a and b for a, b in zip(mismatch, captured)
        ),
        delivery_mismatches_already_out_at_capture=sum(
            a and not b for a, b in zip(mismatch, captured)
        ),
        not_confirmed=sum(row["status"] == "not_confirmed" for _, row in rows),
        unverified=sum(row["status"] == "unverified" for _, row in rows),
    )
    return result


def audit_dataset(root: Path, spec: DatasetSpec) -> tuple[int, int]:
    results = root / "results"
    trial_path, summary_path = results / spec.trial_file, results / spec.summary_file
    trial_fields, trials = _read(trial_path)
    summary_fields, summaries = _read(summary_path)
    if not trials or not summaries:
        raise EvidenceAuditError(f"{spec.name} trial and summary CSVs must not be empty")

    required_summary = set(spec.groups) | set(CORE_COUNTS)
    required_trial = set(spec.groups) | {
        "status",
        "visual_confirmed",
        "physical_success_at_controller_stop",
        "physical_success_at_decision_delivery",
        "physical_success_at_decision_capture",
        "false_confirmation_at_delivery",
        "missed_controller_success",
        "usable_frames",
        "dropped_frames",
    }
    for field in required_summary:
        source = COUNT_SOURCES.get(field)
        if source:
            required_trial.add(source[0])
    for path, required, available in (
        (trial_path, required_trial, set(trial_fields)),
        (summary_path, required_summary, set(summary_fields)),
    ):
        missing = sorted(required - available)
        if missing:
            raise EvidenceAuditError(f"{path} is missing required fields: {', '.join(missing)}")

    grouped: dict[tuple[str, ...], list[tuple[int, dict[str, str]]]] = {}
    for line, row in enumerate(trials, start=2):
        confirmed = _bool(row, "visual_confirmed", trial_path, line)
        if row["status"] not in {"visually_confirmed", "not_confirmed", "unverified"}:
            raise EvidenceAuditError(f"{trial_path}:{line} has unknown status {row['status']!r}")
        if (row["status"] == "visually_confirmed") != confirmed:
            raise EvidenceAuditError(f"{trial_path}:{line} status and confirmation flag disagree")
        delivery_success = _bool(row, "physical_success_at_decision_delivery", trial_path, line)
        false_confirmation = _bool(row, "false_confirmation_at_delivery", trial_path, line)
        if false_confirmation != (confirmed and not delivery_success):
            raise EvidenceAuditError(
                f"{trial_path}:{line} false-confirmation label disagrees with delivery outcome"
            )
        if row.get("policy") in spec.decision_frame_policies:
            frame = row.get("decision_frame_index", "")
            if bool(frame) != confirmed:
                raise EvidenceAuditError(
                    f"{trial_path}:{line} decision_frame_index is inconsistent"
                )
            if frame:
                try:
                    int(frame)
                except ValueError as error:
                    raise EvidenceAuditError(
                        f"{trial_path}:{line} has invalid decision_frame_index"
                    ) from error
        grouped.setdefault(_group_key(row, spec.groups, trial_path), []).append((line, row))

    summaries_by_key = {}
    for line, row in enumerate(summaries, start=2):
        key = _group_key(row, spec.groups, summary_path)
        if key in summaries_by_key:
            raise EvidenceAuditError(f"{summary_path}:{line} repeats group {key}")
        summaries_by_key[key] = row
    if grouped.keys() != summaries_by_key.keys():
        raise EvidenceAuditError(f"{spec.name} trial and summary groups differ")

    for key, rows in grouped.items():
        actual, recorded = _counts(rows, trial_path), summaries_by_key[key]
        for field, value in actual.items():
            if field not in recorded:
                if field in CORE_COUNTS:
                    raise EvidenceAuditError(f"{summary_path} group {key} is missing {field}")
                continue
            try:
                saved = int(recorded[field])
            except ValueError as error:
                raise EvidenceAuditError(
                    f"{summary_path} group {key} has invalid {field}"
                ) from error
            if saved != value:
                raise EvidenceAuditError(f"{spec.name} group {key}: {field} {saved} != {value}")
        for field, (numerator, denominator) in RATES.items():
            if field not in recorded:
                continue
            total = actual[denominator]
            expected = actual[numerator] / total if total else None
            raw = recorded[field]
            if expected is None:
                if raw:
                    raise EvidenceAuditError(f"{summary_path} group {key}: {field} should be blank")
            else:
                try:
                    saved = float(raw)
                except ValueError as error:
                    raise EvidenceAuditError(
                        f"{summary_path} group {key} has invalid {field}: {raw!r}"
                    ) from error
                if not math.isfinite(saved) or not math.isclose(
                    saved, expected, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise EvidenceAuditError(f"{spec.name} group {key}: {field} does not reconcile")
    return len(trials), len(summaries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    try:
        for spec in DATASETS:
            trials, groups = audit_dataset(args.root, spec)
            print(f"PASS {spec.name}: {trials:,} trial rows, {groups:,} summary groups")
    except (EvidenceAuditError, OSError) as error:
        print(f"Evidence audit failed: {error}", file=sys.stderr)
        return 1
    print(f"Validated {len(DATASETS)} trial/summary pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
