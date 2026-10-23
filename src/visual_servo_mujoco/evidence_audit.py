"""Reconcile published benchmark evidence tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median


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


def _finite_float(row: dict[str, str], field: str, path: Path, line: int) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as error:
        raise EvidenceAuditError(
            f"{path}:{line} has invalid {field}: {row.get(field)!r}"
        ) from error
    if not math.isfinite(value):
        raise EvidenceAuditError(f"{path}:{line} has non-finite {field}")
    return value


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _check_numeric_summary(
    summary: dict[str, str], field: str, expected: float | int | None, key: str, path: Path
) -> None:
    raw = summary.get(field, "")
    if expected is None:
        if raw:
            raise EvidenceAuditError(f"{path} group {key}: {field} should be blank")
        return
    try:
        saved = float(raw)
    except ValueError as error:
        raise EvidenceAuditError(
            f"{path} group {key} has invalid {field}: {raw!r}"
        ) from error
    if not math.isfinite(saved) or not math.isclose(
        saved, float(expected), rel_tol=0.0, abs_tol=1e-9
    ):
            raise EvidenceAuditError(f"evidence group {key}: {field} {saved} != {expected}")


def audit_appearance_dataset(root: Path) -> tuple[int, int]:
    """Reconcile the rendered-appearance trial table, summary, and protocol."""
    results = root / "results"
    trial_path = results / "perception-appearance-trials.csv"
    summary_path = results / "perception-appearance-summary.csv"
    report_path = results / "perception-appearance-summary.json"
    trial_fields, trials = _read(trial_path)
    summary_fields, summaries = _read(summary_path)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceAuditError(
            f"cannot read appearance protocol {report_path}: {error}"
        ) from error
    if not trials or not summaries:
        raise EvidenceAuditError("appearance trial and summary CSVs must not be empty")

    factors = (
        report.get("lighting_conditions"),
        report.get("clutter_conditions"),
        report.get("occlusion_bar_width_fractions"),
    )
    if any(not isinstance(values, list) or not values for values in factors):
        raise EvidenceAuditError(f"{report_path} has invalid appearance factor levels")
    lighting_levels, clutter_levels, occlusion_levels = factors
    expected_factors_by_name = {
        f"{lighting}__{clutter}__occlusion-{float(occlusion):.2f}".replace(".", "p"):
        (lighting, clutter, float(occlusion))
        for lighting in lighting_levels
        for clutter in clutter_levels
        for occlusion in occlusion_levels
    }
    expected_condition_names = set(expected_factors_by_name)
    try:
        seeds = int(report["seeds"])
        base_seed = int(report["base_seed"])
        samples_per_seed = int(report["samples_per_seed"])
        paired_samples = int(report["paired_samples_per_condition"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceAuditError(f"{report_path} has invalid seed/sample metadata") from error
    if seeds <= 0 or samples_per_seed <= 0 or paired_samples != seeds * samples_per_seed:
        raise EvidenceAuditError(f"{report_path} has inconsistent paired sample counts")
    expected_seed_values = set(range(base_seed, base_seed + seeds))
    expected_ids = {
        (seed, sample_index)
        for seed in expected_seed_values
        for sample_index in range(samples_per_seed)
    }

    trial_required = {
        "condition",
        "lighting",
        "clutter",
        "occlusion_bar_width_fraction_of_target_diameter",
        "seed",
        "sample_index",
        "target_x_m",
        "target_y_m",
        "target_center_x_px",
        "target_center_y_px",
        "detected",
        "detected_center_x_px",
        "detected_center_y_px",
        "red_components_over_20px",
        "localization_error_px",
        "localization_error_mm",
        "inside_projected_target_radius",
        "projected_target_radius_px",
    }
    summary_required = {
        "condition",
        "lighting",
        "clutter",
        "occlusion_bar_width_fraction_of_target_diameter",
        "trials",
        "detections",
        "detection_rate",
        "within_target_radius_localizations",
        "within_target_radius_rate",
        "within_target_radius_rate_given_detection",
        "median_localization_error_px",
        "p95_localization_error_px",
        "median_localization_error_mm",
        "p95_localization_error_mm",
        "median_red_components_over_20px",
    }
    for path, required, available in (
        (trial_path, trial_required, set(trial_fields)),
        (summary_path, summary_required, set(summary_fields)),
    ):
        missing = sorted(required - available)
        if missing:
            raise EvidenceAuditError(f"{path} is missing fields: {', '.join(missing)}")
    expected_trial_count = len(expected_condition_names) * paired_samples
    if len(trials) != expected_trial_count:
        raise EvidenceAuditError(
            f"appearance trial table has {len(trials)} rows; expected {expected_trial_count}"
        )

    grouped: dict[str, list[tuple[int, dict[str, str]]]] = {}
    paired_positions: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    ids_by_condition: dict[str, set[tuple[int, int]]] = {}
    for line, row in enumerate(trials, start=2):
        key = row["condition"]
        try:
            seed = int(row["seed"])
            sample_index = int(row["sample_index"])
            lighting = row["lighting"]
            clutter = row["clutter"]
            occlusion = float(row["occlusion_bar_width_fraction_of_target_diameter"])
        except (TypeError, ValueError) as error:
            raise EvidenceAuditError(
                f"{trial_path}:{line} has invalid condition identity"
            ) from error
        condition = (lighting, clutter, occlusion)
        if key not in expected_condition_names or condition != expected_factors_by_name[key]:
            raise EvidenceAuditError(
                f"{trial_path}:{line} has unknown condition {(key, *condition)}"
            )
        if seed not in expected_seed_values or not 0 <= sample_index < samples_per_seed:
            raise EvidenceAuditError(f"{trial_path}:{line} has unexpected paired sample id")
        sample_id = (seed, sample_index)
        ids = ids_by_condition.setdefault(key, set())
        if sample_id in ids:
            raise EvidenceAuditError(f"{trial_path}:{line} repeats sample {sample_id} in {key}")
        ids.add(sample_id)
        target = (
            _finite_float(row, "target_x_m", trial_path, line),
            _finite_float(row, "target_y_m", trial_path, line),
            _finite_float(row, "target_center_x_px", trial_path, line),
            _finite_float(row, "target_center_y_px", trial_path, line),
        )
        previous_target = paired_positions.setdefault(sample_id, target)
        if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)
               for a, b in zip(previous_target, target, strict=True)):
            raise EvidenceAuditError(f"{trial_path}:{line} is not paired at target {sample_id}")
        detected = _bool(row, "detected", trial_path, line)
        inside = _bool(row, "inside_projected_target_radius", trial_path, line)
        radius = _finite_float(row, "projected_target_radius_px", trial_path, line)
        try:
            component_count = int(row["red_components_over_20px"])
        except ValueError as error:
            raise EvidenceAuditError(
                f"{trial_path}:{line} has invalid red component count"
            ) from error
        if radius <= 0.0 or component_count < 0:
            raise EvidenceAuditError(f"{trial_path}:{line} has invalid radius/component count")
        if detected:
            detected_x = _finite_float(row, "detected_center_x_px", trial_path, line)
            detected_y = _finite_float(row, "detected_center_y_px", trial_path, line)
            error_px = _finite_float(row, "localization_error_px", trial_path, line)
            error_mm = _finite_float(row, "localization_error_mm", trial_path, line)
            measured_error = math.dist((detected_x, detected_y), target[2:])
            if error_px < 0.0 or error_mm < 0.0 or not math.isclose(
                error_px, measured_error, rel_tol=0.0, abs_tol=1e-7
            ):
                raise EvidenceAuditError(f"{trial_path}:{line} has inconsistent center error")
            if inside != (error_px <= radius):
                raise EvidenceAuditError(f"{trial_path}:{line} has inconsistent target-radius flag")
        elif inside or any(
            row[field]
            for field in (
                "detected_center_x_px",
                "detected_center_y_px",
                "localization_error_px",
                "localization_error_mm",
            )
        ):
            raise EvidenceAuditError(f"{trial_path}:{line} records an estimate without detection")
        grouped.setdefault(key, []).append((line, row))

    if grouped.keys() != expected_condition_names:
        raise EvidenceAuditError("appearance trial conditions do not match protocol factors")
    if any(ids != expected_ids for ids in ids_by_condition.values()):
        raise EvidenceAuditError(
            "appearance conditions do not share the complete paired sample set"
        )
    if len(paired_positions) != paired_samples:
        raise EvidenceAuditError("appearance target positions do not cover the paired sample set")

    summaries_by_key = {}
    for line, row in enumerate(summaries, start=2):
        key = row["condition"]
        if key in summaries_by_key:
            raise EvidenceAuditError(f"{summary_path}:{line} repeats group {key}")
        summaries_by_key[key] = row
    if summaries_by_key.keys() != expected_condition_names:
        raise EvidenceAuditError("appearance trial and summary condition groups differ")
    json_summaries = report.get("condition_summaries")
    if isinstance(json_summaries, list) and any(
        not isinstance(row, dict) for row in json_summaries
    ):
        raise EvidenceAuditError(f"{report_path} has malformed condition summaries")
    json_condition_names = (
        {
            row.get("condition")
            for row in json_summaries
            if isinstance(row, dict)
        }
        if isinstance(json_summaries, list)
        else set()
    )
    if (
        not isinstance(json_summaries, list)
        or len(json_summaries) != len(expected_condition_names)
        or json_condition_names != expected_condition_names
    ):
        raise EvidenceAuditError(f"{report_path} condition summaries do not match protocol")
    json_summaries_by_key = {row["condition"]: row for row in json_summaries}
    for key in expected_condition_names:
        expected_factors = expected_factors_by_name[key]
        for path, row in (
            (summary_path, summaries_by_key[key]),
            (report_path, json_summaries_by_key[key]),
        ):
            actual_factors = (
                row.get("lighting"),
                row.get("clutter"),
                _finite_float(
                    row,
                    "occlusion_bar_width_fraction_of_target_diameter",
                    path,
                    0,
                ),
            )
            if actual_factors != expected_factors:
                raise EvidenceAuditError(f"{path} group {key}: factor labels disagree")

    float_fields = (
        "detection_rate",
        "within_target_radius_rate",
        "within_target_radius_rate_given_detection",
        "median_localization_error_px",
        "p95_localization_error_px",
        "median_localization_error_mm",
        "p95_localization_error_mm",
        "median_red_components_over_20px",
    )
    for key, rows in grouped.items():
        detected_rows = [row for _, row in rows if _bool(row, "detected", trial_path, 0)]
        within_rows = [
            row for _, row in rows if _bool(row, "inside_projected_target_radius", trial_path, 0)
        ]
        errors_px = [float(row["localization_error_px"]) for row in detected_rows]
        errors_mm = [float(row["localization_error_mm"]) for row in detected_rows]
        components = [int(row["red_components_over_20px"]) for _, row in rows]
        expected_values: dict[str, float | int | None] = {
            "trials": len(rows),
            "detections": len(detected_rows),
            "within_target_radius_localizations": len(within_rows),
            "detection_rate": len(detected_rows) / len(rows),
            "within_target_radius_rate": len(within_rows) / len(rows),
            "within_target_radius_rate_given_detection": (
                len(within_rows) / len(detected_rows) if detected_rows else None
            ),
            "median_localization_error_px": median(errors_px) if errors_px else None,
            "p95_localization_error_px": _quantile(errors_px, 0.95),
            "median_localization_error_mm": median(errors_mm) if errors_mm else None,
            "p95_localization_error_mm": _quantile(errors_mm, 0.95),
            "median_red_components_over_20px": median(components),
        }
        summary = summaries_by_key[key]
        json_summary = json_summaries_by_key[key]
        if any(
            summary[field] != json_summary.get(field)
            for field in ("lighting", "clutter")
        ):
            raise EvidenceAuditError(f"{summary_path} group {key}: factor labels disagree")
        for field, expected in expected_values.items():
            _check_numeric_summary(summary, field, expected, key, summary_path)
            json_row = {field: "" if json_summary.get(field) is None else str(json_summary[field])}
            _check_numeric_summary(json_row, field, expected, key, report_path)
    return len(trials), len(summaries)


def audit_candidate_validation_dataset(
    root: Path, *, stem: str = "perception-candidate-validation"
) -> tuple[int, int]:
    """Reconcile candidate-detector validation rows and grouped score summaries."""
    results = root / "results"
    trial_path = results / f"{stem}-trials.csv"
    summary_path = results / f"{stem}-summary.csv"
    report_path = results / f"{stem}-summary.json"
    trial_fields, trials = _read(trial_path)
    summary_fields, summaries = _read(summary_path)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceAuditError(
            f"cannot read candidate validation report {report_path}: {error}"
        ) from error
    if not trials or not summaries:
        raise EvidenceAuditError("candidate validation tables must not be empty")

    try:
        seeds = int(report["seeds"])
        base_seed = int(report["base_seed"])
        samples_per_seed = int(report["samples_per_seed"])
        expected_trials = int(report["trial_count"])
        strict_threshold = float(report["strict_localization_threshold_px"])
        projected_radius = float(report["projected_target_radius_px"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceAuditError(f"{report_path} has invalid validation metadata") from error
    if (
        seeds <= 0
        or samples_per_seed <= 0
        or expected_trials != len(trials)
        or not math.isfinite(strict_threshold)
        or strict_threshold <= 0.0
        or not math.isfinite(projected_radius)
        or projected_radius <= strict_threshold
    ):
        raise EvidenceAuditError(f"{report_path} has inconsistent validation metadata")

    required_trial_fields = {
        "condition", "clutter", "occlusion_fraction", "seed", "sample_index",
        "target_x_m", "target_y_m", "light_scale", "distractor_radius_factor",
        "occlusion_angle_deg", "target_center_x_px", "target_center_y_px",
        "baseline_detected", "baseline_error_px", "baseline_strict_success",
        "baseline_wrong_target", "candidate_detected", "candidate_error_px",
        "candidate_strict_success", "candidate_wrong_target", "candidate_abstained",
        "projected_target_radius_px",
    }
    required_summary_fields = {
        "condition", "clutter", "occlusion_fraction", "trials",
        "baseline_detections", "baseline_coverage_rate", "candidate_detections",
        "candidate_coverage_rate", "paired_strict_improvements",
        "paired_strict_regressions",
    }
    for prefix in ("baseline", "candidate"):
        required_summary_fields.update(
            {
                f"{prefix}_strict_successes", f"{prefix}_strict_success_rate",
                f"{prefix}_strict_success_rate_given_detection",
                f"{prefix}_imprecise_localizations", f"{prefix}_imprecise_rate",
                f"{prefix}_wrong_targets", f"{prefix}_wrong_target_rate",
                f"{prefix}_wrong_target_rate_given_detection",
                f"{prefix}_abstentions", f"{prefix}_abstention_rate",
                f"{prefix}_median_error_px", f"{prefix}_p95_error_px",
            }
        )
    for path, required, available in (
        (trial_path, required_trial_fields, set(trial_fields)),
        (summary_path, required_summary_fields, set(summary_fields)),
    ):
        missing = sorted(required - available)
        if missing:
            raise EvidenceAuditError(f"{path} is missing fields: {', '.join(missing)}")

    report_summaries = report.get("condition_summaries")
    if not isinstance(report_summaries, list) or not report_summaries:
        raise EvidenceAuditError(f"{report_path} has no condition summaries")
    report_by_condition = {
        item.get("condition"): item for item in report_summaries
        if isinstance(item, dict) and isinstance(item.get("condition"), str)
    }
    csv_by_condition = {row["condition"]: row for row in summaries}
    if len(csv_by_condition) != len(summaries) or set(csv_by_condition) != set(report_by_condition):
        raise EvidenceAuditError("candidate validation summary groups do not reconcile")

    expected_ids = {
        (seed, sample)
        for seed in range(base_seed, base_seed + seeds)
        for sample in range(samples_per_seed)
    }
    grouped: dict[str, list[dict[str, str]]] = {}
    identifiers_by_condition: dict[str, set[tuple[int, int]]] = {}
    for line, row in enumerate(trials, start=2):
        condition = row["condition"]
        grouped.setdefault(condition, []).append(row)
        try:
            identifier = (int(row["seed"]), int(row["sample_index"]))
        except ValueError as error:
            raise EvidenceAuditError(f"{trial_path}:{line} has invalid paired sample ID") from error
        ids = identifiers_by_condition.setdefault(condition, set())
        if identifier in ids:
            raise EvidenceAuditError(f"{trial_path}:{line} duplicates a paired sample ID")
        ids.add(identifier)
        for field in (
            "light_scale", "occlusion_fraction", "target_x_m", "target_y_m",
            "target_center_x_px", "target_center_y_px", "projected_target_radius_px",
        ):
            _finite_float(row, field, trial_path, line)
        for prefix in ("baseline", "candidate"):
            detected = _bool(row, f"{prefix}_detected", trial_path, line)
            strict = _bool(row, f"{prefix}_strict_success", trial_path, line)
            wrong = _bool(row, f"{prefix}_wrong_target", trial_path, line)
            error_raw = row.get(f"{prefix}_error_px", "")
            abstained = (
                _bool(row, "candidate_abstained", trial_path, line)
                if prefix == "candidate"
                else not detected
            )
            if detected:
                error_px = _finite_float(row, f"{prefix}_error_px", trial_path, line)
                if error_px < 0.0:
                    raise EvidenceAuditError(f"{trial_path}:{line} has negative localization error")
                if strict != (error_px <= strict_threshold):
                    raise EvidenceAuditError(f"{trial_path}:{line} has inconsistent strict outcome")
                if wrong != (error_px > projected_radius):
                    raise EvidenceAuditError(f"{trial_path}:{line} has inconsistent wrong-target outcome")
            elif error_raw:
                raise EvidenceAuditError(f"{trial_path}:{line} has error for an abstained detector")
            if abstained == detected:
                raise EvidenceAuditError(f"{trial_path}:{line} has inconsistent abstention status")
    if set(grouped) != set(report_by_condition):
        raise EvidenceAuditError("candidate validation trial conditions differ from report")
    if any(ids != expected_ids for ids in identifiers_by_condition.values()):
        raise EvidenceAuditError("candidate validation conditions do not share the full paired sample set")

    count_fields = (
        "trials", "baseline_detections", "baseline_strict_successes",
        "baseline_imprecise_localizations", "baseline_wrong_targets", "baseline_abstentions",
        "candidate_detections", "candidate_strict_successes",
        "candidate_imprecise_localizations", "candidate_wrong_targets",
        "candidate_abstentions", "paired_strict_improvements", "paired_strict_regressions",
    )
    rate_fields = (
        "baseline_coverage_rate", "baseline_strict_success_rate",
        "baseline_strict_success_rate_given_detection", "baseline_imprecise_rate",
        "baseline_wrong_target_rate", "baseline_wrong_target_rate_given_detection",
        "baseline_abstention_rate", "candidate_coverage_rate",
        "candidate_strict_success_rate", "candidate_strict_success_rate_given_detection",
        "candidate_imprecise_rate", "candidate_wrong_target_rate",
        "candidate_wrong_target_rate_given_detection", "candidate_abstention_rate",
    )
    for condition, rows in grouped.items():
        csv_summary = csv_by_condition[condition]
        json_summary = report_by_condition[condition]
        if csv_summary["clutter"] != json_summary.get("clutter"):
            raise EvidenceAuditError(f"candidate summary {condition}: clutter label differs")
        _check_numeric_summary(
            csv_summary, "occlusion_fraction", float(json_summary["occlusion_fraction"]),
            condition, summary_path,
        )
        actual: dict[str, float | int | None] = {"trials": len(rows)}
        for prefix in ("baseline", "candidate"):
            detected_rows = [row for row in rows if _bool(row, f"{prefix}_detected", trial_path, 0)]
            errors = [float(row[f"{prefix}_error_px"]) for row in detected_rows]
            strict_count = sum(error <= strict_threshold for error in errors)
            imprecise_count = sum(strict_threshold < error <= projected_radius for error in errors)
            wrong_count = sum(error > projected_radius for error in errors)
            abstentions = len(rows) - len(detected_rows)
            actual.update(
                {
                    f"{prefix}_detections": len(detected_rows),
                    f"{prefix}_coverage_rate": len(detected_rows) / len(rows),
                    f"{prefix}_strict_successes": strict_count,
                    f"{prefix}_strict_success_rate": strict_count / len(rows),
                    f"{prefix}_strict_success_rate_given_detection": (
                        strict_count / len(detected_rows) if detected_rows else None
                    ),
                    f"{prefix}_imprecise_localizations": imprecise_count,
                    f"{prefix}_imprecise_rate": imprecise_count / len(rows),
                    f"{prefix}_wrong_targets": wrong_count,
                    f"{prefix}_wrong_target_rate": wrong_count / len(rows),
                    f"{prefix}_wrong_target_rate_given_detection": (
                        wrong_count / len(detected_rows) if detected_rows else None
                    ),
                    f"{prefix}_abstentions": abstentions,
                    f"{prefix}_abstention_rate": abstentions / len(rows),
                    f"{prefix}_median_error_px": median(errors) if errors else None,
                    f"{prefix}_p95_error_px": _quantile(errors, 0.95),
                }
            )
        actual["paired_strict_improvements"] = sum(
            _bool(row, "candidate_strict_success", trial_path, 0)
            and not _bool(row, "baseline_strict_success", trial_path, 0)
            for row in rows
        )
        actual["paired_strict_regressions"] = sum(
            _bool(row, "baseline_strict_success", trial_path, 0)
            and not _bool(row, "candidate_strict_success", trial_path, 0)
            for row in rows
        )
        for field in (*count_fields, *rate_fields, "baseline_median_error_px", "baseline_p95_error_px",
                      "candidate_median_error_px", "candidate_p95_error_px"):
            expected = actual.get(field)
            _check_numeric_summary(csv_summary, field, expected, condition, summary_path)
            json_value = json_summary.get(field)
            json_row = {field: "" if json_value is None else str(json_value)}
            _check_numeric_summary(json_row, field, expected, condition, report_path)
    return len(trials), len(summaries)


OCCLUSION_CONDITIONS = ("clean", "marker_occlusion_50pct")


def audit_cooperative_pick_occlusion_dataset(root: Path, *, stage: str) -> tuple[int, int]:
    """Reconcile paired V3/V4 occlusion trials, summaries, and split provenance."""
    if stage not in {"development", "heldout"}:
        raise ValueError("stage must be 'development' or 'heldout'")
    directory = root / "results" / f"cooperative-pick-occlusion-{stage}"
    trial_path = directory / "cooperative-pick-occlusion-trials.csv"
    summary_path = directory / "cooperative-pick-occlusion-summary.json"
    try:
        report = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceAuditError(f"cannot read occlusion summary {summary_path}: {error}") from error
    _, trials = _read(trial_path)
    if not isinstance(report, dict) or report.get("stage") != stage:
        raise EvidenceAuditError(f"{summary_path} has an invalid experiment stage")
    split = report.get("split")
    if not isinstance(split, dict):
        raise EvidenceAuditError(f"{summary_path} has invalid split metadata")
    try:
        base_seed = int(split["base_seed"])
        seeds = int(split["seeds"])
        samples_per_seed = int(split["samples_per_seed"])
        pose_count = int(split["matched_target_poses"])
        trial_count = int(split["simulation_replay_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceAuditError(f"{summary_path} has invalid seed/sample metadata") from error
    if seeds <= 0 or samples_per_seed <= 0 or pose_count != seeds * samples_per_seed:
        raise EvidenceAuditError(f"{summary_path} has inconsistent matched-pose count")
    if trial_count != pose_count * len(OCCLUSION_CONDITIONS) or len(trials) != trial_count:
        raise EvidenceAuditError(f"{trial_path} row count does not match split metadata")

    verifier = report.get("verifier", {})
    if verifier.get("baseline") != "V3 frozen: 50 px rise, 12 px drift, 4 degree fiducial-angle change":
        raise EvidenceAuditError(f"{summary_path} does not record the frozen V3 thresholds")
    tolerances = verifier.get("candidate_gap_tolerances_px")
    if (
        not isinstance(tolerances, list)
        or not tolerances
        or any(not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 100 for value in tolerances)
        or len(set(tolerances)) != len(tolerances)
    ):
        raise EvidenceAuditError(f"{summary_path} has invalid candidate gap tolerances")
    if stage == "heldout":
        development_path = (
            root / "results" / "cooperative-pick-occlusion-development"
            / "cooperative-pick-occlusion-summary.json"
        )
        try:
            development = json.loads(development_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EvidenceAuditError(
                f"cannot read development provenance {development_path}: {error}"
            ) from error
        frozen_tolerance = development.get("selected_gap_tolerance_px")
        provenance = report.get("heldout_provenance")
        if (
            not isinstance(frozen_tolerance, int)
            or tolerances != [frozen_tolerance]
            or report.get("selected_gap_tolerance_px") != frozen_tolerance
            or not isinstance(provenance, dict)
            or provenance.get("candidate_gap_tolerance_px_frozen") != frozen_tolerance
            or provenance.get("development_base_seed") != development.get("split", {}).get("base_seed")
        ):
            raise EvidenceAuditError(f"{summary_path} does not match the frozen development candidate")

    expected_pairs = {
        f"{seed}-{sample:03d}"
        for seed in range(base_seed, base_seed + seeds)
        for sample in range(samples_per_seed)
    }
    paired: dict[str, dict[str, dict[str, str]]] = {}
    for line, row in enumerate(trials, start=2):
        pair_id = row.get("pair_id", "")
        condition = row.get("condition", "")
        if pair_id not in expected_pairs or condition not in OCCLUSION_CONDITIONS:
            raise EvidenceAuditError(f"{trial_path}:{line} has an unexpected pose or condition")
        if condition in paired.setdefault(pair_id, {}):
            raise EvidenceAuditError(f"{trial_path}:{line} duplicates {pair_id} / {condition}")
        if row.get("seed") != pair_id.split("-")[0]:
            raise EvidenceAuditError(f"{trial_path}:{line} seed does not match pair_id")
        _finite_float(row, "target_x_m", trial_path, line)
        _finite_float(row, "target_y_m", trial_path, line)
        for field in (
            "truth_raised_stable_at_two_frames",
            "payload_stable_lift_success",
            "full_sequence_collision_safe",
            "physical_stable_lift_success",
        ):
            _bool(row, field, trial_path, line)
        for tolerance in tolerances:
            for suffix in ("decision", "reason"):
                field = f"gap_{tolerance}_px_{suffix}"
                if not row.get(field):
                    raise EvidenceAuditError(f"{trial_path}:{line} is missing {field}")
        for classifier in ("v3", *(f"gap_{value}_px" for value in tolerances)):
            decision = row.get(f"{classifier}_decision")
            if decision not in {"confirmed", "not_confirmed", "abstain"}:
                raise EvidenceAuditError(f"{trial_path}:{line} has invalid {classifier} decision")
        paired[pair_id][condition] = row
    if set(paired) != expected_pairs or any(
        set(conditions) != set(OCCLUSION_CONDITIONS) for conditions in paired.values()
    ):
        raise EvidenceAuditError(f"{trial_path} does not contain a complete matched-pose block")

    for pair_id, conditions in paired.items():
        clean, occluded = conditions["clean"], conditions["marker_occlusion_50pct"]
        for field in (
            "target_x_m",
            "target_y_m",
            "truth_raised_stable_at_two_frames",
            "payload_stable_lift_success",
            "full_sequence_collision_safe",
            "physical_stable_lift_success",
            "failure_reason",
        ):
            if clean[field] != occluded[field]:
                raise EvidenceAuditError(f"{trial_path} changed matched pose {pair_id}: {field}")
    if report.get("physical_outcomes_identical_across_clean_and_occluded_replays_for_every_pose") is not True:
        raise EvidenceAuditError(f"{summary_path} does not record paired physical invariance")

    summaries = report.get("conditions")
    if not isinstance(summaries, dict) or set(summaries) != set(OCCLUSION_CONDITIONS):
        raise EvidenceAuditError(f"{summary_path} has incomplete condition summaries")
    for condition in OCCLUSION_CONDITIONS:
        rows = [pair[condition] for pair in paired.values()]
        summary = summaries[condition]
        classifier_fields = {"v3": "v3_decision"}
        classifier_fields.update({f"gap_tolerant_{value}px": f"gap_{value}_px_decision" for value in tolerances})
        for classifier, decision_field in classifier_fields.items():
            recorded = summary.get(classifier)
            if not isinstance(recorded, dict):
                raise EvidenceAuditError(f"{summary_path} is missing {condition} / {classifier}")
            positives = sum(_bool(row, "truth_raised_stable_at_two_frames", trial_path, 0) for row in rows)
            negatives = len(rows) - positives
            confirmed = [row for row in rows if row[decision_field] == "confirmed"]
            rejected = [row for row in rows if row[decision_field] == "not_confirmed"]
            actual = {
                "trials": len(rows),
                "capture_time_truth_positive": positives,
                "capture_time_truth_negative": negatives,
                "true_positive": sum(_bool(row, "truth_raised_stable_at_two_frames", trial_path, 0) for row in confirmed),
                "false_confirmation": sum(not _bool(row, "truth_raised_stable_at_two_frames", trial_path, 0) for row in confirmed),
                "true_negative": sum(not _bool(row, "truth_raised_stable_at_two_frames", trial_path, 0) for row in rejected),
                "false_nonconfirmation": sum(_bool(row, "truth_raised_stable_at_two_frames", trial_path, 0) for row in rejected),
                "abstained": sum(row[decision_field] == "abstain" for row in rows),
            }
            for field, value in actual.items():
                if recorded.get(field) != value:
                    raise EvidenceAuditError(
                        f"{summary_path} {condition} / {classifier}: {field} {recorded.get(field)!r} != {value}"
                    )
        physical = summary.get("physical_outcomes", {})
        expected_physical = {
            "payload_stable_hold_successes": sum(
                _bool(row, "payload_stable_lift_success", trial_path, 0) for row in rows
            ),
            "strict_full_sequence_successes": sum(
                _bool(row, "physical_stable_lift_success", trial_path, 0) for row in rows
            ),
            "trials": len(rows),
        }
        if physical != expected_physical:
            raise EvidenceAuditError(f"{summary_path} {condition}: physical outcomes do not reconcile")

    if stage == "development":
        clean_rows = [pair["clean"] for pair in paired.values()]
        occluded_rows = [pair["marker_occlusion_50pct"] for pair in paired.values()]
        eligible: list[tuple[int, int]] = []
        assessments: dict[str, dict[str, int | bool]] = {}
        for tolerance in tolerances:
            candidate_field = f"gap_{tolerance}_px_decision"
            clean_matches = sum(row[candidate_field] == row["v3_decision"] for row in clean_rows)
            false_confirmations = sum(
                row[candidate_field] == "confirmed"
                and not _bool(row, "truth_raised_stable_at_two_frames", trial_path, 0)
                for row in (*clean_rows, *occluded_rows)
            )
            true_positives = sum(
                row[candidate_field] == "confirmed"
                and _bool(row, "truth_raised_stable_at_two_frames", trial_path, 0)
                for row in occluded_rows
            )
            is_eligible = clean_matches == len(clean_rows) and false_confirmations == 0
            assessments[str(tolerance)] = {
                "clean_decisions_matching_v3": clean_matches,
                "clean_trials": len(clean_rows),
                "occlusion_true_positives": true_positives,
                "occlusion_false_confirmations": sum(
                    row[candidate_field] == "confirmed"
                    and not _bool(row, "truth_raised_stable_at_two_frames", trial_path, 0)
                    for row in occluded_rows
                ),
                "eligible": is_eligible,
            }
            if is_eligible:
                eligible.append((true_positives, tolerance))
        selected = None
        if eligible:
            best = max(value for value, _ in eligible)
            selected = min(tolerance for value, tolerance in eligible if value == best)
        recorded_selection = report.get("development_selection")
        if (
            not isinstance(recorded_selection, dict)
            or recorded_selection.get("candidate_assessments") != assessments
            or recorded_selection.get("selected_gap_tolerance_px") != selected
            or report.get("selected_gap_tolerance_px") != selected
        ):
            raise EvidenceAuditError(f"{summary_path} does not reconcile with the development selection rule")
    return len(trials), len(summaries)


def audit_cooperative_pick_occlusion_shape_challenge(root: Path) -> tuple[int, int]:
    """Reconcile the frozen occlusion-geometry challenge and its paired outcomes."""
    directory = root / "results" / "cooperative-pick-occlusion-shape-challenge"
    trial_path = directory / "cooperative-pick-occlusion-shape-trials.csv"
    summary_path = directory / "cooperative-pick-occlusion-shape-summary.json"
    try:
        report = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceAuditError(f"cannot read occlusion challenge summary {summary_path}: {error}") from error
    _, rows = _read(trial_path)
    if (
        not isinstance(report, dict)
        or report.get("frozen_candidate_provenance", {}).get("gap_tolerance_px") != 28
        or report.get("frozen_candidate_provenance", {}).get("no_threshold_search_in_this_challenge") is not True
    ):
        raise EvidenceAuditError(f"{summary_path} does not preserve the frozen V4 candidate")
    strata = report.get("seed_blocks")
    conditions = report.get("image_conditions")
    if not isinstance(strata, list) or not strata or not isinstance(conditions, dict) or not conditions:
        raise EvidenceAuditError(f"{summary_path} has incomplete seed or mask metadata")
    expected_pose_strata: dict[str, str] = {}
    expected_pad_by_stratum: dict[str, float] = {}
    for item in strata:
        try:
            name = str(item["name"])
            base_seed = int(item["base_seed"])
            seed_count = int(item["seeds"])
            samples_per_seed = int(item["samples_per_seed"])
            expected_count = int(item["unique_physical_pose_runs"])
            pad_half_size = float(item["finger_pad_half_size_m"])
        except (KeyError, TypeError, ValueError) as error:
            raise EvidenceAuditError(f"{summary_path} has invalid stratum metadata") from error
        if seed_count <= 0 or samples_per_seed <= 0 or expected_count != seed_count * samples_per_seed:
            raise EvidenceAuditError(f"{summary_path} has inconsistent pose counts for {name}")
        expected_pad_by_stratum[name] = pad_half_size
        for seed in range(base_seed, base_seed + seed_count):
            for sample in range(samples_per_seed):
                pair_id = f"{seed}-{sample:03d}"
                if pair_id in expected_pose_strata:
                    raise EvidenceAuditError(f"{summary_path} reuses seed/sample pair {pair_id}")
                expected_pose_strata[pair_id] = name
    expected_conditions = set(conditions)
    if not expected_conditions or "clean" not in expected_conditions:
        raise EvidenceAuditError(f"{summary_path} is missing its clean image condition")
    expected_rows = len(expected_pose_strata) * len(expected_conditions)
    if (
        report.get("unique_physical_pose_runs") != len(expected_pose_strata)
        or report.get("image_condition_evaluation_rows") != expected_rows
        or len(rows) != expected_rows
    ):
        raise EvidenceAuditError(f"{trial_path} row count does not match the challenge design")

    paired: dict[str, dict[str, dict[str, str]]] = {}
    for line, row in enumerate(rows, start=2):
        pair_id = row.get("pair_id", "")
        condition = row.get("image_condition", "")
        stratum = expected_pose_strata.get(pair_id)
        if stratum is None or row.get("stratum") != stratum or condition not in expected_conditions:
            raise EvidenceAuditError(f"{trial_path}:{line} has an unexpected pose, stratum, or mask")
        if condition in paired.setdefault(pair_id, {}):
            raise EvidenceAuditError(f"{trial_path}:{line} duplicates {pair_id} / {condition}")
        if row.get("seed") != pair_id.split("-")[0]:
            raise EvidenceAuditError(f"{trial_path}:{line} seed does not match pair_id")
        expected_pad = expected_pad_by_stratum[stratum]
        if abs(_finite_float(row, "finger_pad_half_size_m", trial_path, line) - expected_pad) > 1e-9:
            raise EvidenceAuditError(f"{trial_path}:{line} uses the wrong pad geometry for {stratum}")
        _finite_float(row, "target_x_m", trial_path, line)
        _finite_float(row, "target_y_m", trial_path, line)
        for field in (
            "truth_raised_stable_at_two_frames",
            "payload_stable_lift_success",
            "full_sequence_collision_safe",
            "physical_stable_lift_success",
        ):
            _bool(row, field, trial_path, line)
        if row.get("v3_decision") not in {"confirmed", "not_confirmed", "abstain"}:
            raise EvidenceAuditError(f"{trial_path}:{line} has invalid V3 decision")
        if row.get("v4_decision") not in {"confirmed", "not_confirmed", "abstain"}:
            raise EvidenceAuditError(f"{trial_path}:{line} has invalid V4 decision")
        expected_fault = conditions[condition].get("fault")
        try:
            actual_fault = json.loads(row.get("fault") or "null")
        except json.JSONDecodeError as error:
            raise EvidenceAuditError(f"{trial_path}:{line} has invalid fault JSON") from error
        if actual_fault != expected_fault:
            raise EvidenceAuditError(f"{trial_path}:{line} fault does not match {condition}")
        paired[pair_id][condition] = row
    if set(paired) != set(expected_pose_strata) or any(
        set(images) != expected_conditions for images in paired.values()
    ):
        raise EvidenceAuditError(f"{trial_path} is missing matched image conditions")

    fields = (
        "stratum",
        "finger_pad_half_size_m",
        "target_x_m",
        "target_y_m",
        "truth_raised_stable_at_two_frames",
        "payload_stable_lift_success",
        "full_sequence_collision_safe",
        "physical_stable_lift_success",
        "failure_reason",
    )
    for pair_id, images in paired.items():
        reference = images["clean"]
        for condition, row in images.items():
            if any(row[field] != reference[field] for field in fields):
                raise EvidenceAuditError(f"{trial_path} changed physical pose {pair_id} under {condition}")
    if report.get("physical_outcomes_identical_for_all_image_conditions") is not True:
        raise EvidenceAuditError(f"{summary_path} does not assert physical invariance")

    summary_strata = report.get("strata")
    if not isinstance(summary_strata, dict) or set(summary_strata) != set(expected_pad_by_stratum):
        raise EvidenceAuditError(f"{summary_path} has incomplete stratum summaries")
    for stratum in expected_pad_by_stratum:
        stratum_rows = [row for row in rows if row["stratum"] == stratum]
        unique = {row["pair_id"]: row for row in stratum_rows}
        physical = summary_strata[stratum].get("physical_outcomes", {})
        expected_physical = {
            "unique_physical_pose_runs": len(unique),
            "capture_time_truth_positive": sum(_bool(row, "truth_raised_stable_at_two_frames", trial_path, 0) for row in unique.values()),
            "capture_time_truth_negative": sum(not _bool(row, "truth_raised_stable_at_two_frames", trial_path, 0) for row in unique.values()),
            "payload_stable_hold_successes": sum(_bool(row, "payload_stable_lift_success", trial_path, 0) for row in unique.values()),
            "strict_full_sequence_successes": sum(_bool(row, "physical_stable_lift_success", trial_path, 0) for row in unique.values()),
            "failure_reasons": dict(sorted(Counter(row["failure_reason"] for row in unique.values()).items())),
        }
        if physical != expected_physical:
            raise EvidenceAuditError(f"{summary_path} physical counts do not reconcile for {stratum}")
        condition_summaries = summary_strata[stratum].get("image_conditions", {})
        if set(condition_summaries) != expected_conditions:
            raise EvidenceAuditError(f"{summary_path} omits a mask condition for {stratum}")
        for condition in expected_conditions:
            selected = [row for row in stratum_rows if row["image_condition"] == condition]
            record = condition_summaries[condition]
            for name, field in (("v3", "v3_decision"), ("v4_frozen_28px", "v4_decision")):
                recorded = record.get(name)
                if not isinstance(recorded, dict):
                    raise EvidenceAuditError(f"{summary_path} omits {stratum} / {condition} / {name}")
                positives = sum(_bool(row, "truth_raised_stable_at_two_frames", trial_path, 0) for row in selected)
                negatives = len(selected) - positives
                confirmed = [row for row in selected if row[field] == "confirmed"]
                rejected = [row for row in selected if row[field] == "not_confirmed"]
                expected = {
                    "evaluations": len(selected),
                    "capture_time_truth_positive": positives,
                    "capture_time_truth_negative": negatives,
                    "true_positive": sum(_bool(row, "truth_raised_stable_at_two_frames", trial_path, 0) for row in confirmed),
                    "false_confirmation": sum(not _bool(row, "truth_raised_stable_at_two_frames", trial_path, 0) for row in confirmed),
                    "true_negative": sum(not _bool(row, "truth_raised_stable_at_two_frames", trial_path, 0) for row in rejected),
                    "false_nonconfirmation": sum(_bool(row, "truth_raised_stable_at_two_frames", trial_path, 0) for row in rejected),
                    "abstained": sum(row[field] == "abstain" for row in selected),
                }
                if any(recorded.get(key) != value for key, value in expected.items()):
                    raise EvidenceAuditError(f"{summary_path} counts do not reconcile for {stratum} / {condition} / {name}")
            expected_changes = sum(row["v3_decision"] != row["v4_decision"] for row in selected)
            if record.get("v4_decision_changes_from_v3") != expected_changes:
                raise EvidenceAuditError(f"{summary_path} V3/V4 mismatch count does not reconcile for {stratum} / {condition}")
    return len(rows), len(expected_conditions)


def audit_cooperative_pick_orthogonal_validation(root: Path) -> tuple[int, int]:
    """Reconcile orthogonal-verifier rows, physical labels, and summary counts."""
    directory = root / "results" / "cooperative-pick-orthogonal-verification-heldout"
    trial_path = directory / "cooperative-pick-orthogonal-verification-trials.csv"
    summary_path = directory / "cooperative-pick-orthogonal-verification-summary.json"
    try:
        report = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceAuditError(
            f"cannot read orthogonal-verification summary {summary_path}: {error}"
        ) from error
    _, rows = _read(trial_path)
    if not rows or not isinstance(report, dict):
        raise EvidenceAuditError("orthogonal-verification evidence must not be empty")
    if (
        report.get("protocol_version") != 1
        or report.get("simulator_truth_is_input_to_vision") is not False
        or report.get("occluder", {}).get("collision_enabled") is not False
        or report.get("minimum_unique_capture_time_negative_poses") != 80
    ):
        raise EvidenceAuditError(f"{summary_path} has inconsistent protocol metadata")

    strata = report.get("strata")
    if not isinstance(strata, list) or not strata:
        raise EvidenceAuditError(f"{summary_path} has no seed-block metadata")
    expected_ids_by_stratum: dict[str, set[str]] = {}
    expected_pad_sizes: dict[str, float] = {}
    for item in strata:
        try:
            name = str(item["name"])
            base_seed = int(item["base_seed"])
            seed_count = int(item["seeds"])
            samples_per_seed = int(item["samples_per_seed"])
            expected_poses = int(item["unique_physical_pose_runs"])
            pad_size = float(item["finger_pad_half_size_m"])
        except (KeyError, TypeError, ValueError) as error:
            raise EvidenceAuditError(f"{summary_path} has invalid seed metadata") from error
        if seed_count <= 0 or samples_per_seed <= 0 or expected_poses != seed_count * samples_per_seed:
            raise EvidenceAuditError(f"{summary_path} has invalid pose count for {name}")
        expected_ids_by_stratum[name] = {
            f"{seed}-{sample:03d}"
            for seed in range(base_seed, base_seed + seed_count)
            for sample in range(samples_per_seed)
        }
        expected_pad_sizes[name] = pad_size

    required_fields = {
        "pair_id", "seed", "sample_index", "stratum", "finger_pad_half_size_m",
        "target_x_m", "target_y_m", "capture_truth_raised_stable", "capture_lift_m",
        "capture_tilt_deg", "v3_decision", "v4_side_decision", "orthogonal_v4_decision",
        "payload_stable_lift_success", "full_sequence_collision_safe",
        "physical_stable_lift_success", "failure_reason",
    }
    fields = set(rows[0])
    missing = sorted(required_fields - fields)
    if missing:
        raise EvidenceAuditError(f"{trial_path} is missing fields: {', '.join(missing)}")
    seen: dict[str, dict[str, dict[str, str]]] = {name: {} for name in expected_ids_by_stratum}
    for line, row in enumerate(rows, start=2):
        stratum = row["stratum"]
        pair_id = row["pair_id"]
        if stratum not in seen or pair_id not in expected_ids_by_stratum[stratum]:
            raise EvidenceAuditError(f"{trial_path}:{line} has an unexpected pose identity")
        pose_rows = seen[stratum]
        if pair_id in pose_rows:
            raise EvidenceAuditError(f"{trial_path}:{line} duplicates physical pose {pair_id}")
        if row["seed"] != pair_id.split("-")[0]:
            raise EvidenceAuditError(f"{trial_path}:{line} seed does not match pair_id")
        _finite_float(row, "target_x_m", trial_path, line)
        _finite_float(row, "target_y_m", trial_path, line)
        pad_size = _finite_float(row, "finger_pad_half_size_m", trial_path, line)
        if abs(pad_size - expected_pad_sizes[stratum]) > 1e-9:
            raise EvidenceAuditError(f"{trial_path}:{line} has the wrong finger-pad geometry")
        for field in (
            "capture_truth_raised_stable", "payload_stable_lift_success",
            "full_sequence_collision_safe", "physical_stable_lift_success",
        ):
            _bool(row, field, trial_path, line)
        for method_field in ("v3_decision", "v4_side_decision", "orthogonal_v4_decision"):
            if row[method_field] not in {"confirmed", "not_confirmed", "abstain"}:
                raise EvidenceAuditError(f"{trial_path}:{line} has invalid {method_field}")
        for field in ("capture_lift_m", "capture_tilt_deg"):
            try:
                values = json.loads(row[field])
            except json.JSONDecodeError as error:
                raise EvidenceAuditError(f"{trial_path}:{line} has invalid {field} JSON") from error
            if not isinstance(values, list) or len(values) != 2 or any(
                not isinstance(value, (int, float)) or not math.isfinite(value) for value in values
            ):
                raise EvidenceAuditError(f"{trial_path}:{line} has invalid {field} values")
        pose_rows[pair_id] = row

    if any(set(seen[name]) != ids for name, ids in expected_ids_by_stratum.items()):
        raise EvidenceAuditError(f"{trial_path} does not match the full seed/sample design")
    if report.get("total_unique_physical_pose_runs") != len(rows):
        raise EvidenceAuditError(f"{summary_path} physical pose count does not reconcile")
    results = report.get("results")
    if not isinstance(results, dict) or set(results) != set(expected_ids_by_stratum):
        raise EvidenceAuditError(f"{summary_path} has incomplete strata")

    total_negatives = 0
    for stratum, pose_rows in seen.items():
        unique_rows = list(pose_rows.values())
        positives = sum(_bool(row, "capture_truth_raised_stable", trial_path, 0) for row in unique_rows)
        negatives = len(unique_rows) - positives
        total_negatives += negatives
        summary = results[stratum]
        physical_expected = {
            "unique_physical_pose_runs": len(unique_rows),
            "capture_truth_positive": positives,
            "capture_truth_negative": negatives,
            "stable_hold_successes": sum(_bool(row, "payload_stable_lift_success", trial_path, 0) for row in unique_rows),
            "strict_full_cycle_successes": sum(_bool(row, "physical_stable_lift_success", trial_path, 0) for row in unique_rows),
            "failure_reasons": dict(sorted(Counter(row["failure_reason"] for row in unique_rows).items())),
        }
        if any(summary.get(key) != expected for key, expected in physical_expected.items()):
            raise EvidenceAuditError(f"{summary_path} physical outcomes do not reconcile for {stratum}")
        methods = summary.get("methods")
        if not isinstance(methods, dict) or set(methods) != {"V3_side", "V4_side_28px", "V4_two_view"}:
            raise EvidenceAuditError(f"{summary_path} has incomplete verifier results for {stratum}")
        for method, field in (
            ("V3_side", "v3_decision"),
            ("V4_side_28px", "v4_side_decision"),
            ("V4_two_view", "orthogonal_v4_decision"),
        ):
            confirmed = [row for row in unique_rows if row[field] == "confirmed"]
            rejected = [row for row in unique_rows if row[field] == "not_confirmed"]
            expected = {
                "evaluations": len(unique_rows),
                "capture_truth_positive": positives,
                "capture_truth_negative": negatives,
                "true_positive": sum(_bool(row, "capture_truth_raised_stable", trial_path, 0) for row in confirmed),
                "false_confirmation": sum(not _bool(row, "capture_truth_raised_stable", trial_path, 0) for row in confirmed),
                "true_negative": sum(not _bool(row, "capture_truth_raised_stable", trial_path, 0) for row in rejected),
                "false_nonconfirmation": sum(_bool(row, "capture_truth_raised_stable", trial_path, 0) for row in rejected),
                "abstained": sum(row[field] == "abstain" for row in unique_rows),
            }
            recorded = methods[method]
            if any(recorded.get(key) != value for key, value in expected.items()):
                raise EvidenceAuditError(f"{summary_path} counts do not reconcile for {stratum} / {method}")
    if total_negatives < 80 or report.get("minimum_negative_pose_requirement_met") is not True:
        raise EvidenceAuditError(f"{summary_path} has fewer than 80 unique negative poses")
    return len(rows), len(expected_ids_by_stratum)


def audit_cooperative_pick_material_validation(root: Path) -> tuple[int, int]:
    """Reconcile paired material-stress rows, body mass, and success intervals."""
    directory = root / "results" / "cooperative-pick-material-validation"
    trial_path = directory / "cooperative-pick-material-trials.csv"
    summary_path = directory / "cooperative-pick-material-summary.json"
    try:
        report = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceAuditError(
            f"cannot read material-validation summary {summary_path}: {error}"
        ) from error
    _, rows = _read(trial_path)
    if not rows or not isinstance(report, dict):
        raise EvidenceAuditError("material-validation evidence must not be empty")
    if (
        report.get("protocol_version") != 1
        or report.get("evidence_type") != "paired MuJoCo simulation; not hardware evidence"
        or report.get("paired_design")
        != "same target pose for every material condition and both camera-estimate/oracle controllers"
        or report.get("controller_conditions") != ["camera_estimate", "oracle_xy"]
    ):
        raise EvidenceAuditError(f"{summary_path} has inconsistent protocol metadata")
    sampling = report.get("target_sampling", {})
    try:
        base_seed = int(sampling["base_seed"])
        samples_per_seed = int(sampling["samples_per_seed"])
        masses = tuple(float(value) for value in report["varied_material_assumptions"]["payload_body_mass_kg"])
        frictions = tuple(float(value) for value in report["varied_material_assumptions"]["finger_pad_and_payload_sliding_friction"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceAuditError(f"{summary_path} has invalid material or seed metadata") from error
    if (
        samples_per_seed <= 0
        or not masses
        or not frictions
        or any(not math.isfinite(value) or value <= 0.0 for value in (*masses, *frictions))
    ):
        raise EvidenceAuditError(f"{summary_path} has invalid material or seed values")
    condition_parameters = {
        f"mass-{mass:g}_friction-{friction:g}": (mass, friction)
        for mass in masses
        for friction in frictions
    }
    expected_conditions = set(condition_parameters)
    summaries = report.get("conditions")
    if not isinstance(summaries, dict) or set(summaries) != expected_conditions:
        raise EvidenceAuditError(f"{summary_path} material conditions are incomplete")
    if report.get("unique_target_poses") != samples_per_seed:
        raise EvidenceAuditError(f"{summary_path} target-pose count does not reconcile")
    if report.get("unique_pose_material_pairs") != samples_per_seed * len(expected_conditions):
        raise EvidenceAuditError(f"{summary_path} pose/material-pair count does not reconcile")
    if report.get("total_physics_trials") != len(rows):
        raise EvidenceAuditError(f"{summary_path} trial count does not reconcile")

    required_fields = {
        "pair_id", "seed", "sample_index", "condition", "controller",
        "target_x_m", "target_y_m", "payload_body_mass_kg",
        "finger_pad_half_size_m", "finger_pad_sliding_friction", "approach_clearance_m",
        "release_profile", "payload_stable_lift_success", "full_sequence_collision_safe",
        "physical_stable_lift_success", "failure_reason", "fresh_visual_confirmation",
    }
    missing = sorted(required_fields - set(rows[0]))
    if missing:
        raise EvidenceAuditError(f"{trial_path} is missing fields: {', '.join(missing)}")

    expected_pairs = {
        f"{base_seed}-{sample:03d}" for sample in range(samples_per_seed)
    }
    rows_by_group: dict[tuple[str, str], list[dict[str, str]]] = {}
    target_positions: dict[str, tuple[float, float]] = {}
    for line, row in enumerate(rows, start=2):
        pair_id = row["pair_id"]
        condition = row["condition"]
        controller = row["controller"]
        if pair_id not in expected_pairs or condition not in expected_conditions:
            raise EvidenceAuditError(f"{trial_path}:{line} has an unexpected pose or condition")
        if controller not in {"camera_estimate", "oracle_xy"}:
            raise EvidenceAuditError(f"{trial_path}:{line} has an invalid controller")
        try:
            sample_index = int(row["sample_index"])
        except ValueError as error:
            raise EvidenceAuditError(f"{trial_path}:{line} has an invalid sample index") from error
        if (
            row["seed"] != str(base_seed)
            or sample_index != int(pair_id.split("-")[1])
            or pair_id != f"{base_seed}-{sample_index:03d}"
        ):
            raise EvidenceAuditError(f"{trial_path}:{line} seed/sample identity is inconsistent")
        x = _finite_float(row, "target_x_m", trial_path, line)
        y = _finite_float(row, "target_y_m", trial_path, line)
        if abs(x) > 0.09 + 1e-9 or abs(y) > 0.09 + 1e-9:
            raise EvidenceAuditError(f"{trial_path}:{line} target is outside the frozen sampling range")
        previous_position = target_positions.setdefault(pair_id, (x, y))
        if not (
            math.isclose(x, previous_position[0], abs_tol=1e-12)
            and math.isclose(y, previous_position[1], abs_tol=1e-12)
        ):
            raise EvidenceAuditError(f"{trial_path}:{line} paired pose coordinates changed")
        mass = _finite_float(row, "payload_body_mass_kg", trial_path, line)
        friction = _finite_float(row, "finger_pad_sliding_friction", trial_path, line)
        if condition != f"mass-{mass:g}_friction-{friction:g}":
            raise EvidenceAuditError(f"{trial_path}:{line} material label does not match parameters")
        configured_mass, configured_friction = condition_parameters[condition]
        if not math.isclose(mass, configured_mass, rel_tol=0.0, abs_tol=2e-9):
            raise EvidenceAuditError(f"{trial_path}:{line} payload body mass differs from frozen condition")
        if not math.isclose(
            friction, configured_friction, rel_tol=0.0, abs_tol=1e-12
        ):
            raise EvidenceAuditError(
                f"{trial_path}:{line} friction differs from frozen condition"
            )
        if _finite_float(row, "finger_pad_half_size_m", trial_path, line) != 0.016:
            raise EvidenceAuditError(f"{trial_path}:{line} pad geometry changed")
        if (
            _finite_float(row, "approach_clearance_m", trial_path, line) != 0.020
            or row["release_profile"] != "clearance"
        ):
            raise EvidenceAuditError(
                f"{trial_path}:{line} controller settings changed"
            )
        for field in (
            "payload_stable_lift_success", "full_sequence_collision_safe",
            "physical_stable_lift_success",
        ):
            _bool(row, field, trial_path, line)
        if row["fresh_visual_confirmation"] != "not measured":
            raise EvidenceAuditError(f"{trial_path}:{line} invents a visual confirmation result")
        if not row["failure_reason"]:
            raise EvidenceAuditError(f"{trial_path}:{line} has no failure outcome label")
        rows_by_group.setdefault((condition, controller), []).append(row)

    if set(target_positions) != expected_pairs:
        raise EvidenceAuditError(f"{trial_path} does not cover every expected target pose")
    for condition in expected_conditions:
        condition_summary = summaries[condition]
        if condition_summary.get("trials") != samples_per_seed:
            raise EvidenceAuditError(f"{summary_path} trial count does not reconcile for {condition}")
        expected_mass, _ = condition_parameters[condition]
        reported_mass = condition_summary.get("payload_body_mass_kg")
        if not isinstance(reported_mass, (int, float)) or not math.isclose(
            float(reported_mass), expected_mass, rel_tol=0.0, abs_tol=2e-9
        ):
            raise EvidenceAuditError(
                f"{summary_path} body mass does not reconcile for {condition}"
            )
        for controller in ("camera_estimate", "oracle_xy"):
            selected = rows_by_group.get((condition, controller), [])
            if (
                len(selected) != samples_per_seed
                or {row["pair_id"] for row in selected} != expected_pairs
            ):
                raise EvidenceAuditError(
                    f"{trial_path} is missing paired trials for {condition}/{controller}"
                )
            recorded = condition_summary.get("controllers", {}).get(controller)
            if not isinstance(recorded, dict) or recorded.get("trials") != samples_per_seed:
                raise EvidenceAuditError(f"{summary_path} is missing {condition}/{controller}")
            if recorded.get("fresh_visual_confirmation") != "not measured in material sweep":
                raise EvidenceAuditError(f"{summary_path} conflates physical and visual success")
            for metric in (
                "physical_stable_lift_success", "payload_stable_lift_success",
                "full_sequence_collision_safe",
            ):
                successes = sum(_bool(row, metric, trial_path, 0) for row in selected)
                metric_record = recorded.get(metric)
                if not isinstance(metric_record, dict) or metric_record.get("successes") != successes:
                    raise EvidenceAuditError(
                        f"{summary_path} {metric} counts do not reconcile for {condition}/{controller}"
                    )
                if metric_record.get("rate") != round(successes / samples_per_seed, 4):
                    raise EvidenceAuditError(
                        f"{summary_path} {metric} rate does not reconcile for {condition}/{controller}"
                    )
                if metric_record.get("wilson_95_interval") != _audit_wilson_interval(successes, samples_per_seed):
                    raise EvidenceAuditError(
                        f"{summary_path} {metric} interval does not reconcile for {condition}/{controller}"
                    )
            expected_failures = dict(sorted(Counter(row["failure_reason"] for row in selected).items()))
            if recorded.get("failure_reasons") != expected_failures:
                raise EvidenceAuditError(
                    f"{summary_path} failure reasons do not reconcile for {condition}/{controller}"
                )
    return len(rows), len(expected_conditions)


def audit_cooperative_pick_pad_friction_validation(root: Path) -> tuple[int, int]:
    """Reconcile the held-out pad-size/friction experiment and its replay files."""
    directory = root / "results" / "cooperative-pick-pad-friction-validation"
    trial_path = directory / "cooperative-pick-pad-friction-trials.csv"
    summary_path = directory / "cooperative-pick-pad-friction-summary.json"
    try:
        report = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceAuditError(
            f"cannot read pad-friction summary {summary_path}: {error}"
        ) from error
    _, rows = _read(trial_path)
    if not rows or not isinstance(report, dict):
        raise EvidenceAuditError("pad-friction evidence must not be empty")
    if (
        report.get("protocol_version") != 1
        or report.get("evidence_type") != "paired MuJoCo simulation; not hardware evidence"
        or report.get("paired_design")
        != "same target pose for each pad size, friction level, and camera-estimate/oracle controller"
        or report.get("controllers") != ["camera_estimate", "oracle_xy"]
    ):
        raise EvidenceAuditError(f"{summary_path} has inconsistent protocol metadata")
    sampling = report.get("target_sampling", {})
    settings = report.get("frozen_settings", {})
    varied = report.get("varied_settings", {})
    try:
        base_seed = int(sampling["base_seed"])
        sample_count = int(sampling["samples_per_seed"])
        mass = float(settings["payload_body_mass_kg"])
        pads = tuple(float(value) for value in varied["finger_pad_half_size_m"])
        frictions = tuple(
            float(value)
            for value in varied["finger_pad_and_payload_sliding_friction"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceAuditError(f"{summary_path} has invalid frozen settings") from error
    if (
        base_seed != 40900
        or sample_count != 12
        or pads != (0.016, 0.018)
        or frictions != (0.5, 2.2)
        or not math.isclose(mass, 0.7201166534540158, rel_tol=0.0, abs_tol=1e-12)
        or settings.get("lowest_approach_tool_height_offset_m") != 0.020
        or settings.get("release_profile") != "clearance"
        or settings.get("baseline_pad_half_size_m") != pads[0]
        or settings.get("candidate_pad_half_size_m") != pads[1]
    ):
        raise EvidenceAuditError(f"{summary_path} changed the frozen pad/friction protocol")
    expected_pairs = {f"{base_seed}-{index:03d}" for index in range(sample_count)}
    expected_group_rows = {
        (pair_id, controller, pad, friction)
        for pair_id in expected_pairs
        for controller in ("camera_estimate", "oracle_xy")
        for pad in pads
        for friction in frictions
    }
    if len(rows) != len(expected_group_rows) or report.get("total_physics_trials") != len(rows):
        raise EvidenceAuditError(f"{summary_path} does not reconcile to 96 physics trials")
    if report.get("fresh_visual_confirmation") != (
        "not measured; physical stable-lift outcomes do not imply visual confirmation"
    ):
        raise EvidenceAuditError(f"{summary_path} conflates physical and visual outcomes")

    required_fields = {
        "pair_id", "seed", "sample_index", "controller", "finger_pad_half_size_m",
        "friction", "target_x_m", "target_y_m", "payload_body_mass_kg",
        "approach_clearance_m", "release_profile", "physical_stable_lift_success",
        "payload_stable_lift_success", "full_sequence_collision_safe",
        "minimum_finger_payload_contact_points_during_hold",
        "minimum_payload_z_during_hold_m", "maximum_payload_tilt_during_hold_deg",
        "physics_steps_with_arm_table_or_ground_contact",
        "physics_steps_with_nonfinger_arm_payload_contact", "failure_reason",
        "fresh_visual_confirmation",
    }
    missing = sorted(required_fields - set(rows[0]))
    if missing:
        raise EvidenceAuditError(f"{trial_path} is missing fields: {', '.join(missing)}")
    rows_by_group: dict[tuple[str, str, float, float], list[dict[str, str]]] = {}
    target_positions: dict[str, tuple[float, float]] = {}
    seen: set[tuple[str, str, float, float]] = set()
    for line, row in enumerate(rows, start=2):
        pair_id = row["pair_id"]
        controller = row["controller"]
        if pair_id not in expected_pairs or controller not in {"camera_estimate", "oracle_xy"}:
            raise EvidenceAuditError(f"{trial_path}:{line} has an unexpected trial identity")
        seed = int(row["seed"])
        sample_index = int(row["sample_index"])
        if seed != base_seed or pair_id != f"{seed}-{sample_index:03d}":
            raise EvidenceAuditError(f"{trial_path}:{line} seed/sample identity is inconsistent")
        pad = _finite_float(row, "finger_pad_half_size_m", trial_path, line)
        friction = _finite_float(row, "friction", trial_path, line)
        key = (pair_id, controller, pad, friction)
        if key not in expected_group_rows or key in seen:
            raise EvidenceAuditError(f"{trial_path}:{line} has a duplicate or unexpected condition")
        seen.add(key)
        x = _finite_float(row, "target_x_m", trial_path, line)
        y = _finite_float(row, "target_y_m", trial_path, line)
        if abs(x) > 0.09 + 1e-9 or abs(y) > 0.09 + 1e-9:
            raise EvidenceAuditError(f"{trial_path}:{line} target is outside the frozen range")
        previous = target_positions.setdefault(pair_id, (x, y))
        if not (math.isclose(x, previous[0], abs_tol=1e-12) and math.isclose(y, previous[1], abs_tol=1e-12)):
            raise EvidenceAuditError(f"{trial_path}:{line} paired pose coordinates changed")
        if (
            not math.isclose(_finite_float(row, "payload_body_mass_kg", trial_path, line), mass, rel_tol=0.0, abs_tol=1e-12)
            or _finite_float(row, "approach_clearance_m", trial_path, line) != 0.020
            or row["release_profile"] != "clearance"
        ):
            raise EvidenceAuditError(f"{trial_path}:{line} changed a frozen controller or mass setting")
        for field in (
            "physical_stable_lift_success", "payload_stable_lift_success",
            "full_sequence_collision_safe",
        ):
            _bool(row, field, trial_path, line)
        if row["fresh_visual_confirmation"] != "not measured" or not row["failure_reason"]:
            raise EvidenceAuditError(f"{trial_path}:{line} has an invalid perception/failure label")
        rows_by_group.setdefault((f"friction-{friction:g}", controller, pad, friction), []).append(row)
    if seen != expected_group_rows or set(target_positions) != expected_pairs:
        raise EvidenceAuditError(f"{trial_path} does not contain every paired target pose/condition")

    conditions = report.get("conditions")
    paired_report = report.get("paired_success_outcomes")
    if not isinstance(conditions, dict) or not isinstance(paired_report, dict):
        raise EvidenceAuditError(f"{summary_path} is missing condition or paired outcomes")
    for friction in frictions:
        condition = f"friction-{friction:g}"
        if condition not in conditions or condition not in paired_report:
            raise EvidenceAuditError(f"{summary_path} is missing {condition}")
        condition_summary = conditions[condition]
        if condition_summary.get("trials_per_pad_and_controller") != sample_count:
            raise EvidenceAuditError(f"{summary_path} has the wrong trial count for {condition}")
        for controller in ("camera_estimate", "oracle_xy"):
            summaries = condition_summary.get("controllers", {}).get(controller)
            if not isinstance(summaries, dict):
                raise EvidenceAuditError(f"{summary_path} is missing {condition}/{controller}")
            by_pad: dict[float, dict[str, dict[str, str]]] = {}
            for pad in pads:
                selected = rows_by_group.get((condition, controller, pad, friction), [])
                if len(selected) != sample_count or {row["pair_id"] for row in selected} != expected_pairs:
                    raise EvidenceAuditError(f"{trial_path} is incomplete for {condition}/{controller}/{pad:g}")
                summary = summaries.get(f"pad_half_size_m_{pad:g}")
                if not isinstance(summary, dict) or summary.get("trials") != sample_count:
                    raise EvidenceAuditError(f"{summary_path} is missing {condition}/{controller}/{pad:g}")
                if summary.get("fresh_visual_confirmation") != "not measured":
                    raise EvidenceAuditError(f"{summary_path} reports unmeasured vision")
                by_pad[pad] = {row["pair_id"]: row for row in selected}
                for metric in (
                    "physical_stable_lift_success", "payload_stable_lift_success",
                    "full_sequence_collision_safe",
                ):
                    successes = sum(_bool(row, metric, trial_path, 0) for row in selected)
                    record = summary.get(metric)
                    if (
                        not isinstance(record, dict)
                        or record.get("successes") != successes
                        or record.get("rate") != round(successes / sample_count, 4)
                        or record.get("wilson_95_interval") != _audit_wilson_interval(successes, sample_count)
                    ):
                        raise EvidenceAuditError(f"{summary_path} {metric} does not reconcile for {condition}/{controller}/{pad:g}")
                failures = dict(sorted(Counter(row["failure_reason"] for row in selected).items()))
                if summary.get("failure_reasons") != failures:
                    raise EvidenceAuditError(f"{summary_path} failure reasons do not reconcile for {condition}/{controller}/{pad:g}")
            baseline = by_pad[pads[0]]
            candidate = by_pad[pads[1]]
            paired = {"both_success": 0, "candidate_only_success": 0, "baseline_only_success": 0, "both_failure": 0}
            for pair_id in expected_pairs:
                base_success = _bool(baseline[pair_id], "physical_stable_lift_success", trial_path, 0)
                candidate_success = _bool(candidate[pair_id], "physical_stable_lift_success", trial_path, 0)
                if base_success and candidate_success:
                    paired["both_success"] += 1
                elif candidate_success:
                    paired["candidate_only_success"] += 1
                elif base_success:
                    paired["baseline_only_success"] += 1
                else:
                    paired["both_failure"] += 1
            if paired_report[condition].get(controller) != paired:
                raise EvidenceAuditError(f"{summary_path} paired outcomes do not reconcile for {condition}/{controller}")

    demos = report.get("demos")
    artifacts = report.get("artifacts", {})
    if (
        not isinstance(demos, dict)
        or demos.get("fresh_visual_confirmation") != "not measured"
        or demos.get("benchmark_trials") != "replays are not additional trials"
    ):
        raise EvidenceAuditError(f"{summary_path} lacks a valid non-counted demo record")
    media_dir = root / "assets" / "demos"
    metadata_path = media_dir / "cooperative_pick_pad_friction_demo.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceAuditError(f"cannot read pad-friction demo metadata {metadata_path}: {error}") from error
    if (
        metadata.get("source_pair_id") not in expected_pairs
        or metadata.get("fresh_visual_confirmation") != "not measured"
        or len(metadata.get("replays", [])) != 4
        or metadata.get("composite_video") != artifacts.get("demo_video")
        or metadata.get("composite_preview") != artifacts.get("demo_preview")
    ):
        raise EvidenceAuditError(f"{metadata_path} has an inconsistent demo record")
    replay_keys = set()
    for replay in metadata["replays"]:
        if replay.get("source_pair_id") != metadata["source_pair_id"] or replay.get("counted_as_additional_benchmark_trial") is not False:
            raise EvidenceAuditError(f"{metadata_path} contains an unpaired or counted replay")
        if replay.get("fresh_visual_confirmation") != "not measured":
            raise EvidenceAuditError(f"{metadata_path} invents visual confirmation")
        replay_key = (float(replay["finger_pad_half_size_m"]), float(replay["finger_pad_sliding_friction"]))
        replay_keys.add(replay_key)
        source_row = next(
            row for row in rows
            if row["pair_id"] == metadata["source_pair_id"]
            and row["controller"] == "camera_estimate"
            and float(row["finger_pad_half_size_m"]) == replay_key[0]
            and float(row["friction"]) == replay_key[1]
        )
        if replay.get("physical_stable_lift_success") != _bool(source_row, "physical_stable_lift_success", trial_path, 0):
            raise EvidenceAuditError(f"{metadata_path} replay outcome differs from its source trial")
        for filename in (replay.get("video"),):
            if not filename or not (media_dir / filename).is_file():
                raise EvidenceAuditError(f"{metadata_path} references a missing replay video")
    if replay_keys != {(pad, friction) for pad in pads for friction in frictions}:
        raise EvidenceAuditError(f"{metadata_path} does not cover all paired pad/friction settings")
    for filename in (artifacts.get("demo_video"), artifacts.get("demo_preview")):
        if not filename or not (media_dir / filename).is_file():
            raise EvidenceAuditError(f"{metadata_path} references a missing composite demo")
    scorecard = directory / str(artifacts.get("scorecard_svg", ""))
    if not scorecard.is_file():
        raise EvidenceAuditError(f"{summary_path} references a missing scorecard")
    return len(rows), len(frictions) * 2


def audit_cooperative_pick_contact_diagnostics(root: Path) -> tuple[int, int]:
    """Reconcile phase-resolved per-pad contact telemetry and trial summaries."""
    directory = root / "results" / "cooperative-pick-contact-diagnostics"
    trial_path = directory / "cooperative-pick-contact-trials.csv"
    finger_path = directory / "cooperative-pick-contact-by-finger.csv"
    summary_path = directory / "cooperative-pick-contact-summary.json"
    try:
        report = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceAuditError(
            f"cannot read contact-diagnostic summary {summary_path}: {error}"
        ) from error
    _, trials = _read(trial_path)
    _, fingers = _read(finger_path)
    if not trials or not fingers or not isinstance(report, dict):
        raise EvidenceAuditError("contact-diagnostic evidence must not be empty")
    if (
        report.get("protocol_version") != 1
        or report.get("evidence_type") != "paired MuJoCo simulation; not hardware evidence"
        or report.get("paired_design")
        != "same pose, baseline 16 mm pad, nominal mass, clearance approach and release; compare low/high friction under camera-estimate and oracle controllers"
        or report.get("controllers") != ["camera_estimate", "oracle_xy"]
    ):
        raise EvidenceAuditError(f"{summary_path} has inconsistent protocol metadata")
    sampling = report.get("target_sampling", {})
    settings = report.get("frozen_settings", {})
    try:
        seed = int(sampling["base_seed"])
        sample_count = int(sampling["samples_per_seed"])
        mass = float(settings["payload_body_mass_kg"])
        pad_size = float(settings["finger_pad_half_size_m"])
        frictions = tuple(float(value) for value in settings["friction_levels"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceAuditError(f"{summary_path} has invalid frozen settings") from error
    if (
        seed != 41000
        or sample_count != 12
        or pad_size != 0.016
        or frictions != (0.5, 2.2)
        or not math.isclose(mass, 0.7201166534540158, rel_tol=0.0, abs_tol=1e-12)
        or settings.get("lowest_approach_tool_height_offset_m") != 0.020
        or settings.get("release_profile") != "clearance"
    ):
        raise EvidenceAuditError(f"{summary_path} changed the frozen contact protocol")
    if (
        report.get("total_physics_trials") != len(trials)
        or len(trials) != sample_count * len(frictions) * 2
        or report.get("total_finger_trial_rows") != len(fingers)
        or len(fingers) != len(trials) * 6
        or report.get("fresh_visual_confirmation")
        != "not measured; physical stable-lift outcomes and contact traces do not imply visual confirmation"
    ):
        raise EvidenceAuditError(f"{summary_path} trial/trace counts or visual status do not reconcile")
    expected_pairs = {f"{seed}-{index:03d}" for index in range(sample_count)}
    trial_lookup: dict[tuple[str, str, float], dict[str, str]] = {}
    target_positions: dict[str, tuple[float, float]] = {}
    for line, row in enumerate(trials, start=2):
        pair_id = row["pair_id"]
        controller = row["controller"]
        friction = _finite_float(row, "friction", trial_path, line)
        key = (pair_id, controller, friction)
        if (
            pair_id not in expected_pairs
            or controller not in {"camera_estimate", "oracle_xy"}
            or friction not in frictions
            or key in trial_lookup
        ):
            raise EvidenceAuditError(f"{trial_path}:{line} has a duplicate or unexpected trial")
        if (
            int(row["seed"]) != seed
            or pair_id != f"{seed}-{int(row['sample_index']):03d}"
            or _finite_float(row, "finger_pad_half_size_m", trial_path, line) != pad_size
            or not math.isclose(_finite_float(row, "payload_body_mass_kg", trial_path, line), mass, abs_tol=1e-12)
            or _finite_float(row, "approach_clearance_m", trial_path, line) != 0.020
            or row.get("release_profile") != "clearance"
            or row.get("fresh_visual_confirmation") != "not measured"
        ):
            raise EvidenceAuditError(f"{trial_path}:{line} changed a frozen setting or outcome label")
        x = _finite_float(row, "target_x_m", trial_path, line)
        y = _finite_float(row, "target_y_m", trial_path, line)
        if abs(x) > 0.09 + 1e-9 or abs(y) > 0.09 + 1e-9:
            raise EvidenceAuditError(f"{trial_path}:{line} target is outside the frozen range")
        previous = target_positions.setdefault(pair_id, (x, y))
        if not (math.isclose(x, previous[0], abs_tol=1e-12) and math.isclose(y, previous[1], abs_tol=1e-12)):
            raise EvidenceAuditError(f"{trial_path}:{line} paired target pose changed")
        for field in ("physical_stable_lift_success", "payload_stable_lift_success", "full_sequence_collision_safe"):
            _bool(row, field, trial_path, line)
        trial_lookup[key] = row
    expected_trial_keys = {
        (pair_id, controller, friction)
        for pair_id in expected_pairs
        for controller in ("camera_estimate", "oracle_xy")
        for friction in frictions
    }
    if set(trial_lookup) != expected_trial_keys or set(target_positions) != expected_pairs:
        raise EvidenceAuditError(f"{trial_path} does not cover every paired pose/condition")

    finger_groups: dict[tuple[str, str, float], list[dict[str, str]]] = {}
    expected_pad_names = {
        f"arm{arm}_{side}_pad"
        for arm in range(3)
        for side in ("left", "right")
    }
    seen_fingers: set[tuple[str, str, float, str]] = set()
    for line, row in enumerate(fingers, start=2):
        key = (row["pair_id"], row["controller"], _finite_float(row, "friction", finger_path, line))
        pad_name = row["finger_pad_geom"]
        unique_key = (*key, pad_name)
        if (
            key not in expected_trial_keys
            or pad_name not in expected_pad_names
            or unique_key in seen_fingers
        ):
            raise EvidenceAuditError(f"{finger_path}:{line} has a duplicate or unexpected pad row")
        seen_fingers.add(unique_key)
        try:
            phase_steps = json.loads(row["contact_steps_by_phase_json"])
            phase_forces = json.loads(row["maximum_normal_force_by_phase_n_json"])
            phase_impulses = json.loads(row["normal_contact_impulse_by_phase_ns_json"])
            contact_steps = int(row["contact_steps"])
            close_samples = int(row["close_phase_samples"])
            max_tracking = float(row["maximum_absolute_tracking_error_m"])
            max_force = float(row["maximum_normal_contact_force_n"])
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise EvidenceAuditError(f"{finger_path}:{line} has invalid contact telemetry") from error
        if (
            not isinstance(phase_steps, dict)
            or not isinstance(phase_forces, dict)
            or not isinstance(phase_impulses, dict)
            or any(int(value) < 0 for value in phase_steps.values())
            or close_samples <= 0
            or max_tracking < 0.0
            or max_force < 0.0
            or row.get("fresh_visual_confirmation") is not None
        ):
            raise EvidenceAuditError(f"{finger_path}:{line} has invalid contact or slide metrics")
        finger_groups.setdefault(key, []).append(row)
    if len(seen_fingers) != len(expected_trial_keys) * 6:
        raise EvidenceAuditError(f"{finger_path} does not contain exactly six pad traces per trial")

    conditions = report.get("conditions")
    if not isinstance(conditions, dict):
        raise EvidenceAuditError(f"{summary_path} is missing friction conditions")
    closing_phases = {"LOWER TO GRASP", "CLOSE SIX FINGERS", "SECURE CONTACT"}
    for friction in frictions:
        condition_key = f"friction-{friction:g}"
        for controller in ("camera_estimate", "oracle_xy"):
            group_key = (f"friction-{friction:g}", controller, friction)
            selected = [
                trial_lookup[(pair_id, controller, friction)]
                for pair_id in sorted(expected_pairs)
            ]
            pad_rows = [
                row
                for pair_id in expected_pairs
                for row in finger_groups[(pair_id, controller, friction)]
            ]
            summary = conditions.get(condition_key, {}).get(controller)
            if not isinstance(summary, dict) or summary.get("trials") != sample_count:
                raise EvidenceAuditError(f"{summary_path} is missing {condition_key}/{controller}")
            if summary.get("fresh_visual_confirmation") != "not measured":
                raise EvidenceAuditError(f"{summary_path} invents a visual confirmation")
            for metric, source in (
                ("physical_stable_lift_success", "physical_stable_lift_success"),
                ("payload_stable_lift_successes", "payload_stable_lift_success"),
                ("full_sequence_collision_safe_trials", "full_sequence_collision_safe"),
            ):
                observed = sum(_bool(row, source, trial_path, 0) for row in selected)
                record = summary[metric]
                if metric == "physical_stable_lift_success":
                    if (
                        record.get("successes") != observed
                        or record.get("rate") != round(observed / sample_count, 4)
                        or record.get("wilson_95_interval") != _audit_wilson_interval(observed, sample_count)
                    ):
                        raise EvidenceAuditError(f"{summary_path} stable-lift metrics do not reconcile")
                elif record != observed:
                    raise EvidenceAuditError(f"{summary_path} {metric} does not reconcile")
            for summary_field, trace_phases in (
                ("pad_trials_with_contact_during_approach_close_or_secure", closing_phases),
                ("pad_trials_with_contact_during_cooperative_lift", {"COOPERATIVE LIFT"}),
                ("pad_trials_with_contact_during_hold_under_load", {"HOLD UNDER LOAD"}),
            ):
                observed = sum(
                    any(
                        int(json.loads(row["contact_steps_by_phase_json"]).get(phase, 0)) > 0
                        for phase in trace_phases
                    )
                    for row in pad_rows
                )
                if summary.get(summary_field) != observed:
                    raise EvidenceAuditError(f"{summary_path} {summary_field} does not reconcile")
            phase_counts = Counter()
            for row in pad_rows:
                for phase, steps in json.loads(row["contact_steps_by_phase_json"]).items():
                    phase_counts[phase] += int(int(steps) > 0)
            if summary.get("pad_trials_with_contact_by_phase") != dict(sorted(phase_counts.items())):
                raise EvidenceAuditError(f"{summary_path} per-phase contact counts do not reconcile")
            failures = dict(sorted(Counter(row["failure_reason"] for row in selected).items()))
            retention = dict(sorted(Counter(row["contact_retention_outcome"] for row in selected).items()))
            if summary.get("failure_reasons") != failures or summary.get("contact_retention_outcomes") != retention:
                raise EvidenceAuditError(f"{summary_path} failure/contact-retention labels do not reconcile")
            for row in selected:
                pad_rows_for_trial = finger_groups[(row["pair_id"], controller, friction)]
                close_pads = sum(
                    any(int(json.loads(pad["contact_steps_by_phase_json"]).get(phase, 0)) > 0 for phase in closing_phases)
                    for pad in pad_rows_for_trial
                )
                lift_pads = sum(
                    int(json.loads(pad["contact_steps_by_phase_json"]).get("COOPERATIVE LIFT", 0)) > 0
                    for pad in pad_rows_for_trial
                )
                hold_pads = sum(
                    int(json.loads(pad["contact_steps_by_phase_json"]).get("HOLD UNDER LOAD", 0)) > 0
                    for pad in pad_rows_for_trial
                )
                if (
                    int(row["pads_with_contact_during_approach_close_or_secure"]) != close_pads
                    or int(row["pads_with_contact_during_cooperative_lift"]) != lift_pads
                    or int(row["pads_with_contact_during_hold_under_load"]) != hold_pads
                ):
                    raise EvidenceAuditError(f"{trial_path} phase rollup does not reconcile to per-pad rows")

    demos = report.get("demos")
    artifacts = report.get("artifacts", {})
    metadata_path = root / "assets" / "demos" / "cooperative_pick_contact_demos.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceAuditError(f"cannot read contact demo metadata {metadata_path}: {error}") from error
    if (
        not isinstance(demos, dict)
        or demos.get("fresh_visual_confirmation") != "not measured"
        or demos.get("replays_are_additional_trials") is not False
        or metadata.get("source_pair_id") not in expected_pairs
        or metadata.get("fresh_visual_confirmation") != "not measured"
        or len(metadata.get("replays", [])) != 2
        or metadata.get("paired_video") != artifacts.get("demo_video")
        or metadata.get("preview") != artifacts.get("demo_preview")
    ):
        raise EvidenceAuditError(f"{metadata_path} has an inconsistent replay record")
    for replay in metadata["replays"]:
        if replay.get("counted_as_additional_benchmark_trial") is not False or replay.get("source_pair_id") != metadata["source_pair_id"]:
            raise EvidenceAuditError(f"{metadata_path} contains an invalid replay")
        friction = float(replay["finger_pad_sliding_friction"])
        source = trial_lookup[(metadata["source_pair_id"], "camera_estimate", friction)]
        if replay.get("physical_stable_lift_success") != _bool(source, "physical_stable_lift_success", trial_path, 0):
            raise EvidenceAuditError(f"{metadata_path} replay differs from the source trial outcome")
        if not (root / "assets" / "demos" / replay.get("video", "")).is_file():
            raise EvidenceAuditError(f"{metadata_path} references a missing single-condition clip")
    for filename in (artifacts.get("demo_video"), artifacts.get("demo_preview")):
        if not filename or not (root / "assets" / "demos" / filename).is_file():
            raise EvidenceAuditError(f"{metadata_path} references a missing paired demo")
    if not (directory / str(artifacts.get("scorecard_svg", ""))).is_file():
        raise EvidenceAuditError(f"{summary_path} references a missing scorecard")
    return len(trials), len(frictions) * 2


def audit_cooperative_pick_retention_diagnostics(root: Path) -> tuple[int, int]:
    """Reconcile held-out hand-relative motion, force, and finger tracking data."""
    directory = root / "results" / "cooperative-pick-retention-diagnostics"
    trial_path = directory / "cooperative-pick-retention-trials.csv"
    pad_path = directory / "cooperative-pick-retention-by-pad-phase.csv"
    summary_path = directory / "cooperative-pick-retention-summary.json"
    try:
        report = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceAuditError(
            f"cannot read retention-diagnostic summary {summary_path}: {error}"
        ) from error
    trial_fields, trials = _read(trial_path)
    pad_fields, pads = _read(pad_path)
    if not trials or not pads or not isinstance(report, dict):
        raise EvidenceAuditError("retention-diagnostic evidence must not be empty")
    sampling = report.get("target_sampling", {})
    try:
        seed = int(sampling["base_seed"])
        sample_count = int(sampling["samples_per_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceAuditError(f"{summary_path} has invalid seed/sample metadata") from error
    settings = report.get("frozen_settings", {})
    if settings:
        frozen_ok = (
            settings.get("finger_pad_half_size_m") == 0.016
            and math.isclose(
                float(settings.get("payload_body_mass_kg", math.nan)),
                0.7201166534540158,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and settings.get("lowest_approach_tool_height_offset_m") == 0.020
            and settings.get("release_profile") == "clearance"
            and settings.get("friction_levels") == [0.5, 2.2]
        )
    else:
        frozen_ok = False
    if (
        seed != 42000
        or sample_count != 12
        or report.get("protocol_version") != 1
        or report.get("evidence_type") != "paired MuJoCo simulation; not hardware evidence"
        or report.get("controllers") != ["camera_estimate", "oracle_xy"]
        or not frozen_ok
        or report.get("controllers") != ["camera_estimate", "oracle_xy"]
        or report.get("total_physics_trials") != sample_count * 2 * 2
        or report.get("total_pad_phase_rows") != len(pads)
        or len(trials) != sample_count * 2 * 2
        or len(pads) != len(trials) * 6 * 2
        or report.get("fresh_visual_confirmation")
        != "not measured; physical stable-lift outcomes and contact traces do not imply visual confirmation"
    ):
        raise EvidenceAuditError(f"{summary_path} protocol/trial/visual status does not reconcile")
    if report.get("sampling", {}).get("sample_period_s") != 0.02:
        raise EvidenceAuditError(f"{summary_path} has an unexpected telemetry sample period")

    required_trials = {
        "pair_id", "seed", "sample_index", "target_x_m", "target_y_m", "controller", "friction",
        "physical_stable_lift_success", "payload_stable_lift_success", "full_sequence_collision_safe",
        "hold_minimum_pad_contact_points", "hold_contact_sample_fraction_mean_across_pads",
        "hold_maximum_downward_payload_drift_in_hand_frame_m",
        "hold_maximum_horizontal_payload_drift_in_hand_frame_m",
        "mean_tangential_to_normal_force_ratio_when_contacting",
        "maximum_slide_tracking_error_lift_or_hold_m",
        "mean_slide_tracking_error_lift_or_hold_m",
        "maximum_slide_tracking_error_during_hold_m",
        "mean_slide_tracking_error_during_hold_m", "fresh_visual_confirmation",
    }
    required_pads = {
        "pair_id", "controller", "friction", "finger_pad_geom", "phase", "samples",
        "contact_samples", "contact_sample_fraction", "mean_normal_force_n",
        "maximum_normal_force_n", "mean_tangential_force_n", "maximum_tangential_force_n",
        "mean_tangential_to_normal_force_ratio_when_contacting",
        "maximum_tangential_to_normal_force_ratio_when_contacting",
        "mean_slide_tracking_error_m", "maximum_slide_tracking_error_m",
    }
    for path, required, available in (
        (trial_path, required_trials, set(trial_fields)),
        (pad_path, required_pads, set(pad_fields)),
    ):
        missing = sorted(required - available)
        if missing:
            raise EvidenceAuditError(f"{path} is missing fields: {', '.join(missing)}")

    frictions = (0.5, 2.2)
    controllers = ("camera_estimate", "oracle_xy")
    phases = ("COOPERATIVE LIFT", "HOLD UNDER LOAD")
    pad_names = {
        f"arm{arm}_{side}_pad"
        for arm in range(3)
        for side in ("left", "right")
    }
    expected_pairs = {f"{seed}-{index:03d}" for index in range(sample_count)}
    expected_keys = {
        (pair_id, controller, friction)
        for pair_id in expected_pairs
        for controller in controllers
        for friction in frictions
    }
    trial_by_key: dict[tuple[str, str, float], dict[str, str]] = {}
    positions: dict[str, tuple[float, float]] = {}
    for line, row in enumerate(trials, start=2):
        pair_id = row["pair_id"]
        controller = row["controller"]
        friction = _finite_float(row, "friction", trial_path, line)
        key = (pair_id, controller, friction)
        sample_index = int(row["sample_index"])
        if (
            key not in expected_keys
            or key in trial_by_key
            or int(row["seed"]) != seed
            or pair_id != f"{seed}-{sample_index:03d}"
            or not 0 <= sample_index < sample_count
            or row["fresh_visual_confirmation"] != "not measured"
        ):
            raise EvidenceAuditError(f"{trial_path}:{line} has an unexpected identity or outcome label")
        x = _finite_float(row, "target_x_m", trial_path, line)
        y = _finite_float(row, "target_y_m", trial_path, line)
        if abs(x) > 0.09 + 1e-9 or abs(y) > 0.09 + 1e-9:
            raise EvidenceAuditError(f"{trial_path}:{line} target is outside the frozen range")
        previous = positions.setdefault(pair_id, (x, y))
        if not (math.isclose(x, previous[0], abs_tol=1e-12) and math.isclose(y, previous[1], abs_tol=1e-12)):
            raise EvidenceAuditError(f"{trial_path}:{line} paired target coordinates disagree")
        for field in (
            "physical_stable_lift_success", "payload_stable_lift_success",
            "full_sequence_collision_safe",
        ):
            _bool(row, field, trial_path, line)
        for field in (
            "hold_contact_sample_fraction_mean_across_pads",
            "hold_maximum_downward_payload_drift_in_hand_frame_m",
            "hold_maximum_horizontal_payload_drift_in_hand_frame_m",
            "mean_tangential_to_normal_force_ratio_when_contacting",
            "maximum_slide_tracking_error_lift_or_hold_m",
            "mean_slide_tracking_error_lift_or_hold_m",
            "maximum_slide_tracking_error_during_hold_m",
            "mean_slide_tracking_error_during_hold_m",
        ):
            value = _finite_float(row, field, trial_path, line)
            if value < 0.0 or (
                field == "hold_contact_sample_fraction_mean_across_pads" and value > 1.0
            ):
                raise EvidenceAuditError(f"{trial_path}:{line} has out-of-range {field}")
        trial_by_key[key] = row
    if set(trial_by_key) != expected_keys or set(positions) != expected_pairs:
        raise EvidenceAuditError(f"{trial_path} does not contain every paired pose/condition")

    expected_pad_keys = {
        (*trial_key, pad_name, phase)
        for trial_key in expected_keys
        for pad_name in pad_names
        for phase in phases
    }
    pad_by_key: dict[tuple[str, str, float, str, str], dict[str, str]] = {}
    for line, row in enumerate(pads, start=2):
        trial_key = (
            row["pair_id"], row["controller"], _finite_float(row, "friction", pad_path, line)
        )
        key = (*trial_key, row["finger_pad_geom"], row["phase"])
        if trial_key not in expected_keys or key not in expected_pad_keys or key in pad_by_key:
            raise EvidenceAuditError(f"{pad_path}:{line} has an unexpected or duplicate pad/phase row")
        try:
            samples = int(row["samples"])
            contact_samples = int(row["contact_samples"])
        except ValueError as error:
            raise EvidenceAuditError(f"{pad_path}:{line} has invalid sample counts") from error
        fraction = _finite_float(row, "contact_sample_fraction", pad_path, line)
        if (
            samples <= 0 or not 0 <= contact_samples <= samples or not 0.0 <= fraction <= 1.0
            or not math.isclose(fraction, round(contact_samples / samples, 6), rel_tol=0.0, abs_tol=1e-9)
        ):
            raise EvidenceAuditError(f"{pad_path}:{line} has inconsistent contact-sample counts")
        for field in (
            "mean_normal_force_n", "maximum_normal_force_n", "mean_tangential_force_n",
            "maximum_tangential_force_n",
            "mean_tangential_to_normal_force_ratio_when_contacting",
            "maximum_tangential_to_normal_force_ratio_when_contacting",
            "mean_slide_tracking_error_m", "maximum_slide_tracking_error_m",
        ):
            if _finite_float(row, field, pad_path, line) < 0.0:
                raise EvidenceAuditError(f"{pad_path}:{line} has negative {field}")
        pad_by_key[key] = row
    if set(pad_by_key) != expected_pad_keys:
        raise EvidenceAuditError(f"{pad_path} does not contain all six pads in both measured phases")

    conditions = report.get("conditions")
    if not isinstance(conditions, dict):
        raise EvidenceAuditError(f"{summary_path} is missing condition summaries")
    paired_report = report.get("paired_friction_deltas")
    if not isinstance(paired_report, dict):
        raise EvidenceAuditError(f"{summary_path} is missing paired friction outcomes")
    for friction in frictions:
        condition_key = f"friction-{friction:g}"
        for controller in controllers:
            selected = [
                trial_by_key[(pair_id, controller, friction)]
                for pair_id in sorted(expected_pairs)
            ]
            summary = conditions.get(condition_key, {}).get(controller)
            if not isinstance(summary, dict) or summary.get("trials") != sample_count:
                raise EvidenceAuditError(f"{summary_path} is missing {condition_key}/{controller}")
            if summary.get("fresh_visual_confirmation") != "not measured":
                raise EvidenceAuditError(f"{summary_path} invents visual confirmation")
            successes = sum(_bool(row, "physical_stable_lift_success", trial_path, 0) for row in selected)
            collision_safe = sum(_bool(row, "full_sequence_collision_safe", trial_path, 0) for row in selected)
            if (
                summary.get("physical_stable_lift_successes") != successes
                or summary.get("collision_safe_trials") != collision_safe
                or summary.get("physical_stable_lift_success_rate") != round(successes / sample_count, 6)
            ):
                raise EvidenceAuditError(f"{summary_path} outcome counts do not reconcile for {condition_key}/{controller}")
            expected_values: dict[str, float | int | None] = {
                "hold_contact_sample_fraction_mean": round(
                    sum(float(row["hold_contact_sample_fraction_mean_across_pads"]) for row in selected) / sample_count, 8
                ),
                "hold_contact_sample_fraction_median": round(
                    median(float(row["hold_contact_sample_fraction_mean_across_pads"]) for row in selected), 8
                ),
                "maximum_downward_payload_drift_in_hand_frame_m_median": round(
                    median(float(row["hold_maximum_downward_payload_drift_in_hand_frame_m"]) for row in selected), 8
                ),
                "maximum_downward_payload_drift_in_hand_frame_m_p90": round(
                    _quantile([float(row["hold_maximum_downward_payload_drift_in_hand_frame_m"]) for row in selected], 0.9), 8
                ),
                "mean_tangential_to_normal_force_ratio_when_contacting_median": round(
                    median(float(row["mean_tangential_to_normal_force_ratio_when_contacting"]) for row in selected), 8
                ),
                "mean_tangential_to_normal_force_ratio_when_contacting_p90": round(
                    _quantile([float(row["mean_tangential_to_normal_force_ratio_when_contacting"]) for row in selected], 0.9), 8
                ),
                "maximum_slide_tracking_error_lift_or_hold_m_median": round(
                    median(float(row["maximum_slide_tracking_error_lift_or_hold_m"]) for row in selected), 8
                ),
                "maximum_slide_tracking_error_during_hold_m_median": round(
                    median(float(row["maximum_slide_tracking_error_during_hold_m"]) for row in selected), 8
                ),
            }
            for field, expected in expected_values.items():
                _check_numeric_summary(
                    {field: str(summary.get(field, ""))}, field, expected,
                    f"{condition_key}/{controller}", summary_path,
                )

    for controller in controllers:
        deltas = {
            "high_minus_low_hold_downward_drift_m": [],
            "high_minus_low_hold_contact_fraction": [],
            "high_minus_low_hold_slide_tracking_error_m": [],
            "high_minus_low_tangential_to_normal_ratio": [],
        }
        transitions = {
            "low_only_success": 0, "high_only_success": 0,
            "both_success": 0, "neither_success": 0,
        }
        for pair_id in sorted(expected_pairs):
            low = trial_by_key[(pair_id, controller, 0.5)]
            high = trial_by_key[(pair_id, controller, 2.2)]
            for key, field in (
                ("high_minus_low_hold_downward_drift_m", "hold_maximum_downward_payload_drift_in_hand_frame_m"),
                ("high_minus_low_hold_contact_fraction", "hold_contact_sample_fraction_mean_across_pads"),
                ("high_minus_low_hold_slide_tracking_error_m", "maximum_slide_tracking_error_during_hold_m"),
                ("high_minus_low_tangential_to_normal_ratio", "mean_tangential_to_normal_force_ratio_when_contacting"),
            ):
                deltas[key].append(float(high[field]) - float(low[field]))
            low_success = _bool(low, "physical_stable_lift_success", trial_path, 0)
            high_success = _bool(high, "physical_stable_lift_success", trial_path, 0)
            transition = (
                "both_success" if low_success and high_success else
                "low_only_success" if low_success else
                "high_only_success" if high_success else "neither_success"
            )
            transitions[transition] += 1
        recorded = paired_report.get(controller)
        if not isinstance(recorded, dict) or recorded.get("matched_pose_pairs") != sample_count:
            raise EvidenceAuditError(f"{summary_path} paired {controller} trial count does not reconcile")
        if recorded.get("physical_stable_lift_transitions") != transitions:
            raise EvidenceAuditError(f"{summary_path} paired {controller} success transitions do not reconcile")
        for field, values in deltas.items():
            expected = round(median(values), 8)
            _check_numeric_summary(
                {field: str(recorded.get("median_high_minus_low", {}).get(field, ""))},
                field, expected, controller, summary_path,
            )

    metadata_path = root / "assets" / "demos" / "cooperative_pick_retention_diagnostic_demos.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceAuditError(f"cannot read retention demo metadata {metadata_path}: {error}") from error
    demo_pair = metadata.get("source_pair_id")
    if (
        demo_pair not in expected_pairs
        or metadata.get("replays_are_additional_trials") is not False
        or metadata.get("fresh_visual_confirmation") != "not measured"
    ):
        raise EvidenceAuditError(f"{metadata_path} has an invalid source pose or replay label")
    for friction in frictions:
        condition = metadata.get("conditions", {}).get(f"friction-{friction:g}", {})
        source = trial_by_key[(demo_pair, "camera_estimate", friction)]
        if (
            condition.get("fresh_visual_confirmation") != "not measured"
            or condition.get("physical_stable_lift_success")
            != _bool(source, "physical_stable_lift_success", trial_path, 0)
            or condition.get("full_sequence_collision_safe")
            != _bool(source, "full_sequence_collision_safe", trial_path, 0)
        ):
            raise EvidenceAuditError(f"{metadata_path} does not match its source trial")
    artifacts = report.get("artifacts", {})
    artifact_paths = (
        root / "assets" / "demos" / str(artifacts.get("demo_video", "")),
        root / "assets" / "demos" / str(artifacts.get("demo_gif", "")),
        root / "assets" / "demos" / str(artifacts.get("demo_metadata", "")),
        directory / str(artifacts.get("scorecard_svg", "")),
    )
    for artifact_path in artifact_paths:
        if not artifact_path.is_file():
            raise EvidenceAuditError(f"{summary_path} references a missing diagnostic artifact")
    return len(trials), len(frictions) * len(controllers)


def audit_terminal_verification_robustness(root: Path) -> tuple[int, int]:
    """Check that verifier stress rows share one saved reaching run per trial."""
    from .verification_robustness_benchmark import VERIFIER_STRESS_CONDITIONS

    path = root / "results" / "terminal-verification-robustness-benchmark.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceAuditError(f"cannot read terminal-verifier benchmark {path}: {error}") from error

    if not isinstance(report, dict):
        raise EvidenceAuditError(f"{path} must contain a JSON object")
    try:
        seeds = int(report["seeds"])
        episodes_per_seed = int(report["episodes_per_seed"])
        base_seed = int(report["base_seed"])
        rows = report["results"]
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceAuditError(f"{path} has invalid trial metadata") from error
    expected_trial_count = seeds * episodes_per_seed
    if (
        report.get("schema_version") != 3
        or report.get("benchmark") != "terminal_verifier_quality_coverage_and_image_stress"
        or report.get("control_controller") != "image_feedback"
        or report.get("control_policy_changed") is not False
        or report.get("control_trajectory_shared_across_conditions") is not True
        or report.get("control_trajectories_per_seed_episode") != 1
        or report.get("stress_conditions_are_one_factor_at_a_time") is not True
        or seeds <= 0
        or episodes_per_seed <= 0
        or base_seed < 0
        or not isinstance(rows, list)
        or len(rows) != len(VERIFIER_STRESS_CONDITIONS)
    ):
        raise EvidenceAuditError(f"{path} protocol metadata does not reconcile")

    expected_ids = {
        f"{seed}:{episode}"
        for seed in range(base_seed, base_seed + seeds)
        for episode in range(episodes_per_seed)
    }
    reference_outcomes: dict[str, tuple[object, ...]] = {}
    seen_conditions: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise EvidenceAuditError(f"{path} contains a malformed condition row")
        condition = row.get("condition")
        if not isinstance(condition, str):
            raise EvidenceAuditError(f"{path} contains an invalid condition name")
        expected_settings = VERIFIER_STRESS_CONDITIONS.get(condition)
        if condition in seen_conditions or expected_settings is None:
            raise EvidenceAuditError(f"{path} contains an unknown or repeated condition")
        seen_conditions.add(condition)
        if row.get("stress_settings") != json.loads(json.dumps(expected_settings)):
            raise EvidenceAuditError(f"{path} condition {condition} has changed stress settings")
        trials = row.get("trial_results")
        if (
            not isinstance(trials, list)
            or len(trials) != expected_trial_count
            or row.get("trials") != expected_trial_count
            or row.get("controller_outcomes_shared_across_stress_conditions")
            != expected_trial_count
        ):
            raise EvidenceAuditError(f"{path} condition {condition} has an incomplete trial set")
        condition_ids: set[str] = set()
        for trial in trials:
            if not isinstance(trial, dict):
                raise EvidenceAuditError(f"{path} condition {condition} contains a malformed trial")
            trial_id = trial.get("shared_control_trial_id")
            expected_id = f"{trial.get('seed')}:{trial.get('episode')}"
            if (
                trial_id != expected_id
                or trial_id not in expected_ids
                or trial_id in condition_ids
                or trial.get("controller_outcome_shared_across_stress_conditions") is not True
            ):
                raise EvidenceAuditError(f"{path} condition {condition} has an invalid shared-trial identity")
            condition_ids.add(trial_id)
            outcome = (
                trial.get("physical_success"),
                trial.get("physical_error_m"),
                trial.get("controller_stop_reason"),
                trial.get("controller_iterations"),
            )
            previous = reference_outcomes.setdefault(trial_id, outcome)
            if previous != outcome:
                raise EvidenceAuditError(
                    f"{path} changes a physical/control outcome across verifier conditions"
                )
        if condition_ids != expected_ids:
            raise EvidenceAuditError(f"{path} condition {condition} does not share the full trial set")
        if row.get("physical_successes") != sum(
            trial.get("physical_success") is True for trial in trials
        ):
            raise EvidenceAuditError(f"{path} condition {condition} physical-success count does not reconcile")
        visible = [trial for trial in trials if trial.get("visual_success_estimate") is not None]
        if row.get("visual_estimate_available_trials") != len(visible):
            raise EvidenceAuditError(f"{path} condition {condition} availability count does not reconcile")
        expected_confusion = {
            "true_positive": sum(
                trial.get("visual_success_estimate") is True and trial.get("physical_success") is True
                for trial in visible
            ),
            "false_positive": sum(
                trial.get("visual_success_estimate") is True and trial.get("physical_success") is False
                for trial in visible
            ),
            "true_negative": sum(
                trial.get("visual_success_estimate") is False and trial.get("physical_success") is False
                for trial in visible
            ),
            "false_negative": sum(
                trial.get("visual_success_estimate") is False and trial.get("physical_success") is True
                for trial in visible
            ),
        }
        if row.get("confusion_counts_when_estimate_available") != expected_confusion:
            raise EvidenceAuditError(f"{path} condition {condition} confusion counts do not reconcile")

    if seen_conditions != set(VERIFIER_STRESS_CONDITIONS) or set(reference_outcomes) != expected_ids:
        raise EvidenceAuditError(f"{path} condition set or shared physics block does not reconcile")
    return expected_trial_count, len(rows)


def _audit_wilson_interval(successes: int, trials: int) -> list[float] | None:
    if trials <= 0:
        return None
    z = 1.96
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return [round(max(0.0, center - radius), 4), round(min(1.0, center + radius), 4)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    try:
        for spec in DATASETS:
            trials, groups = audit_dataset(args.root, spec)
            print(f"PASS {spec.name}: {trials:,} trial rows, {groups:,} summary groups")
        trials, groups = audit_appearance_dataset(args.root)
        print(f"PASS appearance stress: {trials:,} trial rows, {groups:,} condition groups")
        for stem, label in (
            ("perception-candidate-pilot-v1", "candidate detector pilot v1"),
            ("perception-candidate-development-v2", "candidate detector development v2"),
            ("perception-candidate-adversarial-review", "candidate detector adversarial review"),
            ("perception-candidate-validation", "candidate detector validation"),
        ):
            trials, groups = audit_candidate_validation_dataset(args.root, stem=stem)
            print(f"PASS {label}: {trials:,} trial rows, {groups:,} condition groups")
        for stage in ("development", "heldout"):
            trials, groups = audit_cooperative_pick_occlusion_dataset(args.root, stage=stage)
            print(f"PASS cooperative-pick occlusion {stage}: {trials:,} trial rows, {groups:,} image conditions")
        trials, groups = audit_cooperative_pick_occlusion_shape_challenge(args.root)
        print(f"PASS cooperative-pick occlusion shape challenge: {trials:,} trial rows, {groups:,} strata")
        trials, groups = audit_cooperative_pick_orthogonal_validation(args.root)
        print(f"PASS cooperative-pick orthogonal verification: {trials:,} trial rows, {groups:,} strata")
        trials, groups = audit_cooperative_pick_material_validation(args.root)
        print(f"PASS cooperative-pick material validation: {trials:,} trial rows, {groups:,} material conditions")
        trials, groups = audit_cooperative_pick_pad_friction_validation(args.root)
        print(f"PASS cooperative-pick pad/friction validation: {trials:,} trial rows, {groups:,} controller/friction groups")
        trials, groups = audit_cooperative_pick_contact_diagnostics(args.root)
        print(f"PASS cooperative-pick contact diagnostics: {trials:,} trial rows, {groups:,} controller/friction groups")
        trials, groups = audit_cooperative_pick_retention_diagnostics(args.root)
        print(f"PASS cooperative-pick retention diagnostics: {trials:,} trial rows, {groups:,} controller/friction groups")
        trials, conditions = audit_terminal_verification_robustness(args.root)
        print(
            f"PASS terminal-verifier robustness: {trials:,} shared physics trials, "
            f"{trials * conditions:,} condition evaluations"
        )
    except (EvidenceAuditError, OSError) as error:
        print(f"Evidence audit failed: {error}", file=sys.stderr)
        return 1
    print(
        f"Validated {len(DATASETS)} trial/summary pairs, 1 appearance dataset, "
        "4 candidate detector datasets, 2 post-pick occlusion splits, 1 occlusion-shape challenge, "
        "1 orthogonal-view challenge, 1 material-stress challenge, "
        "1 pad/friction challenge, 1 contact-diagnostic challenge, 1 retention-diagnostic challenge, "
        "and 1 shared-trajectory verifier benchmark."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
