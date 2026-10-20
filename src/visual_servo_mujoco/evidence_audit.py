"""Reconcile published benchmark evidence tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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
        raise EvidenceAuditError(f"appearance group {key}: {field} {saved} != {expected}")


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
    except (EvidenceAuditError, OSError) as error:
        print(f"Evidence audit failed: {error}", file=sys.stderr)
        return 1
    print(
        f"Validated {len(DATASETS)} trial/summary pairs, 1 appearance dataset, "
        "and 4 candidate detector datasets."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
