"""Calibrate a terminal-verifier quality gate on seeds separate from evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run import SUCCESS_THRESHOLD_METERS
from .verification_robustness_benchmark import (
    QUALITY_THRESHOLDS,
    run_verification_robustness_benchmark,
)


MINIMUM_USABLE_COVERAGE = 0.5


def _summarize_records(records: list[dict], threshold: float) -> dict:
    candidates = [
        record
        for record in records
        if record["candidate_estimated_distance_m"] is not None
        and record["detection_quality_score"] is not None
    ]
    accepted = [
        record
        for record in candidates
        if record["detection_quality_score"] >= threshold
    ]
    accepted_ids = {id(record) for record in accepted}
    abstained = [record for record in candidates if id(record) not in accepted_ids]
    accepted_physical_successes = sum(record["physical_success"] for record in accepted)
    accepted_physical_failures = len(accepted) - accepted_physical_successes
    abstained_physical_successes = sum(record["physical_success"] for record in abstained)
    abstained_physical_failures = len(abstained) - abstained_physical_successes
    failed_candidates = [record for record in candidates if not record["physical_success"]]
    false_successes = [
        record
        for record in accepted
        if not record["physical_success"]
        and record["candidate_estimated_distance_m"] <= SUCCESS_THRESHOLD_METERS
    ]
    missed_successes = [
        record
        for record in accepted
        if record["physical_success"]
        and record["candidate_estimated_distance_m"] > SUCCESS_THRESHOLD_METERS
    ]
    return {
        "trials": len(records),
        "candidate_estimates_before_gate": len(candidates),
        "accepted_estimates": len(accepted),
        "abstained_candidate_estimates": len(abstained),
        "coverage_fraction_of_candidates": (
            len(accepted) / len(candidates) if candidates else None
        ),
        "coverage_fraction_of_trials": (
            len(accepted) / len(records) if records else None
        ),
        "false_success_decisions": len(false_successes),
        "failed_candidates": len(failed_candidates),
        "accepted_physical_successes": accepted_physical_successes,
        "accepted_physical_failures": accepted_physical_failures,
        "abstained_physical_successes": abstained_physical_successes,
        "abstained_physical_failures": abstained_physical_failures,
        "false_success_rate_per_failed_candidate": (
            len(false_successes) / len(failed_candidates)
            if failed_candidates
            else None
        ),
        "missed_success_decisions": len(missed_successes),
    }


def summarize_quality_threshold(results: list[dict], threshold: float) -> dict:
    """Summarize a fixed gate overall and separately for each image condition."""
    all_records = [
        record
        for condition in results
        for record in condition["trial_results"]
    ]
    return {
        **_summarize_records(all_records, threshold),
        "conditions": [
            {
                "condition": condition["condition"],
                **_summarize_records(condition["trial_results"], threshold),
            }
            for condition in results
        ],
    }


def select_calibration_threshold(results: list[dict]) -> dict:
    """Choose maximum coverage among gates with zero observed false successes.

    This empirical selection rule is for a simulation experiment, not a safety
    guarantee. A gate is flagged as usable only when it also retains at least
    half of the candidate estimates on the calibration split.
    """
    curve = [
        {
            "minimum_quality_score": threshold,
            **_summarize_records(
                [
                    record
                    for condition in results
                    for record in condition["trial_results"]
                ],
                threshold,
            ),
        }
        for threshold in QUALITY_THRESHOLDS
    ]
    eligible = [row for row in curve if row["false_success_decisions"] == 0]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                row["coverage_fraction_of_candidates"] or 0.0,
                -row["minimum_quality_score"],
            ),
        )
        zero_false_success_constraint_met = True
    else:
        selected = min(
            curve,
            key=lambda row: (
                row["false_success_decisions"],
                -(row["coverage_fraction_of_candidates"] or 0.0),
                row["minimum_quality_score"],
            ),
        )
        zero_false_success_constraint_met = False
    coverage = selected["coverage_fraction_of_candidates"] or 0.0
    return {
        "selection_rule": (
            "Choose the threshold with the highest candidate-estimate coverage among "
            "thresholds with zero observed false-success decisions; break ties toward "
            "the lower threshold. If none meets the zero-false-success constraint, "
            "choose the fewest false-success decisions, then the highest coverage, "
            "and flag the calibration criterion as unmet."
        ),
        "minimum_coverage_fraction_for_usable_gate": MINIMUM_USABLE_COVERAGE,
        "selected_threshold": selected["minimum_quality_score"],
        "selected_calibration_metrics": selected,
        "zero_false_success_constraint_met": zero_false_success_constraint_met,
        "meets_minimum_coverage_for_usable_gate": (
            zero_false_success_constraint_met and coverage >= MINIMUM_USABLE_COVERAGE
        ),
        "threshold_curve": curve,
    }


def _validate_seed_splits(
    *,
    calibration_base_seed: int,
    validation_base_seed: int,
    seeds: int,
) -> None:
    if calibration_base_seed < 0 or validation_base_seed < 0:
        raise ValueError("base seeds must be non-negative")
    calibration_seeds = set(range(calibration_base_seed, calibration_base_seed + seeds))
    validation_seeds = set(range(validation_base_seed, validation_base_seed + seeds))
    if calibration_seeds & validation_seeds:
        raise ValueError("calibration and validation seed ranges must be disjoint")


def run_frozen_quality_validation(
    *,
    seeds: int = 5,
    episodes_per_seed: int = 20,
    calibration_base_seed: int = 200,
    validation_base_seed: int = 300,
    pixel_noise_std_px: float = 20.0,
    output_path: Path | None = None,
) -> dict:
    """Select the quality gate on one seed block and evaluate it unchanged on another."""
    if seeds <= 0 or episodes_per_seed <= 0:
        raise ValueError("seeds and episodes_per_seed must be positive")
    _validate_seed_splits(
        calibration_base_seed=calibration_base_seed,
        validation_base_seed=validation_base_seed,
        seeds=seeds,
    )

    calibration = run_verification_robustness_benchmark(
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
        base_seed=calibration_base_seed,
        pixel_noise_std_px=pixel_noise_std_px,
    )
    selection = select_calibration_threshold(calibration["results"])
    frozen_threshold = selection["selected_threshold"]

    validation = run_verification_robustness_benchmark(
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
        base_seed=validation_base_seed,
        pixel_noise_std_px=pixel_noise_std_px,
    )
    validation_baseline_metrics = summarize_quality_threshold(
        validation["results"], 0.0
    )
    validation_metrics = summarize_quality_threshold(
        validation["results"], frozen_threshold
    )

    report = {
        "schema_version": 1,
        "benchmark": "frozen_terminal_verifier_quality_threshold_validation",
        "protocol": {
            "threshold_selected_before_validation_run": True,
            "calibration_seed_range": [
                calibration_base_seed,
                calibration_base_seed + seeds - 1,
            ],
            "validation_seed_range": [
                validation_base_seed,
                validation_base_seed + seeds - 1,
            ],
            "seed_ranges_disjoint": True,
            "conditions": list(
                row["condition"] for row in calibration["results"]
            ),
            "episodes_per_seed": episodes_per_seed,
            "pixel_noise_std_px": pixel_noise_std_px,
            "threshold_usable_coverage_floor": MINIMUM_USABLE_COVERAGE,
            "selection_uses_validation_labels": False,
        },
        "threshold_selection": selection,
        "validation_frozen_threshold": frozen_threshold,
        "validation_ungated_baseline_metrics": validation_baseline_metrics,
        "validation_metrics": validation_metrics,
        "control_policy_changed": (
            calibration["control_policy_changed"]
            or validation["control_policy_changed"]
        ),
        "evidence_scope": (
            "Seed-separated MuJoCo RGB-render evidence with controlled one-factor image "
            "stresses. This is not real-camera or safety-certification evidence."
        ),
        "calibration_benchmark": calibration,
        "validation_benchmark": validation,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes-per-seed", type=int, default=20)
    parser.add_argument("--calibration-base-seed", type=int, default=200)
    parser.add_argument("--validation-base-seed", type=int, default=300)
    parser.add_argument("--pixel-noise-std-px", type=float, default=20.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/verification-quality-calibration-validation.json"),
    )
    args = parser.parse_args()
    try:
        report = run_frozen_quality_validation(
            seeds=args.seeds,
            episodes_per_seed=args.episodes_per_seed,
            calibration_base_seed=args.calibration_base_seed,
            validation_base_seed=args.validation_base_seed,
            pixel_noise_std_px=args.pixel_noise_std_px,
            output_path=args.output,
        )
    except ValueError as error:
        parser.error(str(error))

    selected = report["validation_frozen_threshold"]
    metrics = report["validation_metrics"]
    counts = metrics["conditions"]
    print(
        f"Frozen quality threshold: {selected:.2f}; "
        f"calibration usable: "
        f"{report['threshold_selection']['meets_minimum_coverage_for_usable_gate']}"
    )
    print(
        f"Validation coverage: {metrics['accepted_estimates']}/"
        f"{metrics['candidate_estimates_before_gate']} candidate estimates "
        f"({metrics['coverage_fraction_of_trials']:.1%} of all trials); "
        f"false-success decisions: {metrics['false_success_decisions']}"
    )
    print(f"Validation conditions measured: {len(counts)}; report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
