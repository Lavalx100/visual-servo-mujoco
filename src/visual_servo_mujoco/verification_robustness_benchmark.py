"""Stress the opt-in terminal verifier without changing reaching control."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from .run import SUCCESS_THRESHOLD_METERS, run_trials


VERIFIER_STRESS_CONDITIONS = {
    "nominal": {},
    "fovy_plus_5deg": {"terminal_verification_fovy_error_deg": 5.0},
    "fovy_minus_5deg": {"terminal_verification_fovy_error_deg": -5.0},
    "camera_x_plus_2cm": {
        "terminal_verification_camera_position_error_m": (0.02, 0.0, 0.0)
    },
    "camera_yaw_plus_3deg": {
        "terminal_verification_camera_rotation_error_deg": (0.0, 0.0, 3.0)
    },
    "dim_exposure_x0_25": {"terminal_verification_brightness_scale": 0.25},
    "bright_exposure_x1_5": {"terminal_verification_brightness_scale": 1.5},
    "target_occluded": {"terminal_verification_occluded_markers": ("target",)},
    "end_effector_occluded": {
        "terminal_verification_occluded_markers": ("end_effector",)
    },
}
QUALITY_THRESHOLDS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6)


def _summarize(records: list[dict]) -> dict:
    available = [record for record in records if record["visual_success_estimate"] is not None]
    true_positive = sum(
        record["visual_success_estimate"] is True and record["physical_success"]
        for record in available
    )
    false_positive = sum(
        record["visual_success_estimate"] is True and not record["physical_success"]
        for record in available
    )
    true_negative = sum(
        record["visual_success_estimate"] is False and not record["physical_success"]
        for record in available
    )
    false_negative = sum(
        record["visual_success_estimate"] is False and record["physical_success"]
        for record in available
    )
    visible_successes = sum(record["physical_success"] for record in available)
    visible_failures = len(available) - visible_successes
    distance_errors = [
        record["estimated_minus_physical_error_m"]
        for record in available
        if record["estimated_minus_physical_error_m"] is not None
    ]
    return {
        "trials": len(records),
        "both_markers_visible_trials": sum(record["both_markers_visible"] for record in records),
        "visual_estimate_available_trials": len(available),
        "visual_estimate_availability_rate": len(available) / len(records) if records else None,
        "physical_successes": sum(record["physical_success"] for record in records),
        "confusion_counts_when_estimate_available": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
        "false_success_rate_when_failed_trial_visible": (
            false_positive / visible_failures if visible_failures else None
        ),
        "missed_success_rate_when_successful_trial_visible": (
            false_negative / visible_successes if visible_successes else None
        ),
        "estimated_minus_physical_error_p50_abs_m": (
            float(np.quantile(np.abs(distance_errors), 0.50))
            if distance_errors
            else None
        ),
        "estimated_minus_physical_error_p95_abs_m": (
            float(np.quantile(np.abs(distance_errors), 0.95))
            if distance_errors
            else None
        ),
        "controller_outcomes_shared_across_stress_conditions": sum(
            record["controller_outcome_shared_across_stress_conditions"]
            for record in records
        ),
    }


def _apply_quality_threshold(records: list[dict], threshold: float) -> list[dict]:
    """Re-score candidate image estimates after an abstention threshold."""
    qualified_records = []
    for record in records:
        candidate_distance = record["candidate_estimated_distance_m"]
        quality_score = record["detection_quality_score"]
        available = (
            candidate_distance is not None
            and quality_score is not None
            and quality_score >= threshold
        )
        qualified_records.append(
            {
                **record,
                "estimated_minus_physical_error_m": (
                    candidate_distance - record["physical_error_m"]
                    if available
                    else None
                ),
                "visual_success_estimate": (
                    candidate_distance <= SUCCESS_THRESHOLD_METERS if available else None
                ),
            }
        )
    return qualified_records


def run_verification_robustness_benchmark(
    *,
    seeds: int = 5,
    episodes_per_seed: int = 20,
    base_seed: int = 100,
    pixel_noise_std_px: float = 20.0,
    output_path: Path | None = None,
) -> dict:
    """Measure one-factor verifier stress cases on repeatable seeded targets."""
    if seeds <= 0 or episodes_per_seed <= 0:
        raise ValueError("seeds and episodes_per_seed must be positive")
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    if not math.isfinite(pixel_noise_std_px) or pixel_noise_std_px < 0.0:
        raise ValueError("pixel_noise_std_px must be finite and non-negative")

    records_by_condition = {name: [] for name in VERIFIER_STRESS_CONDITIONS}
    with tempfile.TemporaryDirectory(prefix="visual-servo-verification-robustness-") as temp_dir:
        for seed in range(base_seed, base_seed + seeds):
            report = run_trials(
                episodes_per_seed,
                seed,
                Path(temp_dir) / str(seed),
                controller="image_feedback",
                pixel_noise_std_px=pixel_noise_std_px,
                verify_terminal=True,
                terminal_verification_stress_candidates={
                    name: settings
                    for name, settings in VERIFIER_STRESS_CONDITIONS.items()
                    if name != "nominal"
                },
                save_media=False,
            )
            for trial in report["results"]:
                episode = int(trial["episode"])
                candidates = trial["terminal_verification_stress_candidates"]
                for condition_name in VERIFIER_STRESS_CONDITIONS:
                    evidence = (
                        trial["terminal_visual_verification"]
                        if condition_name == "nominal"
                        else candidates[condition_name]
                    )
                    estimate_m = evidence["estimated_distance_m"]
                    records_by_condition[condition_name].append(
                        {
                            "seed": seed,
                            "episode": episode,
                            "shared_control_trial_id": f"{seed}:{episode}",
                            "physical_error_m": trial["reaching_error_m"],
                            "physical_success": bool(trial["success"]),
                            "controller_stop_reason": trial["controller_stop_reason"],
                            "controller_iterations": trial["controller_iterations"],
                            "controller_outcome_shared_across_stress_conditions": True,
                            "target_visible": evidence["target_visible"],
                            "end_effector_visible": evidence["end_effector_visible"],
                            "both_markers_visible": evidence["both_markers_visible"],
                            "estimated_distance_m": estimate_m,
                            "candidate_estimated_distance_m": evidence[
                                "candidate_estimated_distance_m_for_evaluation_only"
                            ],
                            "detection_quality_score": evidence[
                                "combined_detection_quality_score"
                            ],
                            "target_detection_quality": evidence[
                                "target_detection_quality"
                            ],
                            "end_effector_detection_quality": evidence[
                                "end_effector_detection_quality"
                            ],
                            "estimated_minus_physical_error_m": (
                                estimate_m - trial["reaching_error_m"]
                                if estimate_m is not None
                                else None
                            ),
                            "visual_success_estimate": evidence[
                                "visual_success_estimate"
                            ],
                            "visual_estimate_status": evidence[
                                "visual_estimate_status"
                            ],
                        }
                    )

    rows = []
    for condition_name, stress_settings in VERIFIER_STRESS_CONDITIONS.items():
        records = records_by_condition[condition_name]
        rows.append(
            {
                "condition": condition_name,
                "stress_settings": stress_settings,
                **_summarize(records),
                "per_seed": [
                    {
                        "seed": seed,
                        **_summarize(
                            [record for record in records if record["seed"] == seed]
                        ),
                    }
                    for seed in range(base_seed, base_seed + seeds)
                ],
                "quality_threshold_sweep": [
                    {
                        "minimum_quality_score": threshold,
                        **_summarize(_apply_quality_threshold(records, threshold)),
                    }
                    for threshold in QUALITY_THRESHOLDS
                ],
                "trial_results": records,
            }
        )

    result = {
        "schema_version": 3,
        "benchmark": "terminal_verifier_quality_coverage_and_image_stress",
        "control_controller": "image_feedback",
        "control_pixel_noise_std_px": pixel_noise_std_px,
        "control_policy_changed": False,
        "control_trajectory_shared_across_conditions": True,
        "control_trajectories_per_seed_episode": 1,
        "pairing_protocol": (
            "Render the post-control verification view once per seed/episode, then score all "
            "image and estimator stress conditions from that same frame and final physics state."
        ),
        "physical_success_definition": "MuJoCo final distance <= 4.5 cm",
        "visual_success_definition": "calibrated verifier distance estimate <= 4.5 cm",
        "camera_image_age_s": 0.0,
        "visual_effects_scope": (
            "Calibration offsets affect only back-projection. Exposure scaling is a uniform "
            "post-render brightness gain; occlusion removes marker-colored pixels with "
            "inpainting. These are controlled image stresses, not real-camera evidence."
        ),
        "quality_score_definition": (
            "The per-marker score is the minimum of normalized contour area, circularity, "
            "median saturation, and median brightness. The combined score is the minimum "
            "of the target and end-effector scores. This is a heuristic, not a calibrated probability."
        ),
        "stress_conditions_are_one_factor_at_a_time": True,
        "seeds": seeds,
        "base_seed": base_seed,
        "episodes_per_seed": episodes_per_seed,
        "results": rows,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes-per-seed", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=100)
    parser.add_argument("--pixel-noise-std-px", type=float, default=20.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/verification-robustness-benchmark.json"),
    )
    args = parser.parse_args()
    try:
        report = run_verification_robustness_benchmark(
            seeds=args.seeds,
            episodes_per_seed=args.episodes_per_seed,
            base_seed=args.base_seed,
            pixel_noise_std_px=args.pixel_noise_std_px,
            output_path=args.output,
        )
    except ValueError as error:
        parser.error(str(error))

    print("Condition                    Visible  Available  False +  False -  Error P95")
    for row in report["results"]:
        p95 = row["estimated_minus_physical_error_p95_abs_m"]
        p95_text = "n/a" if p95 is None else f"{p95 * 100:.2f} cm"
        print(
            f"{row['condition']:28} "
            f"{row['both_markers_visible_trials']:3}/{row['trials']:<3} "
            f"{row['visual_estimate_available_trials']:3}/{row['trials']:<3} "
            f"{row['confusion_counts_when_estimate_available']['false_positive']:7} "
            f"{row['confusion_counts_when_estimate_available']['false_negative']:7} "
            f"{p95_text:>10}"
        )
    print(f"Saved verifier robustness report to {args.output}")
    nominal = report["results"][0]
    print("Nominal quality threshold   Available  False +  False -")
    for row in nominal["quality_threshold_sweep"]:
        counts = row["confusion_counts_when_estimate_available"]
        print(
            f"{row['minimum_quality_score']:>7.2f} "
            f"{row['visual_estimate_available_trials']:11}/{row['trials']:<3} "
            f"{counts['false_positive']:7} {counts['false_negative']:8}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
