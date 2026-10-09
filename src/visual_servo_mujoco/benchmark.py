"""Reproducible stress benchmark for camera-based reaching."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .run import (
    FEEDBACK_MAX_JOINT_STEP_RADIANS,
    FEEDBACK_REQUIRED_TOLERANCE_CHECKS,
    FEEDBACK_TARGET_FILTER_WINDOW,
    SUCCESS_THRESHOLD_METERS,
    run_trials,
)


@dataclass(frozen=True)
class StressCondition:
    """A controlled sensor or calibration perturbation."""

    name: str
    pixel_noise_std_px: float = 0.0
    camera_fovy_error_deg: float = 0.0


CONDITIONS = (
    StressCondition("clean"),
    StressCondition("pixel_noise_2px", pixel_noise_std_px=2.0),
    StressCondition("pixel_noise_8px", pixel_noise_std_px=8.0),
    StressCondition("pixel_noise_20px", pixel_noise_std_px=20.0),
    StressCondition("fovy_error_minus_5deg", camera_fovy_error_deg=-5.0),
    StressCondition("fovy_error_plus_5deg", camera_fovy_error_deg=5.0),
    StressCondition(
        "pixel_noise_8px_fovy_error_plus_5deg",
        pixel_noise_std_px=8.0,
        camera_fovy_error_deg=5.0,
    ),
)
CONTROLLERS = ("open_loop", "image_feedback")


def _summarize(trials: list[dict], threshold_m: float) -> dict:
    errors = np.asarray([trial["reaching_error_m"] for trial in trials], dtype=float)
    final_pixel_errors = [
        trial["final_pixel_error_px"]
        for trial in trials
        if trial["final_pixel_error_px"] is not None
    ]
    successes = int(np.count_nonzero(errors <= threshold_m))
    return {
        "trials": len(trials),
        "successes": successes,
        "failures": len(trials) - successes,
        "success_rate": successes / len(trials),
        "median_reaching_error_m": float(np.median(errors)),
        "p95_reaching_error_m": float(np.quantile(errors, 0.95, method="higher")),
        "max_reaching_error_m": float(np.max(errors)),
        "median_final_pixel_error_px": (
            float(np.median(final_pixel_errors)) if final_pixel_errors else None
        ),
        "median_controller_iterations": float(
            np.median([trial["controller_iterations"] for trial in trials])
        ),
        "median_simulated_motion_time_s": float(
            np.median([trial["simulated_motion_time_s"] for trial in trials])
        ),
        "final_pixel_error_observations": len(final_pixel_errors),
        "median_perception_error_m": float(
            np.median([trial["perception_error_m"] for trial in trials])
        ),
        "failure_reasons": dict(
            Counter(
                trial["failure_reason"]
                for trial in trials
                if trial["failure_reason"] is not None
            )
        ),
    }


def run_benchmark(
    *,
    seeds: int = 5,
    episodes_per_seed: int = 20,
    base_seed: int = 0,
    output_path: Path | None = None,
    feedback_target_filter_window: int = FEEDBACK_TARGET_FILTER_WINDOW,
    feedback_required_tolerance_checks: int = FEEDBACK_REQUIRED_TOLERANCE_CHECKS,
    feedback_max_joint_step_radians: float = FEEDBACK_MAX_JOINT_STEP_RADIANS,
) -> dict:
    """Run every stress condition on paired target sequences and save a report."""
    if seeds <= 0 or episodes_per_seed <= 0:
        raise ValueError("seeds and episodes_per_seed must be positive")
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    if (
        isinstance(feedback_target_filter_window, bool)
        or not isinstance(feedback_target_filter_window, int)
        or feedback_target_filter_window <= 0
    ):
        raise ValueError("feedback_target_filter_window must be a positive integer")
    if (
        isinstance(feedback_required_tolerance_checks, bool)
        or not isinstance(feedback_required_tolerance_checks, int)
        or feedback_required_tolerance_checks <= 0
    ):
        raise ValueError("feedback_required_tolerance_checks must be a positive integer")
    if (
        isinstance(feedback_max_joint_step_radians, bool)
        or not isinstance(feedback_max_joint_step_radians, (int, float))
        or not math.isfinite(feedback_max_joint_step_radians)
        or feedback_max_joint_step_radians <= 0.0
    ):
        raise ValueError("feedback_max_joint_step_radians must be a positive finite number")

    condition_reports = []
    with tempfile.TemporaryDirectory(prefix="visual-servo-benchmark-") as temp_dir:
        for controller in CONTROLLERS:
            for condition in CONDITIONS:
                trials = []
                for seed in range(base_seed, base_seed + seeds):
                    report = run_trials(
                        episodes_per_seed,
                        seed,
                        Path(temp_dir) / controller / condition.name / str(seed),
                        pixel_noise_std_px=condition.pixel_noise_std_px,
                        camera_fovy_error_deg=condition.camera_fovy_error_deg,
                        controller=controller,
                        feedback_target_filter_window=feedback_target_filter_window,
                        feedback_required_tolerance_checks=feedback_required_tolerance_checks,
                        feedback_max_joint_step_radians=feedback_max_joint_step_radians,
                        save_media=False,
                    )
                    trials.extend({"seed": seed, **trial} for trial in report["results"])

                condition_reports.append({
                    "controller": controller,
                    "condition": asdict(condition),
                    **_summarize(trials, SUCCESS_THRESHOLD_METERS),
                    "per_seed": [
                        {
                            "seed": seed,
                            **_summarize(
                                [trial for trial in trials if trial["seed"] == seed],
                                SUCCESS_THRESHOLD_METERS,
                            ),
                        }
                        for seed in range(base_seed, base_seed + seeds)
                    ],
                    "trial_results": trials,
                })

    result = {
        "schema_version": 5,
        "benchmark": "camera_based_reaching_robustness",
        "success_threshold_m": SUCCESS_THRESHOLD_METERS,
        "feedback_target_filter": {
            "method": "rolling_coordinate_median",
            "window": feedback_target_filter_window,
        },
        "feedback_required_tolerance_checks": feedback_required_tolerance_checks,
        "feedback_max_joint_step_radians": feedback_max_joint_step_radians,
        "base_seed": base_seed,
        "seeds": seeds,
        "episodes_per_seed": episodes_per_seed,
        "controllers": list(CONTROLLERS),
        "target_sequences_are_paired_across_controllers_and_conditions": True,
        "conditions": condition_reports,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5, help="number of deterministic seeds")
    parser.add_argument(
        "--episodes-per-seed", type=int, default=20, help="target trials for each seed"
    )
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--feedback-target-filter-window",
        type=int,
        default=FEEDBACK_TARGET_FILTER_WINDOW,
        help="number of recent target detections used by the feedback median",
    )
    parser.add_argument(
        "--feedback-required-tolerance-checks",
        type=int,
        default=FEEDBACK_REQUIRED_TOLERANCE_CHECKS,
        help="consecutive image-space checks required before stopping",
    )
    parser.add_argument(
        "--feedback-max-joint-step-radians",
        type=float,
        default=FEEDBACK_MAX_JOINT_STEP_RADIANS,
        help="maximum joint-angle change for one feedback motion update",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    args = parser.parse_args()
    if args.seeds <= 0 or args.episodes_per_seed <= 0:
        parser.error("--seeds and --episodes-per-seed must be positive")
    if args.base_seed < 0:
        parser.error("--base-seed must be non-negative")
    if args.feedback_target_filter_window <= 0:
        parser.error("--feedback-target-filter-window must be a positive integer")
    if args.feedback_required_tolerance_checks <= 0:
        parser.error("--feedback-required-tolerance-checks must be a positive integer")
    if (
        not math.isfinite(args.feedback_max_joint_step_radians)
        or args.feedback_max_joint_step_radians <= 0.0
    ):
        parser.error("--feedback-max-joint-step-radians must be positive and finite")

    report = run_benchmark(
        seeds=args.seeds,
        episodes_per_seed=args.episodes_per_seed,
        base_seed=args.base_seed,
        output_path=args.output,
        feedback_target_filter_window=args.feedback_target_filter_window,
        feedback_required_tolerance_checks=args.feedback_required_tolerance_checks,
        feedback_max_joint_step_radians=args.feedback_max_joint_step_radians,
    )
    print(
        "Controller     Condition                                  "
        "Success   Median   P95     Max   Updates  Sim s"
    )
    for condition in report["conditions"]:
        name = condition["condition"]["name"]
        print(
            f"{condition['controller']:14} {name:42} {condition['success_rate']:6.1%}"
            f"   {condition['median_reaching_error_m']:.3f} m"
            f"  {condition['p95_reaching_error_m']:.3f} m"
            f"  {condition['max_reaching_error_m']:.3f} m"
            f"  {condition['median_controller_iterations']:6.1f}"
            f"  {condition['median_simulated_motion_time_s']:5.2f}"
        )
    print(f"Saved benchmark report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
