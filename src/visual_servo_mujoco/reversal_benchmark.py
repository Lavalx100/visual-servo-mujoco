"""Compare target-motion predictors across reversals, delays, and pixel noise."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from .benchmark import _summarize
from .model import SIMULATION_TIMESTEP_S
from .run import FEEDBACK_STEPS_PER_UPDATE, SUCCESS_THRESHOLD_METERS, run_trials


PREDICTION_MODES = ("none", "constant_velocity", "reversal_aware")
TARGET_PROFILES = ("sinusoidal", "piecewise_linear")
PIXEL_NOISE_LEVELS_PX = (0.0, 2.0)
CAMERA_DELAYS = (1, 2)
DEFAULT_MOTION_AMPLITUDE_M = 0.04
DEFAULT_MOTION_FREQUENCY_HZ = 0.25


def _paired_comparison(predicted_trials: list[dict], baseline_trials: list[dict]) -> dict:
    """Summarize per-episode paired error and success differences."""
    def trial_key(trial: dict) -> tuple[int, int]:
        return trial["seed"], trial["episode"]

    predicted_by_key = {trial_key(trial): trial for trial in predicted_trials}
    baseline_by_key = {trial_key(trial): trial for trial in baseline_trials}
    if predicted_by_key.keys() != baseline_by_key.keys():
        raise ValueError("paired benchmark modes must contain identical trials")

    ordered_keys = sorted(predicted_by_key)
    error_deltas = np.asarray(
        [
            predicted_by_key[item]["reaching_error_m"]
            - baseline_by_key[item]["reaching_error_m"]
            for item in ordered_keys
        ],
        dtype=float,
    )
    success_deltas = np.asarray(
        [
            int(predicted_by_key[item]["success"])
            - int(baseline_by_key[item]["success"])
            for item in ordered_keys
        ],
        dtype=float,
    )
    deltas_by_seed = {
        seed: np.asarray(
            [
                predicted_by_key[item]["reaching_error_m"]
                - baseline_by_key[item]["reaching_error_m"]
                for item in ordered_keys
                if item[0] == seed
            ],
            dtype=float,
        )
        for seed in sorted({item[0] for item in ordered_keys})
    }
    seed_ids = tuple(deltas_by_seed)
    rng = np.random.default_rng(20261010)
    sampled_seed_indices = rng.integers(
        0, len(seed_ids), size=(2000, len(seed_ids))
    )
    bootstrap_medians = np.asarray(
        [
            np.median(
                np.concatenate(
                    [deltas_by_seed[seed_ids[index]] for index in sampled_indices]
                )
            )
            for sampled_indices in sampled_seed_indices
        ],
        dtype=float,
    )
    return {
        "comparison": "predictor_minus_no_prediction",
        "paired_trials": len(error_deltas),
        "bootstrap_resampling_unit": "seed",
        "median_reaching_error_delta_m": float(np.median(error_deltas)),
        "median_delta_bootstrap_95pct_ci_m": [
            float(value) for value in np.quantile(bootstrap_medians, (0.025, 0.975))
        ],
        "mean_reaching_error_delta_m": float(np.mean(error_deltas)),
        "success_rate_delta_percentage_points": float(100.0 * np.mean(success_deltas)),
    }


def run_reversal_benchmark(
    *,
    seeds: int = 5,
    episodes_per_seed: int = 20,
    base_seed: int = 0,
    output_path: Path | None = None,
    motion_amplitude_m: float = DEFAULT_MOTION_AMPLITUDE_M,
    motion_frequency_hz: float = DEFAULT_MOTION_FREQUENCY_HZ,
) -> dict:
    """Run paired target-estimation trials over smooth and sharp reversals."""
    if seeds <= 0 or episodes_per_seed <= 0:
        raise ValueError("seeds and episodes_per_seed must be positive")
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    if not math.isfinite(motion_amplitude_m) or not 0.0 < motion_amplitude_m <= 0.16:
        raise ValueError("motion_amplitude_m must be finite and in (0, 0.16]")
    if not math.isfinite(motion_frequency_hz) or motion_frequency_hz <= 0.0:
        raise ValueError("motion_frequency_hz must be positive and finite")

    conditions = [
        {
            "name": f"{profile}_delay_{delay}_noise_{noise:g}px",
            "target_motion_profile": profile,
            "camera_observation_delay_observations": delay,
            "pixel_noise_std_px": noise,
        }
        for profile in TARGET_PROFILES
        for delay in CAMERA_DELAYS
        for noise in PIXEL_NOISE_LEVELS_PX
    ]
    condition_reports = []
    with tempfile.TemporaryDirectory(prefix="visual-servo-reversal-benchmark-") as temp_dir:
        for condition in conditions:
            mode_reports = {}
            for prediction_mode in PREDICTION_MODES:
                trials = []
                for seed in range(base_seed, base_seed + seeds):
                    report = run_trials(
                        episodes_per_seed,
                        seed,
                        Path(temp_dir)
                        / prediction_mode
                        / condition["name"]
                        / str(seed),
                        controller="image_feedback",
                        pixel_noise_std_px=condition["pixel_noise_std_px"],
                        target_motion_amplitude_m=motion_amplitude_m,
                        target_motion_frequency_hz=motion_frequency_hz,
                        target_motion_profile=condition["target_motion_profile"],
                        camera_observation_delay_observations=(
                            condition["camera_observation_delay_observations"]
                        ),
                        target_motion_prediction=prediction_mode,
                        use_current_end_effector_state=True,
                        save_media=False,
                    )
                    trials.extend(
                        {"seed": seed, **trial} for trial in report["results"]
                    )
                mode_reports[prediction_mode] = {
                    **_summarize(trials, SUCCESS_THRESHOLD_METERS),
                    "trial_results": [
                        {
                            key: trial[key]
                            for key in ("seed", "episode", "reaching_error_m", "success")
                        }
                        for trial in trials
                    ],
                }
            for prediction_mode in PREDICTION_MODES:
                mode_report = mode_reports[prediction_mode]
                condition_reports.append(
                    {
                        "target_motion_prediction": prediction_mode,
                        "condition": condition,
                        **{
                            name: value
                            for name, value in mode_report.items()
                            if name != "trial_results"
                        },
                        "paired_comparison_to_none": (
                            None
                            if prediction_mode == "none"
                            else _paired_comparison(
                                mode_report["trial_results"],
                                mode_reports["none"]["trial_results"],
                            )
                        ),
                        "trial_results": mode_report["trial_results"],
                    }
                )

    result = {
        "schema_version": 1,
        "benchmark": "target_motion_prediction_with_reversals_and_pixel_noise",
        "success_threshold_m": SUCCESS_THRESHOLD_METERS,
        "feedback_update_simulated_time_s": round(
            FEEDBACK_STEPS_PER_UPDATE * SIMULATION_TIMESTEP_S, 4
        ),
        "target_profiles": list(TARGET_PROFILES),
        "target_profile_definition": {
            "sinusoidal": "x(t) = center + amplitude * sin(2*pi*frequency*t)",
            "piecewise_linear": (
                "x(t) = center + amplitude * (2/pi) * asin(sin(2*pi*frequency*t))"
            ),
        },
        "motion_amplitude_m": motion_amplitude_m,
        "motion_frequency_hz": motion_frequency_hz,
        "camera_delays_observations": list(CAMERA_DELAYS),
        "pixel_noise_std_levels_px": list(PIXEL_NOISE_LEVELS_PX),
        "prediction_modes": list(PREDICTION_MODES),
        "end_effector_feedback_source": "current_joint_state_projection",
        "paired_by_seed_condition_and_episode": True,
        "trial_result_fields": ["seed", "episode", "reaching_error_m", "success"],
        "seeds": seeds,
        "episodes_per_seed": episodes_per_seed,
        "conditions": condition_reports,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes-per-seed", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--motion-amplitude-m", type=float, default=DEFAULT_MOTION_AMPLITUDE_M)
    parser.add_argument("--motion-frequency-hz", type=float, default=DEFAULT_MOTION_FREQUENCY_HZ)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/reversal-benchmark.json")
    )
    args = parser.parse_args()
    try:
        report = run_reversal_benchmark(
            seeds=args.seeds,
            episodes_per_seed=args.episodes_per_seed,
            base_seed=args.base_seed,
            output_path=args.output,
            motion_amplitude_m=args.motion_amplitude_m,
            motion_frequency_hz=args.motion_frequency_hz,
        )
    except ValueError as error:
        parser.error(str(error))

    print("Mode               Condition                         Success  Median error  Paired Δ")
    for row in report["conditions"]:
        paired = row["paired_comparison_to_none"]
        paired_delta = (
            "—"
            if paired is None
            else f"{paired['median_reaching_error_delta_m']:+.3f} m"
        )
        print(
            f"{row['target_motion_prediction']:18} "
            f"{row['condition']['name']:32} "
            f"{row['success_rate']:6.1%}   "
            f"{row['median_reaching_error_m']:.3f} m      "
            f"{paired_delta}"
        )
    print(f"Saved reversal benchmark report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
