"""Compare visual-servo feedback sources under moving-target camera delay."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from .benchmark import _summarize
from .model import SIMULATION_TIMESTEP_S
from .run import FEEDBACK_STEPS_PER_UPDATE, SUCCESS_THRESHOLD_METERS, run_trials


DEFAULT_MOTION_AMPLITUDE_M = 0.04
DEFAULT_MOTION_FREQUENCY_HZ = 0.25
TARGET_PROFILES = ("sinusoidal", "piecewise_linear")
PIXEL_NOISE_LEVELS_PX = (0.0, 2.0)
CAMERA_DELAYS = (1, 2)
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261016


def _trial_key(trial: dict) -> tuple[int, int]:
    return trial["seed"], trial["episode"]


def _paired_comparison(candidate_trials: list[dict], baseline_trials: list[dict]) -> dict:
    """Compare per-episode outcomes and bootstrap over seed groups."""
    candidate_by_key = {_trial_key(trial): trial for trial in candidate_trials}
    baseline_by_key = {_trial_key(trial): trial for trial in baseline_trials}
    if candidate_by_key.keys() != baseline_by_key.keys():
        raise ValueError("paired timing modes must contain identical seed/episode pairs")

    ordered_keys = sorted(candidate_by_key)
    error_deltas = np.asarray(
        [
            candidate_by_key[key]["reaching_error_m"]
            - baseline_by_key[key]["reaching_error_m"]
            for key in ordered_keys
        ],
        dtype=float,
    )
    success_deltas = np.asarray(
        [
            int(candidate_by_key[key]["success"])
            - int(baseline_by_key[key]["success"])
            for key in ordered_keys
        ],
        dtype=float,
    )
    seed_ids = sorted({seed for seed, _ in ordered_keys})
    deltas_by_seed = {
        seed: np.asarray(
            [
                candidate_by_key[key]["reaching_error_m"]
                - baseline_by_key[key]["reaching_error_m"]
                for key in ordered_keys
                if key[0] == seed
            ],
            dtype=float,
        )
        for seed in seed_ids
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled_seed_indices = rng.integers(
        0, len(seed_ids), size=(BOOTSTRAP_SAMPLES, len(seed_ids))
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
        "comparison": "candidate_minus_baseline",
        "paired_trials": len(ordered_keys),
        "bootstrap_resampling_unit": "seed",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "median_reaching_error_delta_m": float(np.median(error_deltas)),
        "median_delta_bootstrap_95pct_ci_m": [
            float(value) for value in np.quantile(bootstrap_medians, (0.025, 0.975))
        ],
        "success_rate_delta_percentage_points": float(100.0 * np.mean(success_deltas)),
    }


def _fresh_visual_confirmation(trial: dict) -> bool:
    """Whether a visual-tolerance stop used a target pixel captured at stop time."""
    if trial["controller_stop_reason"] != "pixel_tolerance_reached":
        return False
    terminal_event = trial["target_visibility_trace"][-1]
    return (
        terminal_event.get("target_measurement_used", False)
        and terminal_event.get("camera_observation_age_observations", math.inf) == 0
        and terminal_event.get("camera_observation_age_simulated_s", math.inf) <= 1e-9
    )


def _summarize_mode(trials: list[dict]) -> dict:
    summary = _summarize(trials, SUCCESS_THRESHOLD_METERS)
    stop_reasons = Counter(trial["controller_stop_reason"] for trial in trials)
    terminal_ages = [
        trial["target_visibility_trace"][-1].get(
            "camera_observation_age_simulated_s", 0.0
        )
        for trial in trials
    ]
    return {
        **summary,
        "controller_stop_reasons": dict(stop_reasons),
        "visual_tolerance_stop_trials": stop_reasons["pixel_tolerance_reached"],
        "fresh_visual_confirmation_trials": sum(
            _fresh_visual_confirmation(trial) for trial in trials
        ),
        "terminal_camera_observation_age_median_s": float(np.median(terminal_ages)),
    }


def _trial_record(trial: dict) -> dict:
    terminal_event = trial["target_visibility_trace"][-1]
    return {
        "seed": trial["seed"],
        "episode": trial["episode"],
        "reaching_error_m": trial["reaching_error_m"],
        "success": trial["success"],
        "controller_stop_reason": trial["controller_stop_reason"],
        "fresh_visual_confirmation": _fresh_visual_confirmation(trial),
        "terminal_camera_observation_age_observations": terminal_event.get(
            "camera_observation_age_observations", 0
        ),
        "terminal_camera_observation_age_simulated_s": terminal_event.get(
            "camera_observation_age_simulated_s", 0.0
        ),
        "natural_target_visibility_loss_observations": trial[
            "natural_target_visibility_loss_observations"
        ],
    }


def run_sensor_timing_benchmark(
    *,
    seeds: int = 5,
    episodes_per_seed: int = 20,
    base_seed: int = 0,
    output_path: Path | None = None,
    motion_amplitude_m: float = DEFAULT_MOTION_AMPLITUDE_M,
    motion_frequency_hz: float = DEFAULT_MOTION_FREQUENCY_HZ,
) -> dict:
    """Pair camera-only and current-joint-state tip feedback under target motion.

    Each profile/noise pair includes a fresh-camera reference. For one- and
    two-observation delays, the benchmark compares a fully delayed RGB frame
    with delayed target pixels and a current joint-state projection of the tip.
    Target-motion prediction and stale-target arrival confirmation stay disabled.
    """
    if seeds <= 0 or episodes_per_seed <= 0:
        raise ValueError("seeds and episodes_per_seed must be positive")
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    if (
        not math.isfinite(motion_amplitude_m)
        or not 0.0 < motion_amplitude_m <= 0.16
    ):
        raise ValueError("motion_amplitude_m must be finite and in (0, 0.16]")
    if not math.isfinite(motion_frequency_hz) or motion_frequency_hz <= 0.0:
        raise ValueError("motion_frequency_hz must be positive and finite")

    rows = []
    with tempfile.TemporaryDirectory(prefix="visual-servo-sensor-timing-") as temp_dir:
        for profile in TARGET_PROFILES:
            for noise in PIXEL_NOISE_LEVELS_PX:
                trial_cache: dict[tuple[int, str], list[dict]] = {}

                def run_mode(
                    mode: str,
                    *,
                    delay_observations: int,
                    use_current_end_effector_state: bool,
                ) -> list[dict]:
                    cache_key = (delay_observations, mode)
                    if cache_key in trial_cache:
                        return trial_cache[cache_key]
                    trials = []
                    for seed in range(base_seed, base_seed + seeds):
                        report = run_trials(
                            episodes_per_seed,
                            seed,
                            Path(temp_dir)
                            / profile
                            / f"noise-{noise:g}"
                            / mode
                            / str(delay_observations)
                            / str(seed),
                            controller="image_feedback",
                            pixel_noise_std_px=noise,
                            target_motion_amplitude_m=motion_amplitude_m,
                            target_motion_frequency_hz=motion_frequency_hz,
                            target_motion_profile=profile,
                            camera_observation_delay_observations=delay_observations,
                            target_motion_prediction="none",
                            use_current_end_effector_state=(
                                use_current_end_effector_state
                            ),
                            feedback_allow_stale_target_confirmation=False,
                            save_media=False,
                        )
                        trials.extend(
                            {"seed": seed, **trial} for trial in report["results"]
                        )
                    trial_cache[cache_key] = trials
                    return trials

                for delay_observations in (0, *CAMERA_DELAYS):
                    mode_specs = (
                        (("fresh_camera", False),)
                        if delay_observations == 0
                        else (
                            ("fully_delayed_camera", False),
                            ("delayed_target_current_joint_state_tip", True),
                        )
                    )
                    for mode, use_current_state in mode_specs:
                        trials = run_mode(
                            mode,
                            delay_observations=delay_observations,
                            use_current_end_effector_state=use_current_state,
                        )
                        comparisons = {}
                        if mode == "fully_delayed_camera":
                            fresh_trials = run_mode(
                                "fresh_camera",
                                delay_observations=0,
                                use_current_end_effector_state=False,
                            )
                            comparisons["vs_fresh_camera"] = _paired_comparison(
                                trials, fresh_trials
                            )
                        elif mode == "delayed_target_current_joint_state_tip":
                            fully_delayed_trials = run_mode(
                                "fully_delayed_camera",
                                delay_observations=delay_observations,
                                use_current_end_effector_state=False,
                            )
                            fresh_trials = run_mode(
                                "fresh_camera",
                                delay_observations=0,
                                use_current_end_effector_state=False,
                            )
                            comparisons = {
                                "vs_fully_delayed_camera": _paired_comparison(
                                    trials, fully_delayed_trials
                                ),
                                "vs_fresh_camera": _paired_comparison(
                                    trials, fresh_trials
                                ),
                            }

                        rows.append({
                            "target_motion_profile": profile,
                            "target_motion_amplitude_m": motion_amplitude_m,
                            "target_motion_frequency_hz": motion_frequency_hz,
                            "pixel_noise_std_px": noise,
                            "mode": mode,
                            "camera_observation_delay_observations": delay_observations,
                            "camera_observation_delay_simulated_s": round(
                                delay_observations
                                * FEEDBACK_STEPS_PER_UPDATE
                                * SIMULATION_TIMESTEP_S,
                                4,
                            ),
                            "end_effector_feedback_source": (
                                "current_joint_state_projection"
                                if use_current_state
                                else "camera_observation"
                            ),
                            **_summarize_mode(trials),
                            "per_seed": [
                                {
                                    "seed": seed,
                                    **_summarize_mode(
                                        [trial for trial in trials if trial["seed"] == seed]
                                    ),
                                }
                                for seed in range(base_seed, base_seed + seeds)
                            ],
                            "paired_comparisons": comparisons,
                            "trial_results": [_trial_record(trial) for trial in trials],
                        })

    result = {
        "schema_version": 1,
        "benchmark": "moving_target_feedback_sensor_timing",
        "success_threshold_m": SUCCESS_THRESHOLD_METERS,
        "feedback_update_simulated_time_s": round(
            FEEDBACK_STEPS_PER_UPDATE * SIMULATION_TIMESTEP_S, 4
        ),
        "target_motion_profiles": list(TARGET_PROFILES),
        "target_motion_definition": {
            "sinusoidal": "x(t) = center + amplitude * sin(2*pi*frequency*t)",
            "piecewise_linear": (
                "x(t) = center + amplitude * (2/pi) * asin(sin(2*pi*frequency*t))"
            ),
        },
        "motion_amplitude_m": motion_amplitude_m,
        "motion_frequency_hz": motion_frequency_hz,
        "peak_target_speed_m_s": (
            2.0 * math.pi * motion_frequency_hz * motion_amplitude_m
        ),
        "camera_delays_observations": [0, *CAMERA_DELAYS],
        "pixel_noise_std_levels_px": list(PIXEL_NOISE_LEVELS_PX),
        "prediction_mode": "none",
        "stale_target_confirmation": False,
        "modes": {
            "fresh_camera": "current RGB observations for target and end effector",
            "fully_delayed_camera": "delayed RGB observations for target and end effector",
            "delayed_target_current_joint_state_tip": (
                "delayed RGB target observation and current joint-state projection of the tip"
            ),
        },
        "target_sequences_paired_by_seed_profile_and_noise": True,
        "measurement_rng_restarted_from_same_seed_for_each_mode": True,
        "fresh_visual_confirmation_definition": (
            "pixel_tolerance_reached using a target measurement with zero observation "
            "and simulated age"
        ),
        "seeds": seeds,
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
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--motion-amplitude-m",
        type=float,
        default=DEFAULT_MOTION_AMPLITUDE_M,
    )
    parser.add_argument(
        "--motion-frequency-hz",
        type=float,
        default=DEFAULT_MOTION_FREQUENCY_HZ,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/sensor-timing-benchmark.json"),
    )
    args = parser.parse_args()
    try:
        report = run_sensor_timing_benchmark(
            seeds=args.seeds,
            episodes_per_seed=args.episodes_per_seed,
            base_seed=args.base_seed,
            output_path=args.output,
            motion_amplitude_m=args.motion_amplitude_m,
            motion_frequency_hz=args.motion_frequency_hz,
        )
    except ValueError as error:
        parser.error(str(error))

    print(
        "Profile         Noise  Mode                                      "
        "Delay  Success  Median   P95"
    )
    for row in report["results"]:
        print(
            f"{row['target_motion_profile']:15} {row['pixel_noise_std_px']:5.1f}px  "
            f"{row['mode']:42} {row['camera_observation_delay_simulated_s']:4.1f}s  "
            f"{row['success_rate']:6.1%}  {row['median_reaching_error_m']:.3f}m  "
            f"{row['p95_reaching_error_m']:.3f}m  "
            f"current-time visual stops {row['fresh_visual_confirmation_trials']}/{row['trials']}"
        )
    print(f"Saved sensor-timing benchmark report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
