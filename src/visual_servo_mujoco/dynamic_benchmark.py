"""Benchmark camera delay and sinusoidal target motion for image feedback."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

from .benchmark import _summarize
from .model import SIMULATION_TIMESTEP_S
from .run import FEEDBACK_STEPS_PER_UPDATE, SUCCESS_THRESHOLD_METERS, run_trials

DEFAULT_MOTION_AMPLITUDE_M = 0.04
DEFAULT_MOTION_FREQUENCY_HZ = 0.25
CAMERA_DELAY_CONDITIONS = (0, 1, 2)
POLICIES = (
    ("strict_fresh_vision", False),
    ("stale_target_confirmation", True),
)


def run_dynamic_benchmark(
    *,
    seeds: int = 5,
    episodes_per_seed: int = 20,
    base_seed: int = 0,
    output_path: Path | None = None,
    motion_amplitude_m: float = DEFAULT_MOTION_AMPLITUDE_M,
    motion_frequency_hz: float = DEFAULT_MOTION_FREQUENCY_HZ,
) -> dict:
    """Compare fresh and stale arrival rules with stationary and moving targets.

    Each policy is evaluated on a stationary control at zero and two-observation
    delay, then on the same moving-target sequence at zero, one, and two
    observations of delay. A feedback update represents 0.4 simulated seconds.
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

    conditions = [
        {
            "name": f"stationary_delay_{delay}",
            "target_motion_amplitude_m": 0.0,
            "target_motion_frequency_hz": motion_frequency_hz,
            "camera_observation_delay_observations": delay,
        }
        for delay in (0, 2)
    ] + [
        {
            "name": f"moving_delay_{delay}",
            "target_motion_amplitude_m": motion_amplitude_m,
            "target_motion_frequency_hz": motion_frequency_hz,
            "camera_observation_delay_observations": delay,
        }
        for delay in CAMERA_DELAY_CONDITIONS
    ]

    condition_reports = []
    with tempfile.TemporaryDirectory(prefix="visual-servo-dynamic-benchmark-") as temp_dir:
        for policy_name, allow_stale_target in POLICIES:
            for condition in conditions:
                trials = []
                for seed in range(base_seed, base_seed + seeds):
                    report = run_trials(
                        episodes_per_seed,
                        seed,
                        Path(temp_dir)
                        / policy_name
                        / condition["name"]
                        / str(seed),
                        controller="image_feedback",
                        feedback_allow_stale_target_confirmation=allow_stale_target,
                        target_motion_amplitude_m=(
                            condition["target_motion_amplitude_m"]
                        ),
                        target_motion_frequency_hz=(
                            condition["target_motion_frequency_hz"]
                        ),
                        camera_observation_delay_observations=(
                            condition["camera_observation_delay_observations"]
                        ),
                        save_media=False,
                    )
                    trials.extend({"seed": seed, **trial} for trial in report["results"])

                condition_reports.append({
                    "policy": policy_name,
                    "condition": condition,
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
        "schema_version": 1,
        "benchmark": "dynamic_target_camera_latency",
        "success_threshold_m": SUCCESS_THRESHOLD_METERS,
        "feedback_update_simulated_time_s": round(
            FEEDBACK_STEPS_PER_UPDATE * SIMULATION_TIMESTEP_S, 4
        ),
        "target_trajectory": "sinusoidal_x",
        "motion_amplitude_m": motion_amplitude_m,
        "motion_frequency_hz": motion_frequency_hz,
        "peak_target_speed_m_s": 2.0 * math.pi * motion_frequency_hz * motion_amplitude_m,
        "seeds": seeds,
        "episodes_per_seed": episodes_per_seed,
        "target_sequences_paired_by_seed_and_motion_condition": True,
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
    parser.add_argument(
        "--motion-amplitude-m",
        type=float,
        default=DEFAULT_MOTION_AMPLITUDE_M,
        help="target's sinusoidal x-motion amplitude in metres",
    )
    parser.add_argument(
        "--motion-frequency-hz",
        type=float,
        default=DEFAULT_MOTION_FREQUENCY_HZ,
        help="target's sinusoidal motion frequency in hertz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/dynamic-target-benchmark.json"),
    )
    args = parser.parse_args()
    try:
        report = run_dynamic_benchmark(
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
        "Policy                      Condition             Success   Median   P95 "
        "  Stale stops  False successes"
    )
    for condition in report["conditions"]:
        print(
            f"{condition['policy']:27} {condition['condition']['name']:21} "
            f"{condition['success_rate']:6.1%}   "
            f"{condition['median_reaching_error_m']:.3f} m  "
            f"{condition['p95_reaching_error_m']:.3f} m  "
            f"{condition['stale_target_confirmation_trials']:11}  "
            f"{condition['stale_target_false_success_trials']:15}"
        )
    print(f"Saved dynamic benchmark report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
