"""Measure timestamped post-stop visual completion across target speeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .model import SIMULATION_TIMESTEP_S
from .run import FEEDBACK_STEPS_PER_UPDATE, SUCCESS_THRESHOLD_METERS, run_trials
from .timestamped_verification_benchmark import (
    AGE_BUDGETS_S,
    CONTROL_MODES,
    DELIVERY_PROFILES,
    MOTION_PHASES_RAD,
    PostStopCameraReplay,
    _evaluate_trial_policies,
    _summarize_policy_records,
)

SEED_BLOCK = 950
PIXEL_NOISE_STD_PX = 2.0


@dataclass(frozen=True)
class MotionCondition:
    """A target path and its analytic peak speed."""

    name: str
    profile: str
    amplitude_m: float
    frequency_hz: float
    phases_rad: tuple[float, ...]

    @property
    def peak_speed_m_s(self) -> float:
        if self.amplitude_m == 0.0:
            return 0.0
        if self.profile == "sinusoidal":
            return 2.0 * math.pi * self.amplitude_m * self.frequency_hz
        if self.profile == "piecewise_linear":
            return 4.0 * self.amplitude_m * self.frequency_hz
        raise ValueError(f"unknown motion profile {self.profile!r}")


MOTION_CONDITIONS = (
    MotionCondition("stationary", "sinusoidal", 0.0, 0.25, (0.0,)),
    MotionCondition("sinusoidal_0p25_hz", "sinusoidal", 0.04, 0.25, MOTION_PHASES_RAD),
    MotionCondition("sinusoidal_0p50_hz", "sinusoidal", 0.04, 0.50, MOTION_PHASES_RAD),
    MotionCondition(
        "piecewise_linear_0p50_hz", "piecewise_linear", 0.04, 0.50, MOTION_PHASES_RAD
    ),
)

TRIAL_FIELDS = (
    "motion_condition",
    "target_motion_profile",
    "target_motion_amplitude_m",
    "target_motion_frequency_hz",
    "peak_target_speed_m_s",
    "target_motion_phase_rad",
    "seed",
    "episode",
    "control_mode",
    "control_delay_observations",
    "control_delay_s",
    "physical_success_at_controller_stop",
    "reaching_error_at_controller_stop_m",
    "delivery_profile",
    "policy",
    "max_frame_age_s",
    "status",
    "visual_confirmed",
    "physical_success_at_decision_delivery",
    "false_confirmation_at_delivery",
    "missed_controller_success",
    "estimated_distance_at_decision_m",
    "physical_success_at_decision_capture",
    "physical_error_at_decision_capture_m",
    "physical_error_at_decision_delivery_m",
    "decision_latency_after_stop_s",
    "usable_frames",
    "dropped_frames",
    "pre_settle_rejections",
    "stale_rejections",
    "out_of_order_rejections",
)

SUMMARY_FIELDS = (
    "motion_condition",
    "target_motion_profile",
    "target_motion_amplitude_m",
    "target_motion_frequency_hz",
    "peak_target_speed_m_s",
    "control_mode",
    "control_delay_observations",
    "control_delay_s",
    "delivery_profile",
    "policy",
    "max_frame_age_s",
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
    "confirmation_latency_median_s",
    "confirmation_latency_p95_s",
)


def _mode_identity(trial: dict) -> dict:
    control_delay = int(trial["camera_observation_delay_observations"])
    return {
        "control_mode": (
            "delayed_target_current_joint_tip"
            if trial["use_current_end_effector_state"]
            else "fully_delayed_rgb"
            if control_delay
            else "fresh_rgb"
        ),
        "control_delay_observations": control_delay,
        "control_delay_s": float(trial["camera_observation_delay_simulated_s"]),
    }


def _aggregate(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    key_fields = (
        "motion_condition",
        "control_mode",
        "control_delay_observations",
        "delivery_profile",
        "policy",
        "max_frame_age_s",
    )
    for record in records:
        grouped[tuple(record[field] for field in key_fields)].append(record)

    conditions = {condition.name: condition for condition in MOTION_CONDITIONS}
    rows = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        condition_name, control_mode, delay, delivery, policy, max_age = key
        condition = conditions[condition_name]
        summary = _summarize_policy_records(group)
        rows.append(
            {
                "motion_condition": condition_name,
                "target_motion_profile": condition.profile,
                "target_motion_amplitude_m": condition.amplitude_m,
                "target_motion_frequency_hz": condition.frequency_hz,
                "peak_target_speed_m_s": condition.peak_speed_m_s,
                "control_mode": control_mode,
                "control_delay_observations": delay,
                "control_delay_s": round(
                    delay * FEEDBACK_STEPS_PER_UPDATE * SIMULATION_TIMESTEP_S, 4
                ),
                "delivery_profile": delivery,
                "policy": policy,
                "max_frame_age_s": max_age,
                **summary,
                "per_seed": [
                    {
                        "seed": seed,
                        **_summarize_policy_records(
                            [record for record in group if record["seed"] == seed]
                        ),
                    }
                    for seed in sorted({record["seed"] for record in group})
                ],
            }
        )
    return rows


def _write_csv(records: list[dict], fields: tuple[str, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)


def _write_summary(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=SUMMARY_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_figure(rows: list[dict], path: Path) -> None:
    """Write paired heatmaps of confirmation coverage and stale delivery decisions."""
    width, height = 1280, 560
    conditions = [condition.name for condition in MOTION_CONDITIONS]
    condition_label = {
        "stationary": "Stationary | 0.000 m/s",
        "sinusoidal_0p25_hz": "Sinusoidal 0.25 Hz | 0.063 m/s",
        "sinusoidal_0p50_hz": "Sinusoidal 0.50 Hz | 0.126 m/s",
        "piecewise_linear_0p50_hz": "Piecewise 0.50 Hz | 0.080 m/s",
    }
    deliveries = ("near_real_time", "moderate_latency", "high_latency")
    policies = (
        ("arrival_first_visible_frame", None, "Arrival"),
        ("two_post_settle_frames", 0.5, "Age ≤0.5"),
        ("two_post_settle_frames", 1.0, "Age ≤1.0"),
    )
    delivery_label = {
        "near_real_time": "~0.05 s",
        "moderate_latency": "~0.40 s",
        "high_latency": "~0.80 s",
    }
    lookup = {}
    for row in rows:
        aggregate_key = (row["motion_condition"], row["delivery_profile"], row["policy"], row["max_frame_age_s"])
        bucket = lookup.setdefault(aggregate_key, [0, 0, 0])
        bucket[0] += int(row["trials"])
        bucket[1] += int(row["visually_confirmed"])
        bucket[2] += int(row["false_confirmations_at_delivery"])

    def color(value: float, *, false_rate: bool) -> str:
        if false_rate:
            red = int(248 - 70 * value)
            green = int(248 - 145 * value)
            blue = int(248 - 175 * value)
        else:
            red = int(244 - 165 * value)
            green = int(246 - 45 * value)
            blue = int(248 - 128 * value)
        return f"#{red:02x}{green:02x}{blue:02x}"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#fbfcfe"/>',
        '<text x="640" y="38" text-anchor="middle" font-family="Arial" font-size="25" font-weight="bold" fill="#15202b">Post-stop verification across target speeds</text>',
        '<text x="640" y="65" text-anchor="middle" font-family="Arial" font-size="13" fill="#4a5560">Rates pool five distinct control modes; each cell combines controller trials from five seeds</text>',
    ]
    x_starts = (250, 755)
    panel_width = 480
    top = 166
    row_h = 62
    cell_w = panel_width / 9
    for panel, is_false in enumerate((False, True)):
        x0 = x_starts[panel]
        title = "Visual confirmation rate" if not is_false else "Positive decisions already out at delivery"
        parts.append(f'<text x="{x0 + panel_width / 2:.1f}" y="112" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold" fill="#263746">{title}</text>')
        for d_index, delivery in enumerate(deliveries):
            center = x0 + (d_index * 3 + 1.5) * cell_w
            parts.append(f'<text x="{center:.1f}" y="132" text-anchor="middle" font-family="Arial" font-size="10" fill="#53626d">{delivery_label[delivery]}</text>')
        for p_index, (_, _, label) in enumerate(policies):
            for d_index in range(3):
                x = x0 + (d_index * 3 + p_index) * cell_w
                parts.append(f'<text x="{x + cell_w / 2:.1f}" y="155" text-anchor="middle" font-family="Arial" font-size="8.5" fill="#53626d">{label}</text>')
        for r_index, condition in enumerate(conditions):
            y = top + r_index * row_h
            for p_index, (policy, age, _) in enumerate(policies):
                for d_index, delivery in enumerate(deliveries):
                    x = x0 + (d_index * 3 + p_index) * cell_w
                    trials, confirmed, false = lookup.get(
                        (condition, delivery, policy, age), [0, 0, 0]
                    )
                    value = ((false / trials) if trials else 0.0) if is_false else ((confirmed / trials) if trials else 0.0)
                    parts.append(f'<rect x="{x + 1:.1f}" y="{y:.1f}" width="{cell_w - 2:.1f}" height="{row_h - 3}" rx="4" fill="{color(value, false_rate=is_false)}" stroke="#ffffff"/>')
                    parts.append(f'<text x="{x + cell_w / 2:.1f}" y="{y + 27:.1f}" text-anchor="middle" font-family="Arial" font-size="11" font-weight="bold" fill="#15202b">{100 * value:.1f}%</text>')
                    parts.append(f'<text x="{x + cell_w / 2:.1f}" y="{y + 43:.1f}" text-anchor="middle" font-family="Arial" font-size="8" fill="#35434e">n={trials}</text>')
            parts.append(f'<text x="235" y="{y + 26:.1f}" text-anchor="end" font-family="Arial" font-size="10" fill="#35434e">{condition_label[condition]}</text>')
    parts.extend([
        '<text x="640" y="442" text-anchor="middle" font-family="Arial" font-size="11" fill="#65737e">A stale-at-delivery positive is still distinct from an inaccurate image at capture. Physical reaching is scored at controller stop.</text>',
        '<text x="640" y="464" text-anchor="middle" font-family="Arial" font-size="11" fill="#65737e">Motion speeds are analytic peak speeds: 2πAf for sinusoidal and 4Af for piecewise-linear paths.</text>',
        '</svg>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def run_completion_speed_benchmark(
    *,
    seeds: int = 5,
    episodes_per_seed: int = 3,
    base_seed: int = SEED_BLOCK,
    output_path: Path | None = None,
    csv_path: Path | None = None,
    summary_csv_path: Path | None = None,
    figure_path: Path | None = None,
) -> dict:
    if seeds <= 0 or episodes_per_seed <= 0:
        raise ValueError("seeds and episodes_per_seed must be positive")
    outcomes: list[dict] = []
    settling_results = []
    replay = PostStopCameraReplay()
    try:
        with tempfile.TemporaryDirectory(prefix="visual-servo-completion-speed-") as temp_dir:
            for condition in MOTION_CONDITIONS:
                for phase in condition.phases_rad:
                    for mode, delay, use_current_tip in CONTROL_MODES:
                        for seed in range(base_seed, base_seed + seeds):
                            report = run_trials(
                                episodes_per_seed,
                                seed,
                                Path(temp_dir)
                                / condition.name
                                / f"phase-{phase:.5f}"
                                / mode
                                / str(delay)
                                / str(seed),
                                controller="image_feedback",
                                pixel_noise_std_px=PIXEL_NOISE_STD_PX,
                                target_motion_amplitude_m=condition.amplitude_m,
                                target_motion_frequency_hz=condition.frequency_hz,
                                target_motion_profile=condition.profile,
                                target_motion_phase_rad=phase,
                                camera_observation_delay_observations=delay,
                                target_motion_prediction="none",
                                use_current_end_effector_state=use_current_tip,
                                feedback_allow_stale_target_confirmation=False,
                                verify_terminal=False,
                                save_media=False,
                            )
                            for raw in report["results"]:
                                trial = {"seed": seed, **raw}
                                stream = replay.replay(trial, seed=seed)
                                identity = {
                                    "motion_condition": condition.name,
                                    "target_motion_profile": condition.profile,
                                    "target_motion_amplitude_m": condition.amplitude_m,
                                    "target_motion_frequency_hz": condition.frequency_hz,
                                    "peak_target_speed_m_s": condition.peak_speed_m_s,
                                    "target_motion_phase_rad": phase,
                                    **_mode_identity(trial),
                                }
                                for outcome in _evaluate_trial_policies(trial, stream):
                                    outcomes.append({**identity, **outcome})
                                settling_results.append(
                                    {
                                        **identity,
                                        "seed": seed,
                                        "episode": int(trial["episode"]),
                                        "settled_time_s": stream["settled_time_s"],
                                        "settle_duration_s": stream["settle_duration_s"],
                                    }
                                )
    finally:
        replay.close()

    rows = _aggregate(outcomes)
    controller_trials = seeds * episodes_per_seed * len(CONTROL_MODES) * sum(
        len(condition.phases_rad) for condition in MOTION_CONDITIONS
    )
    report = {
        "schema_version": 1,
        "benchmark": "timestamped_completion_speed_sweep",
        "protocol": {
            "controller": "unchanged image feedback",
            "motion_conditions": [
                {
                    **asdict(condition),
                    "phases_rad": list(condition.phases_rad),
                    "analytic_peak_speed_m_s": condition.peak_speed_m_s,
                }
                for condition in MOTION_CONDITIONS
            ],
            "control_modes": {
                "fresh_rgb": "current RGB target and tip",
                "fully_delayed_rgb": "delayed RGB target and tip",
                "delayed_target_current_joint_tip": "delayed target pixels with current joint-state tip projection",
            },
            "camera_delivery_profiles": [asdict(profile) for profile in DELIVERY_PROFILES],
            "age_budgets_s": list(AGE_BUDGETS_S),
            "pixel_noise_std_px": PIXEL_NOISE_STD_PX,
            "physical_success_threshold_m": SUCCESS_THRESHOLD_METERS,
            "seeds": seeds,
            "base_seed": base_seed,
            "episodes_per_seed": episodes_per_seed,
            "controller_trial_count": controller_trials,
            "policy_outcome_count": len(outcomes),
            "summary_row_count": len(rows),
            "aggregation": "rates pool the five control modes only within each motion, delivery, and policy cell",
            "interpretation": "physical reaching is evaluated at controller stop; camera decisions use measured image estimates and receive timestamps; ground truth at capture/delivery is evaluation only",
        },
        "results": rows,
        "settling_results": settling_results,
        "trial_results": outcomes,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if csv_path is not None:
        _write_csv(outcomes, TRIAL_FIELDS, csv_path)
    if summary_csv_path is not None:
        _write_summary(rows, summary_csv_path)
    if figure_path is not None:
        _write_figure(rows, figure_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes-per-seed", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=SEED_BLOCK)
    parser.add_argument("--output", type=Path, default=Path("artifacts/timestamped-completion-speed.json"))
    parser.add_argument("--csv", type=Path, default=Path("results/timestamped-completion-speed-trials.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/timestamped-completion-speed-summary.csv"))
    parser.add_argument("--figure", type=Path, default=Path("assets/demos/timestamped_completion_speed_benchmark.svg"))
    args = parser.parse_args()
    try:
        report = run_completion_speed_benchmark(
            seeds=args.seeds,
            episodes_per_seed=args.episodes_per_seed,
            base_seed=args.base_seed,
            output_path=args.output,
            csv_path=args.csv,
            summary_csv_path=args.summary_csv,
            figure_path=args.figure,
        )
    except ValueError as error:
        parser.error(str(error))
    print(
        f"Completion speed sweep: {report['protocol']['controller_trial_count']} controller trials; "
        f"{report['protocol']['policy_outcome_count']} policy outcomes; saved {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
