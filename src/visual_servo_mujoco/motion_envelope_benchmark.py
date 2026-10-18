"""Calibrate and validate a bounded motion envelope for delayed visual completion."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from .completion_speed_benchmark import MOTION_CONDITIONS, _mode_identity
from .model import SIMULATION_TIMESTEP_S
from .run import FEEDBACK_STEPS_PER_UPDATE, SUCCESS_THRESHOLD_METERS, run_trials
from .timestamped_verification_benchmark import (
    AGE_BUDGETS_S,
    CONTROL_MODES,
    DELIVERY_PROFILES,
    MAX_CONFIRMATION_WAIT_S,
    PostStopCameraReplay,
    _summarize_policy_records,
    evaluate_arrival_only_policy,
    evaluate_timestamp_guarded_policy,
    _outcome,
)

SEED_BLOCK = 960
CALIBRATION_QUANTILE = 0.99
TARGET_SPEED_BOUNDS_M_S = (0.0, 0.063, 0.08, 0.126, 0.15)

TRIAL_FIELDS = (
    "split",
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
    "max_target_speed_bound_m_s",
    "target_speed_within_configured_bound",
    "calibrated_margin_m",
    "estimated_upper_distance_at_decision_m",
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
    "image_margin_rejections",
    "motion_margin_rejections",
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
    "max_target_speed_bound_m_s",
    "calibrated_margin_m",
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
    "pre_settle_rejections_total",
    "stale_rejections_total",
    "out_of_order_rejections_total",
    "confirmation_latency_median_s",
    "confirmation_latency_p95_s",
    "image_margin_rejections_total",
    "motion_margin_rejections_total",
)

CALIBRATION_FIELDS = (
    "motion_condition",
    "target_motion_profile",
    "target_motion_phase_rad",
    "seed",
    "episode",
    "control_mode",
    "control_delay_observations",
    "frame_index",
    "capture_time_s",
    "estimated_distance_m",
    "physical_error_at_capture_m_evaluation_only",
    "optimistic_error_m_evaluation_only",
)

DIAGNOSTIC_FIELDS = (
    "motion_condition",
    "peak_target_speed_m_s",
    "delivery_profile",
    "max_target_speed_bound_m_s",
    "target_speed_within_configured_bound",
    "delivered_post_settle_frame_events",
    "image_margin_coverage",
    "image_margin_violations",
    "delivery_envelope_coverage",
    "delivery_envelope_violations",
    "calibrated_margin_m",
)


def evaluate_motion_envelope_policy(
    trial: dict,
    events: list[dict],
    *,
    settled_time_s: float | None,
    calibrated_margin_m: float,
    max_target_speed_m_s: float,
) -> dict:
    """Confirm from two consecutive frames whose distance envelope fits the radius.

    The receiver uses only the estimated image distance, capture-to-delivery age,
    the training-calibrated image margin, and the configured target speed cap.
    Simulator distances remain evaluation-only labels attached to the event.
    """
    if not math.isfinite(calibrated_margin_m) or calibrated_margin_m < 0.0:
        raise ValueError("calibrated_margin_m must be finite and non-negative")
    if not math.isfinite(max_target_speed_m_s) or max_target_speed_m_s < 0.0:
        raise ValueError("max_target_speed_m_s must be finite and non-negative")

    stop_time = float(trial["controller_stop_time_simulated_s"])
    deadline = stop_time + MAX_CONFIRMATION_WAIT_S
    delivered = sorted(
        (
            event
            for event in events
            if not event["transport_dropped"]
            and event.get("delivery_time_s") is not None
            and float(event["delivery_time_s"]) <= deadline + 1e-9
        ),
        key=lambda event: (float(event["delivery_time_s"]), event["frame_index"]),
    )
    dropped_count = sum(
        event["transport_dropped"]
        and float(event["scheduled_arrival_time_s"]) <= deadline + 1e-9
        for event in events
    )
    pre_settle_rejections = 0
    out_of_order_rejections = 0
    image_margin_rejections = 0
    motion_margin_rejections = 0
    usable_frames = 0
    positive_run: list[dict] = []
    newest_capture_index = -1
    last_usable_event: dict | None = None
    last_upper_distance: float | None = None

    for event in delivered:
        frame_index = int(event["frame_index"])
        if frame_index <= newest_capture_index:
            out_of_order_rejections += 1
            positive_run.clear()
            continue
        newest_capture_index = frame_index
        capture_time = float(event["capture_time_s"])
        if settled_time_s is None or capture_time + 1e-9 < settled_time_s:
            pre_settle_rejections += 1
            positive_run.clear()
            continue
        if not event["both_markers_visible"] or event["estimated_distance_m"] is None:
            positive_run.clear()
            continue

        usable_frames += 1
        last_usable_event = event
        age_s = float(event["camera_age_at_delivery_s"])
        estimate = float(event["estimated_distance_m"])
        image_upper_distance = estimate + calibrated_margin_m
        last_upper_distance = image_upper_distance + max_target_speed_m_s * age_s
        if image_upper_distance > SUCCESS_THRESHOLD_METERS + 1e-9:
            image_margin_rejections += 1
            positive_run.clear()
            continue
        if last_upper_distance > SUCCESS_THRESHOLD_METERS + 1e-9:
            motion_margin_rejections += 1
            positive_run.clear()
            continue

        if positive_run and frame_index == int(positive_run[-1]["frame_index"]) + 1:
            positive_run.append(event)
        else:
            positive_run = [event]
        if len(positive_run) == 2:
            outcome = _outcome(
                status="visually_confirmed",
                event=event,
                trial=trial,
                counts={
                    "usable_frames": usable_frames,
                    "dropped_frames": dropped_count,
                    "pre_settle_rejections": pre_settle_rejections,
                    "stale_rejections": 0,
                    "out_of_order_rejections": out_of_order_rejections,
                    "image_margin_rejections": image_margin_rejections,
                    "motion_margin_rejections": motion_margin_rejections,
                },
            )
            outcome["estimated_upper_distance_at_decision_m"] = last_upper_distance
            return outcome

    outcome = _outcome(
        status="not_confirmed" if usable_frames else "unverified",
        event=last_usable_event,
        trial=trial,
        counts={
            "usable_frames": usable_frames,
            "dropped_frames": dropped_count,
            "pre_settle_rejections": pre_settle_rejections,
            "stale_rejections": 0,
            "out_of_order_rejections": out_of_order_rejections,
            "image_margin_rejections": image_margin_rejections,
            "motion_margin_rejections": motion_margin_rejections,
        },
    )
    outcome["estimated_upper_distance_at_decision_m"] = last_upper_distance
    return outcome


def _calibrate_margin(samples: list[dict], quantile: float = CALIBRATION_QUANTILE) -> dict:
    if not samples:
        raise ValueError("training split produced no post-settle image estimates")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between zero and one")
    residuals = np.asarray([sample["optimistic_error_m_evaluation_only"] for sample in samples])
    return {
        "quantile": quantile,
        "method": "numpy higher empirical quantile over post-settle training frames",
        "margin_m": float(np.quantile(residuals, quantile, method="higher")),
        "frame_count": int(residuals.size),
        "seed_count": len({sample["seed"] for sample in samples}),
        "optimistic_error_median": float(np.median(residuals)),
        "optimistic_error_p95_m": float(np.quantile(residuals, 0.95, method="higher")),
        "optimistic_error_p99_m": float(np.quantile(residuals, 0.99, method="higher")),
        "optimistic_error_max_m": float(residuals.max()),
        "interpretation": "empirical training margin, not a statistical or physical guarantee",
    }


def _trial_seed_samples(
    condition: object,
    trial: dict,
    stream: dict,
) -> list[dict]:
    samples = []
    identity = _mode_identity(trial)
    for event in stream["events_by_profile"]["near_real_time"]:
        estimate = event.get("estimated_distance_m")
        if (
            not event.get("capture_after_settle")
            or not event.get("both_markers_visible")
            or estimate is None
        ):
            continue
        physical = float(event["physical_error_at_capture_m"])
        optimistic_error = max(physical - float(estimate), 0.0)
        samples.append(
            {
                "motion_condition": condition.name,
                "target_motion_profile": condition.profile,
                "target_motion_phase_rad": float(trial["target_motion_phase_rad"]),
                "seed": int(trial["seed"]),
                "episode": int(trial["episode"]),
                "control_mode": identity["control_mode"],
                "control_delay_observations": identity["control_delay_observations"],
                "frame_index": int(event["frame_index"]),
                "capture_time_s": float(event["capture_time_s"]),
                "estimated_distance_m": float(estimate),
                "physical_error_at_capture_m_evaluation_only": physical,
                "optimistic_error_m_evaluation_only": optimistic_error,
            }
        )
    return samples


def _trial_identity(condition: object, trial: dict) -> dict:
    return {
        "motion_condition": condition.name,
        "target_motion_profile": condition.profile,
        "target_motion_amplitude_m": condition.amplitude_m,
        "target_motion_frequency_hz": condition.frequency_hz,
        "peak_target_speed_m_s": condition.peak_speed_m_s,
        "target_motion_phase_rad": float(trial["target_motion_phase_rad"]),
        "seed": int(trial["seed"]),
        "episode": int(trial["episode"]),
        **_mode_identity(trial),
    }


def _aggregate(records: list[dict]) -> list[dict]:
    key_fields = (
        "motion_condition",
        "control_mode",
        "control_delay_observations",
        "delivery_profile",
        "policy",
        "max_frame_age_s",
        "max_target_speed_bound_m_s",
    )
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[field] for field in key_fields)].append(record)
    conditions = {condition.name: condition for condition in MOTION_CONDITIONS}
    rows = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        condition_name, mode, delay, delivery, policy, age, speed_bound = key
        condition = conditions[condition_name]
        summary = _summarize_policy_records(group)
        rows.append(
            {
                "motion_condition": condition_name,
                "target_motion_profile": condition.profile,
                "target_motion_amplitude_m": condition.amplitude_m,
                "target_motion_frequency_hz": condition.frequency_hz,
                "peak_target_speed_m_s": condition.peak_speed_m_s,
                "control_mode": mode,
                "control_delay_observations": delay,
                "control_delay_s": round(
                    delay * FEEDBACK_STEPS_PER_UPDATE * SIMULATION_TIMESTEP_S, 4
                ),
                "delivery_profile": delivery,
                "policy": policy,
                "max_frame_age_s": age,
                "max_target_speed_bound_m_s": speed_bound,
                "calibrated_margin_m": group[0]["calibrated_margin_m"],
                **summary,
                "image_margin_rejections_total": sum(
                    item.get("image_margin_rejections", 0) for item in group
                ),
                "motion_margin_rejections_total": sum(
                    item.get("motion_margin_rejections", 0) for item in group
                ),
            }
        )
    return rows


def _envelope_diagnostics(
    records: list[dict],
    *,
    calibrated_margin_m: float,
) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        grouped[
            (
                record["motion_condition"],
                record["peak_target_speed_m_s"],
                record["delivery_profile"],
                record["max_target_speed_bound_m_s"],
                record["target_speed_within_configured_bound"],
            )
        ].append(record)
    rows = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        condition, peak, delivery, speed_bound, bound_is_valid = key
        count = len(group)
        image_violations = sum(item["image_margin_violation"] for item in group)
        delivery_violations = sum(item["delivery_envelope_violation"] for item in group)
        rows.append(
            {
                "motion_condition": condition,
                "peak_target_speed_m_s": peak,
                "delivery_profile": delivery,
                "max_target_speed_bound_m_s": speed_bound,
                "target_speed_within_configured_bound": bound_is_valid,
                "delivered_post_settle_frame_events": count,
                "image_margin_coverage": 1.0 - image_violations / count if count else None,
                "image_margin_violations": image_violations,
                "delivery_envelope_coverage": 1.0 - delivery_violations / count if count else None,
                "delivery_envelope_violations": delivery_violations,
                "calibrated_margin_m": calibrated_margin_m,
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


def _write_figure(
    rows: list[dict],
    path: Path,
    *,
    training_seeds: int = 4,
    validation_seeds: int = 4,
    episodes_per_seed: int = 3,
    base_seed: int = SEED_BLOCK,
) -> None:
    """Draw coverage and false-positive comparisons for the 0.126 m/s envelope."""
    width, height = 1500, 850
    conditions = [condition.name for condition in MOTION_CONDITIONS]
    labels = {
        "stationary": "Stationary | 0.000 m/s",
        "sinusoidal_0p25_hz": "Sine 0.25 Hz | 0.063 m/s",
        "sinusoidal_0p50_hz": "Sine 0.50 Hz | 0.126 m/s",
        "piecewise_linear_0p50_hz": "Triangle 0.50 Hz | 0.080 m/s",
    }
    deliveries = ("near_real_time", "moderate_latency", "high_latency")
    policies = (
        ("two_post_settle_frames", 1.0, None, "Age ≤ 1.0 s", "#5C84D6"),
        ("two_post_settle_motion_envelope", None, 0.126, "Envelope ≤ 0.126 m/s", "#21A179"),
    )
    lookup: dict[tuple, dict] = {}
    for row in rows:
        if (
            row["control_mode"] != "fully_delayed_rgb"
            or row["control_delay_observations"] != 2
        ):
            continue
        key = (
            row["motion_condition"],
            row["delivery_profile"],
            row["policy"],
            row["max_frame_age_s"],
            row["max_target_speed_bound_m_s"],
        )
        bucket = lookup.setdefault(
            key,
            {
                "trials": 0,
                "visually_confirmed": 0,
                "false_confirmations_at_delivery": 0,
            },
        )
        for field in bucket:
            bucket[field] += int(row[field])
    delivery_names = {
        "near_real_time": "Near realtime",
        "moderate_latency": "Moderate latency",
        "high_latency": "High latency",
    }
    bar_x = 375
    bar_w = 250
    row_h = 48
    top_y = 190
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#fbfcfe"/>',
        '<text x="750" y="44" text-anchor="middle" font-family="Arial" font-size="27" font-weight="bold" fill="#15202b">Can a speed envelope reject stale visual confirmations?</text>',
        f'<text x="750" y="73" text-anchor="middle" font-family="Arial" font-size="14" fill="#4a5560">Held-out seeds {base_seed + training_seeds}–{base_seed + training_seeds + validation_seeds - 1} | delayed target and tip, 2 observation delay</text>',
        '<text x="388" y="125" font-family="Arial" font-size="16" font-weight="bold" fill="#263746">Confirmation coverage</text>',
        '<text x="1125" y="125" font-family="Arial" font-size="16" font-weight="bold" fill="#263746">False confirmations at delivery</text>',
    ]
    selected = []
    for condition in conditions:
        for delivery in deliveries:
            selected.append((condition, delivery))
    for row_index, (condition, delivery) in enumerate(selected):
        y = top_y + row_index * row_h
        parts.append(
            f'<text x="350" y="{y + 22}" text-anchor="end" font-family="Arial" font-size="10" fill="#35434e">{labels[condition]} / {delivery_names[delivery]}</text>'
        )
        for panel_index, metric in enumerate(("visual_confirmation_rate", "false_positive_rate")):
            panel_x = bar_x + panel_index * 570
            for policy_index, (policy, age, speed, label, color) in enumerate(policies):
                row = lookup.get((condition, delivery, policy, age, speed))
                if not row:
                    value = 0.0
                    trials = 0
                    count = 0
                else:
                    trials = int(row["trials"])
                    if metric == "visual_confirmation_rate":
                        count = int(row["visually_confirmed"])
                        value = count / trials if trials else 0.0
                    else:
                        count = int(row["false_confirmations_at_delivery"])
                        value = count / trials if trials else 0.0
                bar_y = y + policy_index * 15
                parts.append(
                    f'<rect x="{panel_x}" y="{bar_y}" width="{bar_w * value:.2f}" height="10" rx="4" fill="{color}"/>'
                )
                parts.append(
                    f'<text x="{panel_x + bar_w + 8}" y="{bar_y + 9}" font-family="Arial" font-size="9" fill="#35434e">{100 * value:.1f}% ({count}/{trials})</text>'
                )
    legend_y = 805
    for index, (_, _, _, label, color) in enumerate(policies):
        x = 480 + index * 300
        parts.append(f'<rect x="{x}" y="{legend_y - 9}" width="15" height="12" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{x + 22}" y="{legend_y}" font-family="Arial" font-size="11" fill="#35434e">{label}</text>')
    parts.extend(
        [
            '<text x="750" y="835" text-anchor="middle" font-family="Arial" font-size="10" fill="#65737e">False confirmations count positive camera decisions where MuJoCo distance was outside 4.5 cm at delivery.</text>',
            '</svg>',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _run_trial_set(
    *,
    seeds: range,
    episodes_per_seed: int,
    temp_dir: Path,
    replay: PostStopCameraReplay,
):
    for condition in MOTION_CONDITIONS:
        for phase in condition.phases_rad:
            for mode, delay, use_current_tip in CONTROL_MODES:
                for seed in seeds:
                    report = run_trials(
                        episodes_per_seed,
                        seed,
                        temp_dir
                        / condition.name
                        / f"phase-{phase:.5f}"
                        / mode
                        / str(delay)
                        / str(seed),
                        controller="image_feedback",
                        pixel_noise_std_px=2.0,
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
                        yield condition, trial, stream


def run_motion_envelope_benchmark(
    *,
    training_seeds: int = 4,
    validation_seeds: int = 4,
    episodes_per_seed: int = 3,
    base_seed: int = SEED_BLOCK,
    output_path: Path | None = None,
    csv_path: Path | None = None,
    summary_csv_path: Path | None = None,
    calibration_csv_path: Path | None = None,
    diagnostics_csv_path: Path | None = None,
    figure_path: Path | None = None,
) -> dict:
    if training_seeds <= 0 or validation_seeds <= 0 or episodes_per_seed <= 0:
        raise ValueError("training_seeds, validation_seeds, and episodes_per_seed must be positive")
    train_range = range(base_seed, base_seed + training_seeds)
    validation_range = range(base_seed + training_seeds, base_seed + training_seeds + validation_seeds)
    train_ids = set(train_range)
    validation_ids = set(validation_range)
    if train_ids & validation_ids:
        raise ValueError("training and validation seed splits must not overlap")

    calibration_samples: list[dict] = []
    validation_trials: list[dict] = []
    diagnostic_records: list[dict] = []
    controller_trial_count = 0
    replay = PostStopCameraReplay()
    try:
        with tempfile.TemporaryDirectory(prefix="visual-servo-motion-envelope-") as temp_dir:
            for condition, trial, stream in _run_trial_set(
                seeds=train_range,
                episodes_per_seed=episodes_per_seed,
                temp_dir=Path(temp_dir) / "training",
                replay=replay,
            ):
                controller_trial_count += 1
                calibration_samples.extend(_trial_seed_samples(condition, trial, stream))

            calibration = _calibrate_margin(calibration_samples)

            for condition, trial, stream in _run_trial_set(
                seeds=validation_range,
                episodes_per_seed=episodes_per_seed,
                temp_dir=Path(temp_dir) / "validation",
                replay=replay,
            ):
                controller_trial_count += 1
                identity = _trial_identity(condition, trial)
                for policy_row in (
                    {
                        "policy": "arrival_first_visible_frame",
                        "max_frame_age_s": None,
                        "max_target_speed_bound_m_s": None,
                        "calibrated_margin_m": None,
                        "evaluator": lambda events: evaluate_arrival_only_policy(trial, events),
                    },
                    *(
                        {
                            "policy": "two_post_settle_frames",
                            "max_frame_age_s": max_age,
                            "max_target_speed_bound_m_s": None,
                            "calibrated_margin_m": None,
                            "evaluator": lambda events, age=max_age: evaluate_timestamp_guarded_policy(
                                trial,
                                events,
                                settled_time_s=stream["settled_time_s"],
                                max_frame_age_s=age,
                            ),
                        }
                        for max_age in AGE_BUDGETS_S
                    ),
                    *(
                        {
                            "policy": "two_post_settle_motion_envelope",
                            "max_frame_age_s": None,
                            "max_target_speed_bound_m_s": speed_bound,
                            "calibrated_margin_m": calibration["margin_m"],
                            "evaluator": lambda events, bound=speed_bound: evaluate_motion_envelope_policy(
                                trial,
                                events,
                                settled_time_s=stream["settled_time_s"],
                                calibrated_margin_m=calibration["margin_m"],
                                max_target_speed_m_s=bound,
                            ),
                        }
                        for speed_bound in TARGET_SPEED_BOUNDS_M_S
                    ),
                ):
                    for delivery in DELIVERY_PROFILES:
                        for record in [policy_row["evaluator"](stream["events_by_profile"][delivery.name])]:
                            validation_trials.append(
                                {
                                    "split": "held_out_validation",
                                    **identity,
                                    "physical_success_at_controller_stop": bool(trial["success"]),
                                    "reaching_error_at_controller_stop_m": float(trial["reaching_error_m"]),
                                    "delivery_profile": delivery.name,
                                    "policy": policy_row["policy"],
                                    "max_frame_age_s": policy_row["max_frame_age_s"],
                                    "max_target_speed_bound_m_s": policy_row["max_target_speed_bound_m_s"],
                                    "target_speed_within_configured_bound": (
                                        None
                                        if policy_row["max_target_speed_bound_m_s"] is None
                                        else condition.peak_speed_m_s
                                        <= policy_row["max_target_speed_bound_m_s"] + 1e-9
                                    ),
                                    "calibrated_margin_m": policy_row["calibrated_margin_m"],
                                    **record,
                                }
                            )

                for delivery in DELIVERY_PROFILES:
                    for event in stream["events_by_profile"][delivery.name]:
                        if (
                            event["transport_dropped"]
                            or event.get("delivery_time_s") is None
                            or event.get("physical_error_at_delivery_m") is None
                            or not event.get("capture_after_settle")
                            or not event.get("both_markers_visible")
                            or event.get("estimated_distance_m") is None
                            or float(event["delivery_time_s"])
                            > float(trial["controller_stop_time_simulated_s"])
                            + MAX_CONFIRMATION_WAIT_S
                        ):
                            continue
                        age = float(event["camera_age_at_delivery_s"])
                        estimate = float(event["estimated_distance_m"])
                        image_upper = estimate + calibration["margin_m"]
                        for speed_bound in TARGET_SPEED_BOUNDS_M_S:
                            delivery_upper = image_upper + speed_bound * age
                            diagnostic_records.append(
                                {
                                    "motion_condition": condition.name,
                                    "peak_target_speed_m_s": condition.peak_speed_m_s,
                                    "delivery_profile": delivery.name,
                                    "max_target_speed_bound_m_s": speed_bound,
                                    "target_speed_within_configured_bound": condition.peak_speed_m_s
                                    <= speed_bound + 1e-9,
                                    "image_margin_violation": float(
                                        event["physical_error_at_capture_m"]
                                    )
                                    > image_upper + 1e-9,
                                    "delivery_envelope_violation": float(
                                        event["physical_error_at_delivery_m"]
                                    )
                                    > delivery_upper + 1e-9,
                                }
                            )
    finally:
        replay.close()

    rows = _aggregate(validation_trials)
    diagnostics = _envelope_diagnostics(
        diagnostic_records, calibrated_margin_m=calibration["margin_m"]
    )
    expected_trial_count = (training_seeds + validation_seeds) * episodes_per_seed * len(CONTROL_MODES) * sum(
        len(condition.phases_rad) for condition in MOTION_CONDITIONS
    )
    if controller_trial_count != expected_trial_count:
        raise RuntimeError(
            f"expected {expected_trial_count} controller trials, generated {controller_trial_count}"
        )
    report = {
        "schema_version": 1,
        "benchmark": "held_out_motion_envelope_visual_completion",
        "calibration": calibration,
        "protocol": {
            "controller": "unchanged image feedback; no target motion prediction",
            "training_seeds": sorted(train_ids),
            "validation_seeds": sorted(validation_ids),
            "episodes_per_seed": episodes_per_seed,
            "controller_trial_count": controller_trial_count,
            "training_controller_trials": training_seeds
            * episodes_per_seed
            * len(CONTROL_MODES)
            * sum(len(condition.phases_rad) for condition in MOTION_CONDITIONS),
            "validation_controller_trials": validation_seeds
            * episodes_per_seed
            * len(CONTROL_MODES)
            * sum(len(condition.phases_rad) for condition in MOTION_CONDITIONS),
            "validation_policy_outcome_count": len(validation_trials),
            "summary_row_count": len(rows),
            "calibration_frame_count": len(calibration_samples),
            "calibration_quantile": CALIBRATION_QUANTILE,
            "calibration_margin_source": "training-seed post-settle camera frames, unique near-real-time stream, optimistic image-distance error only",
            "calibration_independence": "training and validation seed ranges are disjoint",
            "target_motion_speed_bounds_m_s": list(TARGET_SPEED_BOUNDS_M_S),
            "motion_envelope_rule": "estimated_image_distance + training_margin + configured_target_speed_bound * camera_age_at_delivery <= 0.045 m for two consecutive post-settle frame indices",
            "camera_delivery_profiles": [
                {
                    "name": profile.name,
                    "latency_s": profile.latency_s,
                    "jitter_s": profile.jitter_s,
                    "frame_drop_probability": profile.frame_drop_probability,
                }
                for profile in DELIVERY_PROFILES
            ],
            "control_modes": {
                "fresh_rgb": "current RGB target and tip",
                "fully_delayed_rgb": "delayed RGB target and tip",
                "delayed_target_current_joint_tip": "delayed target pixels with current joint-state tip projection",
            },
            "target_conditions": [
                {
                    "name": condition.name,
                    "profile": condition.profile,
                    "amplitude_m": condition.amplitude_m,
                    "frequency_hz": condition.frequency_hz,
                    "phases_rad": list(condition.phases_rad),
                    "analytic_peak_speed_m_s": condition.peak_speed_m_s,
                }
                for condition in MOTION_CONDITIONS
            ],
            "pixel_noise_std_px": 2.0,
            "physical_success_threshold_m": SUCCESS_THRESHOLD_METERS,
            "interpretation": "controller-stop physical reaching, image capture estimates, post-stop camera decisions, and MuJoCo delivery labels are reported separately; delivery labels are evaluation-only",
            "limitations": [
                "The q99 training margin is an empirical calibration statistic over correlated frames, not a guarantee.",
                "The target speed cap must be valid for the motion; an under-specified cap is reported as outside its assumption.",
                "The envelope assumes the arm remains at its settled stop pose; held-out delivery diagnostics check the resulting distance bound.",
                "The benchmark uses rendered camera frames and simulated latency, not physical hardware.",
                "All policies are offline comparisons; runtime controller behavior is unchanged.",
            ],
        },
        "calibration_frame_samples": calibration_samples,
        "validation_envelope_diagnostics": diagnostics,
        "results": rows,
        "validation_trial_results": validation_trials,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if csv_path is not None:
        _write_csv(validation_trials, TRIAL_FIELDS, csv_path)
    if summary_csv_path is not None:
        _write_csv(rows, SUMMARY_FIELDS, summary_csv_path)
    if calibration_csv_path is not None:
        _write_csv(calibration_samples, CALIBRATION_FIELDS, calibration_csv_path)
    if diagnostics_csv_path is not None:
        _write_csv(diagnostics, DIAGNOSTIC_FIELDS, diagnostics_csv_path)
    if figure_path is not None:
        _write_figure(
            rows,
            figure_path,
            training_seeds=training_seeds,
            validation_seeds=validation_seeds,
            episodes_per_seed=episodes_per_seed,
            base_seed=base_seed,
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-seeds", type=int, default=4)
    parser.add_argument("--validation-seeds", type=int, default=4)
    parser.add_argument("--episodes-per-seed", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=SEED_BLOCK)
    parser.add_argument("--output", type=Path, default=Path("artifacts/motion-envelope-benchmark.json"))
    parser.add_argument("--csv", type=Path, default=Path("results/motion-envelope-validation-trials.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/motion-envelope-validation-summary.csv"))
    parser.add_argument("--calibration-csv", type=Path, default=Path("results/motion-envelope-calibration-frames.csv"))
    parser.add_argument("--diagnostics-csv", type=Path, default=Path("results/motion-envelope-validation-diagnostics.csv"))
    parser.add_argument("--figure", type=Path, default=Path("assets/demos/motion_envelope_validation.svg"))
    args = parser.parse_args()
    try:
        report = run_motion_envelope_benchmark(
            training_seeds=args.training_seeds,
            validation_seeds=args.validation_seeds,
            episodes_per_seed=args.episodes_per_seed,
            base_seed=args.base_seed,
            output_path=args.output,
            csv_path=args.csv,
            summary_csv_path=args.summary_csv,
            calibration_csv_path=args.calibration_csv,
            diagnostics_csv_path=args.diagnostics_csv,
            figure_path=args.figure,
        )
    except ValueError as error:
        parser.error(str(error))
    print(
        f"Motion-envelope benchmark: {report['protocol']['controller_trial_count']} controller trials; "
        f"{report['protocol']['validation_policy_outcome_count']} held-out policy outcomes; "
        f"training margin {report['calibration']['margin_m']:.4f} m; saved {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
