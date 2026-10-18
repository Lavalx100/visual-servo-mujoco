"""Held-out calibration of image-distance uncertainty from pixels and marker quality."""

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
from .motion_envelope_benchmark import _run_trial_set, _trial_identity, _write_csv
from .run import FEEDBACK_STEPS_PER_UPDATE, SUCCESS_THRESHOLD_METERS
from .timestamped_verification_benchmark import (
    CONTROL_MODES,
    DELIVERY_PROFILES,
    MAX_CONFIRMATION_WAIT_S,
    PostStopCameraReplay,
    _outcome,
    _summarize_policy_records,
    evaluate_timestamp_guarded_policy,
)

SEED_BLOCK = 970
CALIBRATION_QUANTILE = 0.99
CONDITIONAL_BINS = 3
MIN_BIN_SAMPLES = 100
MIN_BIN_SEEDS = 3
SPEED_BOUND_M_S = 0.126
# Locked from results/motion-envelope-validation-summary.csv. This is the
# earlier four-seed global q99, retained as a truly prior baseline.
PRIOR_GLOBAL_Q99_MARGIN_M = 0.025413

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
    "target_pixel_x",
    "target_pixel_y",
    "end_effector_pixel_x",
    "end_effector_pixel_y",
    "target_quality_score",
    "end_effector_quality_score",
    "combined_detection_quality_score",
    "target_area_px",
    "end_effector_area_px",
    "target_circularity",
    "end_effector_circularity",
    "pixel_noise_distance_sigma_m",
    "quality_adjusted_pixel_noise_scale_m",
    "conditional_bin",
    "physical_error_at_capture_m_evaluation_only",
    "optimistic_error_m_evaluation_only",
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
    "decision_frame_index",
    "speed_bound_m_s",
    "calibrated_margin_m",
    "status",
    "visual_confirmed",
    "physical_success_at_decision_delivery",
    "false_confirmation_at_delivery",
    "missed_controller_success",
    "estimated_distance_at_decision_m",
    "estimated_upper_distance_at_decision_m",
    "decision_margin_m",
    "physical_success_at_decision_capture",
    "physical_error_at_decision_capture_m",
    "physical_error_at_decision_delivery_m",
    "decision_latency_after_stop_s",
    "usable_frames",
    "dropped_frames",
    "pre_settle_rejections",
    "out_of_order_rejections",
    "image_margin_rejections",
    "motion_margin_rejections",
)


def _calibration_samples(condition: object, trial: dict, stream: dict) -> list[dict]:
    samples = []
    identity = _mode_identity(trial)
    for event in stream["events_by_profile"]["near_real_time"]:
        feature = event.get("quality_adjusted_pixel_noise_scale_m")
        estimate = event.get("estimated_distance_m")
        target_px = event.get("target_pixel_xy")
        tip_px = event.get("end_effector_pixel_xy")
        if (
            not event.get("capture_after_settle")
            or not event.get("both_markers_visible")
            or estimate is None
            or feature is None
            or target_px is None
            or tip_px is None
        ):
            continue
        physical = float(event["physical_error_at_capture_m"])
        target_quality = event["target_detection_quality"]
        tip_quality = event["end_effector_detection_quality"]
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
                "target_pixel_x": float(target_px[0]),
                "target_pixel_y": float(target_px[1]),
                "end_effector_pixel_x": float(tip_px[0]),
                "end_effector_pixel_y": float(tip_px[1]),
                "target_quality_score": float(target_quality["score"]),
                "end_effector_quality_score": float(tip_quality["score"]),
                "combined_detection_quality_score": float(
                    event["combined_detection_quality_score"]
                ),
                "target_area_px": float(target_quality["area_px"]),
                "end_effector_area_px": float(tip_quality["area_px"]),
                "target_circularity": float(target_quality["circularity"]),
                "end_effector_circularity": float(tip_quality["circularity"]),
                "pixel_noise_distance_sigma_m": float(
                    event["pixel_noise_distance_sigma_m"]
                ),
                "quality_adjusted_pixel_noise_scale_m": float(feature),
                "physical_error_at_capture_m_evaluation_only": physical,
                "optimistic_error_m_evaluation_only": max(
                    physical - float(estimate), 0.0
                ),
            }
        )
    return samples


def _fit_conditional_margins(
    samples: list[dict],
    *,
    quantile: float = CALIBRATION_QUANTILE,
    bin_count: int = CONDITIONAL_BINS,
    minimum_samples: int = MIN_BIN_SAMPLES,
    minimum_seeds: int = MIN_BIN_SEEDS,
) -> dict:
    if not samples:
        raise ValueError("training split produced no post-settle image estimates")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between zero and one")
    if bin_count <= 0 or minimum_samples <= 0 or minimum_seeds <= 0:
        raise ValueError("bin_count, minimum_samples, and minimum_seeds must be positive")

    features = np.asarray(
        [sample["quality_adjusted_pixel_noise_scale_m"] for sample in samples],
        dtype=float,
    )
    residuals = np.asarray(
        [sample["optimistic_error_m_evaluation_only"] for sample in samples],
        dtype=float,
    )
    global_margin = float(np.quantile(residuals, quantile, method="higher"))
    cut_points = np.unique(
        np.quantile(
            features,
            np.arange(1, bin_count) / bin_count,
            method="higher",
        )
    )
    bin_indices = np.searchsorted(cut_points, features, side="right")
    bins = []
    raw_margins = []
    for index in range(len(cut_points) + 1):
        selected = [
            sample
            for sample, assigned in zip(samples, bin_indices, strict=True)
            if int(assigned) == index
        ]
        errors = np.asarray(
            [sample["optimistic_error_m_evaluation_only"] for sample in selected],
            dtype=float,
        )
        seeds = sorted({int(sample["seed"]) for sample in selected})
        fallback = len(selected) < minimum_samples or len(seeds) < minimum_seeds
        raw_margin = (
            float(np.quantile(errors, quantile, method="higher"))
            if selected
            else global_margin
        )
        raw_margins.append(global_margin if fallback else raw_margin)
        bins.append(
            {
                "index": index,
                "feature_lower_bound_m": (
                    float(cut_points[index - 1]) if index > 0 else None
                ),
                "feature_upper_bound_m": (
                    float(cut_points[index]) if index < len(cut_points) else None
                ),
                "sample_count": len(selected),
                "seed_count": len(seeds),
                "seeds": seeds,
                "raw_q99_margin_m": raw_margin,
                "fallback_to_global": fallback,
            }
        )
    # Enforce monotonicity with the image-only uncertainty proxy. This avoids
    # assigning a smaller margin to a higher predicted pixel-noise scale.
    monotone_margins = np.maximum.accumulate(np.asarray(raw_margins, dtype=float))
    for item, margin in zip(bins, monotone_margins, strict=True):
        item["margin_m"] = float(margin)

    return {
        "method": "three empirical feature bands of a first-order pixel-noise uncertainty proxy; higher empirical q99 within each training band",
        "quantile": quantile,
        "feature": "2 px noise standard deviation times local distance Jacobian norm, divided by the minimum of the two heuristic marker quality scores (floor 0.25)",
        "feature_edges_m": [float(value) for value in cut_points],
        "global_q99_margin_m": global_margin,
        "sample_count": len(samples),
        "seed_count": len({int(sample["seed"]) for sample in samples}),
        "minimum_bin_samples": minimum_samples,
        "minimum_bin_seeds": minimum_seeds,
        "bins": bins,
        "interpretation": "empirical conditional calibration over correlated rendered frames, not a confidence guarantee",
    }


def _assign_margin(event: dict, model: dict) -> tuple[float, int]:
    feature = event.get("quality_adjusted_pixel_noise_scale_m")
    if feature is None or not math.isfinite(float(feature)):
        index = 0
    else:
        index = int(np.searchsorted(model["feature_edges_m"], float(feature), side="right"))
        index = min(index, len(model["bins"]) - 1)
    return float(model["bins"][index]["margin_m"]), index


def evaluate_conditional_envelope_policy(
    trial: dict,
    events: list[dict],
    *,
    settled_time_s: float | None,
    margin_model: dict,
    max_target_speed_m_s: float = SPEED_BOUND_M_S,
    fixed_margin_m: float | None = None,
) -> dict:
    """Evaluate an offline policy using image features and delivery age only."""
    if (
        not math.isfinite(max_target_speed_m_s)
        or max_target_speed_m_s < 0.0
        or (fixed_margin_m is not None and (not math.isfinite(fixed_margin_m) or fixed_margin_m < 0.0))
    ):
        raise ValueError("margin and speed bound must be finite and non-negative")
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
    pre_settle = out_of_order = image_rejections = motion_rejections = usable = 0
    positive_run: list[dict] = []
    newest_capture_index = -1
    last_event = None
    last_upper = last_margin = None

    for event in delivered:
        frame_index = int(event["frame_index"])
        if frame_index <= newest_capture_index:
            out_of_order += 1
            positive_run.clear()
            continue
        newest_capture_index = frame_index
        capture_time = float(event["capture_time_s"])
        if settled_time_s is None or capture_time + 1e-9 < settled_time_s:
            pre_settle += 1
            positive_run.clear()
            continue
        if not event["both_markers_visible"] or event["estimated_distance_m"] is None:
            positive_run.clear()
            continue
        usable += 1
        last_event = event
        margin = (
            float(fixed_margin_m)
            if fixed_margin_m is not None
            else _assign_margin(event, margin_model)[0]
        )
        estimate = float(event["estimated_distance_m"])
        age = float(event["camera_age_at_delivery_s"])
        image_upper = estimate + margin
        last_margin = margin
        last_upper = image_upper + max_target_speed_m_s * age
        if image_upper > SUCCESS_THRESHOLD_METERS + 1e-9:
            image_rejections += 1
            positive_run.clear()
            continue
        if last_upper > SUCCESS_THRESHOLD_METERS + 1e-9:
            motion_rejections += 1
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
                    "usable_frames": usable,
                    "dropped_frames": dropped_count,
                    "pre_settle_rejections": pre_settle,
                    "stale_rejections": 0,
                    "out_of_order_rejections": out_of_order,
                    "image_margin_rejections": image_rejections,
                    "motion_margin_rejections": motion_rejections,
                },
            )
            outcome["estimated_upper_distance_at_decision_m"] = last_upper
            outcome["decision_margin_m"] = margin
            outcome["decision_frame_index"] = frame_index
            return outcome

    outcome = _outcome(
        status="not_confirmed" if usable else "unverified",
        event=last_event,
        trial=trial,
        counts={
            "usable_frames": usable,
            "dropped_frames": dropped_count,
            "pre_settle_rejections": pre_settle,
            "stale_rejections": 0,
            "out_of_order_rejections": out_of_order,
            "image_margin_rejections": image_rejections,
            "motion_margin_rejections": motion_rejections,
        },
    )
    outcome["estimated_upper_distance_at_decision_m"] = last_upper
    outcome["decision_margin_m"] = last_margin
    outcome["decision_frame_index"] = None
    return outcome


def _summarize(records: list[dict]) -> list[dict]:
    keys = (
        "control_mode",
        "control_delay_observations",
        "delivery_profile",
        "policy",
    )
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in keys)].append(record)
    rows = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        mode, delay, delivery, policy = key
        rows.append(
            {
                "control_mode": mode,
                "control_delay_observations": delay,
                "control_delay_s": round(
                    delay * FEEDBACK_STEPS_PER_UPDATE * SIMULATION_TIMESTEP_S, 4
                ),
                "delivery_profile": delivery,
                "policy": policy,
                "trials": len(group),
                **_summarize_policy_records(group),
                "false_confirmation_rate": sum(
                    bool(item["false_confirmation_at_delivery"]) for item in group
                )
                / len(group),
                "calibrated_margin_m": group[0]["calibrated_margin_m"],
            }
        )
    return rows


def _write_figure(summary: list[dict], calibration: dict, path: Path) -> None:
    width, height = 1500, 780
    deliveries = ("near_real_time", "moderate_latency", "high_latency")
    policies = (
        ("two_post_settle_age_1s", "Age ≤ 1 s", "#5C84D6"),
        ("two_post_settle_prior_global_q99", "Prior global q99", "#D38A47"),
        ("two_post_settle_new_global_q99", "New global q99", "#9A6BC4"),
        ("two_post_settle_conditional_q99", "Conditional q99", "#21A179"),
    )
    lookup = {
        (row["delivery_profile"], row["policy"]): row
        for row in summary
        if row["control_mode"] == "fully_delayed_rgb"
        and row["control_delay_observations"] == 2
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#fbfcfe"/>',
        '<text x="750" y="42" text-anchor="middle" font-family="Arial" font-size="27" font-weight="bold" fill="#15202b">Can image-local uncertainty improve fresh visual confirmation?</text>',
        '<text x="750" y="70" text-anchor="middle" font-family="Arial" font-size="14" fill="#4a5560">New held-out seed block | fully delayed target and tip | two consecutive post-settle frames | speed cap 0.126 m/s</text>',
        '<text x="330" y="117" font-family="Arial" font-size="15" font-weight="bold" fill="#263746">Confirmation coverage</text>',
        '<text x="910" y="117" font-family="Arial" font-size="15" font-weight="bold" fill="#263746">False confirmations at delivery</text>',
    ]
    x_groups = (330, 910)
    group_width = 490
    y_top = 180
    panel_gap = 190
    labels = {"near_real_time": "Near realtime", "moderate_latency": "Moderate latency", "high_latency": "High latency"}
    for delivery_index, delivery in enumerate(deliveries):
        y = y_top + delivery_index * panel_gap
        parts.append(
            f'<text x="60" y="{y + 12}" font-family="Arial" font-size="13" font-weight="bold" fill="#35434e">{labels[delivery]}</text>'
        )
        for policy_index, (policy, label, color) in enumerate(policies):
            row = lookup.get((delivery, policy), {})
            trials = int(row.get("trials", 0))
            for metric_index, field in enumerate(
                ("visual_confirmation_rate", "false_confirmation_rate")
            ):
                x = x_groups[metric_index]
                value = float(row.get(field) or 0.0)
                count = int(
                    row.get("visually_confirmed", 0)
                    if metric_index == 0
                    else row.get("false_confirmations_at_delivery", 0)
                )
                bar_y = y + policy_index * 31
                bar_x = x + 120
                bar_width = 200 * max(0.0, min(1.0, value))
                parts.append(
                    f'<text x="{x}" y="{bar_y + 13}" font-family="Arial" font-size="10" fill="#35434e">{label}</text>'
                )
                parts.append(f'<rect x="{bar_x}" y="{bar_y + 4}" width="200" height="10" rx="4" fill="#e8edf2"/>')
                parts.append(f'<rect x="{bar_x}" y="{bar_y + 4}" width="{bar_width:.2f}" height="10" rx="4" fill="{color}"/>')
                parts.append(
                    f'<text x="{bar_x + 207}" y="{bar_y + 13}" font-family="Arial" font-size="10" fill="#35434e">{100 * value:.1f}% ({count}/{trials})</text>'
                )
    margins = ", ".join(f"{item['margin_m'] * 100:.2f}" for item in calibration["bins"])
    parts.extend(
        [
            f'<text x="750" y="740" text-anchor="middle" font-family="Arial" font-size="12" fill="#35434e">Conditional q99 margins by increasing pixel-noise feature band: {margins} cm. Marker quality is a heuristic image score.</text>',
            '<text x="750" y="760" text-anchor="middle" font-family="Arial" font-size="10" fill="#65737e">False confirmation means the simulator distance was outside 4.5 cm when the camera result arrived. Physical reach at controller stop is reported separately.</text>',
            '</svg>',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _diagnostics(
    records: list[dict],
    *,
    policy: str,
) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        grouped[(row["delivery_profile"], row["conditional_bin"])].append(row)
    result = []
    for (delivery, bin_index), group in sorted(grouped.items()):
        count = len(group)
        image_violations = sum(item["image_margin_violation"] for item in group)
        delivery_violations = sum(item["delivery_envelope_violation"] for item in group)
        result.append(
            {
                "policy": policy,
                "delivery_profile": delivery,
                "conditional_bin": bin_index,
                "delivered_post_settle_frame_events": count,
                "image_margin_violations": image_violations,
                "image_margin_coverage": 1.0 - image_violations / count if count else None,
                "delivery_envelope_violations": delivery_violations,
                "delivery_envelope_coverage": 1.0 - delivery_violations / count if count else None,
            }
        )
    return result


def run_conditional_uncertainty_benchmark(
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
    training_range = range(base_seed, base_seed + training_seeds)
    validation_range = range(base_seed + training_seeds, base_seed + training_seeds + validation_seeds)
    training_ids = set(training_range)
    validation_ids = set(validation_range)
    if training_ids & validation_ids:
        raise ValueError("training and validation seed splits must not overlap")

    samples: list[dict] = []
    records: list[dict] = []
    validation_diagnostics: list[dict] = []
    controller_trial_count = 0
    replay = PostStopCameraReplay()
    try:
        with tempfile.TemporaryDirectory(prefix="visual-servo-conditional-uncertainty-") as temp_dir:
            for condition, trial, stream in _run_trial_set(
                seeds=training_range,
                episodes_per_seed=episodes_per_seed,
                temp_dir=Path(temp_dir) / "training",
                replay=replay,
            ):
                controller_trial_count += 1
                samples.extend(_calibration_samples(condition, trial, stream))
            calibration = _fit_conditional_margins(samples)
            global_margin = calibration["global_q99_margin_m"]

            for condition, trial, stream in _run_trial_set(
                seeds=validation_range,
                episodes_per_seed=episodes_per_seed,
                temp_dir=Path(temp_dir) / "validation",
                replay=replay,
            ):
                controller_trial_count += 1
                identity = _trial_identity(condition, trial)
                for profile_events in stream["events_by_profile"].values():
                    for event in profile_events:
                        event["conditional_margin_m"], event["conditional_bin"] = _assign_margin(
                            event, calibration
                        )

                policy_specs = (
                    (
                        "two_post_settle_age_1s",
                        None,
                        lambda events: evaluate_timestamp_guarded_policy(
                            trial,
                            events,
                            settled_time_s=stream["settled_time_s"],
                            max_frame_age_s=1.0,
                        ),
                    ),
                    (
                        "two_post_settle_prior_global_q99",
                        PRIOR_GLOBAL_Q99_MARGIN_M,
                        None,
                    ),
                    ("two_post_settle_new_global_q99", global_margin, None),
                    ("two_post_settle_conditional_q99", None, None),
                )
                for delivery in DELIVERY_PROFILES:
                    events = stream["events_by_profile"][delivery.name]
                    for policy, fixed_margin, evaluator in policy_specs:
                        if evaluator is not None:
                            outcome = evaluator(events)
                            outcome["estimated_upper_distance_at_decision_m"] = None
                            outcome["decision_margin_m"] = None
                        else:
                            outcome = evaluate_conditional_envelope_policy(
                                trial,
                                events,
                                settled_time_s=stream["settled_time_s"],
                                margin_model=calibration,
                                max_target_speed_m_s=SPEED_BOUND_M_S,
                                fixed_margin_m=fixed_margin,
                            )
                        records.append(
                            {
                                **identity,
                                "physical_success_at_controller_stop": bool(trial["success"]),
                                "reaching_error_at_controller_stop_m": float(trial["reaching_error_m"]),
                                "delivery_profile": delivery.name,
                                "policy": policy,
                                "speed_bound_m_s": None if evaluator else SPEED_BOUND_M_S,
                                "calibrated_margin_m": (
                                    fixed_margin
                                    if fixed_margin is not None
                                    else global_margin
                                    if policy == "two_post_settle_new_global_q99"
                                    else None
                                ),
                                **outcome,
                            }
                        )

                    for event in events:
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
                        conditional_margin = float(event["conditional_margin_m"])
                        image_upper = float(event["estimated_distance_m"]) + conditional_margin
                        delivery_upper = image_upper + SPEED_BOUND_M_S * float(
                            event["camera_age_at_delivery_s"]
                        )
                        validation_diagnostics.append(
                            {
                                "delivery_profile": delivery.name,
                                "conditional_bin": int(event["conditional_bin"]),
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

    summary = _summarize(records)
    diagnostics = _diagnostics(validation_diagnostics, policy="conditional_q99")
    expected_count = (training_seeds + validation_seeds) * episodes_per_seed * len(CONTROL_MODES) * sum(
        len(condition.phases_rad) for condition in MOTION_CONDITIONS
    )
    if controller_trial_count != expected_count:
        raise RuntimeError(f"expected {expected_count} controller trials, generated {controller_trial_count}")
    report = {
        "schema_version": 1,
        "benchmark": "held_out_conditional_image_distance_uncertainty",
        "calibration": calibration,
        "protocol": {
            "training_seeds": sorted(training_ids),
            "validation_seeds": sorted(validation_ids),
            "episodes_per_seed": episodes_per_seed,
            "controller_trial_count": controller_trial_count,
            "validation_policy_outcome_count": len(records),
            "calibration_frame_count": len(samples),
            "calibration_independence": "training and held-out seed ranges are disjoint",
            "locked_prior_global_q99_margin_m": PRIOR_GLOBAL_Q99_MARGIN_M,
            "new_training_global_q99_margin_m": global_margin,
            "target_speed_bound_m_s": SPEED_BOUND_M_S,
            "policy_rule": "image estimate + selected training-only q99 margin + 0.126 m/s times camera age <= 4.5 cm on two consecutive post-settle frames",
            "feature_uses": "detected noisy pixel centers, known camera projection sensitivity, configured 2 px observation noise, and heuristic marker quality; no simulator ground truth",
            "evaluation_labels": "capture and delivery physical distances are used only to score coverage and false confirmations",
            "control_modes": {
                "fresh_rgb": "current RGB target and tip",
                "fully_delayed_rgb": "delayed RGB target and tip",
                "delayed_target_current_joint_tip": "delayed target pixels with current joint-state tip projection",
            },
            "limitations": [
                "The empirical q99 estimates use correlated frames from rendered simulations and are not confidence guarantees.",
                "The detector quality score is a heuristic, not a probability of correct localization.",
                "The speed cap assumes the target peak speed does not exceed 0.126 m/s; trials retain motion condition labels.",
                "This is an offline benchmark only; runtime completion behavior is unchanged.",
            ],
        },
        "validation_conditional_diagnostics": diagnostics,
        "results": summary,
        "validation_trial_results": records,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if csv_path is not None:
        _write_csv(records, TRIAL_FIELDS, csv_path)
    if summary_csv_path is not None:
        _write_csv(summary, tuple(summary[0]) if summary else (), summary_csv_path)
    if calibration_csv_path is not None:
        for sample in samples:
            feature = float(sample["quality_adjusted_pixel_noise_scale_m"])
            sample["conditional_bin"] = int(
                np.searchsorted(calibration["feature_edges_m"], feature, side="right")
            )
        _write_csv(samples, CALIBRATION_FIELDS, calibration_csv_path)
    if diagnostics_csv_path is not None:
        _write_csv(diagnostics, tuple(diagnostics[0]) if diagnostics else (), diagnostics_csv_path)
    if figure_path is not None:
        _write_figure(summary, calibration, figure_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-seeds", type=int, default=4)
    parser.add_argument("--validation-seeds", type=int, default=4)
    parser.add_argument("--episodes-per-seed", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=SEED_BLOCK)
    parser.add_argument("--output", type=Path, default=Path("artifacts/conditional-uncertainty-benchmark.json"))
    parser.add_argument("--csv", type=Path, default=Path("results/conditional-uncertainty-validation-trials.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/conditional-uncertainty-validation-summary.csv"))
    parser.add_argument("--calibration-csv", type=Path, default=Path("results/conditional-uncertainty-calibration-frames.csv"))
    parser.add_argument("--diagnostics-csv", type=Path, default=Path("results/conditional-uncertainty-validation-diagnostics.csv"))
    parser.add_argument("--figure", type=Path, default=Path("assets/demos/conditional_uncertainty_validation.svg"))
    args = parser.parse_args()
    try:
        report = run_conditional_uncertainty_benchmark(
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
    bins = ", ".join(f"{item['margin_m']:.4f} m" for item in report["calibration"]["bins"])
    print(
        f"Conditional uncertainty benchmark: {report['protocol']['controller_trial_count']} controller trials; "
        f"{report['protocol']['validation_policy_outcome_count']} held-out outcomes; "
        f"global q99 {report['calibration']['global_q99_margin_m']:.4f} m; bins {bins}; saved {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
