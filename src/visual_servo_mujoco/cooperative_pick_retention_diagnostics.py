"""Paired, held-out diagnosis of low-friction contact loss during cooperative lift."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .camera_guided_cooperative_pick import (
    FINGER_PAD_HALF_SIZE_M,
    _sample_positions,
    run_trial,
)
from .cooperative_pick_contact_diagnostics import (
    APPROACH_CLEARANCE_M,
    CONTROLLERS,
    FRICTION_LEVELS,
    PAYLOAD_MASS_KG,
    RELEASE_PROFILE,
)

BASE_SEED = 42000
SAMPLES_PER_SEED = 12
OUTPUT_DIRECTORY = Path("results/cooperative-pick-retention-diagnostics")
MEDIA_DIRECTORY = Path("assets/demos")
TRIALS_FILE = "cooperative-pick-retention-trials.csv"
PAD_PHASE_FILE = "cooperative-pick-retention-by-pad-phase.csv"
SUMMARY_FILE = "cooperative-pick-retention-summary.json"
SCORECARD_FILE = "cooperative-pick-retention-scorecard.svg"
VIDEO_FILE = "cooperative_pick_retention_diagnostic_pair.mp4"
GIF_FILE = "cooperative_pick_retention_diagnostic_pair.gif"
METADATA_FILE = "cooperative_pick_retention_diagnostic_demos.json"
VIDEO_FPS = 30
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 730
PANEL_HEIGHT = 342
DASHBOARD_TOP = 420


def _median(values: list[float]) -> float | None:
    return round(float(statistics.median(values)), 8) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    return round(float(np.percentile(values, percentile)), 8) if values else None


def _row_from_metrics(
    *,
    pair_id: str,
    seed: int,
    sample_index: int,
    target_xy: tuple[float, float],
    controller: str,
    friction: float,
    metrics: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    telemetry = metrics["contact_retention_kinematics"]
    pads = telemetry["by_finger_pad_and_phase"]
    hold_drifts = [
        float(arm["by_phase"]["HOLD UNDER LOAD"]["maximum_downward_payload_drift_m"])
        for arm in telemetry["payload_relative_motion_by_arm_and_phase"].values()
    ]
    hold_horizontal_drifts = [
        float(arm["by_phase"]["HOLD UNDER LOAD"]["maximum_horizontal_payload_drift_m"])
        for arm in telemetry["payload_relative_motion_by_arm_and_phase"].values()
    ]
    loaded = [
        pad[phase] for pad in pads.values() for phase in ("COOPERATIVE LIFT", "HOLD UNDER LOAD")
    ]
    contact_ratios = [
        float(record["mean_tangential_to_normal_force_ratio_when_contacting"])
        for record in loaded
        if int(record["contact_samples"]) > 0
    ]
    hold_slide_errors = [
        float(pad["HOLD UNDER LOAD"]["maximum_slide_tracking_error_m"]) for pad in pads.values()
    ]
    hold_slide_errors_mean = [
        float(pad["HOLD UNDER LOAD"]["mean_slide_tracking_error_m"]) for pad in pads.values()
    ]
    row = {
        "pair_id": pair_id,
        "seed": seed,
        "sample_index": sample_index,
        "target_x_m": target_xy[0],
        "target_y_m": target_xy[1],
        "controller": controller,
        "friction": friction,
        "physical_stable_lift_success": metrics["physical_stable_lift_success"],
        "payload_stable_lift_success": metrics["payload_stable_lift_success"],
        "full_sequence_collision_safe": metrics["full_sequence_collision_safe"],
        "hold_minimum_pad_contact_points": metrics[
            "minimum_finger_payload_contact_points_during_hold"
        ],
        "hold_contact_sample_fraction_mean_across_pads": round(
            statistics.mean(
                float(pads[name]["HOLD UNDER LOAD"]["contact_sample_fraction"]) for name in pads
            ),
            8,
        ),
        "hold_maximum_downward_payload_drift_in_hand_frame_m": round(
            max(hold_drifts, default=0.0), 8
        ),
        "hold_maximum_horizontal_payload_drift_in_hand_frame_m": round(
            max(hold_horizontal_drifts, default=0.0), 8
        ),
        "mean_tangential_to_normal_force_ratio_when_contacting": round(
            statistics.mean(contact_ratios), 8
        )
        if contact_ratios
        else 0.0,
        "maximum_slide_tracking_error_lift_or_hold_m": round(
            max(float(record["maximum_slide_tracking_error_m"]) for record in loaded), 8
        ),
        "maximum_slide_tracking_error_during_hold_m": round(max(hold_slide_errors, default=0.0), 8),
        "mean_slide_tracking_error_during_hold_m": round(
            statistics.mean(hold_slide_errors_mean), 8
        ),
        "mean_slide_tracking_error_lift_or_hold_m": round(
            statistics.mean(float(record["mean_slide_tracking_error_m"]) for record in loaded), 8
        ),
        "fresh_visual_confirmation": metrics["fresh_visual_confirmation"],
    }
    pad_rows = []
    for pad_name, phases in pads.items():
        for phase, record in phases.items():
            pad_rows.append(
                {
                    "pair_id": pair_id,
                    "controller": controller,
                    "friction": friction,
                    "finger_pad_geom": pad_name,
                    "phase": phase,
                    **record,
                }
            )
    return row, pad_rows


def _condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty retention condition")
    drifts = [float(row["hold_maximum_downward_payload_drift_in_hand_frame_m"]) for row in rows]
    ratios = [float(row["mean_tangential_to_normal_force_ratio_when_contacting"]) for row in rows]
    hold_contact = [float(row["hold_contact_sample_fraction_mean_across_pads"]) for row in rows]
    slide_error = [float(row["maximum_slide_tracking_error_lift_or_hold_m"]) for row in rows]
    hold_slide_error = [float(row["maximum_slide_tracking_error_during_hold_m"]) for row in rows]
    successes = sum(bool(row["physical_stable_lift_success"]) for row in rows)
    return {
        "trials": len(rows),
        "physical_stable_lift_successes": successes,
        "physical_stable_lift_success_rate": round(successes / len(rows), 6),
        "collision_safe_trials": sum(bool(row["full_sequence_collision_safe"]) for row in rows),
        "hold_contact_sample_fraction_mean": round(statistics.mean(hold_contact), 8),
        "hold_contact_sample_fraction_median": _median(hold_contact),
        "maximum_downward_payload_drift_in_hand_frame_m_median": _median(drifts),
        "maximum_downward_payload_drift_in_hand_frame_m_p90": _percentile(drifts, 90),
        "mean_tangential_to_normal_force_ratio_when_contacting_median": _median(ratios),
        "mean_tangential_to_normal_force_ratio_when_contacting_p90": _percentile(ratios, 90),
        "maximum_slide_tracking_error_lift_or_hold_m_median": _median(slide_error),
        "maximum_slide_tracking_error_during_hold_m_median": _median(hold_slide_error),
        "fresh_visual_confirmation": "not measured",
    }


def _paired_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for controller in CONTROLLERS:
        controller_rows = [row for row in rows if row["controller"] == controller]
        pair_ids = sorted({row["pair_id"] for row in controller_rows})
        deltas: dict[str, list[float]] = {
            "high_minus_low_hold_downward_drift_m": [],
            "high_minus_low_hold_contact_fraction": [],
            "high_minus_low_hold_slide_tracking_error_m": [],
            "high_minus_low_tangential_to_normal_ratio": [],
        }
        transitions = {
            "low_only_success": 0,
            "high_only_success": 0,
            "both_success": 0,
            "neither_success": 0,
        }
        for pair_id in pair_ids:
            low = next(
                row
                for row in controller_rows
                if row["pair_id"] == pair_id and row["friction"] == min(FRICTION_LEVELS)
            )
            high = next(
                row
                for row in controller_rows
                if row["pair_id"] == pair_id and row["friction"] == max(FRICTION_LEVELS)
            )
            for key, metric in (
                (
                    "high_minus_low_hold_downward_drift_m",
                    "hold_maximum_downward_payload_drift_in_hand_frame_m",
                ),
                (
                    "high_minus_low_hold_contact_fraction",
                    "hold_contact_sample_fraction_mean_across_pads",
                ),
                (
                    "high_minus_low_hold_slide_tracking_error_m",
                    "maximum_slide_tracking_error_during_hold_m",
                ),
                (
                    "high_minus_low_tangential_to_normal_ratio",
                    "mean_tangential_to_normal_force_ratio_when_contacting",
                ),
            ):
                deltas[key].append(float(high[metric]) - float(low[metric]))
            low_success, high_success = (
                bool(low["physical_stable_lift_success"]),
                bool(high["physical_stable_lift_success"]),
            )
            transitions[
                "both_success"
                if low_success and high_success
                else "low_only_success"
                if low_success
                else "high_only_success"
                if high_success
                else "neither_success"
            ] += 1
        out[controller] = {
            "matched_pose_pairs": len(pair_ids),
            "median_high_minus_low": {name: _median(values) for name, values in deltas.items()},
            "physical_stable_lift_transitions": transitions,
        }
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def _write_scorecard(path: Path, conditions: dict[str, Any], sample_count: int) -> None:
    width, height = 1180, 650
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1220"/>',
        "<style>text{font-family:Arial,Helvetica,sans-serif}.title{fill:#f3f6fb}.muted{fill:#a9b6c8}.blue{fill:#73b7ff}.green{fill:#68dfb3}.orange{fill:#ffbd72}</style>",
        '<text x="38" y="48" class="title" font-size="27" font-weight="700">What moves when low-friction contact is lost?</text>',
        f'<text x="38" y="80" class="muted" font-size="14">{sample_count} new target poses · paired camera/oracle controllers · 16 mm pads · nominal payload</text>',
        '<text x="38" y="108" class="muted" font-size="13">Loaded-hold contact, payload drift relative to each hand, and actuator tracking are measured separately.</text>',
    ]
    col_x = {0.5: 45, 2.2: 620}
    row_y = {"camera_estimate": 150, "oracle_xy": 380}
    max_drift = 0.24
    for friction in FRICTION_LEVELS:
        x = col_x[friction]
        parts.append(
            f'<text x="{x}" y="{row_y["camera_estimate"] - 18}" class="blue" font-size="20" font-weight="700">μ = {friction:g}</text>'
        )
        for controller in CONTROLLERS:
            summary = conditions[f"friction-{friction:g}"][controller]
            y = row_y[controller]
            label = "Camera estimate" if controller == "camera_estimate" else "Oracle XY diagnostic"
            success = summary["physical_stable_lift_successes"]
            collision_safe = summary["collision_safe_trials"]
            drift = summary["maximum_downward_payload_drift_in_hand_frame_m_median"] or 0.0
            contact = summary["hold_contact_sample_fraction_mean"]
            ratio = summary["mean_tangential_to_normal_force_ratio_when_contacting_median"] or 0.0
            slide = summary["maximum_slide_tracking_error_lift_or_hold_m_median"] or 0.0
            parts.extend(
                [
                    f'<text x="{x}" y="{y}" class="title" font-size="16" font-weight="700">{label} · stable lift {success}/{sample_count}</text>',
                    f'<text x="{x}" y="{y + 29}" class="muted" font-size="13">Median max downward payload drift in hand frame · 0–240 mm</text>',
                    f'<rect x="{x}" y="{y + 39}" width="470" height="24" rx="6" fill="#253343"/>',
                    f'<rect x="{x}" y="{y + 39}" width="{470 * min(drift, max_drift) / max_drift:.1f}" height="24" rx="6" fill="#ffbd72"/>',
                    f'<text x="{x + 480}" y="{y + 56}" class="title" font-size="13">{drift * 1000:.1f} mm</text>',
                    f'<text x="{x}" y="{y + 91}" class="green" font-size="13">hold contact fraction {contact:.3f} · median tangential/normal {ratio:.3f}</text>',
                    f'<text x="{x}" y="{y + 115}" class="muted" font-size="13">median max finger slide-command error {slide * 1000:.2f} mm</text>',
                    f'<text x="{x}" y="{y + 139}" class="orange" font-size="13">full-sequence collision-safe {collision_safe}/{sample_count}</text>',
                ]
            )
    parts.extend(
        [
            '<text x="38" y="628" class="orange" font-size="13">Forces and motion are simulated diagnostics. Physical lift success is separate from fresh visual confirmation (not measured).</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def _sample_trace_at_time(samples: list[dict[str, Any]], time_s: float) -> dict[str, float]:
    selected = [sample for sample in samples if abs(float(sample["time_s"]) - time_s) <= 0.011]
    contacted = [sample for sample in selected if int(sample["contact_points"]) > 0]
    drifts = [
        float(sample["downward_payload_drift_in_hand_m"])
        for sample in selected
        if sample["downward_payload_drift_in_hand_m"] is not None
    ]
    ratios = [float(sample["tangential_to_normal_force_ratio"]) for sample in contacted]
    errors = [float(sample["slide_tracking_error_m"]) for sample in selected]
    return {
        "downward_drift_m": max(drifts, default=0.0),
        "contact_fraction": len(contacted) / len(selected) if selected else 0.0,
        "tangential_ratio": statistics.mean(ratios) if ratios else 0.0,
        "slide_error_m": statistics.mean(errors) if errors else 0.0,
    }


def _chart_panel(
    canvas: np.ndarray,
    x: int,
    y: int,
    width: int,
    title: str,
    series: list[tuple[str, list[float], float, tuple[int, int, int]]],
    current_index: int,
) -> None:
    height = 215
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (23, 33, 48), -1)
    cv2.putText(
        canvas,
        title,
        (x + 16, y + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.59,
        (242, 246, 250),
        1,
        cv2.LINE_AA,
    )
    chart_left, chart_top, chart_w, chart_h = x + 54, y + 42, width - 78, 142
    cv2.rectangle(
        canvas,
        (chart_left, chart_top),
        (chart_left + chart_w, chart_top + chart_h),
        (32, 46, 64),
        -1,
    )
    for frac in (0.0, 0.5, 1.0):
        yy = chart_top + int(frac * chart_h)
        cv2.line(canvas, (chart_left, yy), (chart_left + chart_w, yy), (52, 68, 87), 1)
    for series_index, (name, values, limit, color) in enumerate(series):
        points = []
        for index, value in enumerate(values[: current_index + 1]):
            px = chart_left + int(index * chart_w / max(1, len(values) - 1))
            py = chart_top + chart_h - int(min(max(value / limit, 0.0), 1.0) * chart_h)
            points.append((px, py))
        if len(points) >= 2:
            cv2.polylines(
                canvas, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA
            )
        cv2.putText(
            canvas,
            name,
            (chart_left, y + height - 12 - 20 * series_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )


def _write_diagnostic_video(
    low_video: Path,
    high_video: Path,
    low_samples: list[dict[str, Any]],
    high_samples: list[dict[str, Any]],
    output_video: Path,
    output_gif: Path,
    *,
    pair_id: str,
    seed: int,
) -> None:
    sources = [cv2.VideoCapture(str(path)) for path in (low_video, high_video)]
    if not all(source.isOpened() for source in sources):
        for source in sources:
            source.release()
        raise RuntimeError("could not open both friction diagnostic replay videos")
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT)
    )
    if not writer.isOpened():
        for source in sources:
            source.release()
        raise RuntimeError("could not open the retention diagnostic video writer")
    previews: list[Image.Image] = []
    histories = {"low": [], "high": []}
    max_frames = min(int(source.get(cv2.CAP_PROP_FRAME_COUNT)) for source in sources)
    for frame_index in range(max_frames):
        frames = [source.read() for source in sources]
        if not all(ok for ok, _ in frames):
            break
        time_s = frame_index / VIDEO_FPS
        paired_stats = []
        for name, samples in (("low", low_samples), ("high", high_samples)):
            stats = _sample_trace_at_time(samples, time_s)
            histories[name].append(stats)
            paired_stats.append(stats)
        canvas = np.full((VIDEO_HEIGHT, VIDEO_WIDTH, 3), (12, 18, 28), dtype=np.uint8)
        cv2.putText(
            canvas,
            f"HELD-OUT POSE {pair_id} | SEED {seed} | SAME CAMERA-ESTIMATE TRAJECTORY",
            (22, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.60,
            (240, 245, 250),
            1,
            cv2.LINE_AA,
        )
        for index, (label, color) in enumerate(
            (
                ("LOW FRICTION | mu=0.5", (100, 190, 255)),
                ("HIGH FRICTION | mu=2.2", (100, 226, 181)),
            )
        ):
            cv2.putText(
                canvas,
                label,
                (22 + index * 642, 61),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.57,
                color,
                1,
                cv2.LINE_AA,
            )
            canvas[72:414, index * 640 : (index + 1) * 640] = cv2.resize(
                frames[index][1], (640, PANEL_HEIGHT), interpolation=cv2.INTER_AREA
            )
        _chart_panel(
            canvas,
            20,
            DASHBOARD_TOP,
            395,
            "PAYLOAD DOWNWARD DRIFT IN HAND FRAME | mm",
            [
                (
                    "low",
                    [s["downward_drift_m"] * 1000 for s in histories["low"]],
                    240,
                    (100, 190, 255),
                )
            ],
            frame_index,
        )
        _chart_panel(
            canvas,
            442,
            DASHBOARD_TOP,
            395,
            "MEAN CONTACTING PAD Ft/Fn | ratio",
            [("low", [s["tangential_ratio"] for s in histories["low"]], 2.2, (100, 190, 255))],
            frame_index,
        )
        _chart_panel(
            canvas,
            864,
            DASHBOARD_TOP,
            395,
            "FINGER SLIDE POSITION ERROR | mm",
            [("low", [s["slide_error_m"] * 1000 for s in histories["low"]], 25, (100, 190, 255))],
            frame_index,
        )
        # High-friction traces are shown as matching dashed/solid overlays in a second row of color.
        for chart_x, key, limit in (
            (20, "downward_drift_m", 0.24),
            (442, "tangential_ratio", 2.2),
            (864, "slide_error_m", 0.025),
        ):
            chart_left, chart_top, chart_w, chart_h = chart_x + 54, DASHBOARD_TOP + 42, 317, 142
            vals = [
                s[key] * (1000 if key in {"downward_drift_m", "slide_error_m"} else 1)
                for s in histories["high"]
            ]
            scaled_limit = limit * (1000 if key in {"downward_drift_m", "slide_error_m"} else 1)
            points = [
                (
                    chart_left + int(i * chart_w / max(1, len(vals) - 1)),
                    chart_top + chart_h - int(min(max(v / scaled_limit, 0), 1) * chart_h),
                )
                for i, v in enumerate(vals)
            ]
            if len(points) >= 2:
                cv2.polylines(
                    canvas,
                    [np.asarray(points, dtype=np.int32)],
                    False,
                    (100, 226, 181),
                    2,
                    cv2.LINE_AA,
                )
        low, high = paired_stats
        cv2.putText(
            canvas,
            f"t={time_s:4.1f}s | hold pad-contact fraction {low['contact_fraction']:.2f} / {high['contact_fraction']:.2f} | green lines = high friction",
            (26, 706),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            (192, 204, 217),
            1,
            cv2.LINE_AA,
        )
        writer.write(canvas)
        if frame_index % 6 == 0:
            previews.append(
                Image.fromarray(
                    cv2.cvtColor(
                        cv2.resize(canvas, (768, 414), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2RGB,
                    )
                )
            )
    for source in sources:
        source.release()
    writer.release()
    if not previews:
        raise RuntimeError("retention diagnostic replays contained no paired video frames")
    previews[0].save(
        output_gif, save_all=True, append_images=previews[1:], duration=200, loop=0, optimize=True
    )


def _render_demo(position: tuple[int, int, tuple[float, float]], media_dir: Path) -> dict[str, Any]:
    seed, sample_index, target_xy = position
    pair_id = f"{seed}-{sample_index:03d}"
    media_dir.mkdir(parents=True, exist_ok=True)
    output_video, output_gif = media_dir / VIDEO_FILE, media_dir / GIF_FILE
    with tempfile.TemporaryDirectory(prefix="cooperative-pick-retention-") as scratch:
        low_path, high_path = Path(scratch) / "low.mp4", Path(scratch) / "high.mp4"
        telemetry_by_friction = {}
        outcomes = {}
        for friction, path in ((min(FRICTION_LEVELS), low_path), (max(FRICTION_LEVELS), high_path)):
            metrics = run_trial(
                target_xy,
                controller="camera_estimate",
                video_path=path,
                preview_path=path.with_suffix(".gif"),
                finger_pad_half_size_m=FINGER_PAD_HALF_SIZE_M,
                finger_pad_sliding_friction=friction,
                payload_body_mass_kg=PAYLOAD_MASS_KG,
                release_profile=RELEASE_PROFILE,
                lowest_approach_tool_height_offset_m=APPROACH_CLEARANCE_M,
                record_contact_retention_kinematics=True,
                capture_contact_retention_trace=True,
            )
            telemetry_by_friction[friction] = metrics["contact_retention_kinematics"]["samples"]
            outcomes[f"friction-{friction:g}"] = {
                "physical_stable_lift_success": metrics["physical_stable_lift_success"],
                "full_sequence_collision_safe": metrics["full_sequence_collision_safe"],
                "fresh_visual_confirmation": metrics["fresh_visual_confirmation"],
                "hold_minimum_pad_contact_points": metrics[
                    "minimum_finger_payload_contact_points_during_hold"
                ],
            }
        _write_diagnostic_video(
            low_path,
            high_path,
            telemetry_by_friction[min(FRICTION_LEVELS)],
            telemetry_by_friction[max(FRICTION_LEVELS)],
            output_video,
            output_gif,
            pair_id=pair_id,
            seed=seed,
        )
    metadata = {
        "purpose": "paired camera-estimate replay with force, finger tracking, and payload-to-hand motion diagnostics",
        "source_pair_id": pair_id,
        "replays_are_additional_trials": False,
        "conditions": outcomes,
        "video": output_video.name,
        "preview": output_gif.name,
        "fresh_visual_confirmation": "not measured",
    }
    (media_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def run_validation(
    *,
    output_dir: Path = OUTPUT_DIRECTORY,
    media_dir: Path = MEDIA_DIRECTORY,
    base_seed: int = BASE_SEED,
    samples_per_seed: int = SAMPLES_PER_SEED,
    render_demo: bool = True,
) -> dict[str, Any]:
    positions = _sample_positions(seeds=1, samples_per_seed=samples_per_seed, base_seed=base_seed)
    rows: list[dict[str, Any]] = []
    pad_rows: list[dict[str, Any]] = []
    for friction in FRICTION_LEVELS:
        for seed, sample_index, target_xy in positions:
            pair_id = f"{seed}-{sample_index:03d}"
            for controller in CONTROLLERS:
                metrics = run_trial(
                    target_xy,
                    controller=controller,
                    finger_pad_half_size_m=FINGER_PAD_HALF_SIZE_M,
                    finger_pad_sliding_friction=friction,
                    payload_body_mass_kg=PAYLOAD_MASS_KG,
                    release_profile=RELEASE_PROFILE,
                    lowest_approach_tool_height_offset_m=APPROACH_CLEARANCE_M,
                    record_contact_retention_kinematics=True,
                )
                row, per_pad = _row_from_metrics(
                    pair_id=pair_id,
                    seed=seed,
                    sample_index=sample_index,
                    target_xy=target_xy,
                    controller=controller,
                    friction=friction,
                    metrics=metrics,
                )
                rows.append(row)
                pad_rows.extend(per_pad)
    conditions = {
        f"friction-{friction:g}": {
            controller: _condition_summary(
                [
                    row
                    for row in rows
                    if row["friction"] == friction and row["controller"] == controller
                ]
            )
            for controller in CONTROLLERS
        }
        for friction in FRICTION_LEVELS
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / TRIALS_FILE,
        rows,
        (
            "pair_id",
            "seed",
            "sample_index",
            "target_x_m",
            "target_y_m",
            "controller",
            "friction",
            "physical_stable_lift_success",
            "payload_stable_lift_success",
            "full_sequence_collision_safe",
            "hold_minimum_pad_contact_points",
            "hold_contact_sample_fraction_mean_across_pads",
            "hold_maximum_downward_payload_drift_in_hand_frame_m",
            "hold_maximum_horizontal_payload_drift_in_hand_frame_m",
            "mean_tangential_to_normal_force_ratio_when_contacting",
            "maximum_slide_tracking_error_lift_or_hold_m",
            "mean_slide_tracking_error_lift_or_hold_m",
            "maximum_slide_tracking_error_during_hold_m",
            "mean_slide_tracking_error_during_hold_m",
            "fresh_visual_confirmation",
        ),
    )
    _write_csv(
        output_dir / PAD_PHASE_FILE,
        pad_rows,
        (
            "pair_id",
            "controller",
            "friction",
            "finger_pad_geom",
            "phase",
            "samples",
            "contact_samples",
            "contact_sample_fraction",
            "mean_normal_force_n",
            "maximum_normal_force_n",
            "mean_tangential_force_n",
            "maximum_tangential_force_n",
            "mean_tangential_to_normal_force_ratio_when_contacting",
            "maximum_tangential_to_normal_force_ratio_when_contacting",
            "mean_slide_tracking_error_m",
            "maximum_slide_tracking_error_m",
        ),
    )
    demo_metadata = _render_demo(positions[0], media_dir) if render_demo else None
    summary = {
        "experiment": "held-out diagnosis of friction-related contact loss during a three-arm cooperative lift",
        "protocol_version": 1,
        "evidence_type": "paired MuJoCo simulation; not hardware evidence",
        "target_sampling": {
            "distribution": "independent uniform XY samples in a square",
            "x_range_m": [-0.09, 0.09],
            "y_range_m": [-0.09, 0.09],
            "base_seed": base_seed,
            "samples_per_seed": samples_per_seed,
            "unique_target_poses": len(positions),
        },
        "paired_design": "same held-out poses at friction 0.5 and 2.2, paired across camera-estimate and oracle-XY controls; nominal payload, 16 mm pads, 20 mm approach clearance, and clearance release",
        "controllers": list(CONTROLLERS),
        "frozen_settings": {
            "finger_pad_half_size_m": FINGER_PAD_HALF_SIZE_M,
            "payload_body_mass_kg": PAYLOAD_MASS_KG,
            "lowest_approach_tool_height_offset_m": APPROACH_CLEARANCE_M,
            "release_profile": RELEASE_PROFILE,
            "friction_levels": list(FRICTION_LEVELS),
        },
        "total_physics_trials": len(rows),
        "total_pad_phase_rows": len(pad_rows),
        "sampling": {
            "physics_step_stride": 10,
            "sample_period_s": 0.02,
            "contact_forces": "contact-frame normal magnitude and tangential norm from mj_contactForce, summed across contacts for each named pad",
            "relative_motion": "payload center in each hand frame relative to its first sample during cooperative lift",
            "slide_tracking": "per-pad position-actuator command minus measured slide-joint position",
        },
        "conditions": conditions,
        "paired_friction_deltas": _paired_deltas(rows),
        "interpretation": "Compare tangential-to-normal force, payload displacement relative to each hand, pad contact persistence, and finger slide tracking. These correlated simulation diagnostics can distinguish patterns consistent with slip from finger opening; they do not establish a universal physical root cause.",
        "fresh_visual_confirmation": "not measured; physical stable-lift outcomes and contact traces do not imply visual confirmation",
        "limitations": [
            "The experiment is simulation-only and uses an idealized fixed cylinder, rigid pads, a scripted trajectory, and author-selected friction values.",
            "Contact forces and finger positions are MuJoCo solver states, not calibrated physical sensor readings.",
            "Twelve poses per condition provide a focused mechanism diagnostic, not a field-reliability estimate.",
            "The oracle-XY condition is a diagnostic control and shares all contact and motion behavior with the camera condition.",
        ],
        "artifacts": {
            "trials_csv": TRIALS_FILE,
            "pad_phase_csv": PAD_PHASE_FILE,
            "scorecard_svg": SCORECARD_FILE,
            "demo_video": VIDEO_FILE if render_demo else None,
            "demo_gif": GIF_FILE if render_demo else None,
            "demo_metadata": METADATA_FILE if render_demo else None,
        },
        "demo": demo_metadata,
    }
    _write_scorecard(output_dir / SCORECARD_FILE, conditions, len(positions))
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--media-dir", type=Path, default=MEDIA_DIRECTORY)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--samples-per-seed", type=int, default=SAMPLES_PER_SEED)
    parser.add_argument("--no-demo", action="store_true", help="skip the replay video and GIF")
    args = parser.parse_args()
    summary = run_validation(
        output_dir=args.output_dir,
        media_dir=args.media_dir,
        base_seed=args.base_seed,
        samples_per_seed=args.samples_per_seed,
        render_demo=not args.no_demo,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
