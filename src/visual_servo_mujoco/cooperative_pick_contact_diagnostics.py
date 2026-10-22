"""Record individual finger contact and slide tracking on a held-out friction split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .camera_guided_cooperative_pick import (
    FINGER_PAD_HALF_SIZE_M,
    _failure_reason,
    _sample_positions,
    run_trial,
)
from .cooperative_pick_demo import DEFAULT_PAYLOAD_BODY_MASS_KG
from .cooperative_pick_visual_validation import _wilson_interval


BASE_SEED = 41000
SAMPLES_PER_SEED = 12
FRICTION_LEVELS = (0.5, 2.2)
CONTROLLERS = ("camera_estimate", "oracle_xy")
PAYLOAD_MASS_KG = DEFAULT_PAYLOAD_BODY_MASS_KG
APPROACH_CLEARANCE_M = 0.020
RELEASE_PROFILE = "clearance"
CONTACT_CLOSING_PHASES = ("LOWER TO GRASP", "CLOSE SIX FINGERS", "SECURE CONTACT")
COOPERATIVE_LIFT_PHASE = "COOPERATIVE LIFT"
HOLD_PHASE = "HOLD UNDER LOAD"
OUTPUT_DIRECTORY = Path("results/cooperative-pick-contact-diagnostics")
TRIALS_FILE = "cooperative-pick-contact-trials.csv"
FINGERS_FILE = "cooperative-pick-contact-by-finger.csv"
SUMMARY_FILE = "cooperative-pick-contact-summary.json"
SCORECARD_FILE = "cooperative-pick-contact-scorecard.svg"
DEMO_DIRECTORY = Path("assets/demos")
DEMO_LOW_VIDEO = "cooperative_pick_contact_low_friction.mp4"
DEMO_HIGH_VIDEO = "cooperative_pick_contact_high_friction.mp4"
DEMO_PAIR_VIDEO = "cooperative_pick_contact_friction_pair.mp4"
DEMO_PAIR_GIF = "cooperative_pick_contact_friction_pair.gif"
DEMO_METADATA_FILE = "cooperative_pick_contact_demos.json"


def _count_contacting_pads(
    trace: dict[str, dict[str, Any]], phases: tuple[str, ...]
) -> int:
    return sum(
        any(int(pad["contact_steps_by_phase"].get(phase, 0)) > 0 for phase in phases)
        for pad in trace.values()
    )


def _trial_record(metrics: dict[str, Any]) -> dict[str, Any]:
    trace = metrics["finger_pad_payload_contact_trace"]
    tracking = metrics["finger_slide_tracking_during_close"]
    close_errors = [
        float(record["maximum_absolute_tracking_error_m"])
        for record in tracking.values()
    ]
    first_times = [
        float(pad["first_contact_time_s"])
        for pad in trace.values()
        if pad["first_contact_time_s"] is not None
    ]
    return {
        "pads_with_contact_during_approach_close_or_secure": _count_contacting_pads(trace, CONTACT_CLOSING_PHASES),
        "pads_with_contact_during_cooperative_lift": _count_contacting_pads(
            trace, (COOPERATIVE_LIFT_PHASE,)
        ),
        "pads_with_contact_during_hold_under_load": _count_contacting_pads(
            trace, (HOLD_PHASE,)
        ),
        "first_pad_contact_time_s": min(first_times) if first_times else None,
        "maximum_slide_tracking_error_during_close_m": max(close_errors, default=0.0),
        "median_slide_tracking_error_during_close_m": median(close_errors) if close_errors else 0.0,
        "peak_simulated_pad_normal_force_during_close_n": max(
            (
                float(pad["maximum_normal_force_by_phase_n"].get(phase, 0.0))
                for pad in trace.values()
                for phase in ("CLOSE SIX FINGERS", "SECURE CONTACT")
            ),
            default=0.0,
        ),
    }


def _contact_retention_outcome(row: dict[str, Any]) -> str:
    if row["physical_stable_lift_success"]:
        return "stable_lift"
    rollup = row["finger_contact_rollup"]
    closure_contacts = rollup["pads_with_contact_during_approach_close_or_secure"]
    hold_contacts = rollup["pads_with_contact_during_hold_under_load"]
    if closure_contacts and not hold_contacts:
        return "contact_not_retained_to_hold"
    if hold_contacts:
        return "contact_reached_hold_but_stable_lift_failed"
    return "no_pad_contact_during_closing_or_hold"


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty contact-diagnostic condition")
    trial_count = len(rows)
    contact_phase_counts: Counter[str] = Counter()
    force_peaks_by_phase: dict[str, list[float]] = {}
    tracking_maxima = []
    pad_trials_before_lift = 0
    pad_trials_during_lift = 0
    pad_trials_during_hold = 0
    for row in rows:
        for pad in row["finger_pad_payload_contact_trace"].values():
            phase_keys = set(pad["contact_steps_by_phase"])
            pad_trials_before_lift += int(
                any(phase in phase_keys for phase in CONTACT_CLOSING_PHASES)
            )
            pad_trials_during_lift += int(COOPERATIVE_LIFT_PHASE in phase_keys)
            pad_trials_during_hold += int(HOLD_PHASE in phase_keys)
            for phase, steps in pad["contact_steps_by_phase"].items():
                contact_phase_counts[phase] += int(steps > 0)
            for phase, force_n in pad["maximum_normal_force_by_phase_n"].items():
                force_peaks_by_phase.setdefault(phase, []).append(float(force_n))
        for record in row["finger_slide_tracking_during_close"].values():
            tracking_maxima.append(
                float(record["maximum_absolute_tracking_error_m"])
            )
    successes = sum(bool(row["physical_stable_lift_success"]) for row in rows)
    return {
        "trials": trial_count,
        "physical_stable_lift_success": {
            "successes": successes,
            "rate": round(successes / trial_count, 4),
            "wilson_95_interval": _wilson_interval(successes, trial_count),
        },
        "payload_stable_lift_successes": sum(
            bool(row["payload_stable_lift_success"]) for row in rows
        ),
        "full_sequence_collision_safe_trials": sum(
            bool(row["full_sequence_collision_safe"]) for row in rows
        ),
        "trials_with_any_pad_contact_during_approach_close_or_secure": sum(
            _trial_record(row)["pads_with_contact_during_approach_close_or_secure"] > 0
            for row in rows
        ),
        "pad_trials_with_contact_by_phase": dict(sorted(contact_phase_counts.items())),
        "pad_trials_with_contact_during_approach_close_or_secure": pad_trials_before_lift,
        "pad_trials_with_contact_during_cooperative_lift": pad_trials_during_lift,
        "pad_trials_with_contact_during_hold_under_load": pad_trials_during_hold,
        "median_peak_simulated_pad_normal_force_by_phase_n": {
            phase: round(median(values), 6)
            for phase, values in sorted(force_peaks_by_phase.items())
        },
        "median_max_slide_tracking_error_during_close_m": round(
            median(tracking_maxima), 8
        ),
        "failure_reasons": dict(
            sorted(Counter(row["failure_reason"] for row in rows).items())
        ),
        "contact_retention_outcomes": dict(
            sorted(Counter(_contact_retention_outcome(row) for row in rows).items())
        ),
        "fresh_visual_confirmation": "not measured",
        "force_interpretation": "MuJoCo contact-solver normal force; a simulation diagnostic, not a calibrated sensor value",
    }


def _write_trials(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "pair_id", "seed", "sample_index", "target_x_m", "target_y_m",
        "controller", "finger_pad_half_size_m", "friction", "payload_body_mass_kg",
        "approach_clearance_m", "release_profile",
        "physical_stable_lift_success", "payload_stable_lift_success",
        "full_sequence_collision_safe", "failure_reason", "contact_retention_outcome",
        "pads_with_contact_during_approach_close_or_secure",
        "pads_with_contact_during_cooperative_lift", "pads_with_contact_during_hold_under_load",
        "first_pad_contact_time_s",
        "maximum_slide_tracking_error_during_close_m",
        "median_slide_tracking_error_during_close_m",
        "peak_simulated_pad_normal_force_during_close_n", "fresh_visual_confirmation",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _write_fingers(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "pair_id", "controller", "friction", "finger_pad_geom",
        "first_contact_time_s", "first_contact_phase", "last_contact_time_s",
        "contact_steps", "maximum_contact_points", "maximum_normal_contact_force_n",
        "normal_contact_impulse_ns", "contact_steps_by_phase_json",
        "maximum_normal_force_by_phase_n_json", "normal_contact_impulse_by_phase_ns_json",
        "close_phase_samples", "commanded_slide_start_m", "commanded_slide_end_m",
        "measured_slide_start_m", "measured_slide_end_m",
        "maximum_absolute_tracking_error_m", "mean_absolute_tracking_error_m",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _write_scorecard(path: Path, conditions: dict[str, Any], sample_count: int) -> None:
    width, height = 1160, 650
    fragments = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1220"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}.title{fill:#f3f6fb}.muted{fill:#a9b6c8}.blue{fill:#73b7ff}.green{fill:#68dfb3}.orange{fill:#ffbd72}</style>',
        '<text x="36" y="46" class="title" font-size="27" font-weight="700">Where does contact disappear?</text>',
        f'<text x="36" y="76" class="muted" font-size="14">{sample_count} new poses · 16 mm pads · nominal 0.720 kg payload · closure, lift, and hold sampled at physics-step rate</text>',
        '<text x="36" y="100" class="muted" font-size="13">Bars count unique finger-pad/pose pairs with any contact in each phase; this separates closure, early lift, and loaded hold.</text>',
    ]
    x_positions = {0.5: 40, 2.2: 590}
    y_positions = {"camera_estimate": 155, "oracle_xy": 375}
    maximum_pad_trials = sample_count * 6
    for friction in FRICTION_LEVELS:
        x = x_positions[friction]
        fragments.append(f'<text x="{x}" y="132" class="blue" font-size="19" font-weight="700">Sliding friction μ = {friction:g}</text>')
        for controller in CONTROLLERS:
            y = y_positions[controller]
            summary = conditions[f"friction-{friction:g}"][controller]
            label = "Camera-estimate" if controller == "camera_estimate" else "Oracle XY diagnostic"
            fragments.append(f'<text x="{x}" y="{y}" class="title" font-size="15" font-weight="700">{label}</text>')
            strict = summary["physical_stable_lift_success"]
            for row_index, (phase_name, summary_key, color) in enumerate((
                (
                    "Approach / close / secure",
                    "pad_trials_with_contact_during_approach_close_or_secure",
                    "#73b7ff",
                ),
                (
                    "Cooperative lift",
                    "pad_trials_with_contact_during_cooperative_lift",
                    "#ffbd72",
                ),
                (
                    "Hold under load",
                    "pad_trials_with_contact_during_hold_under_load",
                    "#68dfb3",
                ),
            )):
                count = summary[summary_key]
                bar_y = y + 22 + row_index * 38
                bar_width = 330 * count / maximum_pad_trials if maximum_pad_trials else 0
                fragments.extend([
                    f'<text x="{x}" y="{bar_y + 16}" class="muted" font-size="12">{phase_name}</text>',
                    f'<rect x="{x + 170}" y="{bar_y}" width="330" height="26" rx="6" fill="#253343"/>',
                    f'<rect x="{x + 170}" y="{bar_y}" width="{bar_width:.1f}" height="26" rx="6" fill="{color}"/>',
                    f'<text x="{x + 510}" y="{bar_y + 18}" class="title" font-size="13" font-weight="700">{count}/{maximum_pad_trials}</text>',
                ])
            fragments.append(f'<text x="{x}" y="{y + 148}" class="orange" font-size="13">Stable lift {strict["successes"]}/{sample_count} · hold {summary["payload_stable_lift_successes"]} · cycle-safe {summary["full_sequence_collision_safe_trials"]}</text>')
    fragments.extend([
        '<text x="36" y="620" class="orange" font-size="13">Hold contact and full-sequence collision safety are separate from fresh visual confirmation, which was not measured.</text>',
        '</svg>',
    ])
    path.write_text("\n".join(fragments), encoding="utf-8")


def _write_paired_demo(
    low_video: Path,
    high_video: Path,
    output_video: Path,
    output_gif: Path,
    *,
    source_pair_id: str,
) -> None:
    low = cv2.VideoCapture(str(low_video))
    high = cv2.VideoCapture(str(high_video))
    if not low.isOpened() or not high.isOpened():
        low.release()
        high.release()
        raise RuntimeError("OpenCV could not open both contact diagnostic clips")
    fps = low.get(cv2.CAP_PROP_FPS) or 30.0
    size = (1280, 424)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        low.release()
        high.release()
        raise RuntimeError("OpenCV could not open the contact diagnostic comparison writer")
    previews: list[Image.Image] = []
    frame_index = 0
    while True:
        has_low, low_frame = low.read()
        has_high, high_frame = high.read()
        if not has_low or not has_high:
            break
        canvas = np.full((size[1], size[0], 3), (14, 21, 32), dtype=np.uint8)
        cv2.putText(canvas, f"SAME HELD-OUT TARGET {source_pair_id}  |  camera-estimate controller  |  16 mm pads", (22, 29), cv2.FONT_HERSHEY_DUPLEX, 0.66, (244, 247, 250), 1, cv2.LINE_AA)
        cv2.putText(canvas, "LOW FRICTION  |  mu 0.5", (22, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (100, 190, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "HIGH FRICTION  |  mu 2.2", (660, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (100, 226, 181), 1, cv2.LINE_AA)
        canvas[78:390, 0:640] = cv2.resize(low_frame, (640, 312), interpolation=cv2.INTER_AREA)
        canvas[78:390, 640:1280] = cv2.resize(high_frame, (640, 312), interpolation=cv2.INTER_AREA)
        cv2.putText(canvas, "Contact traces are simulator diagnostics; the overlays report grasp phase and pad contacts.", (22, 414), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (190, 202, 215), 1, cv2.LINE_AA)
        writer.write(canvas)
        if frame_index % 6 == 0:
            preview = cv2.resize(canvas, (960, 318), interpolation=cv2.INTER_AREA)
            previews.append(Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)))
        frame_index += 1
    low.release()
    high.release()
    writer.release()
    if not previews:
        raise RuntimeError("contact diagnostic clips contained no paired frames")
    previews[0].save(output_gif, save_all=True, append_images=previews[1:], duration=200, loop=0, optimize=True)


def _render_demos(
    positions: list[tuple[int, int, tuple[float, float]]],
    rows: list[dict[str, Any]],
    *,
    media_dir: Path,
) -> dict[str, Any]:
    camera_rows = [
        row for row in rows
        if row["controller"] == "camera_estimate" and row["friction"] == min(FRICTION_LEVELS)
    ]
    selected_row = next(
        (
            row for row in camera_rows
            if row["finger_contact_rollup"]["pads_with_contact_during_hold_under_load"] == 0
        ),
        camera_rows[0],
    )
    source_pair_id = selected_row["pair_id"]
    selected_position = next(
        item for item in positions if f"{item[0]}-{item[1]:03d}" == source_pair_id
    )
    media_dir.mkdir(parents=True, exist_ok=True)
    low_path = media_dir / DEMO_LOW_VIDEO
    high_path = media_dir / DEMO_HIGH_VIDEO
    replay_records = []
    for friction, path in ((0.5, low_path), (2.2, high_path)):
        metrics = run_trial(
            selected_position[2],
            controller="camera_estimate",
            video_path=path,
            finger_pad_half_size_m=FINGER_PAD_HALF_SIZE_M,
            finger_pad_sliding_friction=friction,
            payload_body_mass_kg=PAYLOAD_MASS_KG,
            release_profile=RELEASE_PROFILE,
            lowest_approach_tool_height_offset_m=APPROACH_CLEARANCE_M,
        )
        replay_records.append({
            "video": path.name,
            "source_pair_id": source_pair_id,
            "controller": "camera_estimate",
            "finger_pad_half_size_m": FINGER_PAD_HALF_SIZE_M,
            "finger_pad_sliding_friction": friction,
            "physical_stable_lift_success": metrics["physical_stable_lift_success"],
            "fresh_visual_confirmation": metrics["fresh_visual_confirmation"],
            "counted_as_additional_benchmark_trial": False,
        })
    pair_path = media_dir / DEMO_PAIR_VIDEO
    gif_path = media_dir / DEMO_PAIR_GIF
    _write_paired_demo(low_path, high_path, pair_path, gif_path, source_pair_id=source_pair_id)
    metadata = {
        "purpose": "same-pose camera-estimate replay comparing low and high friction with baseline 16 mm pads",
        "source_pair_id": source_pair_id,
        "replays_are_additional_trials": False,
        "fresh_visual_confirmation": "not measured",
        "paired_video": pair_path.name,
        "preview": gif_path.name,
        "replays": replay_records,
    }
    (media_dir / DEMO_METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def run_validation(
    *,
    output_dir: Path = OUTPUT_DIRECTORY,
    media_dir: Path = DEMO_DIRECTORY,
    base_seed: int = BASE_SEED,
    samples_per_seed: int = SAMPLES_PER_SEED,
    render_demos: bool = True,
) -> dict[str, Any]:
    if samples_per_seed <= 0:
        raise ValueError("samples_per_seed must be positive")
    positions = _sample_positions(seeds=1, samples_per_seed=samples_per_seed, base_seed=base_seed)
    rows: list[dict[str, Any]] = []
    finger_rows: list[dict[str, Any]] = []
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
                )
                row = {
                    "pair_id": pair_id,
                    "seed": seed,
                    "sample_index": sample_index,
                    "target_x_m": target_xy[0],
                    "target_y_m": target_xy[1],
                    "controller": controller,
                    "friction": friction,
                    "finger_pad_half_size_m": FINGER_PAD_HALF_SIZE_M,
                    "payload_body_mass_kg": metrics["payload_body_mass_kg"],
                    "approach_clearance_m": APPROACH_CLEARANCE_M,
                    "release_profile": RELEASE_PROFILE,
                    "physical_stable_lift_success": metrics["physical_stable_lift_success"],
                    "payload_stable_lift_success": metrics["payload_stable_lift_success"],
                    "full_sequence_collision_safe": metrics["full_sequence_collision_safe"],
                    "failure_reason": _failure_reason(metrics),
                    "finger_pad_payload_contact_trace": metrics["finger_pad_payload_contact_trace"],
                    "finger_slide_tracking_during_close": metrics["finger_slide_tracking_during_close"],
                    "fresh_visual_confirmation": metrics["fresh_visual_confirmation"],
                }
                row["finger_contact_rollup"] = _trial_record(metrics)
                row.update(row["finger_contact_rollup"])
                row["contact_retention_outcome"] = _contact_retention_outcome(row)
                rows.append(row)
                for pad_name, contact in metrics["finger_pad_payload_contact_trace"].items():
                    tracking = metrics["finger_slide_tracking_during_close"][pad_name]
                    finger_rows.append({
                        "pair_id": pair_id,
                        "controller": controller,
                        "friction": friction,
                        "finger_pad_geom": pad_name,
                        "first_contact_time_s": contact["first_contact_time_s"],
                        "first_contact_phase": contact["first_contact_phase"],
                        "last_contact_time_s": contact["last_contact_time_s"],
                        "contact_steps": contact["contact_steps"],
                        "maximum_contact_points": contact["maximum_contact_points"],
                        "maximum_normal_contact_force_n": contact["maximum_normal_contact_force_n"],
                        "normal_contact_impulse_ns": contact["normal_contact_impulse_ns"],
                        "contact_steps_by_phase_json": json.dumps(contact["contact_steps_by_phase"], sort_keys=True),
                        "maximum_normal_force_by_phase_n_json": json.dumps(contact["maximum_normal_force_by_phase_n"], sort_keys=True),
                        "normal_contact_impulse_by_phase_ns_json": json.dumps(contact["normal_contact_impulse_by_phase_ns"], sort_keys=True),
                        **tracking,
                    })

    conditions = {
        f"friction-{friction:g}": {
            controller: _aggregate([
                row for row in rows
                if row["friction"] == friction and row["controller"] == controller
            ])
            for controller in CONTROLLERS
        }
        for friction in FRICTION_LEVELS
    }
    paired_friction = {}
    for controller in CONTROLLERS:
        paired = {}
        for pair_id in (row["pair_id"] for row in rows if row["controller"] == controller and row["friction"] == FRICTION_LEVELS[0]):
            low = next(row for row in rows if row["pair_id"] == pair_id and row["controller"] == controller and row["friction"] == FRICTION_LEVELS[0])
            high = next(row for row in rows if row["pair_id"] == pair_id and row["controller"] == controller and row["friction"] == FRICTION_LEVELS[1])
            paired[pair_id] = {
                "low_friction_pads_contact_during_close": low["finger_contact_rollup"]["pads_with_contact_during_approach_close_or_secure"],
                "high_friction_pads_contact_during_close": high["finger_contact_rollup"]["pads_with_contact_during_approach_close_or_secure"],
                "low_friction_pads_contact_during_cooperative_lift": low["finger_contact_rollup"]["pads_with_contact_during_cooperative_lift"],
                "high_friction_pads_contact_during_cooperative_lift": high["finger_contact_rollup"]["pads_with_contact_during_cooperative_lift"],
                "low_friction_pads_contact_during_hold_under_load": low["finger_contact_rollup"]["pads_with_contact_during_hold_under_load"],
                "high_friction_pads_contact_during_hold_under_load": high["finger_contact_rollup"]["pads_with_contact_during_hold_under_load"],
                "low_friction_physical_stable_lift_success": low["physical_stable_lift_success"],
                "high_friction_physical_stable_lift_success": high["physical_stable_lift_success"],
                "low_friction_contact_retention_outcome": low["contact_retention_outcome"],
                "high_friction_contact_retention_outcome": high["contact_retention_outcome"],
            }
        paired_friction[controller] = paired

    output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = output_dir / TRIALS_FILE
    fingers_path = output_dir / FINGERS_FILE
    _write_trials(trials_path, rows)
    _write_fingers(fingers_path, finger_rows)
    _write_scorecard(output_dir / SCORECARD_FILE, conditions, len(positions))
    demos = _render_demos(positions, rows, media_dir=media_dir) if render_demos else None
    summary = {
        "experiment": "per-finger contact onset and gripper slide tracking under friction stress",
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
        "paired_design": "same pose, baseline 16 mm pad, nominal mass, clearance approach and release; compare low/high friction under camera-estimate and oracle controllers",
        "controllers": list(CONTROLLERS),
        "frozen_settings": {
            "finger_pad_half_size_m": FINGER_PAD_HALF_SIZE_M,
            "payload_body_mass_kg": PAYLOAD_MASS_KG,
            "lowest_approach_tool_height_offset_m": APPROACH_CLEARANCE_M,
            "release_profile": RELEASE_PROFILE,
            "friction_levels": list(FRICTION_LEVELS),
        },
        "physics_sampling": {
            "timestep_s": 0.002,
            "pad_contact": "payload-pad contacts sampled after every MuJoCo physics step, attributed to each of six named pad geoms and desired-pick phase",
            "normal_force": "mujoco.mj_contactForce contact-frame normal component magnitude, summed across pad/payload contact points per step",
            "slide_tracking": "position-actuator control target compared with measured finger slide joint position on every physics step during CLOSE SIX FINGERS",
        },
        "total_physics_trials": len(rows),
        "total_finger_trial_rows": len(finger_rows),
        "conditions": conditions,
        "paired_friction_outcomes": paired_friction,
        "failure_reason_interpretation": (
            "The legacy no_grasp_contact label means the score had zero pad contacts during the hold window; the per-phase traces may still show transient contact during closing or early lift."
        ),
        "fresh_visual_confirmation": "not measured; physical stable-lift outcomes and contact traces do not imply visual confirmation",
        "limitations": [
            "Contact force values come from MuJoCo's solver and are not calibrated force-sensor measurements.",
            "The study repeats the fixed cylinder, gripper, actuators, and ideal camera; the friction levels are author-selected stresses.",
            "Twelve poses per friction/controller condition provide diagnostic evidence, not reliability estimates.",
            "Oracle XY shares the same arm motion and only removes target-position estimation error.",
            "Fresh visual confirmation and real-hardware behavior are not measured.",
        ],
        "artifacts": {
            "trials_csv": TRIALS_FILE,
            "finger_detail_csv": FINGERS_FILE,
            "scorecard_svg": SCORECARD_FILE,
            "demo_metadata": DEMO_METADATA_FILE if demos is not None else None,
            "demo_video": DEMO_PAIR_VIDEO if demos is not None else None,
            "demo_preview": DEMO_PAIR_GIF if demos is not None else None,
        },
        "demos": demos,
    }
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--media-dir", type=Path, default=DEMO_DIRECTORY)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--samples-per-seed", type=int, default=SAMPLES_PER_SEED)
    parser.add_argument("--skip-demos", action="store_true")
    args = parser.parse_args()
    summary = run_validation(
        output_dir=args.output_dir,
        media_dir=args.media_dir,
        base_seed=args.base_seed,
        samples_per_seed=args.samples_per_seed,
        render_demos=not args.skip_demos,
    )
    print(json.dumps({"trials": summary["total_physics_trials"], "finger_rows": summary["total_finger_trial_rows"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
