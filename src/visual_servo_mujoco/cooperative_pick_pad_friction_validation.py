"""Held-out paired test of finger-pad size under nominal and low friction."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
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


BASE_SEED = 40900
SAMPLES_PER_SEED = 12
PAD_HALF_SIZES_M = (FINGER_PAD_HALF_SIZE_M, 0.018)
SLIDING_FRICTION_LEVELS = (0.5, 2.2)
CONTROLLERS = ("camera_estimate", "oracle_xy")
PAYLOAD_MASS_KG = DEFAULT_PAYLOAD_BODY_MASS_KG
APPROACH_CLEARANCE_M = 0.020
RELEASE_PROFILE = "clearance"
OUTPUT_DIRECTORY = Path("results/cooperative-pick-pad-friction-validation")
TRIALS_FILE = "cooperative-pick-pad-friction-trials.csv"
SUMMARY_FILE = "cooperative-pick-pad-friction-summary.json"
SCORECARD_FILE = "cooperative-pick-pad-friction-scorecard.svg"
DEMO_DIRECTORY = Path("assets/demos")
DEMO_VIDEO_FILE = "cooperative_pick_pad_friction_matched.mp4"
DEMO_GIF_FILE = "cooperative_pick_pad_friction_matched.gif"
DEMO_METADATA_FILE = "cooperative_pick_pad_friction_demo.json"


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty pad/friction condition")
    trials = len(rows)
    measures = {}
    for name, field in (
        ("physical_stable_lift_success", "physical_stable_lift_success"),
        ("payload_stable_lift_success", "payload_stable_lift_success"),
        ("full_sequence_collision_safe", "full_sequence_collision_safe"),
    ):
        successes = sum(bool(row[field]) for row in rows)
        measures[name] = {
            "successes": successes,
            "rate": round(successes / trials, 4),
            "wilson_95_interval": _wilson_interval(successes, trials),
        }
    measures["trials"] = trials
    measures["failure_reasons"] = dict(
        sorted(Counter(row["failure_reason"] for row in rows).items())
    )
    measures["fresh_visual_confirmation"] = "not measured"
    return measures


def _paired_outcomes(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> dict[str, int]:
    if set(baseline) != set(candidate):
        raise ValueError("pad-size conditions must contain the same paired trial ids")
    outcomes = {
        "both_success": 0,
        "candidate_only_success": 0,
        "baseline_only_success": 0,
        "both_failure": 0,
    }
    for pair_id, baseline_row in baseline.items():
        baseline_success = bool(baseline_row["physical_stable_lift_success"])
        candidate_success = bool(candidate[pair_id]["physical_stable_lift_success"])
        if baseline_success and candidate_success:
            outcomes["both_success"] += 1
        elif candidate_success:
            outcomes["candidate_only_success"] += 1
        elif baseline_success:
            outcomes["baseline_only_success"] += 1
        else:
            outcomes["both_failure"] += 1
    return outcomes


def _write_scorecard(
    path: Path,
    conditions: dict[str, Any],
    samples: int,
    pad_sizes_m: tuple[float, ...],
    friction_levels: tuple[float, ...],
) -> None:
    width, height = 1160, 700
    fragments = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1220"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}.title{fill:#f3f6fb}.muted{fill:#a9b6c8}.blue{fill:#73b7ff}.green{fill:#68dfb3}.orange{fill:#ffbd72}.cell{fill:#12243a}</style>',
        '<text x="36" y="48" class="title" font-size="27" font-weight="700">Held-out pad-size test under friction stress</text>',
        f'<text x="36" y="78" class="muted" font-size="14">{samples} paired poses · nominal {PAYLOAD_MASS_KG:.3f} kg payload · 20 mm approach clearance · clearance release</text>',
        '<text x="36" y="102" class="muted" font-size="13">Only pad half-size varies within each friction/controller pair; all trials use the same sampled target poses.</text>',
    ]
    x_positions = {friction: 40 + index * 550 for index, friction in enumerate(friction_levels)}
    y_positions = {"camera_estimate": 170, "oracle_xy": 418}
    for friction in friction_levels:
        x = x_positions[friction]
        fragments.append(f'<text x="{x}" y="139" class="blue" font-size="19" font-weight="700">Sliding friction μ = {friction:g}</text>')
        for controller in CONTROLLERS:
            y = y_positions[controller]
            label = "Camera estimate" if controller == "camera_estimate" else "Oracle XY diagnostic"
            fragments.append(f'<text x="{x}" y="{y}" class="title" font-size="15" font-weight="700">{label}</text>')
            for index, pad_size in enumerate(pad_sizes_m):
                cell = conditions[f"friction-{friction:g}"]["controllers"][controller][
                    f"pad_half_size_m_{pad_size:g}"
                ]
                strict = cell["physical_stable_lift_success"]
                hold = cell["payload_stable_lift_success"]
                safe = cell["full_sequence_collision_safe"]
                bar_y = y + 25 + index * 74
                fill = "#49c9ab" if index else "#8496a8"
                bar_width = 360 * strict["successes"] / samples
                fragments.extend([
                    f'<text x="{x}" y="{bar_y + 18}" class="muted" font-size="13">{pad_size * 1000:.0f} mm pad</text>',
                    f'<rect x="{x + 93}" y="{bar_y}" width="360" height="32" rx="7" fill="#253343"/>',
                    f'<rect x="{x + 93}" y="{bar_y}" width="{bar_width:.1f}" height="32" rx="7" fill="{fill}"/>',
                    f'<text x="{x + 465}" y="{bar_y + 21}" class="title" font-size="14" font-weight="700">{strict["successes"]}/{samples}</text>',
                    f'<text x="{x + 93}" y="{bar_y + 49}" class="muted" font-size="11">hold {hold["successes"]} · cycle-safe {safe["successes"]}</text>',
                ])
    fragments.extend([
        '<text x="36" y="654" class="orange" font-size="13">Physical stable-lift success requires both hold and full-cycle collision safety. Fresh visual confirmation was not measured.</text>',
        '</svg>',
    ])
    path.write_text("\n".join(fragments), encoding="utf-8")


def _write_matched_demo(
    clips: list[tuple[str, float, float, Path]],
    video_path: Path,
    gif_path: Path,
    *,
    source_pair_id: str,
) -> None:
    captures = [(label, cv2.VideoCapture(str(path))) for label, _, _, path in clips]
    if not all(capture.isOpened() for _, capture in captures):
        for _, capture in captures:
            capture.release()
        raise RuntimeError("OpenCV could not open every pad/friction demo clip")
    fps = captures[0][1].get(cv2.CAP_PROP_FPS) or 30.0
    output_size = (1280, 832)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size
    )
    if not writer.isOpened():
        for _, capture in captures:
            capture.release()
        raise RuntimeError("OpenCV could not open the matched pad/friction video writer")
    previews: list[Image.Image] = []
    frame_index = 0
    while True:
        frames = []
        for _, capture in captures:
            valid, frame = capture.read()
            if not valid:
                frames = []
                break
            frames.append(frame)
        if not frames:
            break
        canvas = np.full((output_size[1], output_size[0], 3), (14, 21, 32), dtype=np.uint8)
        cv2.putText(canvas, f"MATCHED HELD-OUT POSE {source_pair_id}  |  camera estimate", (24, 30), cv2.FONT_HERSHEY_DUPLEX, 0.68, (244, 247, 250), 1, cv2.LINE_AA)
        for index, ((_, friction, pad, _), frame) in enumerate(zip(clips, frames)):
            col, row = index % 2, index // 2
            x, y = col * 640, 44 + row * 386
            label = f"mu {friction:g}  |  pad half-size {pad * 1000:.0f} mm"
            cv2.putText(canvas, label, (x + 18, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (100, 226, 181) if pad > FINGER_PAD_HALF_SIZE_M else (190, 202, 215), 1, cv2.LINE_AA)
            small = cv2.resize(frame, (640, 354), interpolation=cv2.INTER_AREA)
            canvas[y + 30:y + 384, x:x + 640] = small
        writer.write(canvas)
        if frame_index % 6 == 0:
            small_canvas = cv2.resize(canvas, (960, 624), interpolation=cv2.INTER_AREA)
            previews.append(Image.fromarray(cv2.cvtColor(small_canvas, cv2.COLOR_BGR2RGB)))
        frame_index += 1
    for _, capture in captures:
        capture.release()
    writer.release()
    if not previews:
        raise RuntimeError("the matched pad/friction videos contained no frames")
    previews[0].save(gif_path, save_all=True, append_images=previews[1:], duration=200, loop=0, optimize=True)


def _render_demos(
    rows: list[dict[str, Any]],
    positions: list[tuple[int, int, tuple[float, float]]],
    *,
    media_dir: Path,
) -> dict[str, Any]:
    low_friction = min(SLIDING_FRICTION_LEVELS)
    camera_rows = [
        row for row in rows
        if row["controller"] == "camera_estimate" and row["friction"] == low_friction
    ]
    baseline = {
        row["pair_id"]: row for row in camera_rows
        if row["finger_pad_half_size_m"] == FINGER_PAD_HALF_SIZE_M
    }
    candidate = {
        row["pair_id"]: row for row in camera_rows
        if row["finger_pad_half_size_m"] != FINGER_PAD_HALF_SIZE_M
    }
    flips = [
        pair_id for pair_id in sorted(baseline)
        if baseline[pair_id]["physical_stable_lift_success"]
        != candidate[pair_id]["physical_stable_lift_success"]
    ]
    improvements = [
        pair_id for pair_id in flips
        if candidate[pair_id]["physical_stable_lift_success"]
    ]
    source_pair_id = improvements[0] if improvements else (flips[0] if flips else sorted(baseline)[0])
    selected = next(
        item for item in positions
        if f"{item[0]}-{item[1]:03d}" == source_pair_id
    )
    media_dir.mkdir(parents=True, exist_ok=True)
    clips: list[tuple[str, float, float, Path]] = []
    replay_metrics = []
    for friction in SLIDING_FRICTION_LEVELS:
        for pad_size in PAD_HALF_SIZES_M:
            name = f"cooperative_pick_pad_friction_mu_{friction:g}_pad_{pad_size * 1000:.0f}mm.mp4"
            path = media_dir / name
            metrics = run_trial(
                selected[2],
                controller="camera_estimate",
                video_path=path,
                finger_pad_half_size_m=pad_size,
                finger_pad_sliding_friction=friction,
                payload_body_mass_kg=PAYLOAD_MASS_KG,
                release_profile=RELEASE_PROFILE,
                lowest_approach_tool_height_offset_m=APPROACH_CLEARANCE_M,
            )
            clips.append((name, friction, pad_size, path))
            replay_metrics.append({
                "video": name,
                "source_pair_id": source_pair_id,
                "controller": "camera_estimate",
                "finger_pad_half_size_m": metrics["finger_pad_half_size_m"],
                "finger_pad_sliding_friction": friction,
                "payload_body_mass_kg": metrics["payload_body_mass_kg"],
                "physical_stable_lift_success": metrics["physical_stable_lift_success"],
                "payload_stable_lift_success": metrics["payload_stable_lift_success"],
                "full_sequence_collision_safe": metrics["full_sequence_collision_safe"],
                "fresh_visual_confirmation": metrics["fresh_visual_confirmation"],
                "counted_as_additional_benchmark_trial": False,
            })
    video_path = media_dir / DEMO_VIDEO_FILE
    gif_path = media_dir / DEMO_GIF_FILE
    _write_matched_demo(clips, video_path, gif_path, source_pair_id=source_pair_id)
    metadata = {
        "purpose": "four-way replay at one held-out pose comparing 16 mm and 18 mm pad half-sizes under low and high friction",
        "source_pair_id": source_pair_id,
        "selection_rule": "prefer a camera-estimate low-friction pair where only the candidate succeeds; otherwise show the first discordant pair, else the first pair",
        "benchmark_trials": "replays are not additional trials",
        "fresh_visual_confirmation": "not measured",
        "composite_video": video_path.name,
        "composite_preview": gif_path.name,
        "replays": replay_metrics,
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
    pad_half_sizes_m = PAD_HALF_SIZES_M
    friction_levels = SLIDING_FRICTION_LEVELS
    if samples_per_seed <= 0:
        raise ValueError("samples_per_seed must be positive")
    positions = _sample_positions(seeds=1, samples_per_seed=samples_per_seed, base_seed=base_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for friction in friction_levels:
        for pad_size in pad_half_sizes_m:
            for seed, sample_index, target_xy in positions:
                pair_id = f"{seed}-{sample_index:03d}"
                for controller in CONTROLLERS:
                    metrics = run_trial(
                        target_xy,
                        controller=controller,
                        finger_pad_half_size_m=pad_size,
                        finger_pad_sliding_friction=friction,
                        payload_body_mass_kg=PAYLOAD_MASS_KG,
                        release_profile=RELEASE_PROFILE,
                        lowest_approach_tool_height_offset_m=APPROACH_CLEARANCE_M,
                    )
                    rows.append({
                        "pair_id": pair_id,
                        "seed": seed,
                        "sample_index": sample_index,
                        "controller": controller,
                        "finger_pad_half_size_m": pad_size,
                        "friction": friction,
                        "target_x_m": target_xy[0],
                        "target_y_m": target_xy[1],
                        "payload_body_mass_kg": metrics["payload_body_mass_kg"],
                        "approach_clearance_m": APPROACH_CLEARANCE_M,
                        "release_profile": RELEASE_PROFILE,
                        "physical_stable_lift_success": metrics["physical_stable_lift_success"],
                        "payload_stable_lift_success": metrics["payload_stable_lift_success"],
                        "full_sequence_collision_safe": metrics["full_sequence_collision_safe"],
                        "minimum_finger_payload_contact_points_during_hold": metrics["minimum_finger_payload_contact_points_during_hold"],
                        "minimum_payload_z_during_hold_m": metrics["minimum_payload_z_during_hold_m"],
                        "maximum_payload_tilt_during_hold_deg": metrics["maximum_payload_tilt_during_hold_deg"],
                        "physics_steps_with_arm_table_or_ground_contact": metrics["physics_steps_with_arm_table_or_ground_contact"],
                        "physics_steps_with_nonfinger_arm_payload_contact": metrics["physics_steps_with_nonfinger_arm_payload_contact"],
                        "failure_reason": _failure_reason(metrics),
                        "fresh_visual_confirmation": metrics["fresh_visual_confirmation"],
                    })

    conditions: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for friction in friction_levels:
        condition = f"friction-{friction:g}"
        conditions[condition] = {
            "payload_body_mass_kg": PAYLOAD_MASS_KG,
            "finger_pad_and_payload_sliding_friction": friction,
            "trials_per_pad_and_controller": len(positions),
            "controllers": {},
        }
        paired[condition] = {}
        for controller in CONTROLLERS:
            base_rows = [
                row for row in rows
                if row["friction"] == friction and row["controller"] == controller
                and row["finger_pad_half_size_m"] == pad_half_sizes_m[0]
            ]
            candidate_rows = [
                row for row in rows
                if row["friction"] == friction and row["controller"] == controller
                and row["finger_pad_half_size_m"] == pad_half_sizes_m[1]
            ]
            base_by_pair = {row["pair_id"]: row for row in base_rows}
            candidate_by_pair = {row["pair_id"]: row for row in candidate_rows}
            if set(base_by_pair) != set(candidate_by_pair):
                raise ValueError("pad-size conditions do not contain identical paired poses")
            for pair_id in base_by_pair:
                base_pose = (base_by_pair[pair_id]["target_x_m"], base_by_pair[pair_id]["target_y_m"])
                candidate_pose = (candidate_by_pair[pair_id]["target_x_m"], candidate_by_pair[pair_id]["target_y_m"])
                if base_pose != candidate_pose:
                    raise ValueError(f"target pose changed for paired trial {pair_id}")
            conditions[condition]["controllers"][controller] = {
                f"pad_half_size_m_{pad_half_sizes_m[0]:g}": _aggregate(base_rows),
                f"pad_half_size_m_{pad_half_sizes_m[1]:g}": _aggregate(candidate_rows),
            }
            paired[condition][controller] = _paired_outcomes(base_by_pair, candidate_by_pair)

    trials_path = output_dir / TRIALS_FILE
    with trials_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_scorecard(
        output_dir / SCORECARD_FILE,
        conditions,
        len(positions),
        pad_half_sizes_m,
        friction_levels,
    )
    demos = _render_demos(rows, positions, media_dir=media_dir) if render_demos else None
    summary = {
        "experiment": "held-out paired finger-pad geometry test under grasp-friction stress",
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
        "paired_design": "same target pose for each pad size, friction level, and camera-estimate/oracle controller",
        "controllers": list(CONTROLLERS),
        "frozen_settings": {
            "payload_body_mass_kg": PAYLOAD_MASS_KG,
            "lowest_approach_tool_height_offset_m": APPROACH_CLEARANCE_M,
            "release_profile": RELEASE_PROFILE,
            "baseline_pad_half_size_m": pad_half_sizes_m[0],
            "candidate_pad_half_size_m": pad_half_sizes_m[1],
        },
        "varied_settings": {
            "finger_pad_half_size_m": list(pad_half_sizes_m),
            "finger_pad_and_payload_sliding_friction": list(friction_levels),
            "interpretation": "synthetic simulator friction values, not measurements for a physical gripper or object",
        },
        "total_physics_trials": len(rows),
        "conditions": conditions,
        "paired_success_outcomes": paired,
        "fresh_visual_confirmation": "not measured; physical stable-lift outcomes do not imply visual confirmation",
        "limitations": [
            "The 18 mm candidate was identified during prior exploratory work; this independent seed evaluates it but is not a preregistered hardware study.",
            "Only pad size changes in this comparison; friction, nominal payload mass, target poses, approach, release, and controller are held fixed within each pair.",
            "The two chosen friction values are synthetic stress points, not calibrated contact properties.",
            "Twelve poses provide a screening result, not a precise reliability estimate.",
            "The oracle controller is a diagnostic upper bound and does not represent camera-based deployment.",
            "The benchmark measures physical hold and collision outcomes, not fresh visual confirmation or real-robot performance.",
        ],
        "artifacts": {
            "trials_csv": trials_path.name,
            "scorecard_svg": SCORECARD_FILE,
            "demo_metadata": DEMO_METADATA_FILE if demos is not None else None,
            "demo_video": DEMO_VIDEO_FILE if demos is not None else None,
            "demo_preview": DEMO_GIF_FILE if demos is not None else None,
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
    print(json.dumps({"trials": summary["total_physics_trials"], "results": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
