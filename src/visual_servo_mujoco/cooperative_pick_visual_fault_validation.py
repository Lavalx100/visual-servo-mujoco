"""Evaluate frozen post-pick V3 under controlled side-camera image faults."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .camera_guided_cooperative_pick import (
    DEMO_TARGET_XY,
    FINGER_PAD_HALF_SIZE_M,
    _sample_positions,
    run_trial,
)
from .cooperative_pick_demo import APPROACH_TOOL_Z_OFFSETS_M
from .cooperative_pick_visual_validation import _wilson_interval
from .cooperative_pick_visual_verification import V3_MAX_MARKER_ROTATION_FROM_START_DEG


DEFAULT_BASE_SEED = 39600
DEFAULT_SEEDS = 3
DEFAULT_SAMPLES_PER_SEED = 20
RELEASE_PROFILE = "clearance"
MIN_UPWARD_DISPLACEMENT_PX = 50.0
MAX_INTERFRAME_DRIFT_PX = 12.0
OUTPUT_CSV = "cooperative-pick-visual-fault-trials.csv"
OUTPUT_JSON = "cooperative-pick-visual-fault-summary.json"
OUTPUT_SVG = "cooperative-pick-visual-fault-scorecard.svg"
FAULT_CONDITIONS: tuple[tuple[str, dict[str, Any] | None], ...] = (
    ("clean", None),
    (
        "dim_exposure_x0_25",
        {"kind": "exposure_scale", "scale": 0.25, "label": "dim exposure x0.25"},
    ),
    (
        "bright_exposure_x1_5",
        {"kind": "exposure_scale", "scale": 1.5, "label": "bright exposure x1.5"},
    ),
    (
        "marker_occlusion_50pct",
        {
            "kind": "marker_occlusion",
            "occluded_fraction": 0.5,
            "label": "50% fiducial occlusion",
        },
    ),
    (
        "horizontal_blur_9px",
        {
            "kind": "horizontal_motion_blur",
            "kernel_size": 9,
            "label": "9 px horizontal blur",
        },
    ),
    (
        "camera_shift_up_25px",
        {
            "kind": "camera_translation",
            "dx_px": 0,
            "dy_px": -25,
            "label": "fresh-frame camera shift up 25 px",
        },
    ),
)


def _summarize(rows: list[dict[str, Any]], fault_name: str) -> dict[str, Any]:
    selected = [row for row in rows if row["fault_condition"] == fault_name]
    positives = [row for row in selected if row["truth_raised_stable_at_two_frames"]]
    negatives = [row for row in selected if not row["truth_raised_stable_at_two_frames"]]
    confirmed = [row for row in selected if row["decision"] == "confirmed"]
    rejected = [row for row in selected if row["decision"] == "not_confirmed"]
    abstained = [row for row in selected if row["decision"] == "abstain"]
    true_positive = sum(row["truth_raised_stable_at_two_frames"] for row in confirmed)
    false_confirmation = len(confirmed) - true_positive
    true_negative = sum(not row["truth_raised_stable_at_two_frames"] for row in rejected)
    false_nonconfirmation = len(rejected) - true_negative
    resolved = len(confirmed) + len(rejected)
    return {
        "trials": len(selected),
        "capture_time_truth_positive": len(positives),
        "capture_time_truth_negative": len(negatives),
        "true_positive": int(true_positive),
        "false_confirmation": int(false_confirmation),
        "true_negative": int(true_negative),
        "false_nonconfirmation": int(false_nonconfirmation),
        "abstained": len(abstained),
        "coverage": round(resolved / len(selected), 4) if selected else None,
        "coverage_wilson_95": _wilson_interval(resolved, len(selected)),
        "positive_recall_including_abstentions": (
            round(true_positive / len(positives), 4) if positives else None
        ),
        "positive_recall_wilson_95": _wilson_interval(true_positive, len(positives)),
        "negative_specificity_including_abstentions": (
            round(true_negative / len(negatives), 4) if negatives else None
        ),
        "negative_specificity_wilson_95": _wilson_interval(true_negative, len(negatives)),
        "physical_hold_successes": sum(row["payload_stable_lift_success"] for row in selected),
        "strict_physical_successes": sum(row["physical_stable_lift_success"] for row in selected),
    }


def _decision_changes(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], {})[row["fault_condition"]] = row
    changes: dict[str, dict[str, int]] = {}
    for fault_name, _ in FAULT_CONDITIONS:
        if fault_name == "clean":
            continue
        transitions: dict[str, int] = {}
        for pair in by_pair.values():
            start = pair["clean"]["decision"]
            end = pair[fault_name]["decision"]
            label = f"{start}_to_{end}"
            transitions[label] = transitions.get(label, 0) + 1
        changes[fault_name] = dict(sorted(transitions.items()))
    return changes


def _assert_physics_unchanged(rows: list[dict[str, Any]]) -> None:
    by_pair: dict[str, set[tuple[Any, ...]]] = {}
    for row in rows:
        outcome = (
            row["truth_raised_stable_at_two_frames"],
            row["payload_stable_lift_success"],
            row["full_sequence_collision_safe"],
            row["physical_stable_lift_success"],
            row["failure_reason"],
        )
        by_pair.setdefault(row["pair_id"], set()).add(outcome)
    if any(len(outcomes) != 1 for outcomes in by_pair.values()):
        raise RuntimeError("image fault injection changed a physical outcome")


def _write_scorecard(
    path: Path,
    summaries: dict[str, dict[str, Any]],
    matched_target_poses: int,
    base_seed: int,
) -> None:
    width, height = 1240, 730
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1220"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}.muted{fill:#a9b6c8}.white{fill:#f3f6fb}.green{fill:#57d6a3}.red{fill:#ff7b72}.blue{fill:#58a6ff}.orange{fill:#ffb86c}</style>',
        '<text x="30" y="42" font-size="25" font-weight="700" class="white">Post-pick verifier - frozen V3 under side-camera image faults</text>',
        f'<text x="30" y="70" font-size="14" class="muted">Seed {base_seed} | {matched_target_poses} matched poses | {matched_target_poses * len(FAULT_CONDITIONS)} trials | V3 frozen: rise 50 px, drift 12 px, angle 4 deg</text>',
    ]
    xs, ys = (28, 440, 852), (98, 382)
    condition_labels = {
        name: str((fault or {}).get("label", "clean RGB"))
        for name, fault in FAULT_CONDITIONS
    }
    for index, (fault_name, summary) in enumerate(summaries.items()):
        x, y = xs[index % 3], ys[index // 3]
        color = "blue" if fault_name == "clean" else "orange"
        label = condition_labels[fault_name]
        chunks.extend(
            [
                f'<rect x="{x}" y="{y}" width="388" height="258" rx="13" fill="#141e30" stroke="#2b3850"/>',
                f'<text x="{x + 18}" y="{y + 31}" font-size="18" font-weight="700" class="{color}">{label}</text>',
                f'<text x="{x + 18}" y="{y + 58}" font-size="12" class="muted">N={summary["trials"]} | truth +{summary["capture_time_truth_positive"]} / -{summary["capture_time_truth_negative"]}</text>',
                f'<text x="{x + 18}" y="{y + 96}" font-size="15" class="green">TP {summary["true_positive"]}  |  TN {summary["true_negative"]}</text>',
                f'<text x="{x + 18}" y="{y + 125}" font-size="14" class="red">False confirm {summary["false_confirmation"]}  |  Missed {summary["false_nonconfirmation"]}</text>',
                f'<text x="{x + 18}" y="{y + 155}" font-size="13" class="muted">Abstain {summary["abstained"]}  |  Coverage {summary["coverage"]:.0%}</text>',
                f'<text x="{x + 18}" y="{y + 185}" font-size="12" class="muted">Hold {summary["physical_hold_successes"]}/{summary["trials"]}  |  Strict cycle {summary["strict_physical_successes"]}/{summary["trials"]}</text>',
                f'<text x="{x + 18}" y="{y + 220}" font-size="11" class="muted">Image-only fault; same physical pose replayed.</text>',
            ]
        )
    chunks.extend(
        [
            '<text x="30" y="686" font-size="12" class="orange">Authored image perturbations are not real-camera measurements. Camera translation changes fresh side RGB only; control still uses the original overhead RGB.</text>',
            '<text x="30" y="708" font-size="11" class="muted">Simulator capture-time truth, 1.6 s hold, and full-cycle collision score remain separate. No threshold was retuned on this split.</text>',
            '</svg>',
        ]
    )
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def run_validation(
    output_dir: Path,
    *,
    seeds: int = DEFAULT_SEEDS,
    samples_per_seed: int = DEFAULT_SAMPLES_PER_SEED,
    base_seed: int = DEFAULT_BASE_SEED,
) -> dict[str, Any]:
    if seeds <= 0 or samples_per_seed <= 0:
        raise ValueError("seeds and samples_per_seed must both be positive")
    positions = _sample_positions(
        seeds=seeds,
        samples_per_seed=samples_per_seed,
        base_seed=base_seed,
    )
    rows: list[dict[str, Any]] = []
    for seed, sample_index, target_xy in positions:
        pair_id = f"{seed}-{sample_index:03d}"
        for fault_name, fault in FAULT_CONDITIONS:
            metrics = run_trial(
                target_xy,
                controller="camera_estimate",
                finger_pad_half_size_m=FINGER_PAD_HALF_SIZE_M,
                release_profile=RELEASE_PROFILE,
                lowest_approach_tool_height_offset_m=APPROACH_TOOL_Z_OFFSETS_M[0],
                visual_verification=True,
                visual_verification_min_upward_displacement_px=MIN_UPWARD_DISPLACEMENT_PX,
                visual_verification_max_marker_rotation_from_start_deg=(
                    V3_MAX_MARKER_ROTATION_FROM_START_DEG
                ),
                visual_verification_image_fault=fault,
            )
            visual = metrics["visual_verification"]
            physics = visual["evaluation_only_physics"]
            rises = visual.get("upward_displacement_px") or [None, None]
            rotations = visual.get("marker_rotation_from_start_deg") or [None, None]
            rows.append(
                {
                    "pair_id": pair_id,
                    "seed": seed,
                    "sample_index": sample_index,
                    "target_x_m": target_xy[0],
                    "target_y_m": target_xy[1],
                    "fault_condition": fault_name,
                    "decision": visual["decision"],
                    "decision_reason": visual["reason"],
                    "rise_frame_1_px": rises[0],
                    "rise_frame_2_px": rises[1],
                    "interframe_drift_px": visual.get("interframe_drift_px"),
                    "rotation_frame_1_deg": rotations[0],
                    "rotation_frame_2_deg": rotations[1],
                    "truth_raised_stable_at_two_frames": physics[
                        "raised_and_stable_at_both_frames"
                    ],
                    "payload_stable_lift_success": metrics[
                        "payload_stable_lift_success"
                    ],
                    "full_sequence_collision_safe": metrics[
                        "full_sequence_collision_safe"
                    ],
                    "physical_stable_lift_success": metrics[
                        "physical_stable_lift_success"
                    ],
                    "failure_reason": metrics["failure_reason"],
                }
            )

    _assert_physics_unchanged(rows)
    summaries = {
        fault_name: _summarize(rows, fault_name)
        for fault_name, _ in FAULT_CONDITIONS
    }
    decision_changes = _decision_changes(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / OUTPUT_CSV
    json_path = output_dir / OUTPUT_JSON
    svg_path = output_dir / OUTPUT_SVG
    fields = tuple(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "evidence_type": "held-out MuJoCo evaluation with authored image-only fault injection; no real-camera measurement",
        "split": {
            "base_seed": base_seed,
            "development_probe_base_seed_excluded": 39500,
            "seeds": seeds,
            "samples_per_seed": samples_per_seed,
            "matched_target_poses": len(positions),
            "fault_conditions_per_pose": len(FAULT_CONDITIONS),
            "trial_count": len(rows),
            "target_xy_range_m": [-0.09, 0.09],
        },
        "controller": "camera_estimate using the one initial overhead RGB for planning",
        "approach_offset_m": APPROACH_TOOL_Z_OFFSETS_M[0],
        "release_profile": RELEASE_PROFILE,
        "verifier": {
            "version": "V3",
            "thresholds_frozen_before_validation": True,
            "minimum_upward_displacement_px": MIN_UPWARD_DISPLACEMENT_PX,
            "maximum_interframe_drift_px": MAX_INTERFRAME_DRIFT_PX,
            "maximum_marker_rotation_from_start_deg": V3_MAX_MARKER_ROTATION_FROM_START_DEG,
            "decision_inputs": "initial side RGB and two fresh side RGB frames only",
            "capture_times_s": [8.4, 8.8],
            "simulator_truth_available_to_decision": False,
        },
        "fault_conditions": {
            name: {"specification": fault, "results": summaries[name]}
            for name, fault in FAULT_CONDITIONS
        },
        "decision_changes_relative_to_clean": decision_changes,
        "physical_outcomes_identical_across_faults_for_every_pose": True,
        "limitations": [
            "Image transforms are authored digital perturbations of ideal MuJoCo RGB, not camera hardware measurements.",
            "Camera translation is a 25 px image-plane shift applied only to the two fresh side-camera frames; it approximates view motion and does not simulate a new optical pose or parallax.",
            "The marker occlusion is placed relative to the detected fiducial bounding box and does not model a physical occluder in the scene.",
            "A single 60-pose held-out seed block cannot establish broad camera robustness; negative-class counts remain limited.",
            "Two capture times do not establish continuous holding, release safety, or full-cycle collision safety.",
        ],
        "trial_data": csv_path.name,
        "scorecard": svg_path.name,
        "demo_command": "uv run --locked --extra demo python scripts/generate_cooperative_pick_visual_fault_demos.py",
        "demonstration_pose_excluded": list(DEMO_TARGET_XY),
    }
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_scorecard(svg_path, summaries, len(positions), base_seed)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--samples-per-seed", type=int, default=DEFAULT_SAMPLES_PER_SEED)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir or repository / "results" / "cooperative-pick-visual-fault-validation"
    summary = run_validation(
        output_dir,
        seeds=args.seeds,
        samples_per_seed=args.samples_per_seed,
        base_seed=args.base_seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
