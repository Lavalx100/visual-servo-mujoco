"""Frozen V4 challenge across occlusion geometry and physical outcome strata."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .camera_guided_cooperative_pick import (
    FINGER_PAD_HALF_SIZE_M,
    _sample_positions,
    run_trial,
)
from .cooperative_pick_demo import APPROACH_TOOL_Z_OFFSETS_M
from .cooperative_pick_visual_validation import _wilson_interval
from .cooperative_pick_visual_verification import V3_MAX_MARKER_ROTATION_FROM_START_DEG


NOMINAL_BASE_SEED = 40100
HARD_NEGATIVE_BASE_SEED = 40200
DEFAULT_SEEDS = 5
DEFAULT_SAMPLES_PER_SEED = 20
FROZEN_GAP_TOLERANCE_PX = 28
MINIMUM_PAD_HALF_SIZE_M = 0.014
OUTPUT_CSV = "cooperative-pick-occlusion-shape-trials.csv"
OUTPUT_JSON = "cooperative-pick-occlusion-shape-summary.json"
OUTPUT_SVG = "cooperative-pick-occlusion-shape-scorecard.svg"


IMAGE_CONDITIONS: dict[str, dict[str, Any] | None] = {
    "clean": None,
    "horizontal_25_center": {
        "kind": "marker_occlusion",
        "shape": "horizontal_band",
        "occluded_fraction": 0.25,
        "label": "horizontal band, 25% height, centered",
    },
    "horizontal_50_upper": {
        "kind": "marker_occlusion",
        "shape": "horizontal_band",
        "occluded_fraction": 0.50,
        "offset_y_fraction": -0.20,
        "label": "horizontal band, 50% height, upper offset",
    },
    "horizontal_50_lower": {
        "kind": "marker_occlusion",
        "shape": "horizontal_band",
        "occluded_fraction": 0.50,
        "offset_y_fraction": 0.20,
        "label": "horizontal band, 50% height, lower offset",
    },
    "horizontal_75_center": {
        "kind": "marker_occlusion",
        "shape": "horizontal_band",
        "occluded_fraction": 0.75,
        "label": "horizontal band, 75% height, centered",
    },
    "vertical_30_center": {
        "kind": "marker_occlusion",
        "shape": "vertical_band",
        "occluded_fraction": 0.30,
        "label": "vertical band, 30% width, centered",
    },
    "diagonal_25_plus45": {
        "kind": "marker_occlusion",
        "shape": "diagonal_band",
        "occluded_fraction": 0.25,
        "angle_deg": 45,
        "label": "diagonal band, 25% thickness, +45 degrees",
    },
    "diagonal_25_minus45": {
        "kind": "marker_occlusion",
        "shape": "diagonal_band",
        "occluded_fraction": 0.25,
        "angle_deg": -45,
        "label": "diagonal band, 25% thickness, -45 degrees",
    },
    "patch_45_upper_left": {
        "kind": "marker_occlusion",
        "shape": "corner_patch",
        "occluded_fraction": 0.45,
        "offset_x_fraction": -0.20,
        "offset_y_fraction": -0.20,
        "label": "upper-left patch, 45% bbox width and height",
    },
}

STRATA = (
    {
        "name": "nominal_pad_16mm",
        "base_seed_key": "nominal_base_seed",
        "pad_half_size_m": FINGER_PAD_HALF_SIZE_M,
        "description": "project's nominal 16 mm pad half-size",
    },
    {
        "name": "small_pad_14mm_hard_negative",
        "base_seed_key": "hard_negative_base_seed",
        "pad_half_size_m": MINIMUM_PAD_HALF_SIZE_M,
        "description": "previously measured 14 mm pad configuration with frequent physical lift failures",
    },
)


def _classifier_summary(rows: list[dict[str, Any]], decision_field: str) -> dict[str, Any]:
    positives = [row for row in rows if row["truth_raised_stable_at_two_frames"]]
    negatives = [row for row in rows if not row["truth_raised_stable_at_two_frames"]]
    confirmed = [row for row in rows if row[decision_field] == "confirmed"]
    rejected = [row for row in rows if row[decision_field] == "not_confirmed"]
    true_positive = sum(row["truth_raised_stable_at_two_frames"] for row in confirmed)
    false_confirmation = len(confirmed) - true_positive
    true_negative = sum(not row["truth_raised_stable_at_two_frames"] for row in rejected)
    false_nonconfirmation = len(rejected) - true_negative
    resolved = len(confirmed) + len(rejected)
    return {
        "evaluations": len(rows),
        "capture_time_truth_positive": len(positives),
        "capture_time_truth_negative": len(negatives),
        "true_positive": int(true_positive),
        "false_confirmation": int(false_confirmation),
        "true_negative": int(true_negative),
        "false_nonconfirmation": int(false_nonconfirmation),
        "abstained": len(rows) - resolved,
        "positive_recall": round(true_positive / len(positives), 4) if positives else None,
        "positive_recall_wilson_95": _wilson_interval(true_positive, len(positives)),
        "negative_specificity": round(true_negative / len(negatives), 4) if negatives else None,
        "negative_specificity_wilson_95": _wilson_interval(true_negative, len(negatives)),
        "coverage": round(resolved / len(rows), 4) if rows else None,
        "coverage_wilson_95": _wilson_interval(resolved, len(rows)),
    }


def _physical_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pose: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_pose.setdefault(row["pair_id"], row)
    poses = list(by_pose.values())
    return {
        "unique_physical_pose_runs": len(poses),
        "capture_time_truth_positive": sum(row["truth_raised_stable_at_two_frames"] for row in poses),
        "capture_time_truth_negative": sum(not row["truth_raised_stable_at_two_frames"] for row in poses),
        "payload_stable_hold_successes": sum(row["payload_stable_lift_success"] for row in poses),
        "strict_full_sequence_successes": sum(row["physical_stable_lift_success"] for row in poses),
        "failure_reasons": dict(sorted(Counter(row["failure_reason"] for row in poses).items())),
    }


def _assert_single_run_invariance(rows: list[dict[str, Any]]) -> None:
    outcomes: dict[str, set[tuple[Any, ...]]] = {}
    for row in rows:
        outcome = (
            row["truth_raised_stable_at_two_frames"],
            row["payload_stable_lift_success"],
            row["full_sequence_collision_safe"],
            row["physical_stable_lift_success"],
            row["failure_reason"],
        )
        outcomes.setdefault(row["pair_id"], set()).add(outcome)
    if any(len(values) != 1 for values in outcomes.values()):
        raise RuntimeError("post-render occlusion condition changed a physical outcome")


def _summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stratum in (item["name"] for item in STRATA):
        selected = [row for row in rows if row["stratum"] == stratum]
        result[stratum] = {
            "physical_outcomes": _physical_summary(selected),
            "image_conditions": {},
        }
        for condition in IMAGE_CONDITIONS:
            condition_rows = [row for row in selected if row["image_condition"] == condition]
            v3 = _classifier_summary(condition_rows, "v3_decision")
            v4 = _classifier_summary(condition_rows, "v4_decision")
            result[stratum]["image_conditions"][condition] = {
                "v3": v3,
                "v4_frozen_28px": v4,
                "v4_decision_changes_from_v3": sum(
                    row["v3_decision"] != row["v4_decision"] for row in condition_rows
                ),
                "fault": json.loads(condition_rows[0]["fault"])
                if condition_rows[0]["fault"]
                else None,
            }
    return result


def _write_scorecard(path: Path, summaries: dict[str, Any], poses_per_stratum: int) -> None:
    width, height = 1480, 820
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1220"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}.muted{fill:#a9b6c8}.white{fill:#f3f6fb}.green{fill:#57d6a3}.red{fill:#ff7b72}.blue{fill:#58a6ff}.orange{fill:#ffb86c}</style>',
        '<text x="28" y="40" font-size="25" font-weight="700" class="white">Frozen V4 occlusion geometry challenge</text>',
        f'<text x="28" y="68" font-size="14" class="muted">28 px candidate | {poses_per_stratum} physical pose runs per pad geometry | 9 image masks evaluated from each same capture</text>',
    ]
    columns = [
        ("nominal_pad_16mm", "Nominal 16 mm pads"),
        ("small_pad_14mm_hard_negative", "14 mm hard-negative stratum"),
    ]
    column_x = [28, 746]
    table_y = 115
    for (stratum, title), x in zip(columns, column_x, strict=True):
        physical = summaries[stratum]["physical_outcomes"]
        label = (
            f'{title} | capture positives {physical["capture_time_truth_positive"]} '
            f'| negatives {physical["capture_time_truth_negative"]} '
            f'| strict cycles {physical["strict_full_sequence_successes"]}/{physical["unique_physical_pose_runs"]}'
        )
        rows.append(f'<text x="{x}" y="{table_y}" font-size="17" font-weight="700" class="blue">{label}</text>')
        rows.append(f'<text x="{x}" y="{table_y + 31}" font-size="12" class="muted">image condition</text>')
        rows.append(f'<text x="{x + 380}" y="{table_y + 31}" font-size="12" class="muted">V3 TP / positives · FC</text>')
        rows.append(f'<text x="{x + 560}" y="{table_y + 31}" font-size="12" class="muted">V4 TP / positives · FC</text>')
        for index, condition in enumerate(IMAGE_CONDITIONS):
            y = table_y + 55 + index * 55
            item = summaries[stratum]["image_conditions"][condition]
            v3, v4 = item["v3"], item["v4_frozen_28px"]
            bg = "#141e30" if index % 2 == 0 else "#101929"
            rows.append(f'<rect x="{x}" y="{y - 18}" width="690" height="48" rx="6" fill="{bg}"/>')
            rows.append(f'<text x="{x + 10}" y="{y + 3}" font-size="12" class="white">{condition}</text>')
            rows.append(f'<text x="{x + 390}" y="{y + 3}" font-size="12" class="muted">{v3["true_positive"]}/{v3["capture_time_truth_positive"]} · {v3["false_confirmation"]}</text>')
            rows.append(f'<text x="{x + 570}" y="{y + 3}" font-size="12" class="green">{v4["true_positive"]}/{v4["capture_time_truth_positive"]} · {v4["false_confirmation"]}</text>')
    rows.extend(
        [
            '<text x="28" y="720" font-size="13" class="orange">FC = false confirmation against capture-time simulator truth. Each stratum is reported separately; the pad mix is not a field-prevalence estimate.</text>',
            '<text x="28" y="748" font-size="12" class="muted">Occlusion is post-render RGB editing. Nine image variants share each physical run and cannot alter motion or the simulator outcome.</text>',
            '<text x="28" y="776" font-size="12" class="muted">The 14 mm pad is an intentionally difficult negative-control geometry; actual frame labels still come from measured MuJoCo state.</text>',
            '</svg>',
        ]
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_challenge(
    output_dir: Path,
    *,
    seeds: int = DEFAULT_SEEDS,
    samples_per_seed: int = DEFAULT_SAMPLES_PER_SEED,
    nominal_base_seed: int = NOMINAL_BASE_SEED,
    hard_negative_base_seed: int = HARD_NEGATIVE_BASE_SEED,
) -> dict[str, Any]:
    if seeds <= 0 or samples_per_seed <= 0:
        raise ValueError("seeds and samples_per_seed must be positive")
    if not isinstance(nominal_base_seed, int) or not isinstance(hard_negative_base_seed, int):
        raise ValueError("base seeds must be integers")
    nominal_seeds = set(range(nominal_base_seed, nominal_base_seed + seeds))
    negative_seeds = set(range(hard_negative_base_seed, hard_negative_base_seed + seeds))
    if nominal_seeds & negative_seeds:
        raise ValueError("the two pad-geometry strata must use disjoint seed blocks")

    rows: list[dict[str, Any]] = []
    stratum_provenance = []
    for stratum in STRATA:
        base_seed = (
            nominal_base_seed
            if stratum["base_seed_key"] == "nominal_base_seed"
            else hard_negative_base_seed
        )
        positions = _sample_positions(
            seeds=seeds,
            samples_per_seed=samples_per_seed,
            base_seed=base_seed,
        )
        stratum_provenance.append(
            {
                "name": stratum["name"],
                "description": stratum["description"],
                "finger_pad_half_size_m": stratum["pad_half_size_m"],
                "base_seed": base_seed,
                "seeds": seeds,
                "samples_per_seed": samples_per_seed,
                "unique_physical_pose_runs": len(positions),
            }
        )
        for position_index, (seed, sample_index, target_xy) in enumerate(positions, start=1):
            pair_id = f"{seed}-{sample_index:03d}"
            metrics = run_trial(
                target_xy,
                controller="camera_estimate",
                finger_pad_half_size_m=float(stratum["pad_half_size_m"]),
                release_profile="clearance",
                lowest_approach_tool_height_offset_m=APPROACH_TOOL_Z_OFFSETS_M[0],
                visual_verification=True,
                visual_verification_min_upward_displacement_px=50.0,
                visual_verification_max_marker_rotation_from_start_deg=(
                    V3_MAX_MARKER_ROTATION_FROM_START_DEG
                ),
                visual_verification_gap_tolerance_candidates_px=(FROZEN_GAP_TOLERANCE_PX,),
                visual_verification_image_fault_candidates=IMAGE_CONDITIONS,
            )
            truth = metrics["visual_verification"]["evaluation_only_physics"]
            records = metrics["visual_verification_fault_candidate_records"]
            for condition, fault in IMAGE_CONDITIONS.items():
                record = records[condition]
                candidate = record["gap_tolerant_candidates"][str(FROZEN_GAP_TOLERANCE_PX)]
                rows.append(
                    {
                        "pair_id": pair_id,
                        "seed": seed,
                        "sample_index": sample_index,
                        "stratum": stratum["name"],
                        "finger_pad_half_size_m": float(stratum["pad_half_size_m"]),
                        "target_x_m": target_xy[0],
                        "target_y_m": target_xy[1],
                        "image_condition": condition,
                        "fault": json.dumps(fault, sort_keys=True) if fault is not None else "",
                        "truth_raised_stable_at_two_frames": truth[
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
                        "v3_decision": record["v3"]["decision"],
                        "v3_reason": record["v3"]["reason"],
                        "v4_decision": candidate["decision"],
                        "v4_reason": candidate["reason"],
                    }
                )
            if position_index % 10 == 0 or position_index == len(positions):
                print(
                    f'[{stratum["name"]} {position_index}/{len(positions)}] completed {pair_id}',
                    flush=True,
                )

    _assert_single_run_invariance(rows)
    summaries = _summaries(rows)
    pose_count = sum(item["unique_physical_pose_runs"] for item in stratum_provenance)
    report: dict[str, Any] = {
        "experiment": "Frozen opt-in V4 occlusion geometry and negative-control challenge",
        "evidence_type": "MuJoCo physical outcomes with post-render RGB mask transforms; no real-camera evidence",
        "evaluation_stage": "held-out challenge; V4 tolerance and mask matrix were frozen before these seed blocks",
        "protocol": {
            "controller": "camera_estimate using initial overhead RGB for planning",
            "visual_decision_inputs": "initial side-camera RGB and two fresh side-camera RGB frames at 8.4 s and 8.8 s",
            "candidate": "V4 vertical fragment grouping at the previously frozen 28 px tolerance",
            "baseline": "V3 frozen 50 px rise, 12 px drift, and 4 degree marker rotation gates",
            "image_fault_scope": "post-render side-camera RGB only; same transform on initial and both fresh frames",
            "paired_design": "one physical trajectory per target pose; all nine image conditions are replayed from its same captured frames",
            "truth_source": "simulator payload height, inter-frame height drift, and tilt read after RGB decisions",
            "physical_metrics_are_not_visual_confirmation": True,
        },
        "frozen_candidate_provenance": {
            "development_base_seed": 39700,
            "prior_heldout_base_seed": 39800,
            "gap_tolerance_px": FROZEN_GAP_TOLERANCE_PX,
            "no_threshold_search_in_this_challenge": True,
        },
        "seed_blocks": stratum_provenance,
        "unique_physical_pose_runs": pose_count,
        "image_condition_evaluation_rows": len(rows),
        "image_conditions": {
            name: {
                "label": fault["label"] if fault is not None else "no image transform",
                "fault": fault,
            }
            for name, fault in IMAGE_CONDITIONS.items()
        },
        "strata": summaries,
        "physical_outcomes_identical_for_all_image_conditions": True,
        "outputs": {"csv": OUTPUT_CSV, "json": OUTPUT_JSON, "scorecard_svg": OUTPUT_SVG},
        "limitations": [
            "Image masks are synthetic, gray, and aligned to the detected marker bounding box; they are not rendered physical occluders.",
            "The 14 mm pad stratum is an intentionally failure-prone negative-control geometry; stratum prevalence does not estimate deployment prevalence.",
            "The ideal magenta fiducial, fixed camera, lighting, and renderer do not model real sensor noise, object clutter, or moving occluders.",
            "Positive and negative labels describe raised-and-stable state at two capture instants, not continuous hold or safe release.",
            "V4 remains opt-in; this challenge does not change the default V3 decision path.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / OUTPUT_CSV
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / OUTPUT_JSON).write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_scorecard(output_dir / OUTPUT_SVG, summaries, seeds * samples_per_seed)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cooperative-pick-occlusion-shape-challenge"))
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--samples-per-seed", type=int, default=DEFAULT_SAMPLES_PER_SEED)
    parser.add_argument("--nominal-base-seed", type=int, default=NOMINAL_BASE_SEED)
    parser.add_argument("--hard-negative-base-seed", type=int, default=HARD_NEGATIVE_BASE_SEED)
    args = parser.parse_args()
    report = run_challenge(
        args.output_dir,
        seeds=args.seeds,
        samples_per_seed=args.samples_per_seed,
        nominal_base_seed=args.nominal_base_seed,
        hard_negative_base_seed=args.hard_negative_base_seed,
    )
    compact = {
        "experiment": report["experiment"],
        "unique_physical_pose_runs": report["unique_physical_pose_runs"],
        "image_condition_evaluation_rows": report["image_condition_evaluation_rows"],
        "outputs": report["outputs"],
        "strata": {
            name: {
                "physical_outcomes": details["physical_outcomes"],
                "v3_v4_clean": details["image_conditions"]["clean"],
                "v3_v4_horizontal_50_upper": details["image_conditions"]["horizontal_50_upper"],
                "v3_v4_vertical_30_center": details["image_conditions"]["vertical_30_center"],
            }
            for name, details in report["strata"].items()
        },
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
