"""Held-out two-view verification challenge with a rendered MuJoCo occluder."""

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
from .cooperative_pick_occlusion_challenge import (
    FROZEN_GAP_TOLERANCE_PX,
    MINIMUM_PAD_HALF_SIZE_M,
)
from .cooperative_pick_visual_validation import _wilson_interval
from .cooperative_pick_visual_verification import V3_MAX_MARKER_ROTATION_FROM_START_DEG


DEFAULT_NOMINAL_BASE_SEED = 40500
DEFAULT_NEGATIVE_BASE_SEED = 40600
DEFAULT_NOMINAL_SEEDS = 5
DEFAULT_NEGATIVE_SEEDS = 7
DEFAULT_SAMPLES_PER_SEED = 20
MINIMUM_NEGATIVE_POSES = 80
OUTPUT_DIRECTORY = Path("results/cooperative-pick-orthogonal-verification-heldout")
OUTPUT_CSV = "cooperative-pick-orthogonal-verification-trials.csv"
OUTPUT_JSON = "cooperative-pick-orthogonal-verification-summary.json"
OUTPUT_SVG = "cooperative-pick-orthogonal-verification-scorecard.svg"

STRATA = (
    {
        "name": "nominal_pad_16mm",
        "base_seed_key": "nominal_base_seed",
        "pad_half_size_m": FINGER_PAD_HALF_SIZE_M,
        "description": "nominal 16 mm finger-pad half-size",
    },
    {
        "name": "small_pad_14mm_negative_control",
        "base_seed_key": "negative_base_seed",
        "pad_half_size_m": MINIMUM_PAD_HALF_SIZE_M,
        "description": "previously failure-prone 14 mm pad geometry; enriched negative control",
    },
)

METHOD_FIELDS = {
    "V3_side": "v3_decision",
    "V4_side_28px": "v4_side_decision",
    "V4_two_view": "orthogonal_v4_decision",
}


def _decision_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    positives = [row for row in rows if row["capture_truth_raised_stable"]]
    negatives = [row for row in rows if not row["capture_truth_raised_stable"]]
    confirmed = [row for row in rows if row[field] == "confirmed"]
    rejected = [row for row in rows if row[field] == "not_confirmed"]
    true_positive = sum(row["capture_truth_raised_stable"] for row in confirmed)
    false_confirmation = len(confirmed) - true_positive
    true_negative = sum(not row["capture_truth_raised_stable"] for row in rejected)
    false_nonconfirmation = len(rejected) - true_negative
    resolved = len(confirmed) + len(rejected)
    return {
        "evaluations": len(rows),
        "capture_truth_positive": len(positives),
        "capture_truth_negative": len(negatives),
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


def _summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for stratum in STRATA:
        selected = [row for row in rows if row["stratum"] == stratum["name"]]
        unique: dict[str, dict[str, Any]] = {}
        for row in selected:
            unique.setdefault(row["pair_id"], row)
        poses = list(unique.values())
        summaries[stratum["name"]] = {
            "description": stratum["description"],
            "unique_physical_pose_runs": len(poses),
            "capture_truth_positive": sum(row["capture_truth_raised_stable"] for row in poses),
            "capture_truth_negative": sum(not row["capture_truth_raised_stable"] for row in poses),
            "stable_hold_successes": sum(row["payload_stable_lift_success"] for row in poses),
            "strict_full_cycle_successes": sum(row["physical_stable_lift_success"] for row in poses),
            "failure_reasons": dict(sorted(Counter(row["failure_reason"] for row in poses).items())),
            "methods": {
                method: _decision_summary(selected, field)
                for method, field in METHOD_FIELDS.items()
            },
        }
    return summaries


def _write_scorecard(path: Path, summaries: dict[str, Any]) -> None:
    width, height = 1320, 590
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1220"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}.muted{fill:#a9b6c8}.white{fill:#f3f6fb}.blue{fill:#58a6ff}.green{fill:#57d6a3}.orange{fill:#ffb86c}.red{fill:#ff7b72}</style>',
        '<text x="30" y="42" font-size="25" font-weight="700" class="white">Fresh-view verifier under a MuJoCo occluder</text>',
        '<text x="30" y="70" font-size="14" class="muted">Same physical runs and capture times | rail is rendered MuJoCo geometry, collision disabled | V3 remains default</text>',
    ]
    columns = (
        ("nominal_pad_16mm", "Nominal 16 mm pads", 30),
        ("small_pad_14mm_negative_control", "14 mm negative-control stratum", 685),
    )
    methods = (
        ("V3_side", "V3 | side camera"),
        ("V4_side_28px", "V4 | side, 28 px gap"),
        ("V4_two_view", "V4 + orthogonal view"),
    )
    for stratum, title, x in columns:
        data = summaries[stratum]
        y = 118
        elements.append(
            f'<text x="{x}" y="{y}" font-size="17" font-weight="700" class="blue">{title}</text>'
        )
        elements.append(
            f'<text x="{x}" y="{y + 24}" font-size="13" class="muted">{data["unique_physical_pose_runs"]} unique poses | capture truth +{data["capture_truth_positive"]} / -{data["capture_truth_negative"]} | stable holds {data["stable_hold_successes"]}</text>'
        )
        y += 68
        for index, (method, label) in enumerate(methods):
            metrics = data["methods"][method]
            row_y = y + index * 92
            background = "#141e30" if index % 2 == 0 else "#101929"
            elements.append(f'<rect x="{x}" y="{row_y - 24}" width="600" height="78" rx="7" fill="{background}"/>')
            elements.append(f'<text x="{x + 12}" y="{row_y}" font-size="15" font-weight="700" class="white">{label}</text>')
            elements.append(
                f'<text x="{x + 12}" y="{row_y + 23}" font-size="13" class="green">TP {metrics["true_positive"]}/{metrics["capture_truth_positive"]}</text>'
                f'<text x="{x + 153}" y="{row_y + 23}" font-size="13" class="red">false confirms {metrics["false_confirmation"]}/{metrics["capture_truth_negative"]}</text>'
                f'<text x="{x + 385}" y="{row_y + 23}" font-size="13" class="orange">abstain {metrics["abstained"]}</text>'
            )
            elements.append(
                f'<text x="{x + 12}" y="{row_y + 45}" font-size="12" class="muted">recall {metrics["positive_recall"]} | specificity {metrics["negative_specificity"]} | coverage {metrics["coverage"]}</text>'
            )
    elements.extend(
        [
            '<text x="30" y="535" font-size="13" class="muted">Positive = raised and stable at both fresh-frame instants. Physical hold and strict-cycle success are reported separately.</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(elements), encoding="utf-8")


def run_validation(
    *,
    output_dir: Path = OUTPUT_DIRECTORY,
    nominal_base_seed: int = DEFAULT_NOMINAL_BASE_SEED,
    negative_base_seed: int = DEFAULT_NEGATIVE_BASE_SEED,
    nominal_seeds: int = DEFAULT_NOMINAL_SEEDS,
    negative_seeds: int = DEFAULT_NEGATIVE_SEEDS,
    samples_per_seed: int = DEFAULT_SAMPLES_PER_SEED,
    require_minimum_negative_poses: bool = True,
) -> dict[str, Any]:
    if nominal_seeds <= 0 or negative_seeds <= 0 or samples_per_seed <= 0:
        raise ValueError("seed and sample counts must be positive")
    nominal_seed_block = set(range(nominal_base_seed, nominal_base_seed + nominal_seeds))
    negative_seed_block = set(range(negative_base_seed, negative_base_seed + negative_seeds))
    if nominal_seed_block & negative_seed_block:
        raise ValueError("nominal and negative-control seed blocks must be disjoint")

    rows: list[dict[str, Any]] = []
    provenance = []
    for stratum in STRATA:
        seed_base = (
            nominal_base_seed
            if stratum["base_seed_key"] == "nominal_base_seed"
            else negative_base_seed
        )
        seeds = nominal_seeds if stratum["base_seed_key"] == "nominal_base_seed" else negative_seeds
        positions = _sample_positions(
            seeds=seeds,
            samples_per_seed=samples_per_seed,
            base_seed=seed_base,
        )
        provenance.append(
            {
                "name": stratum["name"],
                "description": stratum["description"],
                "finger_pad_half_size_m": stratum["pad_half_size_m"],
                "base_seed": seed_base,
                "seeds": seeds,
                "samples_per_seed": samples_per_seed,
                "unique_physical_pose_runs": len(positions),
            }
        )
        for position_index, (seed, sample_index, target_xy) in enumerate(positions, start=1):
            print(
                f"{stratum['name']} {position_index}/{len(positions)} "
                f"seed={seed} sample={sample_index}",
                flush=True,
            )
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
                visual_verification_orthogonal_candidate=True,
                visual_verification_occluder=True,
                visual_verification_orthogonal_gap_tolerance_px=FROZEN_GAP_TOLERANCE_PX,
            )
            baseline = metrics["visual_verification"]
            side_v4 = metrics["visual_verification_gap_tolerant_candidates"][
                str(FROZEN_GAP_TOLERANCE_PX)
            ]
            stereo_v4 = metrics["visual_verification_orthogonal_candidate"]
            truth = baseline["evaluation_only_physics"]
            pair_id = f"{seed}-{sample_index:03d}"
            rows.append(
                {
                    "pair_id": pair_id,
                    "seed": seed,
                    "sample_index": sample_index,
                    "stratum": stratum["name"],
                    "finger_pad_half_size_m": float(stratum["pad_half_size_m"]),
                    "target_x_m": target_xy[0],
                    "target_y_m": target_xy[1],
                    "occluder": "static in-scene MuJoCo rail; side camera only; collision disabled",
                    "capture_truth_raised_stable": truth["raised_and_stable_at_both_frames"],
                    "capture_lift_m": json.dumps(truth["payload_lift_m_at_capture"]),
                    "capture_tilt_deg": json.dumps(truth["payload_tilt_deg_at_capture"]),
                    "v3_decision": baseline["decision"],
                    "v3_reason": baseline["reason"],
                    "v4_side_decision": side_v4["decision"],
                    "v4_side_reason": side_v4["reason"],
                    "orthogonal_v4_decision": stereo_v4["decision"],
                    "orthogonal_v4_reason": stereo_v4["reason"],
                    "orthogonal_side_decision": stereo_v4["view_records"].get("side", {}).get("decision", "missing"),
                    "orthogonal_side_rotation_deg": json.dumps(stereo_v4["view_records"].get("side", {}).get("marker_rotation_from_start_deg")),
                    "orthogonal_view_decision": stereo_v4["view_records"].get("orthogonal", {}).get("decision", "missing"),
                    "orthogonal_view_rotation_deg": json.dumps(stereo_v4["view_records"].get("orthogonal", {}).get("marker_rotation_from_start_deg")),
                    "payload_stable_lift_success": metrics["payload_stable_lift_success"],
                    "full_sequence_collision_safe": metrics["full_sequence_collision_safe"],
                    "physical_stable_lift_success": metrics["physical_stable_lift_success"],
                    "failure_reason": metrics["failure_reason"],
                    "maximum_payload_tilt_during_hold_deg": metrics["maximum_payload_tilt_during_hold_deg"],
                    "arm_environment_contacts": metrics["physics_steps_with_arm_table_or_ground_contact"],
                    "nonfinger_payload_contacts": metrics["physics_steps_with_nonfinger_arm_payload_contact"],
                }
            )

    summaries = _summaries(rows)
    negative_count = summaries["small_pad_14mm_negative_control"]["capture_truth_negative"]
    if require_minimum_negative_poses and negative_count < MINIMUM_NEGATIVE_POSES:
        raise RuntimeError(
            f"challenge produced only {negative_count} unique capture-time negatives; "
            f"at least {MINIMUM_NEGATIVE_POSES} are required"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / OUTPUT_CSV
    fieldnames = list(rows[0]) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "experiment": "orthogonal-view fresh visual verification under a rendered MuJoCo occluder",
        "protocol_version": 1,
        "thresholds_frozen_before_run": {
            "minimum_upward_displacement_px": 50.0,
            "maximum_interframe_drift_px": 12.0,
            "maximum_marker_rotation_from_start_deg": V3_MAX_MARKER_ROTATION_FROM_START_DEG,
            "V4_vertical_gap_tolerance_px": FROZEN_GAP_TOLERANCE_PX,
        },
        "image_decision_inputs": "initial and two fresh RGB observations from two perpendicular MuJoCo side cameras",
        "simulator_truth_is_input_to_vision": False,
        "occluder": {
            "geometry": "fixed opaque MuJoCo box named verification_occluder_rail",
            "location_m": [0.0, -0.62, 0.347],
            "half_size_m": [0.12, 0.008, 0.006],
            "view": "verification_side only",
            "collision_enabled": False,
            "role": "optical challenge fixture; contact physics remains isolated",
        },
        "minimum_unique_capture_time_negative_poses": MINIMUM_NEGATIVE_POSES,
        "minimum_negative_pose_requirement_met": negative_count >= MINIMUM_NEGATIVE_POSES,
        "total_unique_physical_pose_runs": len({row["pair_id"] for row in rows}),
        "total_image_method_evaluations": len(rows) * len(METHOD_FIELDS),
        "strata": provenance,
        "results": summaries,
        "limitations": [
            "The two cameras and payload fiducials are ideal rendered sensors; this is not a physical-camera result.",
            "The static optical rail has collision disabled to isolate visual occlusion and does not test obstacle avoidance.",
            "The 14 mm pad stratum is enriched for failures and does not estimate deployment prevalence.",
            "Capture-time raised/stable labels do not establish continuous hold or safe release; physical outcomes are reported separately.",
            "Two views provide only image-plane evidence. Marker visibility and view geometry remain synthetic assumptions.",
            "V3 remains the default; V4 and the two-view candidate are experimental and opt-in.",
        ],
        "artifacts": {
            "trials_csv": csv_path.name,
            "scorecard_svg": OUTPUT_SVG,
        },
    }
    json_path = output_dir / OUTPUT_JSON
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    svg_path = output_dir / OUTPUT_SVG
    _write_scorecard(svg_path, summaries)
    print(json.dumps({"csv": str(csv_path), "summary": str(json_path), "scorecard": str(svg_path), "negative_poses": negative_count}, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--nominal-base-seed", type=int, default=DEFAULT_NOMINAL_BASE_SEED)
    parser.add_argument("--negative-base-seed", type=int, default=DEFAULT_NEGATIVE_BASE_SEED)
    parser.add_argument("--nominal-seeds", type=int, default=DEFAULT_NOMINAL_SEEDS)
    parser.add_argument("--negative-seeds", type=int, default=DEFAULT_NEGATIVE_SEEDS)
    parser.add_argument("--samples-per-seed", type=int, default=DEFAULT_SAMPLES_PER_SEED)
    parser.add_argument("--allow-small", action="store_true", help="skip the 80-negative minimum for smoke runs")
    args = parser.parse_args()
    run_validation(
        output_dir=args.output_dir,
        nominal_base_seed=args.nominal_base_seed,
        negative_base_seed=args.negative_base_seed,
        nominal_seeds=args.nominal_seeds,
        negative_seeds=args.negative_seeds,
        samples_per_seed=args.samples_per_seed,
        require_minimum_negative_poses=not args.allow_small,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
