"""Validate fresh-image pick confirmation on a frozen seeded MuJoCo split."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from .camera_guided_cooperative_pick import (
    DEMO_TARGET_XY,
    FINGER_PAD_HALF_SIZE_M,
    _sample_positions,
    run_trial,
)
from .cooperative_pick_demo import APPROACH_TOOL_Z_OFFSETS_M
from .cooperative_pick_visual_verification import (
    MAX_INTERFRAME_DRIFT_PX,
    MIN_UPWARD_DISPLACEMENT_PX,
    ORIENTATION_AWARE_MIN_UPWARD_DISPLACEMENT_PX,
    V3_MAX_MARKER_ROTATION_FROM_START_DEG,
    VERIFICATION_CAPTURE_TIMES_S,
)


DEFAULT_BASE_SEED = 39400
DEFAULT_SEEDS = 3
DEFAULT_SAMPLES_PER_SEED = 20
RELEASE_PROFILE = "clearance"
OUTPUT_CSV = "cooperative-pick-visual-confirmation-trials.csv"
OUTPUT_JSON = "cooperative-pick-visual-confirmation-summary.json"
OUTPUT_SVG = "cooperative-pick-visual-confirmation-scorecard.svg"
CONDITIONS = (
    ("baseline_5mm", APPROACH_TOOL_Z_OFFSETS_M[0]),
    ("experimental_20mm", 0.020),
)


def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> list[float] | None:
    if trials <= 0:
        return None
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return [round(max(0.0, center - radius), 4), round(min(1.0, center + radius), 4)]


def _percent_or_na(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _classifier_summary(
    selected: list[dict[str, Any]], decision_field: str
) -> dict[str, Any]:
    positives = [row for row in selected if row["truth_raised_stable_at_two_frames"]]
    negatives = [row for row in selected if not row["truth_raised_stable_at_two_frames"]]
    confirmed = [row for row in selected if row[decision_field] == "confirmed"]
    rejected = [row for row in selected if row[decision_field] == "not_confirmed"]
    abstained = [row for row in selected if row[decision_field] == "abstain"]
    true_positive = sum(row["truth_raised_stable_at_two_frames"] for row in confirmed)
    false_confirmation = len(confirmed) - true_positive
    true_negative = sum(not row["truth_raised_stable_at_two_frames"] for row in rejected)
    false_nonconfirmation = len(rejected) - true_negative
    resolved = len(confirmed) + len(rejected)
    return {
        "trials": len(selected),
        "confirmed": len(confirmed),
        "not_confirmed": len(rejected),
        "abstained": len(abstained),
        "true_positive": int(true_positive),
        "false_confirmation": int(false_confirmation),
        "true_negative": int(true_negative),
        "false_nonconfirmation": int(false_nonconfirmation),
        "resolved_accuracy": (
            round((true_positive + true_negative) / resolved, 4) if resolved else None
        ),
        "resolved_accuracy_wilson_95": _wilson_interval(
            int(true_positive + true_negative), resolved
        ),
        "confirmation_precision": (
            round(true_positive / len(confirmed), 4) if confirmed else None
        ),
        "confirmation_precision_wilson_95": _wilson_interval(
            int(true_positive), len(confirmed)
        ),
        "positive_recall_including_abstentions": (
            round(true_positive / len(positives), 4) if positives else None
        ),
        "positive_recall_wilson_95": _wilson_interval(
            int(true_positive), len(positives)
        ),
        "negative_specificity_including_abstentions": (
            round(true_negative / len(negatives), 4) if negatives else None
        ),
        "negative_specificity_wilson_95": _wilson_interval(
            int(true_negative), len(negatives)
        ),
        "coverage": round(resolved / len(selected), 4) if selected else None,
        "coverage_wilson_95": _wilson_interval(resolved, len(selected)),
    }


def _condition_summary(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = [row for row in rows if row["condition"] == condition]
    positives = [row for row in selected if row["truth_raised_stable_at_two_frames"]]
    negatives = [row for row in selected if not row["truth_raised_stable_at_two_frames"]]
    hold_successes = sum(row["payload_stable_lift_success"] for row in selected)
    strict_successes = sum(row["physical_stable_lift_success"] for row in selected)
    collision_safe = sum(row["full_sequence_collision_safe"] for row in selected)

    return {
        "condition": condition,
        "trials": len(selected),
        "capture_time_ground_truth": {
            "positive": len(positives),
            "negative": len(negatives),
            "definition": (
                "MuJoCo evaluation only: payload center is at least 7 cm above its "
                "start at both captures, the two heights differ by at most 15 mm, "
                "and tilt is at most 5 degrees at each capture."
            ),
        },
        "image_classifier": {
            "minimum_upward_displacement_px": (
                ORIENTATION_AWARE_MIN_UPWARD_DISPLACEMENT_PX
            ),
            "maximum_interframe_drift_px": MAX_INTERFRAME_DRIFT_PX,
            "maximum_marker_rotation_from_start_deg": (
                V3_MAX_MARKER_ROTATION_FROM_START_DEG
            ),
            **_classifier_summary(selected, "decision"),
        },
        "translation_only_baseline": _classifier_summary(
            selected, "translation_only_decision"
        ),
        "orientation_only_baseline": _classifier_summary(
            selected, "orientation_only_decision"
        ),
        "rise_orientation_baseline": _classifier_summary(
            selected, "rise_orientation_decision"
        ),
        "physical_outcomes": {
            "payload_stable_hold_successes": hold_successes,
            "payload_stable_hold_rate": (
                round(hold_successes / len(selected), 4) if selected else None
            ),
            "strict_full_sequence_successes": strict_successes,
            "strict_full_sequence_rate": (
                round(strict_successes / len(selected), 4) if selected else None
            ),
            "full_sequence_collision_safe_trials": collision_safe,
        },
    }


def _paired_physical_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paired: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        paired.setdefault(row["pair_id"], {})[row["condition"]] = row
    if any(set(pair) != {"baseline_5mm", "experimental_20mm"} for pair in paired.values()):
        raise ValueError("every target pose must have one row for both approach conditions")

    comparisons: dict[str, Any] = {}
    for outcome, field in (
        ("payload_stable_lift_success", "payload_stable_lift_success"),
        ("strict_full_sequence_success", "physical_stable_lift_success"),
    ):
        candidate_only = sum(
            pair["experimental_20mm"][field] and not pair["baseline_5mm"][field]
            for pair in paired.values()
        )
        baseline_only = sum(
            pair["baseline_5mm"][field] and not pair["experimental_20mm"][field]
            for pair in paired.values()
        )
        both = sum(
            pair["baseline_5mm"][field] and pair["experimental_20mm"][field]
            for pair in paired.values()
        )
        neither = len(paired) - candidate_only - baseline_only - both
        comparisons[outcome] = {
            "experimental_20mm_only_successes": int(candidate_only),
            "baseline_5mm_only_successes": int(baseline_only),
            "both_success": int(both),
            "neither_success": int(neither),
            "net_difference_in_successes": int(candidate_only - baseline_only),
        }
    return {
        "matched_target_pose_pairs": len(paired),
        "interpretation": "Paired physical outcomes compare the 20 mm experimental approach with the 5 mm baseline on the exact same target poses.",
        "outcomes": comparisons,
    }


def _trial_row(
    *,
    pair_id: str,
    seed: int,
    sample_index: int,
    target_xy: tuple[float, float],
    condition: str,
    approach_offset_m: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    visual = metrics["visual_verification"]
    translation_only = metrics["visual_verification_translation_only_baseline"]
    orientation_only = metrics["visual_verification_orientation_only_baseline"]
    rise_orientation = metrics["visual_verification_rise_orientation_baseline"]
    physics = visual["evaluation_only_physics"]
    rises = visual.get("upward_displacement_px") or [None, None]
    capture_lift = physics["payload_lift_m_at_capture"]
    capture_tilt = physics["payload_tilt_deg_at_capture"]
    return {
        "pair_id": pair_id,
        "seed": seed,
        "sample_index": sample_index,
        "condition": condition,
        "approach_offset_m": approach_offset_m,
        "target_x_m": target_xy[0],
        "target_y_m": target_xy[1],
        "decision": visual["decision"],
        "decision_reason": visual["reason"],
        "translation_only_decision": translation_only["decision"],
        "translation_only_decision_reason": translation_only["reason"],
        "orientation_only_decision": orientation_only["decision"],
        "orientation_only_decision_reason": orientation_only["reason"],
        "rise_orientation_decision": rise_orientation["decision"],
        "rise_orientation_decision_reason": rise_orientation["reason"],
        "maximum_marker_rotation_from_start_deg": visual["thresholds_deg"][
            "maximum_marker_rotation_from_start"
        ],
        "marker_rotation_frame_1_deg": (
            visual["marker_rotation_from_start_deg"][0]
            if visual.get("marker_rotation_from_start_deg")
            else None
        ),
        "marker_rotation_frame_2_deg": (
            visual["marker_rotation_from_start_deg"][1]
            if visual.get("marker_rotation_from_start_deg")
            else None
        ),
        "capture_time_1_s": VERIFICATION_CAPTURE_TIMES_S[0],
        "capture_time_2_s": VERIFICATION_CAPTURE_TIMES_S[1],
        "rise_frame_1_px": rises[0],
        "rise_frame_2_px": rises[1],
        "interframe_drift_px": visual.get("interframe_drift_px"),
        "minimum_capture_lift_m": min(capture_lift) if capture_lift else None,
        "capture_height_delta_m": (
            abs(capture_lift[1] - capture_lift[0]) if len(capture_lift) == 2 else None
        ),
        "maximum_capture_tilt_deg": max(capture_tilt) if capture_tilt else None,
        "truth_raised_stable_at_two_frames": physics[
            "raised_and_stable_at_both_frames"
        ],
        "payload_stable_lift_success": metrics["payload_stable_lift_success"],
        "full_sequence_collision_safe": metrics["full_sequence_collision_safe"],
        "physical_stable_lift_success": metrics["physical_stable_lift_success"],
        "failure_reason": metrics["failure_reason"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = tuple(rows[0]) if rows else (
        "pair_id", "seed", "sample_index", "condition", "approach_offset_m",
        "target_x_m", "target_y_m", "decision", "decision_reason",
        "translation_only_decision", "translation_only_decision_reason",
        "orientation_only_decision", "orientation_only_decision_reason",
        "rise_orientation_decision", "rise_orientation_decision_reason",
        "maximum_marker_rotation_from_start_deg", "marker_rotation_frame_1_deg",
        "marker_rotation_frame_2_deg",
        "capture_time_1_s", "capture_time_2_s", "rise_frame_1_px",
        "rise_frame_2_px", "interframe_drift_px", "minimum_capture_lift_m",
        "capture_height_delta_m", "maximum_capture_tilt_deg",
        "truth_raised_stable_at_two_frames", "payload_stable_lift_success",
        "full_sequence_collision_safe", "physical_stable_lift_success", "failure_reason",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_scorecard(
    path: Path,
    summaries: dict[str, dict[str, Any]],
    paired_physics: dict[str, Any],
) -> None:
    width, height = 1180, 620
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1220"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}.muted{fill:#a9b6c8}.white{fill:#f3f6fb}.green{fill:#57d6a3}.red{fill:#ff7b72}.blue{fill:#58a6ff}.orange{fill:#ffb86c}</style>',
        '<text x="32" y="42" font-size="25" font-weight="700" class="white">Fresh visual confirmation - independent seeded validation</text>',
        '<text x="32" y="70" font-size="14" class="muted">60 matched target poses per approach condition | fixed side camera | two new frames at 8.4 s and 8.8 s</text>',
        '<text x="32" y="94" font-size="12" class="muted">Frozen V3 rule: at least 50 px rise; at most 12 px drift and 4 deg marker rotation</text>',
    ]
    panel_xs = (30, 600)
    for x, (condition, summary) in zip(panel_xs, summaries.items(), strict=True):
        image = summary["image_classifier"]
        physical = summary["physical_outcomes"]
        rows = summary["capture_time_ground_truth"]
        label = "5 mm baseline" if condition == "baseline_5mm" else "20 mm experimental approach"
        chunks.extend([
            f'<rect x="{x}" y="116" width="550" height="380" rx="14" fill="#141e30" stroke="#2b3850"/>',
            f'<text x="{x + 22}" y="150" font-size="19" font-weight="700" class="white">{label}</text>',
            f'<text x="{x + 22}" y="178" font-size="13" class="muted">N={summary["trials"]} | truth positive {rows["positive"]} | truth negative {rows["negative"]}</text>',
            f'<text x="{x + 22}" y="214" font-size="14" font-weight="700" class="blue">Image-only classifier vs. two-frame physical truth</text>',
            f'<text x="{x + 22}" y="231" font-size="8" class="muted">False confirms V0/V1/V2/V3: {summary["translation_only_baseline"]["false_confirmation"]}/{summary["orientation_only_baseline"]["false_confirmation"]}/{summary["rise_orientation_baseline"]["false_confirmation"]}/{image["false_confirmation"]}</text>',
            f'<text x="{x + 22}" y="245" font-size="8" class="muted">Positive recall: {_percent_or_na(summary["translation_only_baseline"]["positive_recall_including_abstentions"])}/{_percent_or_na(summary["orientation_only_baseline"]["positive_recall_including_abstentions"])}/{_percent_or_na(summary["rise_orientation_baseline"]["positive_recall_including_abstentions"])}/{_percent_or_na(image["positive_recall_including_abstentions"])}</text>',
            f'<text x="{x + 22}" y="269" font-size="13" class="muted">Confirmed</text>',
            f'<text x="{x + 200}" y="269" font-size="13" class="green">TP {image["true_positive"]}</text>',
            f'<text x="{x + 300}" y="269" font-size="13" class="red">False confirmation {image["false_confirmation"]}</text>',
            f'<text x="{x + 22}" y="295" font-size="13" class="muted">Not confirmed</text>',
            f'<text x="{x + 200}" y="295" font-size="13" class="green">TN {image["true_negative"]}</text>',
            f'<text x="{x + 300}" y="295" font-size="13" class="orange">Missed confirmation {image["false_nonconfirmation"]}</text>',
            f'<text x="{x + 22}" y="321" font-size="13" class="muted">Abstained {image["abstained"]} | coverage {image["coverage"]:.1%} | accuracy {image["resolved_accuracy"] if image["resolved_accuracy"] is not None else "n/a"}</text>',
            f'<text x="{x + 22}" y="352" font-size="14" font-weight="700" class="blue">Physical outcomes, reported separately</text>',
            f'<text x="{x + 22}" y="380" font-size="13" class="muted">Payload stable during 1.6 s hold</text>',
            f'<text x="{x + 328}" y="380" font-size="14" class="green">{physical["payload_stable_hold_successes"]}/{summary["trials"]}</text>',
            f'<text x="{x + 22}" y="406" font-size="13" class="muted">Strict full-sequence physical success</text>',
            f'<text x="{x + 328}" y="406" font-size="14" class="green">{physical["strict_full_sequence_successes"]}/{summary["trials"]}</text>',
            f'<text x="{x + 22}" y="434" font-size="11" class="muted">Vision reads images only. Physics truth is evaluation-only; two samples do not</text>',
            f'<text x="{x + 22}" y="452" font-size="11" class="muted">prove continuous holding or collision safety over the full motion.</text>',
        ])
    chunks.extend([
        f'<text x="32" y="535" font-size="12" class="orange">Matched physics: 20 mm had {paired_physics["outcomes"]["strict_full_sequence_success"]["experimental_20mm_only_successes"]} strict-only wins / {paired_physics["outcomes"]["strict_full_sequence_success"]["baseline_5mm_only_successes"]} baseline-only wins; keep experimental.</text>',
        f'<text x="32" y="565" font-size="11" class="muted">V3 had 0 false confirmations among {sum(item["capture_time_ground_truth"]["negative"] for item in summaries.values())} negative labels. Two samples are not hardware or full-trajectory verification.</text>',
        '</svg>',
    ])
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def run_validation(
    output_dir: Path,
    *,
    seeds: int = DEFAULT_SEEDS,
    samples_per_seed: int = DEFAULT_SAMPLES_PER_SEED,
    base_seed: int = DEFAULT_BASE_SEED,
) -> dict[str, Any]:
    """Run matched trials, preserving classifier and simulator outcomes separately."""
    positions = _sample_positions(
        seeds=seeds,
        samples_per_seed=samples_per_seed,
        base_seed=base_seed,
    )
    rows: list[dict[str, Any]] = []
    for seed, sample_index, target_xy in positions:
        pair_id = f"{seed}-{sample_index:03d}"
        for condition, approach_offset_m in CONDITIONS:
            metrics = run_trial(
                target_xy,
                controller="camera_estimate",
                finger_pad_half_size_m=FINGER_PAD_HALF_SIZE_M,
                release_profile=RELEASE_PROFILE,
                lowest_approach_tool_height_offset_m=approach_offset_m,
                visual_verification=True,
                visual_verification_min_upward_displacement_px=(
                    ORIENTATION_AWARE_MIN_UPWARD_DISPLACEMENT_PX
                ),
                visual_verification_max_marker_rotation_from_start_deg=(
                    V3_MAX_MARKER_ROTATION_FROM_START_DEG
                ),
            )
            rows.append(
                _trial_row(
                    pair_id=pair_id,
                    seed=seed,
                    sample_index=sample_index,
                    target_xy=target_xy,
                    condition=condition,
                    approach_offset_m=approach_offset_m,
                    metrics=metrics,
                )
            )

    summaries = {
        condition: _condition_summary(rows, condition)
        for condition, _ in CONDITIONS
    }
    paired_physics = _paired_physical_comparison(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / OUTPUT_CSV
    json_path = output_dir / OUTPUT_JSON
    svg_path = output_dir / OUTPUT_SVG
    _write_csv(csv_path, rows)
    summary: dict[str, Any] = {
        "evidence_type": "held-out MuJoCo validation of the V3 image-only verifier",
        "split": {
            "base_seed": base_seed,
            "seeds": seeds,
            "samples_per_seed": samples_per_seed,
            "target_pose_pairs": len(positions),
            "trial_count": len(rows),
            "target_xy_range_m": [-0.09, 0.09],
            "positions_are_matched_across_conditions": True,
            "demonstration_pose_excluded": list(DEMO_TARGET_XY),
        },
        "conditions": {
            condition: {
                "approach_offset_m": offset,
                "finger_pad_half_size_m": FINGER_PAD_HALF_SIZE_M,
                "release_profile": RELEASE_PROFILE,
                "results": summaries[condition],
            }
            for condition, offset in CONDITIONS
        },
        "paired_physical_comparison": paired_physics,
        "image_verifier": {
            "decision_inputs": "initial side RGB and two fresh side RGB frames only",
            "capture_times_s": list(VERIFICATION_CAPTURE_TIMES_S),
            "minimum_upward_displacement_px": MIN_UPWARD_DISPLACEMENT_PX,
            "orientation_aware_minimum_upward_displacement_px": (
                ORIENTATION_AWARE_MIN_UPWARD_DISPLACEMENT_PX
            ),
            "maximum_interframe_drift_px": MAX_INTERFRAME_DRIFT_PX,
            "maximum_marker_rotation_from_start_deg": V3_MAX_MARKER_ROTATION_FROM_START_DEG,
            "ablations": {
                "translation_only_v0": "35 px minimum rise and 12 px maximum drift",
                "orientation_only_v1": "35 px minimum rise, 12 px maximum drift, 12 degree maximum marker rotation",
                "orientation_and_rise_v2": "50 px minimum rise, 12 px maximum drift, 12 degree maximum marker rotation",
                "tight_orientation_v3": "50 px minimum rise, 12 px maximum drift, 4 degree maximum marker rotation",
            },
            "thresholds_frozen_before_validation_split": True,
            "capture_time_truth_is_simulator_evaluation_only": True,
            "truth_available_to_image_decision": False,
        },
        "limitations": [
            "Synthetic fixed MuJoCo side camera and a visual-only payload fiducial; no physical camera, lens, exposure, motion blur, or occlusion validation.",
            "Two fresh samples test raised-and-stable at those capture times, not continuous holding, release quality, or full-cycle collision safety.",
            "The 20 mm approach condition is an experimental setting retained as opt-in; this experiment does not promote it to the default.",
            "Matched poses share one simulation model and ideal calibration; Wilson intervals are descriptive and do not represent hardware generalization.",
        ],
        "trial_data": csv_path.name,
        "scorecard": svg_path.name,
    }
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_scorecard(svg_path, summaries, paired_physics)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--samples-per-seed", type=int, default=DEFAULT_SAMPLES_PER_SEED)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir or repository / "results" / "cooperative-pick-visual-confirmation"
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
