"""Tune and validate opt-in gap-tolerant verification on paired occlusion replays."""

from __future__ import annotations

import argparse
import csv
import json
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


DEVELOPMENT_BASE_SEED = 39700
HELDOUT_BASE_SEED = 39800
DEVELOPMENT_SAMPLES_PER_SEED = 10
HELDOUT_SAMPLES_PER_SEED = 20
DEFAULT_SEEDS = 3
DEFAULT_GAP_TOLERANCES_PX = (12, 20, 28)
OCCLUSION_FAULT = {
    "kind": "marker_occlusion",
    "occluded_fraction": 0.5,
    "label": "50% fiducial occlusion",
}
CONDITIONS = ("clean", "marker_occlusion_50pct")
OUTPUT_CSV = "cooperative-pick-occlusion-trials.csv"
OUTPUT_JSON = "cooperative-pick-occlusion-summary.json"
OUTPUT_SVG = "cooperative-pick-occlusion-scorecard.svg"


def _classifier_summary(rows: list[dict[str, Any]], decision_field: str) -> dict[str, Any]:
    positives = [row for row in rows if row["truth_raised_stable_at_two_frames"]]
    negatives = [row for row in rows if not row["truth_raised_stable_at_two_frames"]]
    confirmed = [row for row in rows if row[decision_field] == "confirmed"]
    rejected = [row for row in rows if row[decision_field] == "not_confirmed"]
    abstained = [row for row in rows if row[decision_field] == "abstain"]
    true_positive = sum(row["truth_raised_stable_at_two_frames"] for row in confirmed)
    false_confirmation = len(confirmed) - true_positive
    true_negative = sum(not row["truth_raised_stable_at_two_frames"] for row in rejected)
    false_nonconfirmation = len(rejected) - true_negative
    resolved = len(confirmed) + len(rejected)
    return {
        "trials": len(rows),
        "capture_time_truth_positive": len(positives),
        "capture_time_truth_negative": len(negatives),
        "true_positive": int(true_positive),
        "false_confirmation": int(false_confirmation),
        "true_negative": int(true_negative),
        "false_nonconfirmation": int(false_nonconfirmation),
        "abstained": len(abstained),
        "coverage": round(resolved / len(rows), 4) if rows else None,
        "coverage_wilson_95": _wilson_interval(resolved, len(rows)),
        "positive_recall_including_abstentions": (
            round(true_positive / len(positives), 4) if positives else None
        ),
        "positive_recall_wilson_95": _wilson_interval(true_positive, len(positives)),
        "negative_specificity_including_abstentions": (
            round(true_negative / len(negatives), 4) if negatives else None
        ),
        "negative_specificity_wilson_95": _wilson_interval(true_negative, len(negatives)),
    }


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
        raise RuntimeError("image-only occlusion replay changed a physical outcome")


def _decision_field(tolerance: int | None) -> str:
    return "v3_decision" if tolerance is None else f"gap_{tolerance}_px_decision"


def _summaries(rows: list[dict[str, Any]], tolerances: tuple[int, ...]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    classifiers: list[tuple[str, int | None]] = [("v3", None)]
    classifiers.extend((f"gap_tolerant_{value}px", value) for value in tolerances)
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        summaries[condition] = {
            name: _classifier_summary(selected, _decision_field(tolerance))
            for name, tolerance in classifiers
        }
        summaries[condition]["physical_outcomes"] = {
            "payload_stable_hold_successes": sum(
                row["payload_stable_lift_success"] for row in selected
            ),
            "strict_full_sequence_successes": sum(
                row["physical_stable_lift_success"] for row in selected
            ),
            "trials": len(selected),
        }
    return summaries


def _select_development_candidate(
    rows: list[dict[str, Any]], tolerances: tuple[int, ...]
) -> dict[str, Any]:
    clean = [row for row in rows if row["condition"] == "clean"]
    occluded = [row for row in rows if row["condition"] == "marker_occlusion_50pct"]
    eligible: list[tuple[int, int]] = []
    assessments: dict[str, Any] = {}
    for tolerance in tolerances:
        field = _decision_field(tolerance)
        clean_matches = sum(row[field] == row["v3_decision"] for row in clean)
        no_new_clean_false_confirmations = all(
            row[field] != "confirmed"
            or row["truth_raised_stable_at_two_frames"]
            for row in clean
        )
        occlusion_false_confirmations = sum(
            row[field] == "confirmed" and not row["truth_raised_stable_at_two_frames"]
            for row in occluded
        )
        occlusion_true_positives = sum(
            row[field] == "confirmed" and row["truth_raised_stable_at_two_frames"]
            for row in occluded
        )
        eligible_for_selection = (
            clean_matches == len(clean)
            and no_new_clean_false_confirmations
            and occlusion_false_confirmations == 0
        )
        assessments[str(tolerance)] = {
            "clean_decisions_matching_v3": clean_matches,
            "clean_trials": len(clean),
            "occlusion_true_positives": occlusion_true_positives,
            "occlusion_false_confirmations": occlusion_false_confirmations,
            "eligible": eligible_for_selection,
        }
        if eligible_for_selection:
            eligible.append((occlusion_true_positives, tolerance))
    selected = None
    if eligible:
        best_true_positives = max(value for value, _ in eligible)
        selected = min(tolerance for value, tolerance in eligible if value == best_true_positives)
    return {
        "selected_gap_tolerance_px": selected,
        "rule": (
            "Among candidates with exact clean-decision parity to V3 and zero false confirmations "
            "on clean and occluded development images, maximize occlusion true positives; "
            "break ties with the smallest gap tolerance."
        ),
        "candidate_assessments": assessments,
    }


def _write_scorecard(
    path: Path,
    *,
    stage: str,
    base_seed: int,
    matched_poses: int,
    summaries: dict[str, Any],
    tolerances: tuple[int, ...],
    selected_tolerance: int | None,
) -> None:
    width, height = 1120, 620
    gap_label = f"V4 gap {selected_tolerance}px" if selected_tolerance else "No eligible V4"
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1220"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}.muted{fill:#a9b6c8}.white{fill:#f3f6fb}.green{fill:#57d6a3}.red{fill:#ff7b72}.blue{fill:#58a6ff}.orange{fill:#ffb86c}</style>',
        '<text x="28" y="40" font-size="24" font-weight="700" class="white">Gap-tolerant fiducial detection | clean and half occlusion</text>',
        f'<text x="28" y="68" font-size="14" class="muted">{stage.title()} split | seed {base_seed} | {matched_poses} matched poses | {matched_poses * 2} simulation replays</text>',
    ]
    for index, condition in enumerate(CONDITIONS):
        x, y = 28 + index * 548, 104
        label = "Clean side-camera RGB" if condition == "clean" else "50% fiducial occlusion"
        result = summaries[condition]
        v3 = result["v3"]
        chosen_name = f"gap_tolerant_{selected_tolerance}px" if selected_tolerance else None
        chosen = result[chosen_name] if chosen_name else None
        lines.extend(
            [
                f'<rect x="{x}" y="{y}" width="520" height="394" rx="14" fill="#141e30" stroke="#2b3850"/>',
                f'<text x="{x + 18}" y="{y + 32}" font-size="18" font-weight="700" class="blue">{label}</text>',
                f'<text x="{x + 18}" y="{y + 63}" font-size="14" class="white">V3 baseline: TP {v3["true_positive"]} | false confirm {v3["false_confirmation"]}</text>',
                f'<text x="{x + 18}" y="{y + 88}" font-size="14" class="muted">False non-confirm {v3["false_nonconfirmation"]} | abstain {v3["abstained"]} | coverage {v3["coverage"]:.0%}</text>',
            ]
        )
        if chosen:
            lines.extend(
                [
                    f'<text x="{x + 18}" y="{y + 139}" font-size="16" font-weight="700" class="green">{gap_label}: TP {chosen["true_positive"]} | false confirm {chosen["false_confirmation"]}</text>',
                    f'<text x="{x + 18}" y="{y + 166}" font-size="14" class="muted">False non-confirm {chosen["false_nonconfirmation"]} | abstain {chosen["abstained"]} | coverage {chosen["coverage"]:.0%}</text>',
                    f'<text x="{x + 18}" y="{y + 214}" font-size="12" class="muted">Positive recall: {chosen["true_positive"]}/{chosen["capture_time_truth_positive"]} | negative specificity: {chosen["true_negative"]}/{chosen["capture_time_truth_negative"]}</text>',
                ]
            )
        else:
            lines.append(f'<text x="{x + 18}" y="{y + 139}" font-size="15" class="orange">No candidate passed the development guardrails.</text>')
        physical = result["physical_outcomes"]
        lines.extend(
            [
                f'<text x="{x + 18}" y="{y + 286}" font-size="13" class="orange">Physical hold {physical["payload_stable_hold_successes"]}/{physical["trials"]} | strict full cycle {physical["strict_full_sequence_successes"]}/{physical["trials"]}</text>',
                f'<text x="{x + 18}" y="{y + 321}" font-size="12" class="muted">Each pose was replayed; image fault changed side RGB only.</text>',
                f'<text x="{x + 18}" y="{y + 348}" font-size="11" class="muted">Candidate tolerances: {", ".join(str(value) for value in tolerances)} px</text>',
            ]
        )
    lines.extend(
        [
            '<text x="28" y="540" font-size="13" class="orange">Simulation-derived labels and authored image occlusion; this is not real-camera or hardware evidence.</text>',
            '<text x="28" y="566" font-size="12" class="muted">V3 remains the default. V4 is a post-capture image-only candidate evaluated on the exact same RGB frames.</text>',
            '</svg>',
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_split(
    output_dir: Path,
    *,
    stage: str,
    base_seed: int,
    seeds: int,
    samples_per_seed: int,
    tolerances: tuple[int, ...],
    development_base_seed: int | None = None,
) -> dict[str, Any]:
    if stage not in {"development", "heldout"}:
        raise ValueError("stage must be 'development' or 'heldout'")
    if seeds <= 0 or samples_per_seed <= 0:
        raise ValueError("seeds and samples_per_seed must both be positive")
    if not tolerances or any(
        not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 100
        for value in tolerances
    ) or len(set(tolerances)) != len(tolerances):
        raise ValueError("tolerances must be unique integers from 1 to 100")

    positions = _sample_positions(
        seeds=seeds,
        samples_per_seed=samples_per_seed,
        base_seed=base_seed,
    )
    rows: list[dict[str, Any]] = []
    for position_index, (seed, sample_index, target_xy) in enumerate(positions, start=1):
        pair_id = f"{seed}-{sample_index:03d}"
        for condition, fault in (("clean", None), ("marker_occlusion_50pct", OCCLUSION_FAULT)):
            metrics = run_trial(
                target_xy,
                controller="camera_estimate",
                finger_pad_half_size_m=FINGER_PAD_HALF_SIZE_M,
                release_profile="clearance",
                lowest_approach_tool_height_offset_m=APPROACH_TOOL_Z_OFFSETS_M[0],
                visual_verification=True,
                visual_verification_min_upward_displacement_px=50.0,
                visual_verification_max_marker_rotation_from_start_deg=(
                    V3_MAX_MARKER_ROTATION_FROM_START_DEG
                ),
                visual_verification_image_fault=fault,
                visual_verification_gap_tolerance_candidates_px=tolerances,
            )
            visual = metrics["visual_verification"]
            physics = visual["evaluation_only_physics"]
            row: dict[str, Any] = {
                "pair_id": pair_id,
                "seed": seed,
                "sample_index": sample_index,
                "target_x_m": target_xy[0],
                "target_y_m": target_xy[1],
                "condition": condition,
                "v3_decision": visual["decision"],
                "v3_reason": visual["reason"],
                "truth_raised_stable_at_two_frames": physics[
                    "raised_and_stable_at_both_frames"
                ],
                "payload_stable_lift_success": metrics["payload_stable_lift_success"],
                "full_sequence_collision_safe": metrics["full_sequence_collision_safe"],
                "physical_stable_lift_success": metrics["physical_stable_lift_success"],
                "failure_reason": metrics["failure_reason"],
            }
            for tolerance in tolerances:
                candidate = metrics["visual_verification_gap_tolerant_candidates"][
                    str(tolerance)
                ]
                row[_decision_field(tolerance)] = candidate["decision"]
                row[f"gap_{tolerance}_px_reason"] = candidate["reason"]
            rows.append(row)
        if position_index % 10 == 0 or position_index == len(positions):
            print(f"[{position_index}/{len(positions)} poses] completed {pair_id}", flush=True)

    _assert_physics_unchanged(rows)
    summaries = _summaries(rows, tolerances)
    selection = (
        _select_development_candidate(rows, tolerances)
        if stage == "development"
        else None
    )
    selected_tolerance = (
        selection["selected_gap_tolerance_px"] if selection is not None else tolerances[0]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / OUTPUT_CSV
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "evidence_type": "paired MuJoCo simulation with authored side-camera RGB occlusion; no real-camera data",
        "stage": stage,
        "split": {
            "base_seed": base_seed,
            "development_base_seed": development_base_seed or (base_seed if stage == "development" else None),
            "seeds": seeds,
            "samples_per_seed": samples_per_seed,
            "matched_target_poses": len(positions),
            "conditions_per_pose": 2,
            "simulation_replay_count": len(rows),
            "target_xy_range_m": [-0.09, 0.09],
        },
        "controller": "camera_estimate using the initial overhead RGB for planning",
        "verifier": {
            "baseline": "V3 frozen: 50 px rise, 12 px drift, 4 degree fiducial-angle change",
            "candidate": "V4 opt-in vertical fragment grouping; same V3 thresholds and RGB frames",
            "candidate_gap_tolerances_px": list(tolerances),
            "occlusion": OCCLUSION_FAULT,
        },
        "physical_outcomes_identical_across_clean_and_occluded_replays_for_every_pose": True,
        "conditions": summaries,
        "development_selection": selection,
        "selected_gap_tolerance_px": selected_tolerance,
        "heldout_provenance": (
            {"development_base_seed": development_base_seed, "candidate_gap_tolerance_px_frozen": selected_tolerance}
            if stage == "heldout"
            else None
        ),
        "outputs": {
            "csv": csv_path.name,
            "svg": OUTPUT_SVG,
            "summary": OUTPUT_JSON,
        },
        "limitations": [
            "Two image conditions and a small negative class do not establish broad specificity.",
            "The marker is synthetic and the occlusion is a deterministic gray band.",
            "Each candidate is evaluated on the same RGB observations as V3; it cannot alter motion or physics.",
            "A positive visual confirmation remains separate from sustained physical hold and full-cycle collision safety.",
        ],
    }
    json_path = output_dir / OUTPUT_JSON
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_scorecard(
        output_dir / OUTPUT_SVG,
        stage=stage,
        base_seed=base_seed,
        matched_poses=len(positions),
        summaries=summaries,
        tolerances=tolerances,
        selected_tolerance=selected_tolerance,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("development", "heldout"), default="development")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--samples-per-seed", type=int)
    parser.add_argument("--gap-tolerances", type=int, nargs="+")
    parser.add_argument("--selected-gap-tolerance", type=int)
    parser.add_argument("--development-base-seed", type=int, default=DEVELOPMENT_BASE_SEED)
    args = parser.parse_args()
    if args.stage == "heldout" and args.selected_gap_tolerance is None:
        parser.error("heldout stage requires --selected-gap-tolerance from development")
    base_seed = args.base_seed or (
        DEVELOPMENT_BASE_SEED if args.stage == "development" else HELDOUT_BASE_SEED
    )
    samples = args.samples_per_seed or (
        DEVELOPMENT_SAMPLES_PER_SEED
        if args.stage == "development"
        else HELDOUT_SAMPLES_PER_SEED
    )
    tolerances = tuple(args.gap_tolerances or (
        DEFAULT_GAP_TOLERANCES_PX
        if args.stage == "development"
        else (args.selected_gap_tolerance,)
    ))
    output_dir = args.output_dir or Path("results") / f"cooperative-pick-occlusion-{args.stage}"
    summary = run_split(
        output_dir,
        stage=args.stage,
        base_seed=base_seed,
        seeds=args.seeds,
        samples_per_seed=samples,
        tolerances=tolerances,
        development_base_seed=(
            args.development_base_seed if args.stage == "heldout" else None
        ),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
