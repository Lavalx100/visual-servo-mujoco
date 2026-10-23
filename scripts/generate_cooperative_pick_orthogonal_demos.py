"""Replay already-counted orthogonal-verification challenge rows as MP4/GIF."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from visual_servo_mujoco.camera_guided_cooperative_pick import run_trial
from visual_servo_mujoco.cooperative_pick_demo import APPROACH_TOOL_Z_OFFSETS_M
from visual_servo_mujoco.cooperative_pick_occlusion_challenge import FROZEN_GAP_TOLERANCE_PX
from visual_servo_mujoco.cooperative_pick_visual_verification import (
    V3_MAX_MARKER_ROTATION_FROM_START_DEG,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPOSITORY / "results" / "cooperative-pick-orthogonal-verification-heldout"
DEFAULT_OUTPUT = REPOSITORY / "assets" / "demos"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _select_examples(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    positive_recoveries = [
        row
        for row in rows
        if row["stratum"] == "nominal_pad_16mm"
        and row["capture_truth_raised_stable"] == "True"
        and row["v3_decision"] != "confirmed"
        and row["orthogonal_v4_decision"] == "confirmed"
    ]
    rejected_false_confirmations = [
        row
        for row in rows
        if row["stratum"] == "small_pad_14mm_negative_control"
        and row["capture_truth_raised_stable"] == "False"
        and row["v3_decision"] == "confirmed"
        and row["orthogonal_v4_decision"] == "not_confirmed"
    ]
    rejected_true_lifts = [
        row
        for row in rows
        if row["stratum"] == "nominal_pad_16mm"
        and row["capture_truth_raised_stable"] == "True"
        and row["orthogonal_v4_decision"] == "not_confirmed"
    ]
    key = lambda row: (int(row["seed"]), int(row["sample_index"]))
    if not positive_recoveries:
        raise RuntimeError("no nominal positive recovered by the two-view candidate was found")
    if not rejected_false_confirmations:
        raise RuntimeError("no measured V3 false confirmation was rejected by the candidate")
    if not rejected_true_lifts:
        raise RuntimeError("no true-lift false non-confirmation was found")
    return [
        {
            "case": "nominal_positive_occlusion_recovery",
            "selection_note": "V3 side-only rule rejects; frozen V4 measurement plus orthogonal view confirms",
            "source": sorted(positive_recoveries, key=key)[0],
            "video_stem": "cooperative_pick_orthogonal_occlusion_recovery",
        },
        {
            "case": "tilted_false_confirmation_rejected",
            "selection_note": "V3 side-only rule confirms; the orthogonal view reveals changed marker orientation",
            "source": sorted(rejected_false_confirmations, key=key)[0],
            "video_stem": "cooperative_pick_orthogonal_false_confirmation_rejected",
        },
        {
            "case": "true_lift_not_confirmed",
            "selection_note": "candidate conservatively rejects a simulator-labeled raised, stable lift",
            "source": sorted(rejected_true_lifts, key=key)[0],
            "video_stem": "cooperative_pick_orthogonal_true_lift_not_confirmed",
        },
    ]


def generate_demos(results_dir: Path, output_dir: Path) -> dict[str, Any]:
    rows = _read_rows(results_dir / "cooperative-pick-orthogonal-verification-trials.csv")
    examples = _select_examples(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    demonstrations = []
    for example in examples:
        source = example["source"]
        stem = example["video_stem"]
        target_xy = (float(source["target_x_m"]), float(source["target_y_m"]))
        metrics = run_trial(
            target_xy,
            controller="camera_estimate",
            finger_pad_half_size_m=float(source["finger_pad_half_size_m"]),
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
            video_path=output_dir / f"{stem}.mp4",
            preview_path=output_dir / f"{stem}.gif",
        )
        v3 = metrics["visual_verification"]["decision"]
        candidate = metrics["visual_verification_orthogonal_candidate"]
        truth = candidate["evaluation_only_physics"]["raised_and_stable_at_both_frames"]
        if (
            v3 != source["v3_decision"]
            or candidate["decision"] != source["orthogonal_v4_decision"]
            or truth != (source["capture_truth_raised_stable"] == "True")
        ):
            raise RuntimeError(
                f"replay of {source['pair_id']} does not reproduce its saved decision and truth"
            )
        demonstrations.append(
            {
                "case": example["case"],
                "selection_note": example["selection_note"],
                "source_split": "held-out orthogonal verification challenge",
                "source_pair_id": source["pair_id"],
                "stratum": source["stratum"],
                "target_xy_m": [float(value) for value in target_xy],
                "occluder": "actual static MuJoCo rail in original side-camera view; collision disabled",
                "v3_side_decision": v3,
                "v4_side_decision": metrics["visual_verification_gap_tolerant_candidates"][
                    str(FROZEN_GAP_TOLERANCE_PX)
                ]["decision"],
                "v4_two_view_decision": candidate["decision"],
                "capture_truth_raised_stable_at_both_frames": truth,
                "payload_stable_hold_success": metrics["payload_stable_lift_success"],
                "strict_full_cycle_success": metrics["physical_stable_lift_success"],
                "failure_reason": metrics["failure_reason"],
                "video": f"{stem}.mp4",
                "preview": f"{stem}.gif",
                "additional_benchmark_trials": False,
            }
        )
    metadata = {
        "evidence_type": "illustrative replays of counted held-out challenge poses; no additional benchmark trials",
        "decision_inputs": "initial and two fresh RGB frames from each of two perpendicular side cameras",
        "thresholds": {
            "upward_displacement_px": 50,
            "interframe_drift_px": 12,
            "marker_rotation_deg": V3_MAX_MARKER_ROTATION_FROM_START_DEG,
            "marker_gap_tolerance_px": FROZEN_GAP_TOLERANCE_PX,
        },
        "simulator_truth": "read after image decisions; never supplied to either verifier",
        "demonstrations": demonstrations,
    }
    path = output_dir / "cooperative_pick_orthogonal_verification_demos.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate_demos(args.results_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
