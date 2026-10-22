"""Render the development pose that best illustrates the opt-in V4 detector."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from visual_servo_mujoco.camera_guided_cooperative_pick import (
    FINGER_PAD_HALF_SIZE_M,
    run_trial,
)
from visual_servo_mujoco.cooperative_pick_demo import APPROACH_TOOL_Z_OFFSETS_M
from visual_servo_mujoco.cooperative_pick_occlusion_validation import (
    OCCLUSION_FAULT,
    OUTPUT_CSV,
    OUTPUT_JSON,
)
from visual_servo_mujoco.cooperative_pick_visual_verification import (
    V3_MAX_MARKER_ROTATION_FROM_START_DEG,
)


def generate_demos(
    output_dir: Path,
    *,
    development_dir: Path,
) -> dict[str, Any]:
    summary_path = development_dir / OUTPUT_JSON
    trials_path = development_dir / OUTPUT_CSV
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tolerance = summary.get("selected_gap_tolerance_px")
    if not isinstance(tolerance, int):
        raise ValueError("development data did not select an eligible V4 gap tolerance")
    with trials_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    occluded_rows = [
        row for row in rows
        if row["condition"] == "marker_occlusion_50pct"
        and row["truth_raised_stable_at_two_frames"] == "True"
        and row["v3_decision"] != "confirmed"
        and row[f"gap_{tolerance}_px_decision"] == "confirmed"
    ]
    if not occluded_rows:
        raise ValueError("no development pose shows a V4 recovery over V3")
    occluded_row = sorted(occluded_rows, key=lambda row: (int(row["seed"]), int(row["sample_index"])))[0]
    pair_id = occluded_row["pair_id"]
    clean_row = next(
        row for row in rows if row["pair_id"] == pair_id and row["condition"] == "clean"
    )
    target_xy = (float(occluded_row["target_x_m"]), float(occluded_row["target_y_m"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for condition, fault, expected in (
        ("clean", None, clean_row),
        ("marker_occlusion_50pct", OCCLUSION_FAULT, occluded_row),
    ):
        stem = f"cooperative_pick_occlusion_v4_{condition}"
        metrics = run_trial(
            target_xy,
            controller="camera_estimate",
            video_path=output_dir / f"{stem}.mp4",
            preview_path=output_dir / f"{stem}.gif",
            finger_pad_half_size_m=FINGER_PAD_HALF_SIZE_M,
            release_profile="clearance",
            lowest_approach_tool_height_offset_m=APPROACH_TOOL_Z_OFFSETS_M[0],
            visual_verification=True,
            visual_verification_min_upward_displacement_px=50.0,
            visual_verification_max_marker_rotation_from_start_deg=(
                V3_MAX_MARKER_ROTATION_FROM_START_DEG
            ),
            visual_verification_image_fault=fault,
            visual_verification_gap_tolerance_candidates_px=(tolerance,),
        )
        visual = metrics["visual_verification"]
        candidate = metrics["visual_verification_gap_tolerant_candidates"][str(tolerance)]
        if visual["decision"] != expected["v3_decision"] or candidate["decision"] != expected[
            f"gap_{tolerance}_px_decision"
        ]:
            raise RuntimeError(f"illustrative replay no longer matches the development row {pair_id}")
        truth = visual["evaluation_only_physics"]
        cases.append(
            {
                "condition": condition,
                "pair_id": pair_id,
                "target_xy_m": list(target_xy),
                "v3_decision": visual["decision"],
                "v3_reason": visual["reason"],
                "v4_decision": candidate["decision"],
                "v4_reason": candidate["reason"],
                "capture_time_truth_raised_and_stable": truth[
                    "raised_and_stable_at_both_frames"
                ],
                "payload_stable_hold_success": metrics["payload_stable_lift_success"],
                "strict_full_sequence_success": metrics["physical_stable_lift_success"],
                "physical_failure_reason": metrics["failure_reason"],
                "video": f"{stem}.mp4",
                "preview": f"{stem}.gif",
            }
        )
    metadata = {
        "evidence_type": "illustrative replay of two already-counted development rows; not additional benchmark trials",
        "pair_id": pair_id,
        "split": "development",
        "development_base_seed": summary["split"]["base_seed"],
        "candidate_gap_tolerance_px": tolerance,
        "controller": "camera_estimate using initial overhead RGB for planning",
        "decision_inputs": "initial side RGB and two fresh side RGB frames only",
        "fault_scope": "the 50% gray band is applied after render to side-camera RGB; physics, overhead control image, and trajectory are unchanged",
        "cases": cases,
        "limitations": [
            "The clips illustrate one development pose and are not held-out evidence.",
            "The occlusion is a synthetic band and the fiducial is an ideal rendered marker.",
            "V3 remains the default; V4 is an opt-in experimental image measurement.",
        ],
    }
    metadata_path = output_dir / "cooperative_pick_occlusion_v4_demos.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/demos"))
    parser.add_argument(
        "--development-dir",
        type=Path,
        default=Path("results/cooperative-pick-occlusion-development"),
    )
    args = parser.parse_args()
    print(json.dumps(generate_demos(args.output_dir, development_dir=args.development_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
