"""Render representative same-pose clips from the frozen occlusion challenge."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from visual_servo_mujoco.camera_guided_cooperative_pick import run_trial
from visual_servo_mujoco.cooperative_pick_occlusion_challenge import (
    FROZEN_GAP_TOLERANCE_PX,
    OUTPUT_CSV,
    OUTPUT_JSON,
)
from visual_servo_mujoco.cooperative_pick_visual_verification import (
    V3_MAX_MARKER_ROTATION_FROM_START_DEG,
)


def _select_cases(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_pose: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_pose.setdefault(row["pair_id"], {})[row["image_condition"]] = row

    nominal_positives = []
    hard_negatives = []
    for pair_id, conditions in by_pose.items():
        clean = conditions["clean"]
        row = dict(clean)
        row["pair_id"] = pair_id
        if row["stratum"] == "nominal_pad_16mm" and row["truth_raised_stable_at_two_frames"] == "True":
            recoveries = [
                candidate
                for name, candidate in conditions.items()
                if name != "clean"
                and candidate["truth_raised_stable_at_two_frames"] == "True"
                and candidate["v3_decision"] != "confirmed"
                and candidate["v4_decision"] == "confirmed"
            ]
            misses = [
                candidate
                for name, candidate in conditions.items()
                if name != "clean"
                and candidate["truth_raised_stable_at_two_frames"] == "True"
                and candidate["v4_decision"] != "confirmed"
            ]
            nominal_positives.append((row, recoveries, misses, conditions))
        elif row["stratum"] == "small_pad_14mm_hard_negative" and row[
            "truth_raised_stable_at_two_frames"
        ] == "False":
            false_confirms = [
                candidate
                for name, candidate in conditions.items()
                if candidate["v4_decision"] == "confirmed"
                and candidate["truth_raised_stable_at_two_frames"] == "False"
            ]
            safe_rejections = [
                candidate
                for name, candidate in conditions.items()
                if candidate["v4_decision"] == "not_confirmed"
            ]
            hard_negatives.append((row, false_confirms, safe_rejections, conditions))

    if not nominal_positives or not hard_negatives:
        raise ValueError("challenge data must contain a nominal positive and a hard negative")
    nominal_positives.sort(
        key=lambda item: (
            -len(item[1]),
            len(item[2]),
            int(item[0]["seed"]),
            int(item[0]["sample_index"]),
        )
    )
    hard_negatives.sort(
        key=lambda item: (
            -len(item[1]),
            -len(item[2]),
            int(item[0]["seed"]),
            int(item[0]["sample_index"]),
        )
    )
    _positive, recoveries, misses, positive_conditions = nominal_positives[0]
    _negative, _false_confirms, safe_rejections, negative_conditions = hard_negatives[0]

    recovery = sorted(recoveries, key=lambda item: item["image_condition"])[0] if recoveries else None
    missed_case = sorted(misses, key=lambda item: item["image_condition"])[0] if misses else None
    positive_mask_case = recovery or missed_case or positive_conditions["clean"]
    all_negative_false_confirms = [
        candidate
        for _, pose_false_confirms, _, _ in hard_negatives
        for candidate in pose_false_confirms
    ]
    clean_false_confirms = [
        row for row in all_negative_false_confirms if row["image_condition"] == "clean"
    ]
    mask_false_confirms = [
        row for row in all_negative_false_confirms if row["image_condition"] != "clean"
    ]
    mask_priority = {
        "horizontal_75_center": 0,
        "diagonal_25_plus45": 1,
        "diagonal_25_minus45": 2,
        "horizontal_50_upper": 3,
        "horizontal_50_lower": 4,
        "horizontal_25_center": 5,
        "vertical_30_center": 6,
        "patch_45_upper_left": 7,
    }
    clean_false_case = (
        sorted(clean_false_confirms, key=lambda row: (int(row["seed"]), int(row["sample_index"])))[0]
        if clean_false_confirms
        else negative_conditions["clean"]
    )
    mask_false_case = (
        sorted(
            mask_false_confirms,
            key=lambda row: (
                mask_priority.get(row["image_condition"], 99),
                int(row["seed"]),
                int(row["sample_index"]),
            ),
        )[0]
        if mask_false_confirms
        else sorted(safe_rejections, key=lambda row: row["image_condition"])[0]
        if safe_rejections
        else negative_conditions["clean"]
    )
    return [
        {"case": "clean_nominal_positive", "source": positive_conditions["clean"]},
        {
            "case": "nominal_positive_mask",
            "source": positive_mask_case,
            "selection_note": (
                "V4 recovered a positive rejected by V3"
                if recovery
                else "best available positive mask case; V4 recovery was not observed"
            ),
        },
        {
            "case": "hard_negative_clean_false_confirm",
            "source": clean_false_case,
            "selection_note": (
                "retained V3/V4 clean-frame false-confirmation example"
                if clean_false_confirms
                else "hard-negative clean reference; no false confirmation selected"
            ),
        },
        {
            "case": "hard_negative_occlusion_false_confirm",
            "source": mask_false_case,
            "selection_note": (
                "retained masked V4 false-confirmation example"
                if mask_false_confirms
                else "hard-negative mask selected with V4 not-confirmed outcome"
                if safe_rejections
                else "no usable hard-negative mask example was present"
            ),
        },
    ]


def generate_demos(
    output_dir: Path,
    *,
    results_dir: Path,
) -> dict[str, Any]:
    summary = json.loads((results_dir / OUTPUT_JSON).read_text(encoding="utf-8"))
    with (results_dir / OUTPUT_CSV).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = _select_cases(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    demonstrations = []
    for item in selected:
        source = item["source"]
        condition = source["image_condition"]
        fault = summary["image_conditions"][condition]["fault"]
        stratum = source["stratum"]
        target_xy = (float(source["target_x_m"]), float(source["target_y_m"]))
        stem = f"cooperative_pick_occlusion_shape_{item['case']}"
        metrics = run_trial(
            target_xy,
            controller="camera_estimate",
            video_path=output_dir / f"{stem}.mp4",
            preview_path=output_dir / f"{stem}.gif",
            finger_pad_half_size_m=float(source["finger_pad_half_size_m"]),
            release_profile="clearance",
            visual_verification=True,
            visual_verification_min_upward_displacement_px=50.0,
            visual_verification_max_marker_rotation_from_start_deg=(
                V3_MAX_MARKER_ROTATION_FROM_START_DEG
            ),
            visual_verification_image_fault=fault,
            visual_verification_gap_tolerance_candidates_px=(FROZEN_GAP_TOLERANCE_PX,),
        )
        candidate = metrics["visual_verification_gap_tolerant_candidates"][
            str(FROZEN_GAP_TOLERANCE_PX)
        ]
        truth = metrics["visual_verification"]["evaluation_only_physics"]
        if (
            metrics["visual_verification"]["decision"] != source["v3_decision"]
            or candidate["decision"] != source["v4_decision"]
            or truth["raised_and_stable_at_both_frames"]
            != (source["truth_raised_stable_at_two_frames"] == "True")
        ):
            raise RuntimeError(
                f"illustrative replay disagrees with counted challenge row {source['pair_id']} / {condition}"
            )
        demonstrations.append(
            {
                "case": item["case"],
                "selection_note": item.get("selection_note", "clean reference case"),
                "source_split": "held-out challenge",
                "source_pair_id": source["pair_id"],
                "stratum": stratum,
                "image_condition": condition,
                "fault": fault,
                "target_xy_m": list(target_xy),
                "v3_decision": source["v3_decision"],
                "v4_decision": source["v4_decision"],
                "capture_time_truth_raised_and_stable": truth[
                    "raised_and_stable_at_both_frames"
                ],
                "payload_stable_hold_success": metrics["payload_stable_lift_success"],
                "strict_full_sequence_success": metrics["physical_stable_lift_success"],
                "failure_reason": metrics["failure_reason"],
                "video": f"{stem}.mp4",
                "preview": f"{stem}.gif",
                "additional_benchmark_trials": False,
            }
        )
    metadata = {
        "evidence_type": "illustrative same-pose replays of rows already counted in the challenge CSV",
        "candidate_gap_tolerance_px": FROZEN_GAP_TOLERANCE_PX,
        "decision_inputs": "initial side-camera RGB and two fresh side-camera RGB frames only",
        "simulator_truth": "reported after the visual decision; not an image-verifier input",
        "demonstrations": demonstrations,
        "limitations": summary["limitations"],
    }
    path = output_dir / "cooperative_pick_occlusion_shape_demos.json"
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/demos"))
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/cooperative-pick-occlusion-shape-challenge"),
    )
    args = parser.parse_args()
    print(json.dumps(generate_demos(args.output_dir, results_dir=args.results_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
