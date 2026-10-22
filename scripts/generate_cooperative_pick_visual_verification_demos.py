"""Render previously counted false-confirmation cases and the gated replays."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from visual_servo_mujoco.camera_guided_cooperative_pick import run_trial


V0_DATA = Path("results/cooperative-pick-visual-confirmation-v0-heldout")
DEVELOPMENT_DATA = Path("results/cooperative-pick-visual-confirmation-development")
OUTPUT_FILES = {
    "tilted_no_grasp": "cooperative_pick_verification_false_tilt",
    "under_lift": "cooperative_pick_verification_false_underlift",
}


def _read_row(path: Path, pair_id: str, *, condition: str) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    matches = [
        row
        for row in rows
        if row["pair_id"] == pair_id
        and row.get("condition", row.get("controller")) == condition
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {condition} row for {pair_id} in {path}")
    return matches[0]


def _render_case(
    *,
    case_name: str,
    pair_id: str,
    source_csv: Path,
    source_condition: str,
    output_dir: Path,
    expected_v0: str,
    expected_v1: str,
    expected_v2: str,
    expected_v3: str,
    expected_failure: str,
) -> dict[str, Any]:
    recorded = _read_row(source_csv, pair_id, condition=source_condition)
    source_v0 = recorded.get(
        "translation_only_decision",
        recorded.get("fresh_visual_confirmation", recorded.get("decision")),
    )
    if source_v0 != expected_v0 or recorded["failure_reason"] != expected_failure:
        raise RuntimeError(f"archived source outcome no longer matches the {case_name} example")

    target_xy = (float(recorded["target_x_m"]), float(recorded["target_y_m"]))
    output_stem = OUTPUT_FILES[case_name]
    metrics = run_trial(
        target_xy,
        controller="camera_estimate",
        video_path=output_dir / f"{output_stem}.mp4",
        preview_path=output_dir / f"{output_stem}.gif",
        release_profile="clearance",
        lowest_approach_tool_height_offset_m=0.005,
        visual_verification=True,
        visual_verification_min_upward_displacement_px=50.0,
        visual_verification_max_marker_rotation_from_start_deg=4.0,
    )
    v0 = metrics["visual_verification_translation_only_baseline"]["decision"]
    v1 = metrics["visual_verification_orientation_only_baseline"]["decision"]
    v2 = metrics["visual_verification_rise_orientation_baseline"]["decision"]
    v3 = metrics["visual_verification"]["decision"]
    if (v0, v1, v2, v3) != (expected_v0, expected_v1, expected_v2, expected_v3):
        raise RuntimeError(
            f"{case_name} replay decisions changed: V0/V1/V2/V3={(v0, v1, v2, v3)}"
        )
    if metrics["failure_reason"] != expected_failure:
        raise RuntimeError(f"{case_name} replay no longer reproduces {expected_failure}")

    physics = metrics["visual_verification"]["evaluation_only_physics"]
    repository = Path(__file__).resolve().parents[1]
    return {
        "evidence_type": "illustrative replay of a previously counted classifier failure; not new benchmark data",
        "case": case_name,
        "source_pair_id": pair_id,
        "source_table": source_csv.relative_to(repository).as_posix(),
        "target_xy_m": list(target_xy),
        "controller": "camera_estimate",
        "approach_offset_m": 0.005,
        "release_profile": "clearance",
        "image_decisions_on_replay": {"V0_translation_only": v0, "V1_orientation_gate": v1, "V2_orientation_plus_50px_rise": v2, "V3_tighter_orientation_gate": v3},
        "physical_failure_reason": metrics["failure_reason"],
        "capture_time_truth": physics["raised_and_stable_at_both_frames"],
        "payload_stable_hold_success": metrics["payload_stable_lift_success"],
        "strict_full_sequence_success": metrics["physical_stable_lift_success"],
        "capture_tilt_deg": physics["payload_tilt_deg_at_capture"],
        "capture_lift_m": physics["payload_lift_m_at_capture"],
        "videos": [f"{output_stem}.mp4"],
        "previews": [f"{output_stem}.gif"],
    }


def generate_demos(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[1]
    v0_csv = repository / V0_DATA / "cooperative-pick-visual-confirmation-trials.csv"
    development_csv = repository / DEVELOPMENT_DATA / "cooperative-pick-visual-confirmation-trials.csv"
    cases = [
        _render_case(
            case_name="tilted_no_grasp",
            pair_id="39100-004",
            source_csv=v0_csv,
            source_condition="baseline_5mm",
            output_dir=output_dir,
            expected_v0="confirmed",
            expected_v1="not_confirmed",
            expected_v2="not_confirmed",
            expected_v3="not_confirmed",
            expected_failure="no_grasp_contact",
        ),
        _render_case(
            case_name="under_lift",
            pair_id="39200-018",
            source_csv=development_csv,
            source_condition="baseline_5mm",
            output_dir=output_dir,
            expected_v0="confirmed",
            expected_v1="confirmed",
            expected_v2="not_confirmed",
            expected_v3="not_confirmed",
            expected_failure="insufficient_lift",
        ),
    ]
    metadata = {
        "evidence_type": "illustrative development-error replays; each pose remains counted only in its original dataset",
        "cases": cases,
        "visual_verification": {
            "frame_times_s": [0.0, 8.4, 8.8],
            "V0": "35 px rise and 12 px inter-frame drift",
            "V1": "V0 plus no more than 12 degrees fiducial rotation",
            "V2": "V1 with at least 50 px rise in both fresh frames",
            "V3": "V2 plus no more than 4 degrees fiducial rotation",
            "simulator_state_is_not_an_input_to_any_image_decision": True,
        },
        "limitations": [
            "Both clips replay selected failures from development or the first validation and are not new evaluation trials.",
            "The side camera, fiducial, image conditions, and physics are synthetic.",
        ],
    }
    (output_dir / "cooperative_pick_visual_verification_failure_demos.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/demos"))
    args = parser.parse_args()
    print(json.dumps(generate_demos(args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
