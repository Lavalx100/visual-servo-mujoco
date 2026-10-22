"""Render one already-counted pose under every frozen side-camera image fault."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from visual_servo_mujoco.camera_guided_cooperative_pick import (
    FINGER_PAD_HALF_SIZE_M,
    _sample_positions,
    run_trial,
)
from visual_servo_mujoco.cooperative_pick_demo import APPROACH_TOOL_Z_OFFSETS_M
from visual_servo_mujoco.cooperative_pick_visual_fault_validation import (
    DEFAULT_BASE_SEED,
    FAULT_CONDITIONS,
    RELEASE_PROFILE,
)
from visual_servo_mujoco.cooperative_pick_visual_verification import (
    V3_MAX_MARKER_ROTATION_FROM_START_DEG,
)


def generate_demos(output_dir: Path, *, base_seed: int = DEFAULT_BASE_SEED) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    validation_dir = repository / "results" / "cooperative-pick-visual-fault-validation"
    summary_path = validation_dir / "cooperative-pick-visual-fault-summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"run the visual fault validation first: {summary_path}")
    split = json.loads(summary_path.read_text(encoding="utf-8"))["split"]
    if split["base_seed"] != base_seed:
        raise ValueError("demo seed must match the saved validation split")
    positions = _sample_positions(seeds=1, samples_per_seed=1, base_seed=base_seed)
    seed, sample_index, target_xy = positions[0]
    pair_id = f"{seed}-{sample_index:03d}"
    trials_path = validation_dir / "cooperative-pick-visual-fault-trials.csv"
    with trials_path.open(encoding="utf-8", newline="") as stream:
        recorded_pairs = {row["pair_id"] for row in csv.DictReader(stream)}
    if pair_id not in recorded_pairs:
        raise ValueError(f"selected demo pose {pair_id} is not in the saved validation table")
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for fault_name, fault in FAULT_CONDITIONS:
        stem = f"cooperative_pick_visual_fault_{fault_name}"
        metrics = run_trial(
            target_xy,
            controller="camera_estimate",
            video_path=output_dir / f"{stem}.mp4",
            preview_path=output_dir / f"{stem}.gif",
            finger_pad_half_size_m=FINGER_PAD_HALF_SIZE_M,
            release_profile=RELEASE_PROFILE,
            lowest_approach_tool_height_offset_m=APPROACH_TOOL_Z_OFFSETS_M[0],
            visual_verification=True,
            visual_verification_min_upward_displacement_px=50.0,
            visual_verification_max_marker_rotation_from_start_deg=(
                V3_MAX_MARKER_ROTATION_FROM_START_DEG
            ),
            visual_verification_image_fault=fault,
        )
        visual = metrics["visual_verification"]
        truth = visual["evaluation_only_physics"]
        cases.append(
            {
                "fault_condition": fault_name,
                "fault_specification": fault,
                "pair_id": pair_id,
                "target_xy_m": list(target_xy),
                "decision": visual["decision"],
                "decision_reason": visual["reason"],
                "truth_raised_stable_at_two_frames": truth[
                    "raised_and_stable_at_both_frames"
                ],
                "payload_stable_hold_success": metrics[
                    "payload_stable_lift_success"
                ],
                "strict_full_sequence_success": metrics[
                    "physical_stable_lift_success"
                ],
                "physical_failure_reason": metrics["failure_reason"],
                "videos": [f"{stem}.mp4"],
                "previews": [f"{stem}.gif"],
            }
        )
    metadata = {
        "evidence_type": "illustrative same-pose replays already counted in the 39500 held-out dataset",
        "pair_id": pair_id,
        "source_table": trials_path.relative_to(repository).as_posix(),
        "controller": "camera_estimate with initial overhead RGB planning",
        "verifier": "frozen V3; 50 px rise, 12 px drift, 4 degree marker rotation",
        "faults_apply_to": "side-camera verification RGB only, after simulation capture; not MuJoCo physics or the overhead controller image",
        "cases": cases,
        "reproduction": "uv run --locked --extra demo python scripts/generate_cooperative_pick_visual_fault_demos.py",
        "limitations": [
            "One selected pose is illustrative and does not replace the aggregate 60-pose results.",
            "All image faults are authored digital transformations of synthetic MuJoCo RGB.",
        ],
    }
    (output_dir / "cooperative_pick_visual_fault_demos.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/demos"))
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    args = parser.parse_args()
    print(json.dumps(generate_demos(args.output_dir, base_seed=args.base_seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
