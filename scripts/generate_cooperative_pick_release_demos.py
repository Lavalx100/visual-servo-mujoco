"""Render matched baseline and clearance-profile release demos for one held-out pose."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from visual_servo_mujoco.camera_guided_cooperative_pick import run_trial


BASELINE_VIDEO = "cooperative_pick_release_baseline.mp4"
BASELINE_GIF = "cooperative_pick_release_baseline.gif"
CLEARANCE_VIDEO = "cooperative_pick_release_clearance.mp4"
CLEARANCE_GIF = "cooperative_pick_release_clearance.gif"
METADATA_FILE = "cooperative_pick_release_pair.json"
TRIALS_FILE = "cooperative-pick-pose-validation-trials.csv"
SUMMARY_FILE = "cooperative-pick-pose-validation-summary.json"


def _read_trials(directory: Path) -> dict[tuple[str, str], dict[str, str]]:
    with (directory / TRIALS_FILE).open(encoding="utf-8", newline="") as stream:
        return {
            (row["pair_id"], row["controller"]): row
            for row in csv.DictReader(stream)
        }


def generate_pair_demos(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    pair_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Replay one paired failure-to-success transition and save both videos."""
    baseline_rows = _read_trials(baseline_dir)
    candidate_rows = _read_trials(candidate_dir)
    key = (pair_id, "camera_estimate")
    if key not in baseline_rows or key not in candidate_rows:
        raise ValueError(f"pose {pair_id} is absent from one of the trial tables")
    baseline_row = baseline_rows[key]
    candidate_row = candidate_rows[key]
    target_xy = (float(baseline_row["target_x_m"]), float(baseline_row["target_y_m"]))
    candidate_xy = (float(candidate_row["target_x_m"]), float(candidate_row["target_y_m"]))
    if target_xy != candidate_xy:
        raise ValueError(f"target pose differs between profiles for {pair_id}")
    if baseline_row["physical_stable_lift_success"] != "False":
        raise ValueError("selected baseline trial must fail the strict physical score")
    if candidate_row["physical_stable_lift_success"] != "True":
        raise ValueError("selected clearance trial must pass the strict physical score")

    pad_size = float(
        json.loads((candidate_dir / SUMMARY_FILE).read_text())["gripper_pad_half_size_m"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = run_trial(
        target_xy,
        controller="camera_estimate",
        video_path=output_dir / BASELINE_VIDEO,
        preview_path=output_dir / BASELINE_GIF,
        finger_pad_half_size_m=pad_size,
        release_profile="baseline",
    )
    clearance = run_trial(
        target_xy,
        controller="camera_estimate",
        video_path=output_dir / CLEARANCE_VIDEO,
        preview_path=output_dir / CLEARANCE_GIF,
        finger_pad_half_size_m=pad_size,
        release_profile="clearance",
    )
    if baseline["physical_stable_lift_success"] or not clearance[
        "physical_stable_lift_success"
    ]:
        raise RuntimeError("rendered replay did not reproduce the recorded paired outcomes")

    metadata = {
        "evidence_type": "illustrative replay of one paired held-out MuJoCo pose",
        "pair_id": pair_id,
        "target_xy_m": list(target_xy),
        "controller": "camera_estimate",
        "finger_pad_half_size_m": pad_size,
        "baseline_recorded_strict_success": baseline_row[
            "physical_stable_lift_success"
        ],
        "clearance_recorded_strict_success": candidate_row[
            "physical_stable_lift_success"
        ],
        "baseline_replay_metrics": baseline,
        "clearance_replay_metrics": clearance,
        "videos": [BASELINE_VIDEO, CLEARANCE_VIDEO],
        "previews": [BASELINE_GIF, CLEARANCE_GIF],
        "overhead_video_display": (
            "Live overhead frames are rendered for video display; control still "
            "uses only the initial image."
        ),
        "fresh_visual_confirmation": "not measured",
        "aggregate_counting": "This replay is already part of the held-out 30-pose results; do not add it as another trial.",
    }
    (output_dir / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = generate_pair_demos(
        args.baseline_dir,
        args.candidate_dir,
        pair_id=args.pair_id,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
