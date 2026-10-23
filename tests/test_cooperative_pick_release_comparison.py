import csv
import json

import pytest

from visual_servo_mujoco.release_profile_comparison import compare_release_profiles


FIELDS = (
    "pair_id",
    "controller",
    "target_x_m",
    "target_y_m",
    "payload_stable_lift_success",
    "full_sequence_collision_safe",
    "physical_stable_lift_success",
)


def _write_run(directory, release_profile, *, target_x=0.01, successes=None):
    directory.mkdir()
    summary = {
        "release_profile": release_profile,
        "target_sampling": {"base_seed": 38200, "unique_target_positions": 1},
        "camera_pipeline": {"detector": "default"},
        "yaw_actuators": {"base_yaw": {"kp": 300}},
        "task_success_definition": {"physical": "strict"},
        "gripper_pad_half_size_m": 0.016,
        "contact_attribution": {},
    }
    (directory / "cooperative-pick-pose-validation-summary.json").write_text(
        json.dumps(summary)
    )
    outcomes = successes or {}
    rows = []
    for condition in ("camera_estimate", "oracle_xy"):
        row = {
            "pair_id": "38200-000",
            "controller": condition,
            "target_x_m": str(target_x),
            "target_y_m": "-0.02",
        }
        row.update(
            {
                key: str(outcomes.get((condition, key), False))
                for key in FIELDS[4:]
            }
        )
        rows.append(row)
    with (directory / "cooperative-pick-pose-validation-trials.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_release_profile_comparison_checks_pairing_and_writes_visual_outputs(tmp_path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "clearance"
    _write_run(
        baseline_dir,
        "baseline",
        successes={("camera_estimate", "payload_stable_lift_success"): True},
    )
    _write_run(
        candidate_dir,
        "clearance",
        successes={
            ("camera_estimate", "payload_stable_lift_success"): True,
            ("camera_estimate", "full_sequence_collision_safe"): True,
            ("camera_estimate", "physical_stable_lift_success"): True,
        },
    )
    json_output = tmp_path / "comparison.json"
    svg_output = tmp_path / "comparison.svg"
    paired_csv = tmp_path / "paired.csv"
    result = compare_release_profiles(
        baseline_dir,
        candidate_dir,
        json_output=json_output,
        svg_output=svg_output,
        paired_csv_output=paired_csv,
    )

    assert result["paired_outcomes"]["camera_estimate"][
        "physical_stable_lift_success"
    ]["candidate_only_success"] == 1
    assert result["paired_outcomes"]["camera_estimate"][
        "payload_stable_lift_success"
    ]["both_success"] == 1
    assert result["fresh_visual_confirmation"] == "not measured"
    assert set(result["contact_attribution_by_profile"]) == {"baseline", "clearance"}
    assert json_output.is_file()
    assert "<svg" in svg_output.read_text()
    with paired_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]["baseline_physical_stable_lift_success"] == "False"


def test_release_profile_comparison_rejects_a_changed_target_pose(tmp_path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "clearance"
    _write_run(baseline_dir, "baseline")
    _write_run(candidate_dir, "clearance", target_x=0.02)

    with pytest.raises(ValueError, match="target pose changed"):
        compare_release_profiles(
            baseline_dir,
            candidate_dir,
            json_output=tmp_path / "comparison.json",
            svg_output=tmp_path / "comparison.svg",
            paired_csv_output=tmp_path / "paired.csv",
        )
