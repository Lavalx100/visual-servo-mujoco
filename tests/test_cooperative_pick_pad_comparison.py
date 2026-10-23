import csv
import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compare_cooperative_pick_pad_sensitivity.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "compare_cooperative_pick_pad_sensitivity", _SCRIPT_PATH
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_COMPARISON = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_COMPARISON)
_paired_outcomes = _COMPARISON._paired_outcomes
_wilson_interval = _COMPARISON._wilson_interval
compare_pad_runs = _COMPARISON.compare_pad_runs


def _write_run(
    directory,
    pad_half_size,
    target_x="0.0123456789012345",
    camera_success="True",
):
    directory.mkdir()
    (directory / "cooperative-pick-pose-validation-summary.json").write_text(
        json.dumps(
            {
                "gripper_pad_half_size_m": pad_half_size,
                "target_sampling": {"base_seed": 38100},
                "camera_pipeline": {"camera": "overhead"},
                "yaw_actuators": {"enabled": True},
                "task_success_definition": {"stable_lift_hold_s": 1.6},
            }
        )
    )
    fields = [
        "pair_id",
        "controller",
        "target_x_m",
        "target_y_m",
        "physical_stable_lift_success",
        "initial_target_projection_error_m",
    ]
    with (directory / "cooperative-pick-pose-validation-trials.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for condition, success in (
            ("camera_estimate", camera_success),
            ("oracle_xy", "False"),
        ):
            writer.writerow(
                {
                    "pair_id": "seed38100-sample000",
                    "controller": condition,
                    "target_x_m": target_x,
                    "target_y_m": "-0.0234567890123456",
                    "physical_stable_lift_success": success,
                    "initial_target_projection_error_m": "0.00048",
                }
            )


def test_wilson_interval_matches_the_reported_small_sample_uncertainty():
    assert _wilson_interval(4, 30) == [0.0531, 0.2968]


def test_paired_outcomes_keep_candidate_only_and_baseline_only_counts_separate():
    baseline = {
        "a": {"physical_stable_lift_success": "True"},
        "b": {"physical_stable_lift_success": "False"},
    }
    candidate = {
        "a": {"physical_stable_lift_success": "False"},
        "b": {"physical_stable_lift_success": "True"},
    }

    assert _paired_outcomes(baseline, candidate) == {
        "both_success": 0,
        "candidate_only_success": 1,
        "baseline_only_success": 1,
        "both_failure": 0,
    }


def test_pad_comparison_requires_identical_full_precision_target_poses(tmp_path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_run(baseline_dir, 0.014)
    _write_run(candidate_dir, 0.016, target_x="0.0123456789012346")

    with pytest.raises(ValueError, match="target pose changed"):
        compare_pad_runs(
            baseline_dir,
            candidate_dir,
            json_output=tmp_path / "comparison.json",
            svg_output=tmp_path / "comparison.svg",
        )


def test_pad_comparison_emits_summary_and_scorecard_for_exactly_paired_trials(tmp_path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_run(baseline_dir, 0.014)
    _write_run(candidate_dir, 0.016, camera_success="False")

    summary = compare_pad_runs(
        baseline_dir,
        candidate_dir,
        json_output=tmp_path / "comparison.json",
        svg_output=tmp_path / "comparison.svg",
    )

    assert summary["unique_target_positions"] == 1
    assert summary["paired_success_outcomes"]["camera_estimate"] == {
        "both_success": 0,
        "candidate_only_success": 0,
        "baseline_only_success": 1,
        "both_failure": 0,
    }
    assert (tmp_path / "comparison.json").exists()
    assert "Fresh visual confirmation was not measured" in (
        tmp_path / "comparison.svg"
    ).read_text()
