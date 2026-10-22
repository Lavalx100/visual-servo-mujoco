"""Compare same-seed cooperative-pick runs with two finger-pad sizes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


CONDITIONS = ("camera_estimate", "oracle_xy")


def _read_trials(directory: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = directory / "cooperative-pick-pose-validation-trials.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    trials = {}
    for row in rows:
        key = (row["pair_id"], row["controller"])
        if key in trials:
            raise ValueError(f"duplicate trial row: {key}")
        trials[key] = row
    return trials


def _wilson_interval(successes: int, trials: int, *, z: float = 1.96) -> list[float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half_width = (
        z
        * ((proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) ** 0.5)
        / denominator
    )
    return [round(max(0.0, center - half_width), 4), round(min(1.0, center + half_width), 4)]


def _paired_outcomes(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> dict[str, int]:
    outcomes = {
        "both_success": 0,
        "candidate_only_success": 0,
        "baseline_only_success": 0,
        "both_failure": 0,
    }
    if set(baseline) != set(candidate):
        raise ValueError("pad-size conditions must contain the same paired trial ids")
    for pair_id, baseline_row in baseline.items():
        baseline_success = baseline_row["physical_stable_lift_success"] == "True"
        candidate_success = candidate[pair_id]["physical_stable_lift_success"] == "True"
        if baseline_success and candidate_success:
            outcomes["both_success"] += 1
        elif candidate_success:
            outcomes["candidate_only_success"] += 1
        elif baseline_success:
            outcomes["baseline_only_success"] += 1
        else:
            outcomes["both_failure"] += 1
    return outcomes


def compare_pad_runs(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    json_output: Path,
    svg_output: Path,
) -> dict[str, Any]:
    """Validate exact split pairing and write outcome statistics and a scorecard."""
    baseline_summary = json.loads(
        (baseline_dir / "cooperative-pick-pose-validation-summary.json").read_text()
    )
    candidate_summary = json.loads(
        (candidate_dir / "cooperative-pick-pose-validation-summary.json").read_text()
    )
    baseline_rows = _read_trials(baseline_dir)
    candidate_rows = _read_trials(candidate_dir)
    if set(baseline_rows) != set(candidate_rows):
        raise ValueError("pad-size runs must contain the same condition/pair rows")
    pair_ids = sorted({pair_id for pair_id, _ in baseline_rows})
    if not pair_ids:
        raise ValueError("pad-size runs must contain at least one paired pose")
    expected_rows = {(pair_id, condition) for pair_id in pair_ids for condition in CONDITIONS}
    if set(baseline_rows) != expected_rows:
        raise ValueError("each pad-size run must include both conditions for every pose")
    for field in (
        "target_sampling",
        "camera_pipeline",
        "yaw_actuators",
        "task_success_definition",
        "release_profile",
    ):
        if baseline_summary.get(field) != candidate_summary.get(field):
            raise ValueError(f"pad-size runs differ in non-pad experiment setting: {field}")
    if float(baseline_summary.get("wrist_center_radial_standoff_m", 0.0)) != float(
        candidate_summary.get("wrist_center_radial_standoff_m", 0.0)
    ):
        raise ValueError("pad-size runs must use the same wrist-center standoff")
    if float(baseline_summary.get("finger_pad_inward_offset_m", 0.022)) != float(
        candidate_summary.get("finger_pad_inward_offset_m", 0.022)
    ):
        raise ValueError("pad-size runs must use the same inward pad offset")
    for pair_id in pair_ids:
        for condition in CONDITIONS:
            base_row = baseline_rows[pair_id, condition]
            candidate_row = candidate_rows[pair_id, condition]
            base_pose = (float(base_row["target_x_m"]), float(base_row["target_y_m"]))
            candidate_pose = (
                float(candidate_row["target_x_m"]),
                float(candidate_row["target_y_m"]),
            )
            if not all(math.isfinite(value) for value in (*base_pose, *candidate_pose)):
                raise ValueError(f"non-finite target pose for {pair_id} / {condition}")
            if base_pose != candidate_pose:
                raise ValueError(f"target pose changed for {pair_id} / {condition}")

    baseline_pad = float(baseline_summary["gripper_pad_half_size_m"])
    candidate_pad = float(candidate_summary["gripper_pad_half_size_m"])
    baseline_conditions = []
    candidate_conditions = []
    paired = {}
    for condition in CONDITIONS:
        base_by_pair = {
            pair_id: baseline_rows[pair_id, condition] for pair_id in pair_ids
        }
        candidate_by_pair = {
            pair_id: candidate_rows[pair_id, condition] for pair_id in pair_ids
        }
        for rows, output, pad in (
            (base_by_pair, baseline_conditions, baseline_pad),
            (candidate_by_pair, candidate_conditions, candidate_pad),
        ):
            successes = sum(
                row["physical_stable_lift_success"] == "True" for row in rows.values()
            )
            errors = sorted(
                float(row["initial_target_projection_error_m"])
                for row in rows.values()
                if row["initial_target_projection_error_m"]
            )
            if errors:
                middle = len(errors) // 2
                median_error = (
                    errors[middle]
                    if len(errors) % 2
                    else (errors[middle - 1] + errors[middle]) / 2
                )
            else:
                median_error = None
            output.append(
                {
                    "controller": condition,
                    "pad_half_size_m": pad,
                    "trials": len(rows),
                    "physical_stable_lift_successes": successes,
                    "physical_stable_lift_success_rate": round(
                        successes / len(rows), 4
                    ),
                    "wilson_95_interval": _wilson_interval(successes, len(rows)),
                    "median_initial_target_projection_error_cm": round(
                        median_error * 100, 3
                    ) if median_error is not None else None,
                }
            )
        paired[condition] = _paired_outcomes(base_by_pair, candidate_by_pair)

    summary = {
        "experiment": "same-pose finger-pad-size comparison",
        "evidence_type": "paired MuJoCo simulator comparison; not hardware evidence",
        "unique_target_positions": len(pair_ids),
        "target_split": candidate_summary["target_sampling"],
        "baseline": {
            "pad_half_size_m": baseline_pad,
            "results_directory": str(baseline_dir),
            "conditions": baseline_conditions,
        },
        "candidate": {
            "pad_half_size_m": candidate_pad,
            "wrist_center_radial_standoff_m": candidate_summary.get(
                "wrist_center_radial_standoff_m", 0.0
            ),
            "results_directory": str(candidate_dir),
            "conditions": candidate_conditions,
        },
        "paired_success_outcomes": paired,
        "fresh_visual_confirmation": "not measured",
        "interpretation_limits": [
            "The candidate pad size was selected after a development sweep; this same-seed comparison estimates its effect on this held-out pose set.",
            "Thirty poses are a small simulator sample; Wilson intervals show substantial uncertainty.",
            "Success is a physical stable lift under the stated thresholds, not fresh visual confirmation.",
        ],
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    svg_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    _write_svg(svg_output, summary)
    return summary


def _write_svg(path: Path, summary: dict[str, Any]) -> None:
    conditions = {row["controller"]: row for row in summary["baseline"]["conditions"]}
    candidates = {row["controller"]: row for row in summary["candidate"]["conditions"]}
    count = summary["unique_target_positions"]
    chart_x, chart_w = 310, 600
    y_positions = {"camera_estimate": 170, "oracle_xy": 320}
    bar_colors = ("#8496a8", "#49c9ab")
    bars = []
    for condition, label in (("camera_estimate", "Camera estimate"), ("oracle_xy", "Oracle XY diagnostic")):
        y = y_positions[condition]
        bars.append(
            f'<text x="54" y="{y + 31}" fill="#edf3f7" font-family="sans-serif" font-size="17">{label}</text>'
        )
        for index, row in enumerate((conditions[condition], candidates[condition])):
            top = y + index * 48
            width = round(chart_w * row["physical_stable_lift_success_rate"])
            bars.append(
                f'<text x="180" y="{top + 25}" fill="#b6c4d3" font-family="sans-serif" font-size="14">{row["pad_half_size_m"] * 1000:.0f} mm</text>'
            )
            bars.append(f'<rect x="{chart_x}" y="{top}" width="{chart_w}" height="34" rx="7" fill="#253343"/>')
            bars.append(
                f'<rect x="{chart_x}" y="{top}" width="{width}" height="34" rx="7" fill="{bar_colors[index]}"/>'
            )
            bars.append(
                f'<text x="{chart_x + chart_w + 16}" y="{top + 24}" fill="#f2f6fa" font-family="sans-serif" font-size="15" font-weight="700">{row["physical_stable_lift_successes"]}/{count}</text>'
            )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="530" viewBox="0 0 1040 530">
<rect width="100%" height="100%" fill="#111923"/>
<text x="54" y="54" fill="#f2f6fa" font-family="sans-serif" font-size="26" font-weight="700">Finger pad size: same-pose comparison</text>
<text x="54" y="88" fill="#b6c4d3" font-family="sans-serif" font-size="16">{count} exact target poses · same camera and oracle conditions · held-out seed set</text>
{''.join(bars)}
<text x="54" y="455" fill="#b6c4d3" font-family="sans-serif" font-size="15">Median target localization error is unchanged between pad settings; this comparison changes contact geometry.</text>
<text x="54" y="483" fill="#f4c976" font-family="sans-serif" font-size="15">Physical stable-lift outcome only. Fresh visual confirmation was not measured.</text>
<text x="{chart_x}" y="515" fill="#8e9cab" font-family="sans-serif" font-size="12">0</text>
<text x="{chart_x + chart_w - 15}" y="515" fill="#8e9cab" font-family="sans-serif" font-size="12">100%</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--svg-output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_pad_runs(
        args.baseline_dir,
        args.candidate_dir,
        json_output=args.json_output,
        svg_output=args.svg_output,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
