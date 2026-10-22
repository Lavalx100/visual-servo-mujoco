"""Compare paired cooperative-pick baseline and clearance release trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


CONDITIONS = ("camera_estimate", "oracle_xy")
OUTCOMES = (
    "payload_stable_lift_success",
    "full_sequence_collision_safe",
    "physical_stable_lift_success",
)
FILES = {
    "trials": "cooperative-pick-pose-validation-trials.csv",
    "summary": "cooperative-pick-pose-validation-summary.json",
}


def _read_trials(directory: Path) -> dict[tuple[str, str], dict[str, str]]:
    with (directory / FILES["trials"]).open(encoding="utf-8", newline="") as stream:
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
        * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials**2))
        / denominator
    )
    return [
        round(max(0.0, center - half_width), 4),
        round(min(1.0, center + half_width), 4),
    ]


def _paired_outcomes(
    baseline_rows: dict[str, dict[str, str]],
    candidate_rows: dict[str, dict[str, str]],
    outcome: str,
) -> dict[str, int]:
    counts = {
        "both_success": 0,
        "candidate_only_success": 0,
        "baseline_only_success": 0,
        "both_failure": 0,
    }
    for pair_id in baseline_rows:
        baseline_success = baseline_rows[pair_id][outcome] == "True"
        candidate_success = candidate_rows[pair_id][outcome] == "True"
        if baseline_success and candidate_success:
            counts["both_success"] += 1
        elif candidate_success:
            counts["candidate_only_success"] += 1
        elif baseline_success:
            counts["baseline_only_success"] += 1
        else:
            counts["both_failure"] += 1
    return counts


def compare_release_profiles(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    json_output: Path,
    svg_output: Path,
    paired_csv_output: Path,
) -> dict[str, Any]:
    """Check exact pose pairing and summarize release-profile outcomes."""
    baseline_summary = json.loads((baseline_dir / FILES["summary"]).read_text())
    candidate_summary = json.loads((candidate_dir / FILES["summary"]).read_text())
    baseline_rows = _read_trials(baseline_dir)
    candidate_rows = _read_trials(candidate_dir)
    if set(baseline_rows) != set(candidate_rows):
        raise ValueError("release profiles must contain the same condition/pair rows")
    pair_ids = sorted({pair_id for pair_id, _ in baseline_rows})
    if not pair_ids:
        raise ValueError("release-profile runs must contain at least one paired pose")
    expected_rows = {(pair_id, condition) for pair_id in pair_ids for condition in CONDITIONS}
    if set(baseline_rows) != expected_rows:
        raise ValueError("each release-profile run must include both conditions per pose")
    for field in (
        "target_sampling",
        "camera_pipeline",
        "yaw_actuators",
        "task_success_definition",
        "gripper_pad_half_size_m",
        "wrist_center_radial_standoff_m",
        "finger_pad_inward_offset_m",
    ):
        if baseline_summary.get(field) != candidate_summary.get(field):
            raise ValueError(f"release-profile runs differ in non-profile setting: {field}")
    if baseline_summary.get("release_profile") != "baseline":
        raise ValueError("baseline input must use release_profile='baseline'")
    if candidate_summary.get("release_profile") != "clearance":
        raise ValueError("candidate input must use release_profile='clearance'")

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

    comparisons: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for condition in CONDITIONS:
        baseline_by_pair = {
            pair_id: baseline_rows[pair_id, condition] for pair_id in pair_ids
        }
        candidate_by_pair = {
            pair_id: candidate_rows[pair_id, condition] for pair_id in pair_ids
        }
        paired[condition] = {}
        comparisons[condition] = {}
        for label, source_rows in (
            ("baseline", baseline_by_pair),
            ("clearance", candidate_by_pair),
        ):
            comparisons[condition][label] = {}
            for outcome in OUTCOMES:
                successes = sum(row[outcome] == "True" for row in source_rows.values())
                comparisons[condition][label][outcome] = {
                    "successes": successes,
                    "trials": len(pair_ids),
                    "rate": round(successes / len(pair_ids), 4),
                    "wilson_95_interval": _wilson_interval(successes, len(pair_ids)),
                }
        for outcome in OUTCOMES:
            paired[condition][outcome] = _paired_outcomes(
                baseline_by_pair, candidate_by_pair, outcome
            )

    summary = {
        "experiment": "same-pose cooperative-pick release-profile comparison",
        "evidence_type": "paired MuJoCo simulation; not hardware evidence",
        "unique_target_positions": len(pair_ids),
        "target_split": candidate_summary["target_sampling"],
        "baseline": {"release_profile": "baseline", "results_directory": str(baseline_dir)},
        "candidate": {
            "release_profile": "clearance",
            "results_directory": str(candidate_dir),
            "minimum_tool_height_offset_m": 0.035,
            "minimum_tool_center_radius_m": 0.110,
        },
        "outcomes_by_condition": comparisons,
        "paired_outcomes": paired,
        "contact_attribution_by_profile": {
            "baseline": baseline_summary["contact_attribution"],
            "clearance": candidate_summary["contact_attribution"],
        },
        "fresh_visual_confirmation": "not measured",
        "interpretation_limits": [
            "The clearance settings were chosen after reviewing release and withdrawal contacts on seed 38100; this comparison uses a separate seed 38200 evaluation set.",
            "Thirty poses are a small simulator sample; Wilson intervals show substantial uncertainty.",
            "Payload stable lift is a physics outcome and does not mean the camera visually confirmed the pick.",
        ],
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    svg_output.parent.mkdir(parents=True, exist_ok=True)
    paired_csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    _write_svg(svg_output, summary)
    _write_paired_csv(paired_csv_output, baseline_rows, candidate_rows, pair_ids)
    return summary


def _write_paired_csv(
    path: Path,
    baseline_rows: dict[tuple[str, str], dict[str, str]],
    candidate_rows: dict[tuple[str, str], dict[str, str]],
    pair_ids: list[str],
) -> None:
    fields = ["pair_id", "controller", "target_x_m", "target_y_m"]
    for outcome in OUTCOMES:
        fields.extend((f"baseline_{outcome}", f"clearance_{outcome}"))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for pair_id in pair_ids:
            for condition in CONDITIONS:
                baseline = baseline_rows[pair_id, condition]
                candidate = candidate_rows[pair_id, condition]
                row: dict[str, Any] = {
                    "pair_id": pair_id,
                    "controller": condition,
                    "target_x_m": baseline["target_x_m"],
                    "target_y_m": baseline["target_y_m"],
                }
                for outcome in OUTCOMES:
                    row[f"baseline_{outcome}"] = baseline[outcome]
                    row[f"clearance_{outcome}"] = candidate[outcome]
                writer.writerow(row)


def _write_svg(path: Path, summary: dict[str, Any]) -> None:
    count = summary["unique_target_positions"]
    rows = []
    metrics = (
        ("Payload stable lift", "payload_stable_lift_success", "#49c9ab"),
        ("Full-sequence collision-safe", "full_sequence_collision_safe", "#f3a849"),
        ("Strict combined score", "physical_stable_lift_success", "#63a9f5"),
    )
    y = 142
    for condition, condition_label in (
        ("camera_estimate", "Camera estimate"),
        ("oracle_xy", "Oracle XY diagnostic"),
    ):
        rows.append(
            f'<text x="52" y="{y}" fill="#edf3f7" font-family="sans-serif" font-size="19" font-weight="700">{condition_label}</text>'
        )
        y += 24
        for label, key, color in metrics:
            outcomes = summary["outcomes_by_condition"][condition]
            baseline = outcomes["baseline"][key]["successes"]
            candidate = outcomes["clearance"][key]["successes"]
            rows.append(
                f'<text x="52" y="{y + 20}" fill="#b6c4d3" font-family="sans-serif" font-size="15">{label}</text>'
            )
            for index, value in enumerate((baseline, candidate)):
                top = y + index * 21
                width = 700 * value / count
                fill = color if index else "#8496a8"
                name = "Baseline" if index == 0 else "Clearance"
                rows.extend(
                    [
                        f'<text x="310" y="{top + 14}" fill="#899bad" font-family="sans-serif" font-size="12">{name}</text>',
                        f'<rect x="380" y="{top}" width="700" height="16" rx="5" fill="#253343"/>',
                        f'<rect x="380" y="{top}" width="{width:.1f}" height="16" rx="5" fill="{fill}"/>',
                        f'<text x="1092" y="{top + 14}" fill="#f2f6fa" font-family="sans-serif" font-size="13" font-weight="700">{value}/{count}</text>',
                    ]
                )
            y += 66
        y += 25
    path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="{y + 115}" viewBox="0 0 1180 {y + 115}">
<rect width="100%" height="100%" fill="#111923"/>
<text x="52" y="55" fill="#f2f6fa" font-family="sans-serif" font-size="26" font-weight="700">Collision-aware release trajectory: paired evaluation</text>
<text x="52" y="88" fill="#b6c4d3" font-family="sans-serif" font-size="16">{count} exact target poses · baseline vs clearance profile · seed 38200</text>
{''.join(rows)}
<text x="52" y="{y + 45}" fill="#b6c4d3" font-family="sans-serif" font-size="15">The new profile changes release and withdrawal geometry only. Other components and success rules are shared.</text>
<text x="52" y="{y + 73}" fill="#f4c976" font-family="sans-serif" font-size="15">Physics outcomes only; fresh visual confirmation was not measured.</text>
</svg>''',
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--svg-output", type=Path, required=True)
    parser.add_argument("--paired-csv-output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_release_profiles(
        args.baseline_dir,
        args.candidate_dir,
        json_output=args.json_output,
        svg_output=args.svg_output,
        paired_csv_output=args.paired_csv_output,
    )
    count = result["unique_target_positions"]
    print(f"Paired release-profile comparison: {count} poses")
    for condition, label in (
        ("camera_estimate", "Camera estimate"),
        ("oracle_xy", "Oracle XY diagnostic"),
    ):
        rates = result["outcomes_by_condition"][condition]
        strict = result["paired_outcomes"][condition]["physical_stable_lift_success"]
        print(
            f"{label}: hold "
            f"{rates['baseline']['payload_stable_lift_success']['successes']}/{count} -> "
            f"{rates['clearance']['payload_stable_lift_success']['successes']}/{count}; "
            f"strict "
            f"{rates['baseline']['physical_stable_lift_success']['successes']}/{count} -> "
            f"{rates['clearance']['physical_stable_lift_success']['successes']}/{count}; "
            f"paired candidate-only={strict['candidate_only_success']}, "
            f"baseline-only={strict['baseline_only_success']}"
        )
    print("Wrote paired JSON, SVG, and CSV outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
