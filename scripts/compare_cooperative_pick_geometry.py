"""Compare one changed cooperative-pick trajectory geometry setting."""

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
TRIALS_FILE = "cooperative-pick-pose-validation-trials.csv"
SUMMARY_FILE = "cooperative-pick-pose-validation-summary.json"


def _read_trials(directory: Path) -> dict[tuple[str, str], dict[str, str]]:
    with (directory / TRIALS_FILE).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    trials: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["pair_id"], row["controller"])
        if key in trials:
            raise ValueError(f"duplicate trial row: {key}")
        trials[key] = row
    return trials


def _wilson_interval(successes: int, trials: int, *, z: float = 1.96) -> list[float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials**2)) / denominator
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def _paired_counts(
    baseline: dict[str, dict[str, str]],
    candidate: dict[str, dict[str, str]],
    outcome: str,
) -> dict[str, int]:
    counts = {
        "both_success": 0,
        "candidate_only_success": 0,
        "baseline_only_success": 0,
        "both_failure": 0,
    }
    for pair_id, base_row in baseline.items():
        base_ok = base_row[outcome] == "True"
        candidate_ok = candidate[pair_id][outcome] == "True"
        if base_ok and candidate_ok:
            counts["both_success"] += 1
        elif candidate_ok:
            counts["candidate_only_success"] += 1
        elif base_ok:
            counts["baseline_only_success"] += 1
        else:
            counts["both_failure"] += 1
    return counts


def compare_runs(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    json_output: Path,
    svg_output: Path,
    paired_csv_output: Path,
    parameter: str = "wrist_center_radial_standoff_m",
) -> dict[str, Any]:
    """Validate a one-parameter paired change and write outcomes and contacts."""
    if parameter not in {
        "wrist_center_radial_standoff_m",
        "lowest_approach_tool_height_offset_m",
    }:
        raise ValueError(f"unsupported changed parameter: {parameter}")
    baseline_summary = json.loads((baseline_dir / SUMMARY_FILE).read_text())
    candidate_summary = json.loads((candidate_dir / SUMMARY_FILE).read_text())
    baseline_rows = _read_trials(baseline_dir)
    candidate_rows = _read_trials(candidate_dir)
    if set(baseline_rows) != set(candidate_rows):
        raise ValueError("geometry runs must contain the same paired trials")
    for key in baseline_rows:
        base_xy = tuple(float(baseline_rows[key][field]) for field in ("target_x_m", "target_y_m"))
        candidate_xy = tuple(float(candidate_rows[key][field]) for field in ("target_x_m", "target_y_m"))
        if base_xy != candidate_xy:
            raise ValueError(f"paired target pose differs for {key}: {base_xy} != {candidate_xy}")
    pair_ids = sorted({pair_id for pair_id, _ in baseline_rows})
    expected = {(pair_id, condition) for pair_id in pair_ids for condition in CONDITIONS}
    if not pair_ids or set(baseline_rows) != expected:
        raise ValueError("each geometry run must include both conditions per pose")

    for field in (
        "target_sampling",
        "camera_pipeline",
        "yaw_actuators",
        "task_success_definition",
        "gripper_pad_half_size_m",
        "release_profile",
    ):
        if baseline_summary.get(field) != candidate_summary.get(field):
            raise ValueError(f"geometry runs differ in matched setting: {field}")

    baseline_standoff = float(baseline_summary.get("wrist_center_radial_standoff_m", 0.0))
    candidate_standoff = float(candidate_summary.get("wrist_center_radial_standoff_m", 0.0))
    baseline_approach = float(
        baseline_summary.get("lowest_approach_tool_height_offset_m", 0.005)
    )
    candidate_approach = float(
        candidate_summary.get("lowest_approach_tool_height_offset_m", 0.005)
    )
    if parameter == "wrist_center_radial_standoff_m":
        if baseline_standoff == candidate_standoff:
            raise ValueError("the candidate must use a different wrist standoff")
        if baseline_approach != candidate_approach:
            raise ValueError("wrist-standoff runs must use the same approach height")
        baseline_value, candidate_value = baseline_standoff, candidate_standoff
    else:
        if baseline_approach == candidate_approach:
            raise ValueError("the candidate must use a different approach height")
        if baseline_standoff != candidate_standoff:
            raise ValueError("approach-height runs must use the same wrist standoff")
        baseline_value, candidate_value = baseline_approach, candidate_approach
    base_pad_half_size = float(baseline_summary["gripper_pad_half_size_m"])
    for summary, standoff, label in (
        (baseline_summary, baseline_standoff, "baseline"),
        (candidate_summary, candidate_standoff, "candidate"),
    ):
        expected_pad_offset = 0.022 + standoff
        recorded_pad_offset = summary.get("finger_pad_inward_offset_m", 0.022 + standoff)
        if not math.isclose(float(recorded_pad_offset), expected_pad_offset, abs_tol=1e-9):
            raise ValueError(f"{label} finger extension does not match wrist standoff")
        if standoff < 0 or standoff > base_pad_half_size:
            raise ValueError(f"{label} wrist standoff exceeds the pad contact envelope")
    if parameter == "lowest_approach_tool_height_offset_m":
        base_offset = float(
            baseline_summary.get("finger_pad_inward_offset_m", 0.022)
        )
        candidate_offset = float(
            candidate_summary.get("finger_pad_inward_offset_m", 0.022)
        )
        if not math.isclose(base_offset, candidate_offset, abs_tol=1e-9):
            raise ValueError("approach-height runs must use identical finger geometry")
        if not 0.005 <= baseline_approach <= 0.050 or not 0.005 <= candidate_approach <= 0.050:
            raise ValueError("approach-height offsets must remain within the supported range")

    by_condition: dict[str, dict[str, Any]] = {}
    paired: dict[str, dict[str, dict[str, int]]] = {}
    for condition in CONDITIONS:
        base = {pair: baseline_rows[pair, condition] for pair in pair_ids}
        candidate = {pair: candidate_rows[pair, condition] for pair in pair_ids}
        by_condition[condition] = {}
        paired[condition] = {}
        for profile_name, rows in (("baseline", base), ("candidate", candidate)):
            by_condition[condition][profile_name] = {}
            for outcome in OUTCOMES:
                successes = sum(row[outcome] == "True" for row in rows.values())
                by_condition[condition][profile_name][outcome] = {
                    "successes": successes,
                    "trials": len(pair_ids),
                    "rate": round(successes / len(pair_ids), 4),
                    "wilson_95_interval": _wilson_interval(successes, len(pair_ids)),
                }
        for outcome in OUTCOMES:
            paired[condition][outcome] = _paired_counts(base, candidate, outcome)

    summary = {
        "experiment": "same-pose cooperative-pick geometry comparison",
        "evidence_type": "paired MuJoCo simulation; not hardware evidence",
        "changed_parameter": parameter,
        "baseline_parameter_value": baseline_value,
        "candidate_parameter_value": candidate_value,
        "unique_target_positions": len(pair_ids),
        "target_split": candidate_summary["target_sampling"],
        "baseline": {
            "wrist_center_radial_standoff_m": baseline_standoff,
            "lowest_approach_tool_height_offset_m": baseline_approach,
            "finger_pad_inward_offset_m": baseline_summary.get("finger_pad_inward_offset_m", 0.022),
            "results_directory": str(baseline_dir),
        },
        "candidate": {
            "wrist_center_radial_standoff_m": candidate_standoff,
            "lowest_approach_tool_height_offset_m": candidate_approach,
            "finger_pad_inward_offset_m": candidate_summary["finger_pad_inward_offset_m"],
            "results_directory": str(candidate_dir),
        },
        "outcomes_by_condition": by_condition,
        "paired_outcomes": paired,
        "contact_attribution": {
            "baseline": baseline_summary["contact_attribution"],
            "candidate": candidate_summary["contact_attribution"],
        },
        "fresh_visual_confirmation": "not measured",
        "interpretation_limits": [
            (
                "The candidate moves the wrist center outward and extends the finger shanks and collidable pads inward by the same distance, preserving nominal pad-center placement."
                if parameter == "wrist_center_radial_standoff_m"
                else "The candidate raises only the lowest open gripper's approach path; the closed grasp target and pad geometry are unchanged."
            ),
            "Thirty poses provide a small simulator sample; Wilson intervals show substantial uncertainty.",
            "A physically stable lift does not mean the camera visually confirmed the pick.",
        ],
    }

    json_output.parent.mkdir(parents=True, exist_ok=True)
    svg_output.parent.mkdir(parents=True, exist_ok=True)
    paired_csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_svg(svg_output, summary)
    _write_paired_csv(paired_csv_output, baseline_rows, candidate_rows, pair_ids)
    return summary


def _write_svg(path: Path, summary: dict[str, Any]) -> None:
    if summary["changed_parameter"] == "lowest_approach_tool_height_offset_m":
        svg = _approach_clearance_svg(summary)
    else:
        svg = _wrist_standoff_svg(summary)
    path.write_text(svg, encoding="utf-8")


def _approach_clearance_svg(summary: dict[str, Any]) -> str:
    """Show the pad/table geometry beside both paired outcome conditions."""
    count = summary["unique_target_positions"]
    base_seed = summary["target_split"]["base_seed"]
    baseline_mm = round(summary["baseline"]["lowest_approach_tool_height_offset_m"] * 1000)
    candidate_mm = round(summary["candidate"]["lowest_approach_tool_height_offset_m"] * 1000)
    outcomes = summary["outcomes_by_condition"]
    paired = summary["paired_outcomes"]
    baseline_contacts = summary["contact_attribution"]["baseline"]["camera_estimate"]["arm_environment"]["trials_with_contact"]
    candidate_contacts = summary["contact_attribution"]["candidate"]["camera_estimate"]["arm_environment"]["trials_with_contact"]
    baseline_payload_contacts = summary["contact_attribution"]["baseline"]["camera_estimate"]["nonfinger_payload"]["trials_with_contact"]
    candidate_payload_contacts = summary["contact_attribution"]["candidate"]["camera_estimate"]["nonfinger_payload"]["trials_with_contact"]

    def panel(x: int, title: str, offset: int, *, candidate: bool) -> str:
        scale_px_per_mm = 1.6
        table_y = 475
        payload_y = table_y - 60 * scale_px_per_mm
        tool_y = payload_y - offset * scale_px_per_mm
        pad_center_y = payload_y + (50 - offset) * scale_px_per_mm
        pad_h = 32 * scale_px_per_mm
        pad_top = pad_center_y - pad_h / 2
        pad_bottom = pad_center_y + pad_h / 2
        color = "#44d4a3" if candidate else "#fa6578"
        clearance_mm = offset - 6
        clearance_label = f"+{clearance_mm} mm clearance" if candidate else f"{clearance_mm} mm nominal overlap"
        marker = (
            f'<path d="M{x + 202} {table_y - 8} l16 16 m0 -16 l-16 16" stroke="{color}" stroke-width="4"/>'
            if not candidate
            else f'<path d="M{x + 202} {pad_bottom + 2} v{table_y - pad_bottom - 7}" stroke="{color}" stroke-width="2" marker-end="url(#arrowGreen)"/>'
        )
        return f'''<rect x="{x}" y="204" width="270" height="340" rx="16" fill="#111e2d" stroke="#29394b"/>
<text x="{x + 18}" y="235" class="paneltitle">{title}</text>
<text x="{x + 18}" y="258" class="small muted">{offset} mm above payload center</text>
<rect x="{x + 58}" y="{payload_y - 60 * scale_px_per_mm}" width="82" height="{120 * scale_px_per_mm}" rx="16" fill="#1f3348" stroke="#4a6d8a" stroke-width="2"/>
<text x="{x + 76}" y="{payload_y + 4}" class="micro">PAYLOAD</text>
<line x1="{x + 42}" y1="{payload_y}" x2="{x + 237}" y2="{payload_y}" stroke="#91a6bd" stroke-width="1" stroke-dasharray="5 5"/>
<rect x="{x + 179}" y="{pad_top:.1f}" width="46" height="{pad_h:.1f}" rx="7" fill="#ced9e5" stroke="#f2f6fa" stroke-width="2"/>
<line x1="{x + 202}" y1="{tool_y}" x2="{x + 202}" y2="{pad_center_y}" stroke="#83b5f3" stroke-width="4"/>
<circle cx="{x + 202}" cy="{tool_y}" r="6" fill="#83b5f3"/>
<line x1="{x + 28}" y1="{table_y}" x2="{x + 246}" y2="{table_y}" stroke="#8d9aa8" stroke-width="6"/>
<rect x="{x + 28}" y="{table_y + 3}" width="218" height="30" fill="#253143"/>
<text x="{x + 18}" y="{table_y + 22}" class="micro">table top | z = 0.180 m</text>
{marker}
<text x="{x + 18}" y="515" class="pill" fill="{color}">{clearance_label}</text>
<text x="{x + 18}" y="535" class="micro">pad 16 mm half-size · Z offset −50 mm</text>'''

    def bars(condition: str, y: int) -> str:
        base = outcomes[condition]["baseline"]["physical_stable_lift_success"]
        cand = outcomes[condition]["candidate"]["physical_stable_lift_success"]
        transitions = paired[condition]["physical_stable_lift_success"]
        base_w = 322 * base["rate"]
        candidate_w = 322 * cand["rate"]
        base_ci = base["wilson_95_interval"]
        cand_ci = cand["wilson_95_interval"]
        label = "Camera estimate" if condition == "camera_estimate" else "Oracle XY diagnostic"
        return f'''<text x="752" y="{y}" class="label">{label} · strict physical score</text>
<text x="752" y="{y + 26}" class="muted small">{baseline_mm} mm</text>
<rect x="934" y="{y + 12}" width="322" height="15" rx="7.5" fill="#263448"/><rect x="934" y="{y + 12}" width="{base_w:.1f}" height="15" rx="7.5" fill="#8291a5"/>
<text x="1272" y="{y + 25}" class="value">{base['successes']}/{count}</text>
<text x="752" y="{y + 58}" class="muted small">{candidate_mm} mm</text>
<rect x="934" y="{y + 44}" width="322" height="15" rx="7.5" fill="#263448"/><rect x="934" y="{y + 44}" width="{candidate_w:.1f}" height="15" rx="7.5" fill="#44d4a3"/>
<text x="1272" y="{y + 57}" class="value">{cand['successes']}/{count}</text>
<text x="934" y="{y + 80}" class="micro">95% Wilson: {base_ci[0]:.0%}-{base_ci[1]:.0%} → {cand_ci[0]:.0%}-{cand_ci[1]:.0%}</text>
<text x="752" y="{y + 103}" class="small muted">paired: <tspan fill="#44d4a3">{transitions['candidate_only_success']} candidate-only</tspan><tspan> · </tspan><tspan fill="#f36d7d">{transitions['baseline_only_success']} baseline-only</tspan></text>'''

    camera_pairs = paired["camera_estimate"]["physical_stable_lift_success"]
    oracle_pairs = paired["oracle_xy"]["physical_stable_lift_success"]
    camera_base = outcomes["camera_estimate"]["baseline"]["physical_stable_lift_success"]
    camera_candidate = outcomes["camera_estimate"]["candidate"]["physical_stable_lift_success"]
    oracle_base = outcomes["oracle_xy"]["baseline"]["physical_stable_lift_success"]
    oracle_candidate = outcomes["oracle_xy"]["candidate"]["physical_stable_lift_success"]
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1015" viewBox="0 0 1400 1015">
<defs><linearGradient id="bg" x2="0" y2="1"><stop stop-color="#111b2a"/><stop offset="1" stop-color="#0a101a"/></linearGradient>
<marker id="arrowGreen" markerWidth="7" markerHeight="7" refX="4" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 Z" fill="#44d4a3"/></marker>
<style>
.title {{font:700 30px Inter,Arial,sans-serif;fill:#f3f7fc}} .subtitle {{font:400 14px Inter,Arial,sans-serif;fill:#a9bacb}}
.paneltitle {{font:700 16px Inter,Arial,sans-serif;fill:#eef4fa}} .section {{font:700 13px Inter,Arial,sans-serif;letter-spacing:1px;fill:#b8c9d9}}
.label {{font:600 14px Inter,Arial,sans-serif;fill:#e5edf5}} .value {{font:700 14px Inter,Arial,sans-serif;fill:#f4f8fc}}
.small {{font:400 12px Inter,Arial,sans-serif}} .micro {{font:400 10px Inter,Arial,sans-serif;fill:#adbdcd}}
.muted {{fill:#9daec0}} .pill {{font:700 13px Inter,Arial,sans-serif}} .body {{font:400 13px Inter,Arial,sans-serif;fill:#cad6e2}}
.big {{font:700 20px Inter,Arial,sans-serif;fill:#f2f6fa}}
</style></defs>
<rect width="1400" height="1015" fill="url(#bg)"/><rect x="60" y="38" width="5" height="72" rx="2.5" fill="#44d4a3"/>
<text x="82" y="65" class="title">A 15 mm approach lift reduces the lowest pad's table contact</text>
<text x="82" y="94" class="subtitle">Camera-guided tri-arm pick · {count} exact paired poses from base seed {base_seed} · release trajectory and finger geometry held fixed · MuJoCo simulation</text>
<rect x="60" y="134" width="620" height="444" rx="18" fill="#172333" stroke="#29394b"/><rect x="700" y="134" width="640" height="444" rx="18" fill="#172333" stroke="#29394b"/>
<text x="88" y="173" class="section">GEOMETRY | SAME ARM, DIFFERENT OPEN-APPROACH HEIGHT</text>
<text x="728" y="173" class="section">HELD-OUT SCORE | {count} IDENTICAL TARGET POSES PER CONDITION</text>
{panel(85, 'BASELINE', baseline_mm, candidate=False)}{panel(390, 'CLEARANCE CANDIDATE', candidate_mm, candidate=True)}
{bars('camera_estimate', 222)}
<line x1="752" y1="351" x2="1300" y2="351" stroke="#2b3b4d"/>
{bars('oracle_xy', 379)}
<rect x="728" y="500" width="584" height="54" rx="10" fill="#2c2830"/><text x="746" y="523" class="small" fill="#ffca82">Camera: {camera_base['successes']}/{count} → {camera_candidate['successes']}/{count};</text>
<text x="746" y="542" class="small" fill="#ffca82">oracle: {oracle_base['successes']}/{count} → {oracle_candidate['successes']}/{count}. Keep opt-in.</text>
<rect x="60" y="598" width="1280" height="345" rx="18" fill="#172333" stroke="#29394b"/><text x="88" y="638" class="section">CONTACT TRACE, INTERPRETATION, AND LIMITS</text>
<line x1="486" y1="662" x2="486" y2="911" stroke="#2b3b4d"/><line x1="912" y1="662" x2="912" y2="911" stroke="#2b3b4d"/>
<text x="90" y="684" class="paneltitle">What happened in the paired run</text><text x="90" y="714" class="big">arm0 right pad → table</text>
<text x="90" y="743" class="body">Trace: lowest pad/table contacts during</text><text x="90" y="765" class="body">LOWER TO GRASP and CLOSE SIX FINGERS.</text>
<text x="90" y="800" class="body">Camera arm/environment: {baseline_contacts}/{count} → {candidate_contacts}/{count}.</text><text x="90" y="822" class="body">Nonfinger arm/payload: {baseline_payload_contacts}/{count} → {candidate_payload_contacts}/{count}.</text>
<text x="90" y="862" class="small" fill="#44d4a3">Per-pose pass/fail pairs are in the linked CSV.</text>
<text x="510" y="684" class="paneltitle">Why the pad reaches the tabletop</text><text x="510" y="714" class="body">Pad center = tool Z − 50 mm; half-height 16 mm.</text>
<text x="510" y="736" class="body">At 5 mm offset, pad bottom is payload center −61 mm.</text><text x="510" y="758" class="body">The table is payload center −60 mm: about 1 mm</text>
<text x="510" y="780" class="body">of nominal overlap. At 20 mm, the pad bottom is</text><text x="510" y="802" class="body">14 mm above the tabletop.</text>
<text x="510" y="850" class="small" fill="#f4c976">Geometric calculation; MuJoCo contact scoring stays active.</text>
<text x="936" y="684" class="paneltitle">What this does not establish</text><text x="936" y="714" class="body">Small sample; camera Wilson intervals overlap.</text>
<text x="936" y="736" class="body">Oracle XY moved {oracle_base['successes']}→{oracle_candidate['successes']}.</text>
<text x="936" y="767" class="body">Ideal fixed camera calibration; no hardware.</text><text x="936" y="789" class="body">Target detected once for planning.</text>
<text x="936" y="811" class="body">Fresh visual confirmation not measured.</text><text x="936" y="862" class="small" fill="#83b5f3">Default stays at 5 mm; candidate is opt-in.</text>
<text x="60" y="982" class="small muted">Paired MuJoCo evidence · physical stable-lift/collision score ≠ visual confirmation · no hardware claims</text></svg>'''


def _wrist_standoff_svg(summary: dict[str, Any]) -> str:
    """Keep the older wrist probe readable and label its mixed evidence."""
    count = summary["unique_target_positions"]
    parameter = "wrist_center_radial_standoff_m"
    rows = []
    y = 164
    for condition, label in (("camera_estimate", "Camera estimate"), ("oracle_xy", "Oracle XY diagnostic")):
        rows.append(f'<text x="64" y="{y}" fill="#edf3f7" font-family="sans-serif" font-size="20" font-weight="700">{label}</text>')
        y += 35
        for outcome, outcome_label in (("physical_stable_lift_success", "Strict physical score"), ("payload_stable_lift_success", "Payload stable lift")):
            rows.append(f'<text x="76" y="{y + 20}" fill="#b6c4d3" font-family="sans-serif" font-size="15">{outcome_label}</text>')
            for index, profile in enumerate(("baseline", "candidate")):
                metric = summary["outcomes_by_condition"][condition][profile][outcome]
                top = y + index * 24
                bar_width = 600 * metric["rate"]
                fill = "#8496a8" if index == 0 else "#49c9ab"
                label_text = f'{summary[profile][parameter] * 1000:.0f} mm'
                rows.extend([
                    f'<text x="310" y="{top + 16}" fill="#9eacbb" font-family="sans-serif" font-size="12">{label_text}</text>',
                    f'<rect x="380" y="{top}" width="600" height="17" rx="6" fill="#253343"/>',
                    f'<rect x="380" y="{top}" width="{bar_width:.1f}" height="17" rx="6" fill="{fill}"/>',
                    f'<text x="996" y="{top + 15}" fill="#f2f6fa" font-family="sans-serif" font-size="13" font-weight="700">{metric["successes"]}/{count}</text>',
                ])
            y += 66
        y += 28
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1260" height="760" viewBox="0 0 1260 760"><rect width="100%" height="100%" fill="#111923"/>
<text x="55" y="58" fill="#f2f6fa" font-family="sans-serif" font-size="28" font-weight="700">Matched-pad wrist setback probe</text>
<text x="55" y="94" fill="#b6c4d3" font-family="sans-serif" font-size="16">Finger-pad centers remain nominal; both controls share {count} target poses.</text>{''.join(rows)}
<rect x="55" y="{y + 4}" width="1150" height="94" rx="12" fill="#25242a"/><text x="78" y="{y + 40}" fill="#f4c976" font-family="sans-serif" font-size="17" font-weight="700">Mixed camera/oracle result; experimental only</text>
<text x="78" y="{y + 70}" fill="#c1ccd7" font-family="sans-serif" font-size="15">Small paired simulator sample; physical lift and fresh visual confirmation are separate.</text></svg>'''


def _write_paired_csv(
    path: Path,
    baseline: dict[tuple[str, str], dict[str, str]],
    candidate: dict[tuple[str, str], dict[str, str]],
    pair_ids: list[str],
) -> None:
    fields = ["pair_id", "controller", "target_x_m", "target_y_m"]
    for outcome in OUTCOMES:
        fields.extend((f"baseline_{outcome}", f"candidate_{outcome}"))
    fields.extend(("baseline_failure_reason", "candidate_failure_reason"))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for pair_id in pair_ids:
            for condition in CONDITIONS:
                base = baseline[pair_id, condition]
                candidate_row = candidate[pair_id, condition]
                row: dict[str, Any] = {
                    "pair_id": pair_id,
                    "controller": condition,
                    "target_x_m": base["target_x_m"],
                    "target_y_m": base["target_y_m"],
                    "baseline_failure_reason": base["failure_reason"],
                    "candidate_failure_reason": candidate_row["failure_reason"],
                }
                for outcome in OUTCOMES:
                    row[f"baseline_{outcome}"] = base[outcome]
                    row[f"candidate_{outcome}"] = candidate_row[outcome]
                writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--svg-output", type=Path, required=True)
    parser.add_argument("--paired-csv-output", type=Path, required=True)
    parser.add_argument(
        "--parameter",
        choices=("wrist_center_radial_standoff_m", "lowest_approach_tool_height_offset_m"),
        default="wrist_center_radial_standoff_m",
    )
    args = parser.parse_args()
    result = compare_runs(
        args.baseline_dir,
        args.candidate_dir,
        json_output=args.json_output,
        svg_output=args.svg_output,
        paired_csv_output=args.paired_csv_output,
        parameter=args.parameter,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
