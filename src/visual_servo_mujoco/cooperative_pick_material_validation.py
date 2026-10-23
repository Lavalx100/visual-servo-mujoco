"""Held-out payload mass and grasp-friction stress test for the three-arm pick."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .camera_guided_cooperative_pick import (
    FINGER_PAD_HALF_SIZE_M,
    _failure_reason,
    _sample_positions,
    run_trial,
)
from .cooperative_pick_demo import (
    DEFAULT_PAYLOAD_BODY_MASS_KG,
)
from .cooperative_pick_visual_validation import _wilson_interval


BASE_SEED = 40800
SAMPLES_PER_SEED = 12
PAYLOAD_BODY_MASSES_KG = (0.45, DEFAULT_PAYLOAD_BODY_MASS_KG, 0.99)
GRASP_SLIDING_FRICTION = (0.5, 1.1, 2.2)
CONTROLLERS = ("camera_estimate", "oracle_xy")
PAD_HALF_SIZE_M = FINGER_PAD_HALF_SIZE_M
APPROACH_CLEARANCE_M = 0.020
OUTPUT_DIRECTORY = Path("results/cooperative-pick-material-validation")
TRIALS_FILE = "cooperative-pick-material-trials.csv"
SUMMARY_FILE = "cooperative-pick-material-summary.json"
SCORECARD_FILE = "cooperative-pick-material-scorecard.svg"
DEMO_METADATA_FILE = Path("assets/demos/cooperative_pick_material_demos.json")
DEMO_DIRECTORY = Path("assets/demos")


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trials = len(rows)
    metrics: dict[str, Any] = {"trials": trials}
    for name, field in (
        ("physical_stable_lift_success", "physical_stable_lift_success"),
        ("payload_stable_lift_success", "payload_stable_lift_success"),
        ("full_sequence_collision_safe", "full_sequence_collision_safe"),
    ):
        successes = sum(bool(row[field]) for row in rows)
        metrics[name] = {
            "successes": successes,
            "rate": round(successes / trials, 4) if trials else None,
            "wilson_95_interval": _wilson_interval(successes, trials),
        }
    metrics["failure_reasons"] = dict(
        sorted(Counter(row["failure_reason"] for row in rows).items())
    )
    metrics["fresh_visual_confirmation"] = "not measured in material sweep"
    return metrics


def _write_scorecard(path: Path, conditions: dict[str, Any]) -> None:
    width, height = 1320, 680
    pose_count = next(iter(conditions.values()))["trials"]
    fragments = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1220"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}.title{fill:#f3f6fb}.muted{fill:#a9b6c8}.blue{fill:#73b7ff}.green{fill:#68dfb3}.orange{fill:#ffbd72}.cell{fill:#12243a}</style>',
        '<text x="36" y="45" class="title" font-size="27" font-weight="700">Three-arm pick: held-out material stress test</text>',
        f'<text x="36" y="76" class="muted" font-size="14">Same {pose_count} target poses at every setting · 16 mm pads · 20 mm approach clearance · clearance release</text>',
        '<text x="36" y="99" class="muted" font-size="13">Author-selected simulator stresses; sliding friction is applied to both pads and payload. No hardware calibration.</text>',
    ]
    controller_panels = (("camera_estimate", 36, "Camera-estimated target"), ("oracle_xy", 678, "Oracle target upper bound"))
    masses = tuple(sorted({float(record["payload_body_mass_kg"]) for record in conditions.values()}))
    frictions = tuple(sorted({float(record["finger_pad_and_payload_sliding_friction"]) for record in conditions.values()}))
    for controller, panel_x, panel_title in controller_panels:
        fragments.append(
            f'<text x="{panel_x}" y="140" class="blue" font-size="18" font-weight="700">{panel_title}</text>'
        )
        fragments.append(
            f'<text x="{panel_x + 186}" y="177" class="muted" font-size="13">pad/payload sliding friction μ</text>'
        )
        for col, friction in enumerate(frictions):
            x = panel_x + 170 + col * 145
            fragments.append(f'<text x="{x}" y="205" class="muted" font-size="14">{friction:g}</text>')
        for row_index, mass in enumerate(masses):
            y = 240 + row_index * 112
            fragments.append(f'<text x="{panel_x}" y="{y + 24}" class="title" font-size="14">{mass:.2f} kg</text>')
            fragments.append(f'<text x="{panel_x}" y="{y + 45}" class="muted" font-size="12">body mass</text>')
            for col, friction in enumerate(frictions):
                key = f"mass-{mass:g}_friction-{friction:g}"
                record = conditions[key]["controllers"][controller]
                result = record["physical_stable_lift_success"]
                rate = result["rate"] or 0.0
                hue = int(12 + 135 * rate)
                fill = f"#{int(18 + 12 * rate):02x}{hue:02x}{int(36 + 54 * rate):02x}"
                x = panel_x + 112 + col * 145
                fragments.append(f'<rect x="{x}" y="{y}" width="132" height="74" rx="8" fill="{fill}"/>')
                fragments.append(f'<text x="{x + 14}" y="{y + 27}" class="title" font-size="17" font-weight="700">{result["successes"]}/{record["trials"]}</text>')
                hold = record["payload_stable_lift_success"]
                safe = record["full_sequence_collision_safe"]
                fragments.append(f'<text x="{x + 14}" y="{y + 50}" class="title" font-size="11">hold {hold["successes"]} · cycle {safe["successes"]}</text>')
        fragments.append(f'<text x="{panel_x}" y="600" class="muted" font-size="12">Each cell: strict success / trials; hold and full-cycle safety are shown separately.</text>')
    fragments.extend(
        [
            '<text x="36" y="646" class="orange" font-size="13">Vision confirmation was not scored. Physical success does not imply fresh visual confirmation.</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(fragments), encoding="utf-8")


def _write_paired_demo(
    nominal_video: Path,
    stress_video: Path,
    output_video: Path,
    output_gif: Path,
    *,
    source_pair_id: str,
    nominal_mass_kg: float,
    nominal_friction: float,
    stress_mass_kg: float,
    stress_friction: float,
) -> None:
    """Compose two matched replay videos with visible material labels."""
    nominal = cv2.VideoCapture(str(nominal_video))
    stress = cv2.VideoCapture(str(stress_video))
    fps = nominal.get(cv2.CAP_PROP_FPS) or 30.0
    output_size = (1280, 424)
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size
    )
    if not writer.isOpened():
        nominal.release()
        stress.release()
        raise RuntimeError("OpenCV could not open the paired material demo writer")
    gif_frames: list[Image.Image] = []
    frame_index = 0
    while True:
        has_nominal, left = nominal.read()
        has_stress, right = stress.read()
        if not has_nominal or not has_stress:
            break
        canvas = np.full((output_size[1], output_size[0], 3), (14, 21, 32), dtype=np.uint8)
        cv2.putText(canvas, f"MATCHED MATERIAL REPLAY  |  held-out pose {source_pair_id}", (24, 31), cv2.FONT_HERSHEY_DUPLEX, 0.7, (244, 247, 250), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"NOMINAL  |  {nominal_mass_kg:.3f} kg  |  mu {nominal_friction:g}", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (100, 226, 181), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"HEAVY + LOW FRICTION  |  {stress_mass_kg:.3f} kg  |  mu {stress_friction:g}", (660, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (93, 191, 255), 1, cv2.LINE_AA)
        left_small = cv2.resize(left, (640, 312), interpolation=cv2.INTER_AREA)
        right_small = cv2.resize(right, (640, 312), interpolation=cv2.INTER_AREA)
        canvas[76:388, 0:640] = left_small
        canvas[76:388, 640:1280] = right_small
        cv2.putText(canvas, "Same target, controller, geometry and motion; only material values differ.", (24, 414), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (190, 202, 215), 1, cv2.LINE_AA)
        writer.write(canvas)
        if frame_index % 6 == 0:
            preview = cv2.resize(canvas, (960, 318), interpolation=cv2.INTER_AREA)
            gif_frames.append(
                Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
            )
        frame_index += 1
    nominal.release()
    stress.release()
    writer.release()
    if not gif_frames:
        raise RuntimeError("the paired material videos contained no matching frames")
    gif_frames[0].save(
        output_gif,
        save_all=True,
        append_images=gif_frames[1:],
        duration=200,
        loop=0,
        optimize=True,
    )


def _run_demos(
    demo_output_dir: Path,
    demo_metadata_path: Path,
    positions: list[tuple[int, int, tuple[float, float]]],
    rows: list[dict[str, Any]],
    payload_masses_kg: tuple[float, ...],
    sliding_friction_levels: tuple[float, ...],
) -> list[dict[str, Any]]:
    demo_output_dir.mkdir(parents=True, exist_ok=True)
    demo_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    stressed_mass = max(payload_masses_kg)
    stressed_friction = min(sliding_friction_levels)
    stressed_key = f"mass-{stressed_mass:g}_friction-{stressed_friction:g}"
    stress_camera_rows = [
        row
        for row in rows
        if row["condition"] == stressed_key and row["controller"] == "camera_estimate"
    ]
    failure_rows = [row for row in stress_camera_rows if not row["physical_stable_lift_success"]]
    selected = failure_rows[0] if failure_rows else stress_camera_rows[0]
    pair_id = selected["pair_id"]
    position = next((item for item in positions if f"{item[0]}-{item[1]:03d}" == pair_id), None)
    if position is None:
        raise RuntimeError(f"demo pose {pair_id} is not in the held-out position block")

    demo_specs = (
        {
            "name": "nominal_material",
            "mass": DEFAULT_PAYLOAD_BODY_MASS_KG,
            "friction": 2.2,
            "condition": f"mass-{DEFAULT_PAYLOAD_BODY_MASS_KG:g}_friction-2.2",
        },
        {
            "name": "heavy_low_friction_stress",
            "mass": stressed_mass,
            "friction": stressed_friction,
            "condition": stressed_key,
        },
    )
    demos: list[dict[str, Any]] = []
    for spec in demo_specs:
        result = run_trial(
            position[2],
            controller="camera_estimate",
            video_path=demo_output_dir / f"cooperative_pick_material_{spec['name']}.mp4",
            preview_path=demo_output_dir / f"cooperative_pick_material_{spec['name']}.gif",
            finger_pad_half_size_m=PAD_HALF_SIZE_M,
            finger_pad_sliding_friction=spec["friction"],
            payload_body_mass_kg=spec["mass"],
            release_profile="clearance",
            lowest_approach_tool_height_offset_m=APPROACH_CLEARANCE_M,
        )
        demos.append(
            {
                "name": spec["name"],
                "source_pair_id": pair_id,
                "condition": spec["condition"],
                "controller": "camera_estimate",
                "payload_body_mass_kg": result["payload_body_mass_kg"],
                "finger_pad_and_payload_sliding_friction": spec["friction"],
                "physical_stable_lift_success": result["physical_stable_lift_success"],
                "payload_stable_lift_success": result["payload_stable_lift_success"],
                "full_sequence_collision_safe": result["full_sequence_collision_safe"],
                "fresh_visual_confirmation": "not measured",
                "mp4": f"cooperative_pick_material_{spec['name']}.mp4",
                "gif": f"cooperative_pick_material_{spec['name']}.gif",
                "counted_as_additional_trial": False,
            }
        )
    paired_mp4 = demo_output_dir / "cooperative_pick_material_matched_comparison.mp4"
    paired_gif = demo_output_dir / "cooperative_pick_material_matched_comparison.gif"
    _write_paired_demo(
        demo_output_dir / "cooperative_pick_material_nominal_material.mp4",
        demo_output_dir / "cooperative_pick_material_heavy_low_friction_stress.mp4",
        paired_mp4,
        paired_gif,
        source_pair_id=pair_id,
        nominal_mass_kg=DEFAULT_PAYLOAD_BODY_MASS_KG,
        nominal_friction=2.2,
        stress_mass_kg=stressed_mass,
        stress_friction=stressed_friction,
    )
    demos.append(
        {
            "name": "matched_nominal_vs_heavy_low_friction",
            "source_pair_id": pair_id,
            "controller": "camera_estimate",
            "comparison": (
                f"same pose; {DEFAULT_PAYLOAD_BODY_MASS_KG:.3f} kg and mu 2.2 versus "
                f"{stressed_mass:.3f} kg and mu {stressed_friction:g}"
            ),
            "counted_as_additional_trial": False,
            "mp4": paired_mp4.name,
            "gif": paired_gif.name,
        }
    )
    demo_metadata_path.write_text(
        json.dumps(
            {
                "purpose": "illustrative replays of one held-out target pose at nominal and stress material settings",
                "heldout_pair_id": pair_id,
                "video_runs_are_benchmark_trials": False,
                "demos": demos,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return demos


def run_validation(
    *,
    output_dir: Path = OUTPUT_DIRECTORY,
    demo_metadata_path: Path = DEMO_METADATA_FILE,
    demo_output_dir: Path = DEMO_DIRECTORY,
    base_seed: int = BASE_SEED,
    samples_per_seed: int = SAMPLES_PER_SEED,
    payload_masses_kg: tuple[float, ...] = PAYLOAD_BODY_MASSES_KG,
    sliding_friction_levels: tuple[float, ...] = GRASP_SLIDING_FRICTION,
    render_demos: bool = True,
) -> dict[str, Any]:
    if samples_per_seed <= 0:
        raise ValueError("samples_per_seed must be positive")
    if not payload_masses_kg or any(
        not math.isfinite(value) or value <= 0.0 for value in payload_masses_kg
    ):
        raise ValueError("payload masses must be finite and positive")
    if not sliding_friction_levels or any(
        not math.isfinite(value) or value <= 0.0 for value in sliding_friction_levels
    ):
        raise ValueError("sliding-friction levels must be finite and positive")
    if len(set(payload_masses_kg)) != len(payload_masses_kg):
        raise ValueError("payload mass levels must be unique")
    if len(set(sliding_friction_levels)) != len(sliding_friction_levels):
        raise ValueError("sliding-friction levels must be unique")
    if render_demos and (
        DEFAULT_PAYLOAD_BODY_MASS_KG not in payload_masses_kg
        or 2.2 not in sliding_friction_levels
    ):
        raise ValueError(
            "rendered matched demos require the nominal payload mass and 2.2 friction settings"
        )

    positions = _sample_positions(
        seeds=1,
        samples_per_seed=samples_per_seed,
        base_seed=base_seed,
    )
    rows: list[dict[str, Any]] = []
    for mass in payload_masses_kg:
        for friction in sliding_friction_levels:
            condition = f"mass-{mass:g}_friction-{friction:g}"
            for seed, sample_index, target_xy in positions:
                pair_id = f"{seed}-{sample_index:03d}"
                for controller in CONTROLLERS:
                    print(
                        f"{condition} {controller} {sample_index + 1}/{samples_per_seed}",
                        flush=True,
                    )
                    metrics = run_trial(
                        target_xy,
                        controller=controller,
                        finger_pad_half_size_m=PAD_HALF_SIZE_M,
                        finger_pad_sliding_friction=friction,
                        payload_body_mass_kg=mass,
                        release_profile="clearance",
                        lowest_approach_tool_height_offset_m=APPROACH_CLEARANCE_M,
                    )
                    rows.append(
                        {
                            "pair_id": pair_id,
                            "seed": seed,
                            "sample_index": sample_index,
                            "condition": condition,
                            "controller": controller,
                            "target_x_m": target_xy[0],
                            "target_y_m": target_xy[1],
                            "payload_body_mass_kg": metrics["payload_body_mass_kg"],
                            "finger_pad_half_size_m": PAD_HALF_SIZE_M,
                            "finger_pad_sliding_friction": friction,
                            "approach_clearance_m": APPROACH_CLEARANCE_M,
                            "release_profile": "clearance",
                            "payload_stable_lift_success": metrics[
                                "payload_stable_lift_success"
                            ],
                            "full_sequence_collision_safe": metrics[
                                "full_sequence_collision_safe"
                            ],
                            "physical_stable_lift_success": metrics[
                                "physical_stable_lift_success"
                            ],
                            "minimum_finger_payload_contact_points_during_hold": metrics[
                                "minimum_finger_payload_contact_points_during_hold"
                            ],
                            "minimum_payload_z_during_hold_m": metrics[
                                "minimum_payload_z_during_hold_m"
                            ],
                            "maximum_payload_tilt_during_hold_deg": metrics[
                                "maximum_payload_tilt_during_hold_deg"
                            ],
                            "physics_steps_with_arm_table_or_ground_contact": metrics[
                                "physics_steps_with_arm_table_or_ground_contact"
                            ],
                            "physics_steps_with_nonfinger_arm_payload_contact": metrics[
                                "physics_steps_with_nonfinger_arm_payload_contact"
                            ],
                            "failure_reason": _failure_reason(metrics),
                            "fresh_visual_confirmation": "not measured",
                        }
                    )

    conditions: dict[str, Any] = {}
    for mass in payload_masses_kg:
        for friction in sliding_friction_levels:
            condition = f"mass-{mass:g}_friction-{friction:g}"
            selected = [row for row in rows if row["condition"] == condition]
            conditions[condition] = {
                "payload_body_mass_kg": mass,
                "finger_pad_and_payload_sliding_friction": friction,
                "trials": len(selected) // len(CONTROLLERS),
                "controllers": {
                    controller: _aggregate(
                        [row for row in selected if row["controller"] == controller]
                    )
                    for controller in CONTROLLERS
                },
            }
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = output_dir / TRIALS_FILE
    with trials_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    demos = (
        _run_demos(
            demo_output_dir,
            demo_metadata_path,
            positions,
            rows,
            payload_masses_kg,
            sliding_friction_levels,
        )
        if render_demos
        else []
    )
    summary = {
        "experiment": "held-out three-arm grasp material stress test",
        "protocol_version": 1,
        "evidence_type": "paired MuJoCo simulation; not hardware evidence",
        "target_sampling": {
            "distribution": "independent uniform XY samples in a square",
            "x_range_m": [-0.09, 0.09],
            "y_range_m": [-0.09, 0.09],
            "base_seed": base_seed,
            "samples_per_seed": samples_per_seed,
        },
        "paired_design": "same target pose for every material condition and both camera-estimate/oracle controllers",
        "controller_conditions": list(CONTROLLERS),
        "frozen_controller_settings": {
            "finger_pad_half_size_m": PAD_HALF_SIZE_M,
            "lowest_approach_tool_height_offset_m": APPROACH_CLEARANCE_M,
            "release_profile": "clearance",
        },
        "varied_material_assumptions": {
            "payload_body_mass_kg": list(payload_masses_kg),
            "finger_pad_and_payload_sliding_friction": list(sliding_friction_levels),
            "interpretation": "author-selected simulator stress values; no real object or hardware calibration",
            "inertia_model": "the nominal mass retains the legacy geometry-derived inertial path; off-nominal masses use explicit inertials with principal moments scaled proportionally",
        },
        "total_physics_trials": len(rows),
        "unique_target_poses": len(positions),
        "unique_pose_material_pairs": len(positions) * len(conditions),
        "conditions": conditions,
        "limitations": [
            "The mass and friction levels are deliberate synthetic stresses, not estimates of a specific real canister.",
            "Off-nominal mass conditions use explicit body inertials and zero-density visual trim; the nominal condition keeps the legacy geometry-derived inertial model to preserve its established contact trajectory.",
            "The cylinder geometry and all controller settings remain fixed; this study does not cover shape variation or dynamic transport.",
            "Target perception remains the existing one-shot detector; oracle_xy is included only as an upper-bound comparator.",
            "Fresh visual confirmation is not measured; a contact-driven stable lift is not a visual success claim.",
            "Twelve poses per material condition provide a stress map, not a precise population reliability estimate.",
        ],
        "artifacts": {
            "trials_csv": trials_path.name,
            "scorecard_svg": SCORECARD_FILE,
            "demo_metadata": str(demo_metadata_path),
        },
        "demos": demos,
    }
    summary_path = output_dir / SUMMARY_FILE
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_scorecard(output_dir / SCORECARD_FILE, conditions)
    print(json.dumps({"trials": len(rows), "summary": str(summary_path), "scorecard": str(output_dir / SCORECARD_FILE)}, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--demo-metadata", type=Path, default=DEMO_METADATA_FILE)
    parser.add_argument("--demo-output-dir", type=Path, default=DEMO_DIRECTORY)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--samples-per-seed", type=int, default=SAMPLES_PER_SEED)
    parser.add_argument("--skip-demos", action="store_true")
    args = parser.parse_args()
    run_validation(
        output_dir=args.output_dir,
        demo_metadata_path=args.demo_metadata,
        demo_output_dir=args.demo_output_dir,
        base_seed=args.base_seed,
        samples_per_seed=args.samples_per_seed,
        render_demos=not args.skip_demos,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
