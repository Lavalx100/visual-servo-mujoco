"""Replay and render a paired held-out camera-guided approach-clearance case."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from visual_servo_mujoco.camera_guided_cooperative_pick import run_trial


DEFAULT_FILE_PREFIX = "cooperative_pick_approach_clearance"
TRIALS_FILE = "cooperative-pick-pose-validation-trials.csv"
SUMMARY_FILE = "cooperative-pick-pose-validation-summary.json"
CONTACTS_FILE = "cooperative-pick-contact-events.csv"
FPS = 30
PANEL_SIZE = (960, 468)
PAIR_SIZE = (1920, 624)


def _read_trials(directory: Path) -> dict[tuple[str, str], dict[str, str]]:
    with (directory / TRIALS_FILE).open(encoding="utf-8", newline="") as stream:
        return {
            (row["pair_id"], row["controller"]): row
            for row in csv.DictReader(stream)
        }


def _read_summaries(baseline_dir: Path, candidate_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = json.loads((baseline_dir / SUMMARY_FILE).read_text(encoding="utf-8"))
    candidate = json.loads((candidate_dir / SUMMARY_FILE).read_text(encoding="utf-8"))
    for field in ("release_profile", "wrist_center_radial_standoff_m", "gripper_pad_half_size_m"):
        if baseline.get(field) != candidate.get(field):
            raise ValueError(f"paired video runs differ in {field}")
    for summary, key in ((baseline, "lowest_approach_tool_height_offset_m"), (candidate, "lowest_approach_tool_height_offset_m")):
        if key not in summary:
            raise ValueError(f"run summary is missing {key}")
    if baseline["lowest_approach_tool_height_offset_m"] == candidate["lowest_approach_tool_height_offset_m"]:
        raise ValueError("paired runs must use different approach offsets")
    return baseline, candidate


def _select_pair(
    baseline_dir: Path,
    baseline_rows: dict[tuple[str, str], dict[str, str]],
    candidate_rows: dict[tuple[str, str], dict[str, str]],
    requested_pair_id: str | None,
) -> str:
    eligible = [
        pair_id
        for pair_id, condition in baseline_rows
        if condition == "camera_estimate"
        and baseline_rows[pair_id, condition]["physical_stable_lift_success"] == "False"
        and candidate_rows.get((pair_id, condition), {}).get("physical_stable_lift_success") == "True"
    ]
    if requested_pair_id is not None:
        if requested_pair_id not in eligible:
            raise ValueError("requested pose is not a camera-estimate baseline-fail/candidate-pass pair")
        return requested_pair_id
    if not eligible:
        raise ValueError("no paired camera-estimate baseline failure became a candidate success")

    contact_path = baseline_dir / CONTACTS_FILE
    table_contacts: set[str] = set()
    if contact_path.exists():
        with contact_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if (
                    row.get("controller") == "camera_estimate"
                    and row.get("category") == "arm_environment"
                    and row.get("other_geom") == "table"
                    and row.get("phase") in {"LOWER TO GRASP", "CLOSE SIX FINGERS"}
                ):
                    table_contacts.add(row["pair_id"])
    return next((pair_id for pair_id in eligible if pair_id in table_contacts), eligible[0])


def _render_pair_video(
    baseline_path: Path,
    candidate_path: Path,
    output_path: Path,
    preview_path: Path,
    *,
    pair_id: str,
    baseline_mm: int,
    candidate_mm: int,
    baseline_result: dict[str, Any],
    candidate_result: dict[str, Any],
) -> dict[str, int]:
    captures = [cv2.VideoCapture(str(baseline_path)), cv2.VideoCapture(str(candidate_path))]
    if not all(capture.isOpened() for capture in captures):
        for capture in captures:
            capture.release()
        raise RuntimeError("could not open one of the rendered individual MP4s")
    frame_counts = [int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) for capture in captures]
    if frame_counts[0] != frame_counts[1] or frame_counts[0] <= 0:
        for capture in captures:
            capture.release()
        raise RuntimeError(f"paired videos have different frame counts: {frame_counts}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, PAIR_SIZE
    )
    if not writer.isOpened():
        writer.release()
        for capture in captures:
            capture.release()
        raise RuntimeError("could not open paired comparison MP4 writer")

    gif_frames: list[Image.Image] = []
    gif_durations: list[int] = []
    contact_frame_indices: set[int] = set()
    for contact in (baseline_result.get("first_unexpected_contact") or {}).values():
        if contact is None:
            continue
        center = round(float(contact["time_s"]) * FPS)
        contact_frame_indices.update(range(max(0, center - 1), center + 3))
    frame_index = 0
    try:
        while True:
            ok_base, base_frame = captures[0].read()
            ok_candidate, candidate_frame = captures[1].read()
            if not ok_base and not ok_candidate:
                break
            if not ok_base or not ok_candidate:
                raise RuntimeError("paired videos ended at different frames")
            base_panel = cv2.resize(base_frame, PANEL_SIZE, interpolation=cv2.INTER_AREA)
            candidate_panel = cv2.resize(candidate_frame, PANEL_SIZE, interpolation=cv2.INTER_AREA)
            paired = np.zeros((PAIR_SIZE[1], PAIR_SIZE[0], 3), dtype=np.uint8)
            paired[:] = (20, 28, 40)
            cv2.putText(
                paired,
                f"MATCHED HELD-OUT POSE {pair_id}  |  CAMERA-ESTIMATE CONTROL  |  SIMULATION",
                (22, 30), cv2.FONT_HERSHEY_DUPLEX, 0.72, (238, 244, 250), 1, cv2.LINE_AA,
            )
            cv2.putText(
                paired, f"BASELINE APPROACH {baseline_mm} mm", (22, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (201, 214, 228), 1, cv2.LINE_AA,
            )
            cv2.putText(
                paired, f"CLEARANCE APPROACH {candidate_mm} mm", (982, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (78, 219, 170), 1, cv2.LINE_AA,
            )
            paired[74 : 74 + PANEL_SIZE[1], : PANEL_SIZE[0]] = base_panel
            paired[74 : 74 + PANEL_SIZE[1], PANEL_SIZE[0] :] = candidate_panel
            cv2.line(paired, (960, 74), (960, 542), (92, 111, 132), 2)
            base_state = "STRICT PHYSICAL SCORE: PASS" if baseline_result["physical_stable_lift_success"] else "STRICT PHYSICAL SCORE: FAIL"
            candidate_state = "STRICT PHYSICAL SCORE: PASS" if candidate_result["physical_stable_lift_success"] else "STRICT PHYSICAL SCORE: FAIL"
            cv2.putText(
                paired, base_state, (22, 579), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (83, 209, 157) if baseline_result["physical_stable_lift_success"] else (113, 121, 234), 1, cv2.LINE_AA,
            )
            cv2.putText(
                paired, candidate_state, (982, 579), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (83, 209, 157) if candidate_result["physical_stable_lift_success"] else (113, 121, 234), 1, cv2.LINE_AA,
            )
            confirmation_note = (
                "FRESH IMAGE CHECK: BASELINE "
                f"{baseline_result['fresh_visual_confirmation'].upper()}  |  CANDIDATE "
                f"{candidate_result['fresh_visual_confirmation'].upper()}  "
                "(TWO SAMPLES; NOT A FULL-HOLD CERTIFICATE)"
            )
            note_size = cv2.getTextSize(
                confirmation_note, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
            )[0]
            cv2.putText(
                paired,
                confirmation_note,
                ((PAIR_SIZE[0] - note_size[0]) // 2, 610),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (175, 190, 205),
                1,
                cv2.LINE_AA,
            )
            writer.write(paired)
            if frame_index % 8 == 0 or frame_index in contact_frame_indices:
                preview = cv2.resize(paired, (1200, 390), interpolation=cv2.INTER_AREA)
                gif_frames.append(Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)))
                gif_durations.append(230 if frame_index in contact_frame_indices else round(8 * 1000 / FPS))
            frame_index += 1
    finally:
        writer.release()
        for capture in captures:
            capture.release()

    if not gif_frames:
        raise RuntimeError("paired replay did not produce GIF frames")
    gif_frames[0].save(
        preview_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=gif_durations,
        loop=0,
        optimize=True,
    )
    return {"paired_video_frames": frame_index, "preview_frames": len(gif_frames)}


def generate_pair_demos(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    output_dir: Path,
    pair_id: str | None = None,
    file_prefix: str = DEFAULT_FILE_PREFIX,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_]+", file_prefix):
        raise ValueError("file prefix may contain only letters, numbers, and underscores")
    baseline_video = f"{file_prefix}_5mm.mp4"
    baseline_gif = f"{file_prefix}_5mm.gif"
    candidate_video = f"{file_prefix}_20mm.mp4"
    candidate_gif = f"{file_prefix}_20mm.gif"
    pair_video = f"{file_prefix}_pair.mp4"
    pair_gif = f"{file_prefix}_pair.gif"
    metadata_file = f"{file_prefix}_pair.json"
    baseline_rows = _read_trials(baseline_dir)
    candidate_rows = _read_trials(candidate_dir)
    baseline_summary, candidate_summary = _read_summaries(baseline_dir, candidate_dir)
    selected = _select_pair(baseline_dir, baseline_rows, candidate_rows, pair_id)
    key = (selected, "camera_estimate")
    baseline_row, candidate_row = baseline_rows[key], candidate_rows[key]
    target_xy = (float(baseline_row["target_x_m"]), float(baseline_row["target_y_m"]))
    if target_xy != (float(candidate_row["target_x_m"]), float(candidate_row["target_y_m"])):
        raise ValueError(f"target coordinates differ between profiles for {selected}")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = run_trial(
        target_xy,
        controller="camera_estimate",
        video_path=output_dir / baseline_video,
        preview_path=output_dir / baseline_gif,
        finger_pad_half_size_m=float(baseline_summary["gripper_pad_half_size_m"]),
        release_profile=baseline_summary["release_profile"],
        wrist_center_radial_standoff_m=float(baseline_summary["wrist_center_radial_standoff_m"]),
        lowest_approach_tool_height_offset_m=float(baseline_summary["lowest_approach_tool_height_offset_m"]),
        visual_verification=True,
        visual_verification_min_upward_displacement_px=50.0,
        visual_verification_max_marker_rotation_from_start_deg=4.0,
    )
    candidate_metrics = run_trial(
        target_xy,
        controller="camera_estimate",
        video_path=output_dir / candidate_video,
        preview_path=output_dir / candidate_gif,
        finger_pad_half_size_m=float(candidate_summary["gripper_pad_half_size_m"]),
        release_profile=candidate_summary["release_profile"],
        wrist_center_radial_standoff_m=float(candidate_summary["wrist_center_radial_standoff_m"]),
        lowest_approach_tool_height_offset_m=float(candidate_summary["lowest_approach_tool_height_offset_m"]),
        visual_verification=True,
        visual_verification_min_upward_displacement_px=50.0,
        visual_verification_max_marker_rotation_from_start_deg=4.0,
    )
    for label, replay, recorded in (
        ("baseline", baseline_metrics, baseline_row),
        ("candidate", candidate_metrics, candidate_row),
    ):
        for field in ("physical_stable_lift_success", "payload_stable_lift_success", "full_sequence_collision_safe"):
            if str(replay[field]) != recorded[field]:
                raise RuntimeError(f"{label} replay did not reproduce {field} for {selected}")

    baseline_mm = round(float(baseline_summary["lowest_approach_tool_height_offset_m"]) * 1000)
    candidate_mm = round(float(candidate_summary["lowest_approach_tool_height_offset_m"]) * 1000)
    counts = _render_pair_video(
        output_dir / baseline_video,
        output_dir / candidate_video,
        output_dir / pair_video,
        output_dir / pair_gif,
        pair_id=selected,
        baseline_mm=baseline_mm,
        candidate_mm=candidate_mm,
        baseline_result=baseline_metrics,
        candidate_result=candidate_metrics,
    )
    metadata: dict[str, Any] = {
        "evidence_type": "deterministic paired held-out MuJoCo replay; illustrative demo is already counted in the benchmark",
        "pair_id": selected,
        "target_xy_m": list(target_xy),
        "controller": "camera_estimate",
        "shared_settings": {
            "release_profile": baseline_summary["release_profile"],
            "wrist_center_radial_standoff_m": baseline_summary["wrist_center_radial_standoff_m"],
            "finger_pad_half_size_m": baseline_summary["gripper_pad_half_size_m"],
        },
        "baseline": {
            "lowest_approach_tool_height_offset_m": baseline_summary["lowest_approach_tool_height_offset_m"],
            "recorded_failure_reason": baseline_row["failure_reason"],
            "recorded_physical_stable_lift_success": baseline_row["physical_stable_lift_success"] == "True",
            "replay_metrics": baseline_metrics,
        },
        "candidate": {
            "lowest_approach_tool_height_offset_m": candidate_summary["lowest_approach_tool_height_offset_m"],
            "recorded_failure_reason": candidate_row["failure_reason"],
            "recorded_physical_stable_lift_success": candidate_row["physical_stable_lift_success"] == "True",
            "replay_metrics": candidate_metrics,
        },
        "file_prefix": file_prefix,
        "videos": [baseline_video, candidate_video, pair_video],
        "previews": [baseline_gif, candidate_gif, pair_gif],
        "paired_video_counts": counts,
        "camera_policy": "one initial overhead image plans the motion; a fixed side camera supplies two fresh RGB samples at 8.4 s and 8.8 s for an image-only post-lift decision; simulator state is used only afterward for evaluation",
        "fresh_visual_confirmation": {
            "baseline_5mm": baseline_metrics["fresh_visual_confirmation"],
            "candidate_20mm": candidate_metrics["fresh_visual_confirmation"],
        },
        "visual_truth_is_evaluation_only": True,
        "visual_verification_scope": "two fresh samples; not continuous hold, release, or full-cycle collision certification",
        "aggregate_counting": "This selected pose is already included in the paired benchmark dataset and must not be counted again.",
    }
    (output_dir / metadata_file).write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--pair-id", help="optional exact paired pose; auto-selects a logged pad/table failure if omitted")
    parser.add_argument("--output-dir", type=Path, default=Path("assets/demos"))
    parser.add_argument("--file-prefix", default=DEFAULT_FILE_PREFIX)
    args = parser.parse_args()
    result = generate_pair_demos(
        args.baseline_dir,
        args.candidate_dir,
        output_dir=args.output_dir,
        pair_id=args.pair_id,
        file_prefix=args.file_prefix,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
