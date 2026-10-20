"""Held-out rendered-scene evaluation of an opt-in geometric red-target detector."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median
from typing import Any
from xml.sax.saxutils import escape

import cv2
import mujoco
import numpy as np

from .controller import pixel_to_table_xy, table_xy_to_pixel
from .perception_appearance_benchmark import (
    CAMERA_FOVY_DEG,
    CAMERA_HEIGHT_M,
    TARGET_CENTER_Z_M,
    TARGET_RADIUS_M,
    TARGET_TOP_Z_M,
)
from .perception_candidate import (
    ACCEPTED_RADIUS_ERROR_FRACTION,
    AMBIGUITY_CENTER_DISTANCE_RADIUS_FRACTION,
    AMBIGUITY_COMPETITOR_RADIUS_ERROR_FRACTION,
    AMBIGUITY_RADIUS_ERROR_DIFFERENCE_PX,
    HOUGH_BLUR_KERNEL,
    HOUGH_BLUR_SIGMA,
    HOUGH_DP,
    HOUGH_MIN_DISTANCE_RADIUS_FRACTION,
    HOUGH_PARAM1,
    HOUGH_PARAM2,
    MAX_TARGET_RADIUS_FRACTION,
    MIN_TARGET_RADIUS_FRACTION,
    RED_BRIGHTNESS_FLOOR,
    RED_HUE_INTERVALS,
    RED_SATURATION_FLOOR,
    detect_red_target_circle,
    expected_target_radius_px,
)
from .run import IMAGE_HEIGHT, IMAGE_WIDTH, detect_red_target


LIGHT_INTENSITY_RANGE = (0.08, 1.30)
DISTRACTOR_RADIUS_FACTOR_RANGE = (0.95, 1.55)
DISTRACTOR_DISTANCE_RANGE_M = (0.115, 0.235)
OCCLUSION_OFFSET_RANGE_M = (-0.005, 0.005)
DEVELOPMENT_OCCLUSION_FRACTIONS = (0.18, 0.42, 0.68)
HELD_OUT_LIGHT_INTENSITY_RANGES = ((0.04, 0.07), (1.31, 1.65))
HELD_OUT_LIGHT_POSITION_RANGES = ((-0.95, 0.95), (-0.85, 0.85), (2.00, 2.25))
HELD_OUT_DISTRACTOR_RADIUS_FACTOR_RANGE = (0.80, 0.94)
HELD_OUT_DISTRACTOR_DISTANCE_RANGES_M = ((0.090, 0.110), (0.240, 0.290))
HELD_OUT_OCCLUSION_OFFSET_RANGE_M = (-0.012, 0.012)
HELD_OUT_OCCLUSION_FRACTIONS = (0.28, 0.54, 0.76)
CLUTTER_MODES = ("none", "red_disk", "red_box")
TARGET_RADIUS_PX = expected_target_radius_px(IMAGE_HEIGHT)
STRICT_LOCALIZATION_THRESHOLD_PX = 4.0


@dataclass(frozen=True)
class ValidationCondition:
    """One held-out clutter and occluder-width cell."""

    clutter: str
    occlusion_fraction: float

    @property
    def name(self) -> str:
        width = f"{self.occlusion_fraction:.2f}".replace(".", "p")
        return f"{self.clutter}__occlusion-{width}"


DEVELOPMENT_CONDITIONS = tuple(
    ValidationCondition(clutter, occlusion)
    for clutter in CLUTTER_MODES
    for occlusion in DEVELOPMENT_OCCLUSION_FRACTIONS
)
HELD_OUT_VALIDATION_CONDITIONS = tuple(
    ValidationCondition(clutter, occlusion)
    for clutter in CLUTTER_MODES
    for occlusion in HELD_OUT_OCCLUSION_FRACTIONS
)


def build_validation_scene_xml(condition: ValidationCondition) -> str:
    """Build a scene whose randomized factors are drawn per rendered trial."""
    if condition.clutter not in CLUTTER_MODES:
        raise ValueError(f"unknown clutter mode: {condition.clutter}")
    if not 0.0 < condition.occlusion_fraction < 0.8:
        raise ValueError("occlusion_fraction must be between 0 and 0.8")

    clutter_xml = ""
    if condition.clutter == "red_disk":
        clutter_xml = (
            '<body name="red_distractor" pos="0 0 0.155" mocap="true">'
            '<geom name="red_distractor_visual" type="cylinder" size="0.055 0.014" '
            'rgba="0.95 0.08 0.04 1" contype="0" conaffinity="0"/></body>'
        )
    elif condition.clutter == "red_box":
        clutter_xml = (
            '<body name="red_distractor" pos="0 0 0.155" mocap="true">'
            '<geom name="red_distractor_visual" type="box" size="0.052 0.044 0.014" '
            'rgba="0.95 0.08 0.04 1" contype="0" conaffinity="0"/></body>'
        )

    half_width = TARGET_RADIUS_M * condition.occlusion_fraction
    return f"""<mujoco model="perception_candidate_validation">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  <visual><global offwidth="{IMAGE_WIDTH}" offheight="{IMAGE_HEIGHT}"/>
    <headlight ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/></visual>
  <worldbody>
    <light name="key" pos="0 0 3" directional="false" diffuse="0.8 0.8 0.8"
           ambient="0.02 0.02 0.02" specular="0.05 0.05 0.05"/>
    <camera name="overhead" pos="0 0 2" fovy="{CAMERA_FOVY_DEG:g}"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.12 0.14 0.17 1"/>
    <geom name="table" type="box" pos="0 0 0.07" size="0.75 0.75 0.07"
          rgba="0.78 0.73 0.62 1"/>
    <body name="target" pos="0 0 {TARGET_CENTER_Z_M:g}" mocap="true">
      <geom name="target_visual" type="cylinder" size="{TARGET_RADIUS_M:g} 0.025"
            rgba="0.95 0.08 0.04 1" contype="0" conaffinity="0"/>
    </body>
    {clutter_xml}
    <body name="target_occluder" pos="0 0 0.202" mocap="true">
      <geom name="target_occluder_visual" type="box" size="{half_width:.8f} 0.060 0.010"
            rgba="0.24 0.27 0.31 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>"""


def _body_mocap_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"scene body is missing: {name}")
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise ValueError(f"scene body is not mocap-controlled: {name}")
    return mocap_id


def _target_pixel(target_xy_m: tuple[float, float]) -> tuple[float, float]:
    return table_xy_to_pixel(
        target_xy_m,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        camera_height=CAMERA_HEIGHT_M,
        point_height=TARGET_TOP_Z_M,
        vertical_fov_degrees=CAMERA_FOVY_DEG,
    )


def _pixel_error_m(
    pixel_xy: tuple[float, float], target_xy_m: tuple[float, float]
) -> float:
    estimated_xy = pixel_to_table_xy(
        pixel_xy,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        camera_height=CAMERA_HEIGHT_M,
        target_height=TARGET_TOP_Z_M,
        vertical_fov_degrees=CAMERA_FOVY_DEG,
    )
    return float(math.dist(estimated_xy, target_xy_m))


def _set_occluder_pose(
    data: mujoco.MjData,
    mocap_id: int,
    target_xy_m: tuple[float, float],
    angle_rad: float,
    offset_xy_m: tuple[float, float],
) -> None:
    data.mocap_pos[mocap_id] = (
        target_xy_m[0] + offset_xy_m[0],
        target_xy_m[1] + offset_xy_m[1],
        0.202,
    )
    data.mocap_quat[mocap_id] = (
        math.cos(angle_rad / 2.0),
        0.0,
        0.0,
        math.sin(angle_rad / 2.0),
    )


def _draw_target_cross(frame: np.ndarray, center: tuple[float, float]) -> None:
    x, y = (int(round(value)) for value in center)
    cv2.drawMarker(
        frame,
        (x, y),
        (0, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=16,
        thickness=2,
    )


def _draw_detection(
    frame: np.ndarray,
    center: tuple[float, float] | None,
    color: tuple[int, int, int],
    label: str,
) -> None:
    if center is None:
        cv2.putText(frame, "ABSTAIN", (14, IMAGE_HEIGHT - 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, cv2.LINE_AA)
        return
    x, y = (int(round(value)) for value in center)
    cv2.circle(frame, (x, y), 9, color, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (14, IMAGE_HEIGHT - 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.66, color, 2, cv2.LINE_AA)


def _demo_frames(
    frames: list[dict[str, Any]],
    mp4_path: Path,
    gif_path: Path,
) -> None:
    from PIL import Image

    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    canvas_size = (IMAGE_WIDTH * 2, IMAGE_HEIGHT + 72)
    writer = cv2.VideoWriter(
        str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, canvas_size
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"could not open video output {mp4_path}")

    gif_images = []
    try:
        for item in frames:
            left = cv2.cvtColor(item["rgb"], cv2.COLOR_RGB2BGR)
            right = left.copy()
            truth = item["target_pixel"]
            _draw_target_cross(left, truth)
            _draw_target_cross(right, truth)
            _draw_detection(left, item["baseline_pixel"], (0, 145, 255), "BASELINE")
            _draw_detection(right, item["candidate_pixel"], (255, 220, 0), "CIRCLE PRIOR")
            combined = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
            combined[72:, :IMAGE_WIDTH] = left
            combined[72:, IMAGE_WIDTH:] = right
            cv2.rectangle(combined, (0, 0), (canvas_size[0], 71), (247, 249, 251), -1)
            title = (
                f"{item['condition']} | light={item['light_scale']:.2f} | "
                f"disk x{item['distractor_radius_factor']:.2f} | "
                f"bar={item['occlusion_fraction']:.0%} at {item['occlusion_angle_deg']:.0f} deg"
            )
            cv2.putText(combined, title[:112], (14, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.54, (27, 39, 51), 1, cv2.LINE_AA)
            cv2.putText(combined, "Yellow cross: rendered target center", (14, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 86, 0), 1, cv2.LINE_AA)
            for _ in range(10):
                writer.write(combined)
            gif_images.append(
                Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)).convert("P")
            )
    finally:
        writer.release()
    if gif_images:
        gif_images[0].save(
            gif_path,
            save_all=True,
            append_images=gif_images[1:],
            duration=700,
            loop=0,
            optimize=True,
        )


def _write_scorecard(summaries: list[dict], path: Path) -> None:
    width = 1180
    row_height = 52
    height = 145 + len(summaries) * row_height
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f5f7f9"/>',
        '<text x="34" y="42" font-family="Arial" font-size="25" font-weight="700" '
        'fill="#17212b">Held-out red-target detector validation</text>',
        '<text x="34" y="70" font-family="Arial" font-size="14" fill="#44515e">'
        'Bars: within 4 px (green), imprecise but within target disk (amber), wrong target '
        '(red), and abstention (gray), across paired rendered trials.</text>',
        '<text x="34" y="112" font-family="Arial" font-size="14" font-weight="700" '
        'fill="#17212b">Condition</text>',
        '<text x="400" y="112" font-family="Arial" font-size="14" font-weight="700" '
        'fill="#17212b">Baseline strict / wrong / abstain</text>',
        '<text x="795" y="112" font-family="Arial" font-size="14" font-weight="700" '
        'fill="#17212b">Circle prior strict / wrong / abstain</text>',
    ]
    for index, row in enumerate(summaries):
        y = 145 + index * row_height
        pieces.append(f'<rect x="20" y="{y-19}" width="1140" height="{row_height}" '
                      f'rx="6" fill="{"#ffffff" if index%2==0 else "#edf1f4"}"/>')
        label = f"{row['clutter']} · bar {row['occlusion_fraction']:.0%}"
        pieces.append(f'<text x="34" y="{y+4}" font-family="Arial" font-size="14" '
                      f'fill="#17212b">{escape(label)}</text>')
        for start_x, prefix in ((400, "baseline"), (795, "candidate")):
            strict = row[f"{prefix}_strict_success_rate"] or 0.0
            wrong = row[f"{prefix}_wrong_target_rate"] or 0.0
            abstain = row[f"{prefix}_abstention_rate"] or 0.0
            imprecise = row[f"{prefix}_imprecise_rate"] or 0.0
            pieces.extend(_score_bar(start_x, y - 11, strict, imprecise, wrong, abstain))
            pieces.append(f'<text x="{start_x+300}" y="{y+4}" font-family="Arial" '
                          f'font-size="12" fill="#33404d">{strict:.0%} / '
                          f'{wrong:.0%} / {abstain:.0%}</text>')
    pieces.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def _score_bar(
    x: int, y: int, strict: float, imprecise: float, wrong: float, abstain: float
) -> list[str]:
    segments = []
    cursor = x
    for rate, color in (
        (strict, "#2d8a57"),
        (imprecise, "#e6b84a"),
        (wrong, "#c94c4c"),
        (abstain, "#aab4be"),
    ):
        bar_width = 245 * max(0.0, min(rate, 1.0))
        if bar_width > 0:
            segments.append(f'<rect x="{cursor:.1f}" y="{y}" width="{bar_width:.1f}" '
                            f'height="18" rx="3" fill="{color}"/>')
            cursor += bar_width
    segments.append(f'<rect x="{x}" y="{y}" width="245" height="18" rx="3" '
                    'fill="none" stroke="#7c8995" stroke-width="1"/>')
    return segments


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _summarize_group(rows: list[dict], condition: ValidationCondition) -> dict:
    total = len(rows)
    summary: dict[str, Any] = {
        "condition": condition.name,
        "clutter": condition.clutter,
        "occlusion_fraction": condition.occlusion_fraction,
        "trials": total,
    }
    for prefix in ("baseline", "candidate"):
        detected = [row for row in rows if row[f"{prefix}_detected"]]
        errors = [float(row[f"{prefix}_error_px"]) for row in detected]
        strict = [error <= STRICT_LOCALIZATION_THRESHOLD_PX for error in errors]
        wrong = [error > TARGET_RADIUS_PX for error in errors]
        imprecise = [
            STRICT_LOCALIZATION_THRESHOLD_PX < error <= TARGET_RADIUS_PX for error in errors
        ]
        summary.update(
            {
                f"{prefix}_detections": len(detected),
                f"{prefix}_coverage_rate": len(detected) / total if total else None,
                f"{prefix}_strict_successes": sum(strict),
                f"{prefix}_strict_success_rate": sum(strict) / total if total else None,
                f"{prefix}_strict_success_rate_given_detection": (
                    sum(strict) / len(detected) if detected else None
                ),
                f"{prefix}_imprecise_localizations": sum(imprecise),
                f"{prefix}_imprecise_rate": sum(imprecise) / total if total else None,
                f"{prefix}_wrong_targets": sum(wrong),
                f"{prefix}_wrong_target_rate": sum(wrong) / total if total else None,
                f"{prefix}_wrong_target_rate_given_detection": (
                    sum(wrong) / len(detected) if detected else None
                ),
                f"{prefix}_abstentions": total - len(detected),
                f"{prefix}_abstention_rate": (
                    (total - len(detected)) / total if total else None
                ),
                f"{prefix}_median_error_px": median(errors) if errors else None,
                f"{prefix}_p95_error_px": (
                    float(np.percentile(errors, 95)) if errors else None
                ),
            }
        )
    summary["paired_strict_improvements"] = sum(
        row["candidate_strict_success"] and not row["baseline_strict_success"]
        for row in rows
    )
    summary["paired_strict_regressions"] = sum(
        row["baseline_strict_success"] and not row["candidate_strict_success"]
        for row in rows
    )
    return summary


def run_perception_candidate_validation(
    *,
    seeds: int = 5,
    samples_per_seed: int = 10,
    base_seed: int = 9400,
    split_name: str = "validation",
    ambiguity_radius_error_px: float = AMBIGUITY_RADIUS_ERROR_DIFFERENCE_PX,
    check_partial_arc_competitors: bool = True,
    conditions: tuple[ValidationCondition, ...] | None = None,
    output_dir: Path | None = None,
    media_dir: Path | None = None,
    save_media: bool = True,
) -> dict:
    """Compare the default detector with the opt-in circle prior on new renders."""
    if seeds <= 0 or samples_per_seed <= 0:
        raise ValueError("seeds and samples_per_seed must be positive")
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    if split_name not in {"development", "validation"}:
        raise ValueError("split_name must be 'development' or 'validation'")
    if not math.isfinite(ambiguity_radius_error_px) or ambiguity_radius_error_px < 0.0:
        raise ValueError("ambiguity_radius_error_px must be finite and non-negative")
    if not isinstance(check_partial_arc_competitors, bool):
        raise ValueError("check_partial_arc_competitors must be a boolean")
    if conditions is None:
        conditions = (
            DEVELOPMENT_CONDITIONS
            if split_name == "development"
            else HELD_OUT_VALIDATION_CONDITIONS
        )
    if not conditions or len({item.name for item in conditions}) != len(conditions):
        raise ValueError("conditions must be non-empty and uniquely named")

    positions: dict[int, list[tuple[float, float]]] = {}
    for seed_offset in range(seeds):
        seed = base_seed + seed_offset
        rng = np.random.default_rng(seed)
        xs = rng.uniform(-0.40, 0.40, samples_per_seed)
        ys = rng.uniform(-0.32, 0.32, samples_per_seed)
        positions[seed] = list(zip(xs.tolist(), ys.tolist(), strict=True))

    nuisance_rng = np.random.default_rng(base_seed + 100_000)
    all_trials: list[dict] = []
    grouped_trials: dict[str, list[dict]] = {}
    demo_frames: list[dict] = []
    for condition in conditions:
        model = mujoco.MjModel.from_xml_string(build_validation_scene_xml(condition))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
        target_mocap_id = _body_mocap_id(model, "target")
        occluder_mocap_id = _body_mocap_id(model, "target_occluder")
        clutter_mocap_id = (
            _body_mocap_id(model, "red_distractor")
            if condition.clutter != "none"
            else None
        )
        light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "key")
        distractor_geom_id = (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "red_distractor_visual")
            if condition.clutter != "none"
            else None
        )
        condition_rows = []
        try:
            for seed, target_positions in positions.items():
                for sample_index, target_xy in enumerate(target_positions):
                    if split_name == "development":
                        light_scale = float(nuisance_rng.uniform(*LIGHT_INTENSITY_RANGE))
                        light_position_ranges = (
                            (-0.65, 0.65),
                            (-0.55, 0.55),
                            (2.35, 3.30),
                        )
                        distractor_radius_range = DISTRACTOR_RADIUS_FACTOR_RANGE
                        distractor_distance_ranges = (DISTRACTOR_DISTANCE_RANGE_M,)
                        occlusion_offset_range = OCCLUSION_OFFSET_RANGE_M
                    else:
                        intensity_range = HELD_OUT_LIGHT_INTENSITY_RANGES[
                            int(nuisance_rng.integers(len(HELD_OUT_LIGHT_INTENSITY_RANGES)))
                        ]
                        light_scale = float(nuisance_rng.uniform(*intensity_range))
                        light_position_ranges = HELD_OUT_LIGHT_POSITION_RANGES
                        distractor_radius_range = HELD_OUT_DISTRACTOR_RADIUS_FACTOR_RANGE
                        distractor_distance_ranges = HELD_OUT_DISTRACTOR_DISTANCE_RANGES_M
                        occlusion_offset_range = HELD_OUT_OCCLUSION_OFFSET_RANGE_M
                    light_pos = (
                        float(nuisance_rng.uniform(*light_position_ranges[0])),
                        float(nuisance_rng.uniform(*light_position_ranges[1])),
                        float(nuisance_rng.uniform(*light_position_ranges[2])),
                    )
                    model.light_diffuse[light_id] = (light_scale, light_scale, light_scale)
                    model.light_pos[light_id] = light_pos

                    data.mocap_pos[target_mocap_id] = (
                        target_xy[0], target_xy[1], TARGET_CENTER_Z_M
                    )
                    angle_deg = float(nuisance_rng.uniform(0.0, 180.0))
                    angle_rad = math.radians(angle_deg)
                    occlusion_offset = (
                        float(nuisance_rng.uniform(*occlusion_offset_range)),
                        float(nuisance_rng.uniform(*occlusion_offset_range)),
                    )
                    _set_occluder_pose(
                        data,
                        occluder_mocap_id,
                        target_xy,
                        angle_rad,
                        occlusion_offset,
                    )

                    distractor_radius_factor = 0.0
                    distractor_offset = (0.0, 0.0)
                    if clutter_mocap_id is not None:
                        bearing = float(nuisance_rng.uniform(0.0, 2.0 * math.pi))
                        distance_range = distractor_distance_ranges[
                            int(nuisance_rng.integers(len(distractor_distance_ranges)))
                        ]
                        distance = float(nuisance_rng.uniform(*distance_range))
                        distractor_offset = (
                            distance * math.cos(bearing), distance * math.sin(bearing)
                        )
                        data.mocap_pos[clutter_mocap_id] = (
                            target_xy[0] + distractor_offset[0],
                            target_xy[1] + distractor_offset[1],
                            0.155,
                        )
                        distractor_radius_factor = float(
                            nuisance_rng.uniform(*distractor_radius_range)
                        )
                        if condition.clutter == "red_disk":
                            model.geom_size[distractor_geom_id, 0] = (
                                TARGET_RADIUS_M * distractor_radius_factor
                            )
                        else:
                            side = TARGET_RADIUS_M * distractor_radius_factor
                            model.geom_size[distractor_geom_id, 0:2] = (side, side * 0.82)

                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera="overhead")
                    rgb = renderer.render().copy()
                    target_pixel = _target_pixel(target_xy)
                    try:
                        baseline_pixel = detect_red_target(rgb)
                    except RuntimeError:
                        baseline_pixel = None
                    candidate = detect_red_target_circle(
                        rgb,
                        expected_radius_px=TARGET_RADIUS_PX,
                        ambiguity_radius_error_px=ambiguity_radius_error_px,
                        check_partial_arc_competitors=check_partial_arc_competitors,
                    )
                    candidate_pixel = (
                        None
                        if candidate is None
                        else (candidate.center_x_px, candidate.center_y_px)
                    )

                    baseline_error = (
                        None if baseline_pixel is None
                        else float(math.dist(baseline_pixel, target_pixel))
                    )
                    candidate_error = (
                        None if candidate_pixel is None
                        else float(math.dist(candidate_pixel, target_pixel))
                    )
                    row = {
                        "condition": condition.name,
                        "clutter": condition.clutter,
                        "occlusion_fraction": condition.occlusion_fraction,
                        "seed": seed,
                        "sample_index": sample_index,
                        "target_x_m": target_xy[0],
                        "target_y_m": target_xy[1],
                        "light_scale": light_scale,
                        "light_x_m": light_pos[0],
                        "light_y_m": light_pos[1],
                        "light_z_m": light_pos[2],
                        "distractor_radius_factor": distractor_radius_factor,
                        "distractor_offset_x_m": distractor_offset[0],
                        "distractor_offset_y_m": distractor_offset[1],
                        "occlusion_angle_deg": angle_deg,
                        "occlusion_offset_x_m": occlusion_offset[0],
                        "occlusion_offset_y_m": occlusion_offset[1],
                        "target_center_x_px": target_pixel[0],
                        "target_center_y_px": target_pixel[1],
                        "baseline_detected": baseline_pixel is not None,
                        "baseline_center_x_px": None if baseline_pixel is None else baseline_pixel[0],
                        "baseline_center_y_px": None if baseline_pixel is None else baseline_pixel[1],
                        "baseline_error_px": baseline_error,
                        "baseline_error_mm": (
                            None if baseline_pixel is None
                            else 1000.0 * _pixel_error_m(baseline_pixel, target_xy)
                        ),
                        "baseline_strict_success": (
                            baseline_error is not None
                            and baseline_error <= STRICT_LOCALIZATION_THRESHOLD_PX
                        ),
                        "baseline_wrong_target": (
                            baseline_error is not None and baseline_error > TARGET_RADIUS_PX
                        ),
                        "candidate_detected": candidate is not None,
                        "candidate_center_x_px": (
                            None if candidate is None else candidate.center_x_px
                        ),
                        "candidate_center_y_px": (
                            None if candidate is None else candidate.center_y_px
                        ),
                        "candidate_radius_px": None if candidate is None else candidate.radius_px,
                        "candidate_radius_error_px": (
                            None if candidate is None else candidate.radius_error_px
                        ),
                        "candidate_confidence": (
                            None if candidate is None else candidate.confidence
                        ),
                        "candidate_error_px": candidate_error,
                        "candidate_error_mm": (
                            None if candidate_pixel is None
                            else 1000.0 * _pixel_error_m(candidate_pixel, target_xy)
                        ),
                        "candidate_strict_success": (
                            candidate_error is not None
                            and candidate_error <= STRICT_LOCALIZATION_THRESHOLD_PX
                        ),
                        "candidate_wrong_target": (
                            candidate_error is not None and candidate_error > TARGET_RADIUS_PX
                        ),
                        "candidate_abstained": candidate is None,
                        "projected_target_radius_px": TARGET_RADIUS_PX,
                    }
                    condition_rows.append(row)
                    all_trials.append(row)
                    if save_media and (
                        (seed == base_seed and sample_index == 0)
                        or row["candidate_wrong_target"]
                    ):
                        demo_frames.append(
                            {
                                "condition": condition.name,
                                "rgb": rgb,
                                "target_pixel": target_pixel,
                                "baseline_pixel": baseline_pixel,
                                "candidate_pixel": candidate_pixel,
                                "light_scale": light_scale,
                                "distractor_radius_factor": distractor_radius_factor,
                                "occlusion_fraction": condition.occlusion_fraction,
                                "occlusion_angle_deg": angle_deg,
                            }
                        )
        finally:
            renderer.close()
        grouped_trials[condition.name] = condition_rows

    summaries = [
        _summarize_group(grouped_trials[condition.name], condition)
        for condition in conditions
    ]
    report = {
        "schema_version": 1,
        "benchmark": "held_out_rendered_candidate_detector_validation",
        "scope": (
            "paired target-center localization only; no controller, physical reaching, "
            "or terminal visual confirmation is measured"
        ),
        "baseline": "run.detect_red_target largest red HSV contour",
        "candidate": "opt-in red Hough-circle detector with known projected-size prior",
        "candidate_status": "experimental; not the runtime default",
        "data_split": split_name,
        "seeds": seeds,
        "base_seed": base_seed,
        "condition_names": [condition.name for condition in conditions],
        "samples_per_seed": samples_per_seed,
        "trial_count": len(all_trials),
        "paired_samples_per_condition": seeds * samples_per_seed,
        "projected_target_radius_px": TARGET_RADIUS_PX,
        "strict_localization_threshold_px": STRICT_LOCALIZATION_THRESHOLD_PX,
        "candidate_parameters": {
            "red_hue_intervals": [list(item) for item in RED_HUE_INTERVALS],
            "red_saturation_floor": RED_SATURATION_FLOOR,
            "red_brightness_floor": RED_BRIGHTNESS_FLOOR,
            "hough_blur_kernel": list(HOUGH_BLUR_KERNEL),
            "hough_blur_sigma": HOUGH_BLUR_SIGMA,
            "hough_dp": HOUGH_DP,
            "hough_min_distance_expected_radius_fraction": (
                HOUGH_MIN_DISTANCE_RADIUS_FRACTION
            ),
            "hough_param1": HOUGH_PARAM1,
            "hough_param2": HOUGH_PARAM2,
            "search_radius_fraction": [
                MIN_TARGET_RADIUS_FRACTION,
                MAX_TARGET_RADIUS_FRACTION,
            ],
            "accepted_radius_error_fraction": ACCEPTED_RADIUS_ERROR_FRACTION,
            "ambiguity_center_distance_expected_radius_fraction": (
                AMBIGUITY_CENTER_DISTANCE_RADIUS_FRACTION
            ),
            "ambiguity_competitor_radius_error_fraction": (
                AMBIGUITY_COMPETITOR_RADIUS_ERROR_FRACTION
            ),
            "ambiguity_radius_error_difference_px": ambiguity_radius_error_px,
            "ambiguity_check_partial_arc_competitors": check_partial_arc_competitors,
        },
        "development_protocol": (
            {
                "parameter_development": {
                    "base_seed": 9200,
                    "seeds": 5,
                    "samples_per_seed": 10,
                    "purpose": "parameter development; excluded from the validation score",
                },
                "same_distribution_adversarial_review": {
                    "base_seed": 9300,
                    "seeds": 5,
                    "samples_per_seed": 10,
                    "purpose": (
                        "identified two accepted near-size red distractors; preserved "
                        "as development evidence"
                    ),
                },
                "validation_design": (
                    "uses disjoint light-intensity, distractor-size/distance, and "
                    "occlusion-width ranges"
                ),
                "archived_initial_negative_run": (
                    "results/perception-candidate-pilot-v1-summary.json"
                ),
            }
            if split_name == "validation"
            else None
        ),
        "randomized_factor_ranges": {
            "light_scale": (
                [list(item) for item in HELD_OUT_LIGHT_INTENSITY_RANGES]
                if split_name == "validation"
                else list(LIGHT_INTENSITY_RANGE)
            ),
            "light_position_x_m": list(light_position_ranges[0]),
            "light_position_y_m": list(light_position_ranges[1]),
            "light_position_z_m": list(light_position_ranges[2]),
            "distractor_radius_factor": list(distractor_radius_range),
            "distractor_center_distance_m": [
                list(item) for item in distractor_distance_ranges
            ],
            "occlusion_angle_deg": [0.0, 180.0],
            "occlusion_center_offset_each_axis_m": list(occlusion_offset_range),
            "occlusion_width_fractions_of_target_diameter": [
                sorted({condition.occlusion_fraction for condition in conditions})
            ],
        },
        "metrics": {
            "strict_success": "detected center within 4 px of rendered target center",
            "wrong_target": "detected center farther than one projected target radius",
            "abstention": "detector returned no estimate",
            "physical_reach": "not measured",
            "fresh_visual_confirmation": "not measured",
        },
        "limitations": [
            "All images are rendered by MuJoCo and are not recorded camera data.",
            "The detector assumes the fixed overhead camera and known target diameter.",
            "A same-size red distractor with no distinguishing visual cue is inherently ambiguous.",
            "The randomized conditions are author-defined stress distributions, not a field survey.",
            "No arm self-occlusion, exposure adaptation, motion blur, lens flare, or sensor noise is simulated.",
        ],
        "condition_summaries": summaries,
    }
    root = Path(__file__).resolve().parents[2]
    output_dir = output_dir or root / "results"
    media_dir = media_dir or root / "assets" / "demos"
    _write_csv(output_dir / "perception-candidate-validation-trials.csv", all_trials)
    _write_csv(output_dir / "perception-candidate-validation-summary.csv", summaries)
    (output_dir / "perception-candidate-validation-summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_scorecard(summaries, media_dir / "perception-candidate-validation.svg")
    if save_media:
        _demo_frames(
            demo_frames,
            media_dir / "perception_candidate_validation.mp4",
            media_dir / "previews" / "perception_candidate_validation.gif",
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--samples-per-seed", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=9400)
    parser.add_argument(
        "--split-name", choices=("development", "validation"), default="validation"
    )
    parser.add_argument(
        "--ambiguity-radius-error-px",
        type=float,
        default=AMBIGUITY_RADIUS_ERROR_DIFFERENCE_PX,
    )
    parser.add_argument(
        "--no-partial-arc-competitor-check",
        action="store_true",
        help="recreate the initial candidate rule used by the archived pilot-v1 run",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--media-dir", type=Path)
    parser.add_argument("--no-media", action="store_true")
    args = parser.parse_args()
    report = run_perception_candidate_validation(
        seeds=args.seeds,
        samples_per_seed=args.samples_per_seed,
        base_seed=args.base_seed,
        split_name=args.split_name,
        ambiguity_radius_error_px=args.ambiguity_radius_error_px,
        check_partial_arc_competitors=not args.no_partial_arc_competitor_check,
        output_dir=args.output_dir,
        media_dir=args.media_dir,
        save_media=not args.no_media,
    )
    print(
        json.dumps(
            {
                "benchmark": report["benchmark"],
                "trial_count": report["trial_count"],
                "condition_summaries": report["condition_summaries"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
