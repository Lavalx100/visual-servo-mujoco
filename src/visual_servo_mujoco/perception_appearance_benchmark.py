"""Rendered-scene appearance stress test for the baseline red-target detector.

This is a perception-localization diagnostic. It does not run the arm controller
or score physical reaching or terminal visual confirmation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median
from typing import Iterable
from xml.sax.saxutils import escape

import cv2
import mujoco
import numpy as np

from .controller import pixel_to_table_xy, table_xy_to_pixel
from .run import IMAGE_HEIGHT, IMAGE_WIDTH, detect_red_target

CAMERA_HEIGHT_M = 2.0
CAMERA_FOVY_DEG = 40.0
TARGET_CENTER_Z_M = 0.165
TARGET_TOP_Z_M = 0.190
TARGET_RADIUS_M = 0.045
TARGET_RADIUS_PX = (
    TARGET_RADIUS_M
    * IMAGE_HEIGHT
    / (2.0 * (CAMERA_HEIGHT_M - TARGET_TOP_Z_M)
       * math.tan(math.radians(CAMERA_FOVY_DEG) / 2.0))
)


@dataclass(frozen=True)
class AppearanceCondition:
    lighting: str
    clutter: str
    occlusion_fraction: float

    @property
    def name(self) -> str:
        fraction = f"{self.occlusion_fraction:.2f}".replace(".", "p")
        return f"{self.lighting}__{self.clutter}__occlusion-{fraction}"


LIGHTING_CONDITIONS = ("nominal", "dim", "bright", "side")
CLUTTER_CONDITIONS = ("none", "neutral", "red_distractor", "mixed")
OCCLUSION_FRACTIONS = (0.0, 0.25, 0.50, 0.75)
ALL_CONDITIONS = tuple(
    AppearanceCondition(lighting, clutter, occlusion)
    for lighting in LIGHTING_CONDITIONS
    for clutter in CLUTTER_CONDITIONS
    for occlusion in OCCLUSION_FRACTIONS
)
DEMO_CONDITIONS = (
    AppearanceCondition("nominal", "none", 0.0),
    AppearanceCondition("dim", "none", 0.0),
    AppearanceCondition("nominal", "neutral", 0.0),
    AppearanceCondition("nominal", "red_distractor", 0.0),
    AppearanceCondition("nominal", "none", 0.50),
    AppearanceCondition("dim", "mixed", 0.50),
)

_LIGHTING = {
    "nominal": ((0.0, 0.0, 3.0), 0.8),
    "dim": ((0.0, 0.0, 3.0), 0.16),
    "bright": ((0.0, 0.0, 3.0), 1.35),
    "side": ((0.65, -0.35, 2.4), 0.8),
}
_NEUTRAL_PROPS = (
    (
        "neutral_box_a",
        '<geom type="box" size="0.035 0.030 0.015" rgba="0.28 0.36 0.46 1" '
        'contype="0" conaffinity="0"/>',
        -0.13,
        0.085,
        0.155,
    ),
    (
        "neutral_cylinder",
        '<geom type="cylinder" size="0.028 0.016" rgba="0.32 0.48 0.39 1" '
        'contype="0" conaffinity="0"/>',
        -0.035,
        -0.145,
        0.156,
    ),
    (
        "neutral_box_b",
        '<geom type="box" size="0.024 0.043 0.018" rgba="0.48 0.41 0.29 1" '
        'contype="0" conaffinity="0"/>',
        0.155,
        -0.105,
        0.158,
    ),
)
_RED_DISTRACTOR = (
    "red_distractor",
    '<geom type="cylinder" size="0.066 0.014" rgba="0.95 0.08 0.04 1" '
    'contype="0" conaffinity="0"/>',
    0.12,
    0.035,
    0.156,
)
DemoFrame = tuple[np.ndarray, tuple[float, float], tuple[float, float] | None]


def _clutter_props(clutter: str) -> tuple[tuple[str, str, float, float, float], ...]:
    if clutter == "none":
        return ()
    if clutter == "neutral":
        return _NEUTRAL_PROPS
    if clutter == "red_distractor":
        return (_RED_DISTRACTOR,)
    if clutter == "mixed":
        return (*_NEUTRAL_PROPS, _RED_DISTRACTOR)
    raise ValueError(f"unknown clutter condition: {clutter}")


def build_appearance_scene_xml(condition: AppearanceCondition) -> str:
    """Build the target-only diagnostic scene with physical lighting/occluders."""
    if condition.lighting not in _LIGHTING:
        raise ValueError(f"unknown lighting condition: {condition.lighting}")
    if not 0.0 <= condition.occlusion_fraction <= 0.75:
        raise ValueError("occlusion_fraction must be between 0 and 0.75")
    light_position, light_scale = _LIGHTING[condition.lighting]
    light_pos = " ".join(f"{value:g}" for value in light_position)
    light_diffuse = " ".join(f"{light_scale:g}" for _ in range(3))

    bodies = [
        '<body name="target" pos="0 0 0.165" mocap="true">'
        '<geom name="target_visual" type="cylinder" size="0.045 0.025" '
        'rgba="0.95 0.08 0.04 1" contype="0" conaffinity="0"/></body>'
    ]
    for name, geom_xml, _, _, _ in _clutter_props(condition.clutter):
        bodies.append(
            f'<body name="{name}" pos="0 0 0.155" mocap="true">{geom_xml}</body>'
        )
    if condition.occlusion_fraction > 0.0:
        half_width = TARGET_RADIUS_M * condition.occlusion_fraction
        bodies.append(
            '<body name="target_occluder" pos="0 0 0.202" mocap="true">'
            f'<geom type="box" size="{half_width:.6f} 0.060 0.010" '
            'rgba="0.24 0.27 0.31 1" contype="0" conaffinity="0"/></body>'
        )

    return f"""<mujoco model="appearance_perception_diagnostic">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  <visual>
    <global offwidth="{IMAGE_WIDTH}" offheight="{IMAGE_HEIGHT}"/>
    <headlight ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/>
  </visual>
  <worldbody>
    <light name="key" pos="{light_pos}" directional="false" diffuse="{light_diffuse}"
           ambient="0.02 0.02 0.02" specular="0.05 0.05 0.05"/>
    <camera name="overhead" pos="0 0 2" fovy="{CAMERA_FOVY_DEG:g}"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.12 0.14 0.17 1"/>
    <geom name="table" type="box" pos="0 0 0.07" size="0.75 0.75 0.07"
          rgba="0.78 0.73 0.62 1"/>
    {''.join(bodies)}
  </worldbody>
</mujoco>"""


def _body_mocap_ids(model: mujoco.MjModel, body_names: Iterable[str]) -> dict[str, int]:
    result = {}
    for name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"scene body is missing: {name}")
        mocap_id = int(model.body_mocapid[body_id])
        if mocap_id < 0:
            raise ValueError(f"scene body is not mocap-controlled: {name}")
        result[name] = mocap_id
    return result


def _set_trial_positions(
    data: mujoco.MjData,
    condition: AppearanceCondition,
    mocap_ids: dict[str, int],
    target_xy_m: tuple[float, float],
) -> None:
    x, y = target_xy_m
    positions = {"target": (x, y, TARGET_CENTER_Z_M)}
    for name, _, offset_x, offset_y, z in _clutter_props(condition.clutter):
        positions[name] = (x + offset_x, y + offset_y, z)
    if condition.occlusion_fraction > 0.0:
        positions["target_occluder"] = (x, y, 0.202)
    for name, position in positions.items():
        data.mocap_pos[mocap_ids[name]] = position


def _red_component_count(rgb: np.ndarray) -> int:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    low = cv2.inRange(hsv, np.array([0, 110, 70]), np.array([12, 255, 255]))
    high = cv2.inRange(hsv, np.array([168, 110, 70]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(low, high)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(float(cv2.contourArea(contour)) >= 20.0 for contour in contours)


def _measure_frame(
    rgb: np.ndarray,
    condition: AppearanceCondition,
    *,
    target_xy_m: tuple[float, float],
    seed: int,
    sample_index: int,
) -> dict:
    target_pixel = table_xy_to_pixel(
        target_xy_m,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        camera_height=CAMERA_HEIGHT_M,
        point_height=TARGET_TOP_Z_M,
        vertical_fov_degrees=CAMERA_FOVY_DEG,
    )
    try:
        detected_pixel = detect_red_target(rgb)
        detection_error_px = math.dist(detected_pixel, target_pixel)
        detected_xy_m = pixel_to_table_xy(
            detected_pixel,
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            camera_height=CAMERA_HEIGHT_M,
            target_height=TARGET_TOP_Z_M,
            vertical_fov_degrees=CAMERA_FOVY_DEG,
        )
        detection_error_mm = 1000.0 * math.dist(detected_xy_m, target_xy_m)
        detected = True
    except RuntimeError:
        detected_pixel = None
        detection_error_px = None
        detection_error_mm = None
        detected = False
    inside_radius = (
        detected and detection_error_px <= TARGET_RADIUS_PX
    )
    return {
        "condition": condition.name,
        "lighting": condition.lighting,
        "clutter": condition.clutter,
        "occlusion_bar_width_fraction_of_target_diameter": condition.occlusion_fraction,
        "seed": seed,
        "sample_index": sample_index,
        "target_x_m": target_xy_m[0],
        "target_y_m": target_xy_m[1],
        "target_center_x_px": target_pixel[0],
        "target_center_y_px": target_pixel[1],
        "detected": detected,
        "detected_center_x_px": None if detected_pixel is None else detected_pixel[0],
        "detected_center_y_px": None if detected_pixel is None else detected_pixel[1],
        "red_components_over_20px": _red_component_count(rgb),
        "localization_error_px": detection_error_px,
        "localization_error_mm": detection_error_mm,
        "inside_projected_target_radius": bool(inside_radius),
        "projected_target_radius_px": TARGET_RADIUS_PX,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _summarize_trials(trials: list[dict], condition: AppearanceCondition) -> dict:
    detected = [row for row in trials if row["detected"]]
    errors_px = [float(row["localization_error_px"]) for row in detected]
    errors_mm = [float(row["localization_error_mm"]) for row in detected]
    within = sum(row["inside_projected_target_radius"] for row in trials)
    return {
        "condition": condition.name,
        "lighting": condition.lighting,
        "clutter": condition.clutter,
        "occlusion_bar_width_fraction_of_target_diameter": condition.occlusion_fraction,
        "trials": len(trials),
        "detections": len(detected),
        "detection_rate": len(detected) / len(trials) if trials else None,
        "within_target_radius_localizations": within,
        "within_target_radius_rate": within / len(trials) if trials else None,
        "within_target_radius_rate_given_detection": (
            within / len(detected) if detected else None
        ),
        "median_localization_error_px": median(errors_px) if errors_px else None,
        "p95_localization_error_px": _percentile(errors_px, 95.0),
        "median_localization_error_mm": median(errors_mm) if errors_mm else None,
        "p95_localization_error_mm": _percentile(errors_mm, 95.0),
        "median_red_components_over_20px": median(
            [row["red_components_over_20px"] for row in trials]
        ) if trials else None,
    }


def _paired_comparisons(
    trials: list[dict], summaries: list[dict], baseline_name: str
) -> list[dict]:
    by_condition = {}
    for trial in trials:
        by_condition.setdefault(trial["condition"], {})[
            (trial["seed"], trial["sample_index"])
        ] = trial
    baseline = by_condition[baseline_name]
    results = []
    for summary in summaries:
        current = by_condition[summary["condition"]]
        shared = sorted(set(baseline) & set(current))
        pairs_with_both = [
            key for key in shared if baseline[key]["detected"] and current[key]["detected"]
        ]
        deltas = [
            float(current[key]["localization_error_px"])
            - float(baseline[key]["localization_error_px"])
            for key in pairs_with_both
        ]
        results.append(
            {
                "condition": summary["condition"],
                "baseline_condition": baseline_name,
                "paired_samples": len(shared),
                "pairs_with_both_detections": len(pairs_with_both),
                "median_error_delta_px_vs_nominal_clear": median(deltas) if deltas else None,
                "lower_error_pairs": sum(delta < -1e-9 for delta in deltas),
                "ties": sum(abs(delta) <= 1e-9 for delta in deltas),
                "higher_error_pairs": sum(delta > 1e-9 for delta in deltas),
                "detection_outcomes_changed_vs_baseline": sum(
                    baseline[key]["detected"] != current[key]["detected"] for key in shared
                ),
                "inside_radius_outcomes_changed_vs_baseline": sum(
                    baseline[key]["inside_projected_target_radius"]
                    != current[key]["inside_projected_target_radius"]
                    for key in shared
                ),
            }
        )
    return results


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _scorecard_svg(summaries: list[dict], path: Path) -> None:
    width, height = 1120, 820
    by_cell = {
        (
            row["clutter"],
            row["lighting"],
            row["occlusion_bar_width_fraction_of_target_diameter"],
        ): row
        for row in summaries
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f4f6f8"/>',
        '<text x="40" y="48" font-family="Arial" font-size="26" font-weight="700" '
        'fill="#16202a">Baseline detector center error under rendered appearance stress</text>',
        '<text x="40" y="78" font-family="Arial" font-size="14" fill="#44515e">'
        'Cells show median center localization error and detection availability. '
        'Missing detections are gray; lower error is greener. '
        'All conditions use paired target positions.</text>',
    ]
    panel_positions = {
        "none": (40, 125),
        "neutral": (580, 125),
        "red_distractor": (40, 470),
        "mixed": (580, 470),
    }
    cell_w, cell_h = 82, 44
    row_gap = 50
    for clutter, (x0, y0) in panel_positions.items():
        parts.append(
            f'<text x="{x0}" y="{y0}" font-family="Arial" font-size="18" '
            f'font-weight="700" fill="#16202a">Clutter: {escape(clutter.replace("_", " "))}</text>'
        )
        grid_x = x0 + 92
        grid_y = y0 + 40
        for col_index, fraction in enumerate(OCCLUSION_FRACTIONS):
            cell_x = grid_x + col_index * (cell_w + 5)
            label = "clear" if fraction == 0.0 else f"bar {fraction:.0%}"
            parts.append(
                f'<text x="{cell_x + cell_w / 2:.1f}" y="{grid_y - 8}" text-anchor="middle" '
                f'font-family="Arial" font-size="11" fill="#44515e">{label}</text>'
            )
        for row_index, lighting in enumerate(LIGHTING_CONDITIONS):
            y = grid_y + row_index * row_gap
            parts.append(
                f'<text x="{x0}" y="{y + 27}" font-family="Arial" font-size="12" '
                f'fill="#293744">{escape(lighting)}</text>'
            )
            for col_index, fraction in enumerate(OCCLUSION_FRACTIONS):
                row = by_cell.get((clutter, lighting, fraction))
                error_px = (
                    row["median_localization_error_px"] if row is not None else None
                )
                severity = min(1.0, float(error_px) / 50.0) if error_px is not None else None
                red = int(48 + 176 * severity) if severity is not None else 151
                green = int(193 - 117 * severity) if severity is not None else 160
                blue = 83 if severity is not None else 170
                fill = f"#{red:02x}{green:02x}{blue:02x}"
                cell_x = grid_x + col_index * (cell_w + 5)
                error_label = (
                    f"{float(error_px):.1f} px" if error_px is not None else "NO DETECT"
                )
                detection_rate = float(row["detection_rate"]) if row is not None else 0.0
                parts.append(
                    f'<rect x="{cell_x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                    f'rx="5" fill="{fill}" stroke="#ffffff" stroke-width="1"/>'
                )
                parts.append(
                    f'<text x="{cell_x + cell_w / 2:.1f}" y="{y + 19}" text-anchor="middle" '
                    f'font-family="Arial" font-size="12" font-weight="700" fill="#101820">'
                    f'{error_label}</text>'
                )
                parts.append(
                    f'<text x="{cell_x + cell_w / 2:.1f}" y="{y + 35}" text-anchor="middle" '
                    f'font-family="Arial" font-size="10" fill="#101820">'
                    f'detect {detection_rate:.0%}</text>'
                )
    parts.extend(
        [
            '<text x="40" y="792" font-family="Arial" font-size="12" fill="#44515e">'
            'Localization-only benchmark, no arm/controller, target-only MuJoCo scene. '
            'Full trial table: results/perception-appearance-trials.csv</text>',
            '</svg>',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def _demo_tile(
    rgb: np.ndarray,
    truth_pixel: tuple[float, float],
    detected_pixel: tuple[float, float] | None,
    condition: AppearanceCondition,
    tile_size: tuple[int, int] = (426, 360),
) -> np.ndarray:
    tile_width, tile_height = tile_size
    header_h, image_h = 38, 292
    tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
    tile[:] = (38, 47, 58)
    cv2.rectangle(tile, (0, 0), (tile_width, header_h), (24, 31, 40), -1)
    title = (
        f"{condition.lighting} | {condition.clutter.replace('_', ' ')} | "
        f"bar {condition.occlusion_fraction:.0%}"
    )
    cv2.putText(tile, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.49,
                (245, 248, 250), 1, cv2.LINE_AA)
    resized = cv2.resize(rgb, (tile_width, image_h), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    sx, sy = tile_width / IMAGE_WIDTH, image_h / IMAGE_HEIGHT
    truth = (int(round(truth_pixel[0] * sx)), int(round(truth_pixel[1] * sy)))
    cv2.drawMarker(image, truth, (0, 235, 255), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
    if detected_pixel is not None:
        detected = (int(round(detected_pixel[0] * sx)), int(round(detected_pixel[1] * sy)))
        cv2.circle(image, detected, 9, (255, 220, 20), 2, cv2.LINE_AA)
    tile[header_h : header_h + image_h] = image
    if detected_pixel is None:
        footer = "NO DETECTION"
    else:
        error = math.dist(truth_pixel, detected_pixel)
        footer = f"center error {error:.1f} px | marker radius {TARGET_RADIUS_PX:.1f} px"
    cv2.putText(
        tile,
        footer,
        (10, tile_height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (215, 225, 235),
        1,
        cv2.LINE_AA,
    )
    return tile


def _write_demo(
    demo_frames: dict[int, dict[str, DemoFrame]],
    conditions: tuple[AppearanceCondition, ...],
    video_path: Path,
    preview_path: Path,
) -> None:
    from PIL import Image

    video_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1278, 720
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create demo video: {video_path}")
    gif_frames = []
    try:
        for sample_index in sorted(demo_frames):
            row = demo_frames[sample_index]
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[:] = (27, 35, 44)
            for condition_index, condition in enumerate(conditions):
                image, truth, detected = row[condition.name]
                tile = _demo_tile(image, truth, detected, condition)
                x = (condition_index % 3) * 426
                y = (condition_index // 3) * 360
                canvas[y : y + 360, x : x + 426] = tile
            cv2.putText(canvas, f"Paired target position {sample_index + 1}",
                        (10, 714), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (168, 185, 202), 1, cv2.LINE_AA)
            for _ in range(2):
                writer.write(canvas)
            gif_frames.append(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
    finally:
        writer.release()
    if gif_frames:
        gif_frames[0].save(
            preview_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=350,
            loop=0,
            optimize=True,
        )


def run_perception_appearance_benchmark(
    *,
    seeds: int = 5,
    samples_per_seed: int = 10,
    base_seed: int = 9100,
    conditions: tuple[AppearanceCondition, ...] = ALL_CONDITIONS,
    output_dir: Path | None = None,
    media_dir: Path | None = None,
    save_media: bool = True,
) -> dict:
    """Run paired target-localization trials across appearance conditions."""
    if seeds <= 0 or samples_per_seed <= 0:
        raise ValueError("seeds and samples_per_seed must be positive")
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    if not conditions:
        raise ValueError("at least one appearance condition is required")
    if len({condition.name for condition in conditions}) != len(conditions):
        raise ValueError("appearance conditions must have unique names")

    paired_positions = {}
    for seed_offset in range(seeds):
        seed = base_seed + seed_offset
        rng = np.random.default_rng(seed)
        xs = rng.uniform(-0.40, 0.40, samples_per_seed)
        ys = rng.uniform(-0.32, 0.32, samples_per_seed)
        paired_positions[seed] = list(zip(xs.tolist(), ys.tolist(), strict=True))

    all_trials = []
    demo_frames: dict[int, dict[str, DemoFrame]] = {}
    condition_trials = {}
    for condition in conditions:
        model = mujoco.MjModel.from_xml_string(build_appearance_scene_xml(condition))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
        body_names = ["target", *(item[0] for item in _clutter_props(condition.clutter))]
        if condition.occlusion_fraction > 0.0:
            body_names.append("target_occluder")
        mocap_ids = _body_mocap_ids(model, body_names)
        rows = []
        try:
            for seed, positions in paired_positions.items():
                for sample_index, target_xy_m in enumerate(positions):
                    _set_trial_positions(data, condition, mocap_ids, target_xy_m)
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera="overhead")
                    rgb = renderer.render()
                    trial = _measure_frame(
                        rgb,
                        condition,
                        target_xy_m=target_xy_m,
                        seed=seed,
                        sample_index=sample_index,
                    )
                    rows.append(trial)
                    all_trials.append(trial)
                    if (
                        save_media
                        and seed == base_seed
                        and sample_index < min(10, samples_per_seed)
                        and condition in DEMO_CONDITIONS
                    ):
                        truth = (
                            float(trial["target_center_x_px"]),
                            float(trial["target_center_y_px"]),
                        )
                        detected = (
                            None
                            if not trial["detected"]
                            else (
                                float(trial["detected_center_x_px"]),
                                float(trial["detected_center_y_px"]),
                            )
                        )
                        demo_frames.setdefault(sample_index, {})[condition.name] = (
                            rgb.copy(), truth, detected
                        )
        finally:
            renderer.close()
        condition_trials[condition.name] = rows

    summaries = [
        _summarize_trials(condition_trials[condition.name], condition)
        for condition in conditions
    ]
    baseline = AppearanceCondition("nominal", "none", 0.0).name
    if baseline not in condition_trials:
        baseline = conditions[0].name
    paired = _paired_comparisons(all_trials, summaries, baseline)
    report = {
        "schema_version": 1,
        "benchmark": "mujoco_rendered_appearance_target_localization",
        "detector": (
            "largest red HSV contour and minimum enclosing circle from "
            "run.detect_red_target"
        ),
        "scope": (
            "target-center localization only; no arm controller, physical reach, "
            "or terminal visual confirmation"
        ),
        "seeds": seeds,
        "base_seed": base_seed,
        "samples_per_seed": samples_per_seed,
        "paired_samples_per_condition": seeds * samples_per_seed,
        "projected_target_radius_px": TARGET_RADIUS_PX,
        "lighting_conditions": sorted({condition.lighting for condition in conditions}),
        "clutter_conditions": sorted({condition.clutter for condition in conditions}),
        "occlusion_bar_width_fractions": sorted(
            {condition.occlusion_fraction for condition in conditions}
        ),
        "scene_limitations": [
            "The scene is rendered by MuJoCo; it is not a physical camera or "
            "measured lighting setup.",
            "A target-only tabletop scene isolates perception and excludes arm "
            "self-occlusion.",
            "Clutter placement, object colors, and lighting strengths are authored "
            "benchmark conditions.",
            "The 4.5 cm target-radius localization criterion is not a physical "
            "reaching-success measure.",
        ],
        "condition_summaries": summaries,
        "paired_comparisons_vs_nominal_clear": paired,
    }

    root = Path(__file__).resolve().parents[2]
    output_dir = output_dir or root / "results"
    media_dir = media_dir or root / "assets" / "demos"
    _write_csv(output_dir / "perception-appearance-trials.csv", all_trials)
    _write_csv(output_dir / "perception-appearance-summary.csv", summaries)
    (output_dir / "perception-appearance-summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _scorecard_svg(summaries, media_dir / "perception-appearance-scorecard.svg")
    if save_media:
        available_demo_conditions = tuple(
            condition for condition in DEMO_CONDITIONS if condition.name in condition_trials
        )
        if len(available_demo_conditions) == len(DEMO_CONDITIONS):
            _write_demo(
                demo_frames,
                available_demo_conditions,
                media_dir / "perception_appearance_comparison.mp4",
                media_dir / "previews" / "perception_appearance_comparison.gif",
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--samples-per-seed", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=9100)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--media-dir", type=Path, default=None)
    parser.add_argument("--no-media", action="store_true")
    args = parser.parse_args()
    report = run_perception_appearance_benchmark(
        seeds=args.seeds,
        samples_per_seed=args.samples_per_seed,
        base_seed=args.base_seed,
        output_dir=args.output_dir,
        media_dir=args.media_dir,
        save_media=not args.no_media,
    )
    print(json.dumps({
        "conditions": len(report["condition_summaries"]),
        "trials": sum(row["trials"] for row in report["condition_summaries"]),
        "trial_csv": str((args.output_dir or Path("results")) / "perception-appearance-trials.csv"),
    }, indent=2))


if __name__ == "__main__":
    main()
