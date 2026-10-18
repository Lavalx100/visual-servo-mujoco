"""Held-out calibration checks on recorded checkerboard images.

The default protocol uses a small public OpenCV sample sequence. It fits camera
intrinsics on named development images, freezes those intrinsics, then estimates
only each held-out board pose before measuring pixel reprojection residuals.
This is an image-model consistency check, not ground-truth calibration or robot
reaching evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence
from xml.sax.saxutils import escape

import cv2
import numpy as np

OPENCV_SAMPLE_REVISION = "9b6287ec500ef7b327ac74c4f6c9cd84a2cbc9cd"
OPENCV_SAMPLE_SOURCE = (
    "https://github.com/opencv/opencv/tree/"
    f"{OPENCV_SAMPLE_REVISION}/samples/data"
)
OPENCV_SAMPLE_EXCLUDED_IMAGES = {
    "left11.jpg": (
        "The 9x6 target pattern is not present; the sample contains a "
        "9x7 inner-corner pattern instead."
    )
}
OPENCV_SAMPLE_EXCLUDED_PATTERN_SIZES = {"left11.jpg": (9, 7)}
DEFAULT_TRAIN_IMAGES = tuple(f"left{index:02d}.jpg" for index in range(1, 9))
DEFAULT_VALIDATION_IMAGES = ("left09.jpg", "left12.jpg", "left13.jpg", "left14.jpg")
DEFAULT_PATTERN_SIZE = (9, 6)  # internal corners: columns, rows
RADIUS_BINS_PX = (0.0, 80.0, 160.0, 240.0, 320.0, math.inf)
MODEL_FLAGS = {
    "zero_distortion": (
        cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
    ),
    "radial_k1_k2": cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,
}


def _validate_image_names(
    train_images: Sequence[str], validation_images: Sequence[str]
) -> None:
    if len(train_images) < 3:
        raise ValueError("at least three training images are required")
    if not validation_images:
        raise ValueError("at least one held-out validation image is required")
    all_names = [*train_images, *validation_images]
    if any(not name or Path(name).name != name for name in all_names):
        raise ValueError("image names must be plain filenames within image_dir")
    if len(set(all_names)) != len(all_names):
        raise ValueError("training and validation filenames must be unique and disjoint")


def _detect_corners(
    image_bgr: np.ndarray, pattern_size: tuple[int, int]
) -> tuple[np.ndarray | None, str | None]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
    found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
    if found and corners is not None:
        return corners.reshape(-1, 2).astype(np.float32), "sector_based"

    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=classic_flags)
    if not found or corners is None:
        return None, None
    refined = cv2.cornerSubPix(
        gray,
        corners.astype(np.float32),
        (5, 5),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001),
    )
    return refined.reshape(-1, 2), "classic_refined"


def _object_points(pattern_size: tuple[int, int]) -> np.ndarray:
    columns, rows = pattern_size
    grid_x, grid_y = np.meshgrid(np.arange(columns), np.arange(rows))
    points = np.zeros((rows * columns, 3), dtype=np.float32)
    points[:, :2] = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
    # Intrinsic calibration is invariant to this arbitrary scale. No physical
    # checker-square measurement is needed for the reprojection experiment.
    return points


def _radius_label(lower: float, upper: float) -> str:
    upper_text = f"{upper:.0f}" if math.isfinite(upper) else "inf"
    return f"{lower:.0f}_{upper_text}px"


def _residual_metrics(
    observed: np.ndarray,
    projected: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[dict, list[float]]:
    residual = observed - projected
    errors = np.linalg.norm(residual, axis=1)
    principal = camera_matrix[:2, 2]
    radial_vectors = observed - principal
    radii = np.linalg.norm(radial_vectors, axis=1)
    radial_unit = radial_vectors / np.maximum(radii[:, None], 1e-12)
    radial_residual = np.sum(residual * radial_unit, axis=1)
    tangential_residual = (
        radial_unit[:, 0] * residual[:, 1]
        - radial_unit[:, 1] * residual[:, 0]
    )

    by_radius = []
    for lower, upper in zip(RADIUS_BINS_PX[:-1], RADIUS_BINS_PX[1:]):
        mask = (radii >= lower) & (radii < upper)
        selected = errors[mask]
        by_radius.append(
            {
                "radius_bin": _radius_label(lower, upper),
                "corner_count": int(np.count_nonzero(mask)),
                "reprojection_rms_px": (
                    float(np.sqrt(np.mean(np.square(selected))))
                    if selected.size
                    else None
                ),
                "reprojection_p95_px": (
                    float(np.quantile(selected, 0.95)) if selected.size else None
                ),
                "median_signed_radial_residual_px": (
                    float(np.median(radial_residual[mask]))
                    if selected.size
                    else None
                ),
                "median_signed_tangential_residual_px": (
                    float(np.median(tangential_residual[mask]))
                    if selected.size
                    else None
                ),
            }
        )

    return (
        {
            "corner_count": int(len(errors)),
            "reprojection_rms_px": float(np.sqrt(np.mean(np.square(errors)))),
            "reprojection_median_px": float(np.median(errors)),
            "reprojection_p95_px": float(np.quantile(errors, 0.95)),
            "radius_bins": by_radius,
        },
        errors.astype(float).tolist(),
    )


def _fit_model(
    model_name: str,
    object_points: np.ndarray,
    training_corners: list[np.ndarray],
    image_size: tuple[int, int],
) -> dict:
    object_views = [object_points.copy() for _ in training_corners]
    rms, camera_matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_views,
        [corners.reshape(-1, 1, 2).copy() for corners in training_corners],
        image_size,
        None,
        None,
        flags=MODEL_FLAGS[model_name],
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-10),
    )
    train_errors = []
    for corners, rotation, translation in zip(
        training_corners, rotations, translations
    ):
        projected, _ = cv2.projectPoints(
            object_points, rotation, translation, camera_matrix, distortion
        )
        train_errors.extend(
            np.linalg.norm(corners - projected.reshape(-1, 2), axis=1).tolist()
        )
    return {
        "name": model_name,
        "calibration_rms_px": float(rms),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "training_corner_rms_px": float(np.sqrt(np.mean(np.square(train_errors)))),
        "_camera_matrix": camera_matrix,
        "_distortion": distortion,
    }


def run_public_camera_calibration_validation(
    image_dir: Path,
    *,
    train_images: Sequence[str] = DEFAULT_TRAIN_IMAGES,
    validation_images: Sequence[str] = DEFAULT_VALIDATION_IMAGES,
    pattern_size: tuple[int, int] = DEFAULT_PATTERN_SIZE,
    output_path: Path | None = None,
) -> dict:
    """Fit and compare frozen camera models on held-out recorded images."""
    _validate_image_names(train_images, validation_images)
    if len(pattern_size) != 2 or any(value < 2 for value in pattern_size):
        raise ValueError("pattern_size must be (columns, rows), each at least two")
    root = Path(image_dir)
    if not root.is_dir():
        raise ValueError(f"image_dir does not exist or is not a directory: {root}")

    detections = {}
    image_size = None
    for split, names in (("training", train_images), ("validation", validation_images)):
        for name in names:
            image_path = root / name
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                if split == "training":
                    raise ValueError(f"cannot read required training image: {image_path}")
                detections[name] = {
                    "split": split,
                    "status": "unavailable_image_decode_failed",
                    "detector": None,
                    "corners": None,
                }
                continue
            current_size = (int(image.shape[1]), int(image.shape[0]))
            if image_size is None:
                image_size = current_size
            elif current_size != image_size:
                raise ValueError(
                    f"all images must share a size; {name} is {current_size}, "
                    f"expected {image_size}"
                )
            corners, detector = _detect_corners(image, pattern_size)
            detections[name] = {
                "split": split,
                "status": (
                    "available"
                    if corners is not None
                    else "unavailable_pattern_not_detected"
                ),
                "detector": detector,
                "corners": corners,
            }
    if image_size is None:
        raise ValueError("no readable images were found")

    missing_training = [
        name for name in train_images if detections[name]["corners"] is None
    ]
    if missing_training:
        raise ValueError(
            "checkerboard detection failed for training images: "
            + ", ".join(missing_training)
        )

    object_points = _object_points(pattern_size)
    training_corners = [detections[name]["corners"] for name in train_images]
    models = {}
    for model_name in MODEL_FLAGS:
        fitted = _fit_model(model_name, object_points, training_corners, image_size)
        model_report = {
            key: value
            for key, value in fitted.items()
            if not key.startswith("_")
        }
        validation_rows = []
        all_validation_errors = []
        for name in validation_images:
            detection = detections[name]
            corners = detection["corners"]
            if corners is None:
                validation_rows.append(
                    {
                        "image": name,
                        "status": detection["status"],
                        "detector": detection["detector"],
                        "reprojection_rms_px": None,
                        "reprojection_median_px": None,
                        "reprojection_p95_px": None,
                        "corner_count": 0,
                        "radius_bins": [],
                    }
                )
                continue
            solved, rotation, translation = cv2.solvePnP(
                object_points,
                corners.reshape(-1, 1, 2),
                fitted["_camera_matrix"],
                fitted["_distortion"],
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not solved:
                validation_rows.append(
                    {
                        "image": name,
                        "status": "unavailable_pose_estimation_failed",
                        "detector": detection["detector"],
                        "reprojection_rms_px": None,
                        "reprojection_median_px": None,
                        "reprojection_p95_px": None,
                        "corner_count": 0,
                        "radius_bins": [],
                    }
                )
                continue
            projected, _ = cv2.projectPoints(
                object_points,
                rotation,
                translation,
                fitted["_camera_matrix"],
                fitted["_distortion"],
            )
            metrics, errors = _residual_metrics(
                corners, projected.reshape(-1, 2), fitted["_camera_matrix"]
            )
            all_validation_errors.extend(errors)
            validation_rows.append(
                {
                    "image": name,
                    "status": "available",
                    "detector": detection["detector"],
                    **metrics,
                }
            )
        available = [
            row for row in validation_rows if row["reprojection_rms_px"] is not None
        ]
        model_report["validation"] = {
            "requested_images": len(validation_images),
            "available_images": len(available),
            "availability_rate": len(available) / len(validation_images),
            "pooled_corner_rms_px": (
                float(np.sqrt(np.mean(np.square(all_validation_errors))))
                if all_validation_errors
                else None
            ),
            "median_image_rms_px": (
                float(np.median([row["reprojection_rms_px"] for row in available]))
                if available
                else None
            ),
            "p95_image_rms_px": (
                float(
                    np.quantile(
                        [row["reprojection_rms_px"] for row in available], 0.95
                    )
                )
                if available
                else None
            ),
            "images": validation_rows,
        }
        models[model_name] = model_report

    paired = []
    zero_by_name = {
        row["image"]: row
        for row in models["zero_distortion"]["validation"]["images"]
    }
    radial_by_name = {
        row["image"]: row for row in models["radial_k1_k2"]["validation"]["images"]
    }
    for name in validation_images:
        zero = zero_by_name[name]["reprojection_rms_px"]
        radial = radial_by_name[name]["reprojection_rms_px"]
        paired.append(
            {
                "image": name,
                "zero_distortion_rms_px": zero,
                "radial_k1_k2_rms_px": radial,
                "radial_minus_zero_px": (
                    None if zero is None or radial is None else radial - zero
                ),
            }
        )

    detected_training = [
        name for name in train_images if detections[name]["corners"] is not None
    ]
    excluded_audit = {}
    for name, reason in OPENCV_SAMPLE_EXCLUDED_IMAGES.items():
        image_path = root / name
        audit = {
            "reason": reason,
            "detected_pattern_size_columns_rows": list(
                OPENCV_SAMPLE_EXCLUDED_PATTERN_SIZES[name]
            ),
            "status": "unavailable_image_missing",
            "detector": None,
        }
        if image_path.is_file():
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is not None:
                other_size = OPENCV_SAMPLE_EXCLUDED_PATTERN_SIZES[name]
                corners, detector = _detect_corners(image, other_size)
                audit["status"] = (
                    "excluded_other_pattern_detected"
                    if corners is not None
                    else "excluded_reason_not_reproduced"
                )
                audit["detector"] = detector
        excluded_audit[name] = audit
    report = {
        "schema_version": 1,
        "experiment": "recorded_checkerboard_intrinsics_heldout_validation",
        "scope": (
            "Real recorded pixels from the OpenCV sample camera; descriptive held-out "
            "reprojection consistency only. This is not calibration ground truth, "
            "this project's camera, robot reaching, or fresh visual confirmation."
        ),
        "dataset": {
            "name": "OpenCV left-camera calibration sample images",
            "repository": "opencv/opencv",
            "revision": OPENCV_SAMPLE_REVISION,
            "source": OPENCV_SAMPLE_SOURCE,
            "image_size_px": list(image_size),
            "pattern_inner_corners_columns_rows": list(pattern_size),
            "training_images": list(train_images),
            "validation_images": list(validation_images),
            "excluded_images": excluded_audit,
            "pattern_detection": {
                name: {
                    "split": detections[name]["split"],
                    "status": detections[name]["status"],
                    "detector": detections[name]["detector"],
                }
                for name in [*train_images, *validation_images]
            },
        },
        "models": models,
        "model_definitions": {
            "zero_distortion": (
                "Pinhole intrinsics with radial and tangential distortion fixed to zero."
            ),
            "radial_k1_k2": (
                "Pinhole intrinsics with k1 and k2 fitted; tangential distortion and "
                "k3 fixed to zero."
            ),
        },
        "paired_validation": paired,
        "interpretation": {
            "selection": (
                "The image split and model families are fixed in this benchmark; "
                "held-out corners do not enter calibrateCamera."
            ),
            "training_images_with_detected_pattern": len(detected_training),
            "heldout_image_count": len(validation_images),
            "confidence_intervals": None,
            "limitations": [
                "Only one public camera sequence and four held-out images are evaluated.",
                "The camera's true intrinsics are not supplied as independent ground truth.",
                "Held-out board pose is estimated with solvePnP while intrinsics stay frozen.",
                "The held-out sequence has no checkerboard corners beyond 240 px radius.",
                "This is an exploratory split, not a preregistered study.",
                (
                    "Low reprojection error does not establish accurate metric depth "
                    "or robot reaching."
                ),
            ],
        },
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def render_validation_figure(report: dict, output_path: Path) -> None:
    """Write a compact SVG scorecard without redistributing source photographs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    zero = report["models"]["zero_distortion"]["validation"]["images"]
    radial = report["models"]["radial_k1_k2"]["validation"]["images"]
    radial_by_name = {row["image"]: row for row in radial}
    zero_by_name = {row["image"]: row for row in zero}
    names = report["dataset"]["validation_images"]

    width, height = 1120, 510
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#17212b}.muted{fill:#526171}'
        '.axis{stroke:#aab5c1;stroke-width:1}.grid{stroke:#dce2e9;stroke-width:1}'
        '.panel{fill:#fff;stroke:#d7dee7;stroke-width:1.2}</style>',
        '<text x="36" y="42" font-size="24" font-weight="700">'
        "Recorded-camera calibration: held-out images</text>",
        '<text x="36" y="69" class="muted" font-size="14">OpenCV sample sequence · '
        f'{len(report["dataset"]["training_images"])} calibration images · '
        f'{len(names)} held-out images · 640×480</text>',
        '<rect class="panel" x="30" y="92" width="520" height="338" rx="8"/>',
        '<rect class="panel" x="570" y="92" width="520" height="338" rx="8"/>',
        '<text x="52" y="125" font-size="17" font-weight="700">Per-image held-out RMS</text>',
        '<text x="592" y="125" font-size="17" font-weight="700">Error by corner radius</text>',
    ]

    zero_color = "#687789"
    radial_color = "#117a9b"
    svg.extend(
        [
            f'<rect x="345" y="108" width="12" height="12" fill="{zero_color}"/>',
            '<text x="363" y="119" font-size="12">zero distortion</text>',
            f'<rect x="452" y="108" width="12" height="12" fill="{radial_color}"/>',
            '<text x="470" y="119" font-size="12">radial k₁,k₂</text>',
        ]
    )

    chart_x, chart_y, chart_w = 158, 160, 346
    max_rms = max(
        2.0,
        max(
            (
                value
                for row in [*zero, *radial]
                if (value := row.get("reprojection_rms_px")) is not None
            ),
            default=0.0,
        )
        * 1.15,
    )
    for tick in range(5):
        value = max_rms * tick / 4.0
        x = chart_x + chart_w * tick / 4.0
        svg.append(f'<line class="grid" x1="{x:.1f}" y1="150" x2="{x:.1f}" y2="390"/>')
        svg.append(
            f'<text x="{x:.1f}" y="410" text-anchor="middle" class="muted" '
            f'font-size="11">{value:.1f}</text>'
        )
    svg.append(
        '<text x="331" y="423" text-anchor="middle" class="muted" '
        'font-size="11">reprojection RMS (px)</text>'
    )
    for index, name in enumerate(names):
        y = 166 + index * 54
        svg.append(f'<text x="52" y="{y+20}" font-size="12">{escape(name)}</text>')
        for offset, (row, color) in enumerate(
            ((zero_by_name[name], zero_color), (radial_by_name[name], radial_color))
        ):
            value = row.get("reprojection_rms_px")
            if value is None:
                svg.append(
                    f'<text x="{chart_x}" y="{y+17+offset*20}" class="muted" '
                    'font-size="11">unavailable</text>'
                )
                continue
            bar_y = y + offset * 20
            bar_w = chart_w * value / max_rms
            svg.append(
                f'<rect x="{chart_x}" y="{bar_y}" width="{bar_w:.2f}" '
                f'height="13" rx="2" fill="{color}"/>'
            )
            svg.append(
                f'<text x="{chart_x+bar_w+5:.1f}" y="{bar_y+11}" '
                f'font-size="11">{value:.2f}</text>'
            )

    radius_bins = []
    radius_labels = [
        _radius_label(lower, upper)
        for lower, upper in zip(RADIUS_BINS_PX[:-1], RADIUS_BINS_PX[1:])
    ]
    for bin_index, bin_label in enumerate(radius_labels):
        zero_count = 0
        radial_count = 0
        zero_sum_sq = 0.0
        radial_sum_sq = 0.0
        for row in zero:
            per_image_bins = row.get("radius_bins", [])
            if bin_index >= len(per_image_bins):
                continue
            bin_row = per_image_bins[bin_index]
            count = bin_row["corner_count"]
            if count and bin_row["reprojection_rms_px"] is not None:
                zero_count += count
                zero_sum_sq += count * bin_row["reprojection_rms_px"] ** 2
        for row in radial:
            per_image_bins = row.get("radius_bins", [])
            if bin_index >= len(per_image_bins):
                continue
            bin_row = per_image_bins[bin_index]
            count = bin_row["corner_count"]
            if count and bin_row["reprojection_rms_px"] is not None:
                radial_count += count
                radial_sum_sq += count * bin_row["reprojection_rms_px"] ** 2
        zero_rms = math.sqrt(zero_sum_sq / zero_count) if zero_count else None
        radial_rms = math.sqrt(radial_sum_sq / radial_count) if radial_count else None
        radius_bins.append(
            (bin_label, zero_rms, zero_count, radial_rms, radial_count)
        )

    radius_x, radius_y, radius_h, radius_w = 620, 170, 196, 410
    max_radius_rms = max(
        2.0,
        max(
            (
                value
                for row in radius_bins
                for value in (row[1], row[3])
                if value is not None
            ),
            default=0.0,
        )
        * 1.15,
    )
    for tick in range(5):
        value = max_radius_rms * tick / 4.0
        y = radius_y + radius_h - radius_h * tick / 4.0
        svg.append(f'<line class="grid" x1="612" y1="{y:.1f}" x2="1067" y2="{y:.1f}"/>')
        svg.append(
            f'<text x="604" y="{y+4:.1f}" text-anchor="end" class="muted" '
            f'font-size="10">{value:.1f}</text>'
        )
    group_w = radius_w / max(1, len(radius_bins))
    bar_w = min(27.0, group_w * 0.28)
    for index, (
        label,
        zero_rms,
        zero_count,
        radial_rms,
        radial_count,
    ) in enumerate(radius_bins):
        center_x = radius_x + group_w * (index + 0.5)
        for offset, value, count, color in (
            (-bar_w / 2 - 2, zero_rms, zero_count, zero_color),
            (2, radial_rms, radial_count, radial_color),
        ):
            if value is None:
                continue
            bar_h = radius_h * value / max_radius_rms
            x = center_x + offset
            y = radius_y + radius_h - bar_h
            svg.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{bar_h:.1f}" rx="2" fill="{color}"/>'
            )
            svg.append(
                f'<text x="{x+bar_w/2:.1f}" '
                f'y="{max(radius_y+11,y-4):.1f}" text-anchor="middle" '
                f'font-size="9">{value:.2f}</text>'
            )
        svg.append(
            f'<text x="{center_x:.1f}" y="392" text-anchor="middle" '
            f'class="muted" font-size="10">{escape(label)}</text>'
        )
        svg.append(
            f'<text x="{center_x:.1f}" y="407" text-anchor="middle" '
            f'class="muted" font-size="9">n={zero_count}/{radial_count}</text>'
        )
    svg.append(
        '<text x="823" y="423" text-anchor="middle" class="muted" '
        'font-size="11">radius from estimated principal point · '
        'corners per model</text>'
    )
    zero_pooled = report["models"]["zero_distortion"]["validation"]["pooled_corner_rms_px"]
    radial_pooled = report["models"]["radial_k1_k2"]["validation"]["pooled_corner_rms_px"]
    lower_count = sum(
        1
        for name in names
        if (
            zero_by_name[name].get("reprojection_rms_px") is not None
            and radial_by_name[name].get("reprojection_rms_px") is not None
            and radial_by_name[name]["reprojection_rms_px"]
            < zero_by_name[name]["reprojection_rms_px"]
        )
    )
    image_count = len(names)
    nonempty_radius = [
        index
        for index, row in enumerate(radius_bins)
        if row[2] or row[4]
    ]
    max_radius_text = (
        radius_bins[max(nonempty_radius)][0].split("_", 1)[1].removesuffix("px")
        if nonempty_radius
        else "unknown"
    )
    zero_text = "unavailable" if zero_pooled is None else f"{zero_pooled:.3f} px"
    radial_text = "unavailable" if radial_pooled is None else f"{radial_pooled:.3f} px"
    svg.append(
        '<text x="36" y="462" font-size="13" font-weight="700">'
        f'Radial k₁,k₂ pooled held-out RMS: {radial_text}; '
        f'zero-distortion: {zero_text}; lower on {lower_count}/{image_count} images.'
        '</text>'
    )
    svg.append(
        '<text x="36" y="484" class="muted" font-size="12">'
        'Four held-out images; no independent camera ground truth; '
        f'no corners beyond {max_radius_text} px.'
        '</text>'
    )
    svg.append("</svg>")
    output_path.write_text("\n".join(svg) + "\n")


def export_validation_csv(report: dict, output_path: Path) -> None:
    """Export per-image and per-radius validation summaries as a compact CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "model",
        "record_type",
        "validation_image",
        "radius_bin",
        "corner_count",
        "reprojection_rms_px",
        "reprojection_median_px",
        "reprojection_p95_px",
        "median_signed_radial_residual_px",
        "median_signed_tangential_residual_px",
        "status",
    )
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for model_name, model in report["models"].items():
            for row in model["validation"]["images"]:
                writer.writerow(
                    {
                        "model": model_name,
                        "record_type": "image",
                        "validation_image": row["image"],
                        "radius_bin": "",
                        "corner_count": row["corner_count"],
                        "reprojection_rms_px": row["reprojection_rms_px"],
                        "reprojection_median_px": row["reprojection_median_px"],
                        "reprojection_p95_px": row["reprojection_p95_px"],
                        "status": row["status"],
                    }
                )
                for radius_row in row.get("radius_bins", []):
                    writer.writerow(
                        {
                            "model": model_name,
                            "record_type": "radius_bin",
                            "validation_image": row["image"],
                            "radius_bin": radius_row["radius_bin"],
                            "corner_count": radius_row["corner_count"],
                            "reprojection_rms_px": radius_row["reprojection_rms_px"],
                            "reprojection_p95_px": radius_row["reprojection_p95_px"],
                            "median_signed_radial_residual_px": radius_row[
                                "median_signed_radial_residual_px"
                            ],
                            "median_signed_tangential_residual_px": radius_row[
                                "median_signed_tangential_residual_px"
                            ],
                            "status": row["status"],
                        }
                    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare frozen camera models on held-out checkerboard photos."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("artifacts/opencv-calibration-samples"),
    )
    parser.add_argument("--train-images", nargs="+", default=DEFAULT_TRAIN_IMAGES)
    parser.add_argument(
        "--validation-images", nargs="+", default=DEFAULT_VALIDATION_IMAGES
    )
    parser.add_argument("--pattern-columns", type=int, default=DEFAULT_PATTERN_SIZE[0])
    parser.add_argument("--pattern-rows", type=int, default=DEFAULT_PATTERN_SIZE[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/public-camera-calibration-validation.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        help="Optional SVG output path for the held-out residual scorecard.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional CSV output path for per-image and per-radius scores.",
    )
    args = parser.parse_args()
    report = run_public_camera_calibration_validation(
        args.image_dir,
        train_images=args.train_images,
        validation_images=args.validation_images,
        pattern_size=(args.pattern_columns, args.pattern_rows),
        output_path=args.output,
    )
    if args.figure is not None:
        render_validation_figure(report, args.figure)
        print(f"Wrote {args.figure}")
    if args.csv is not None:
        export_validation_csv(report, args.csv)
        print(f"Wrote {args.csv}")
    for model_name, model in report["models"].items():
        validation = model["validation"]
        print(
            f"{model_name}: train RMS={model['training_corner_rms_px']:.3f}px, "
            f"held-out images={validation['available_images']}/"
            f"{validation['requested_images']}, pooled held-out RMS="
            f"{validation['pooled_corner_rms_px']:.3f}px"
        )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
