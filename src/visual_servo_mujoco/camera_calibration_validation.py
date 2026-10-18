"""Seed-separated threshold selection and held-out validation for camera health."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .camera_calibration_health import CalibrationScene

DEFAULT_CALIBRATION_BASE_SEED = 400
DEFAULT_VALIDATION_BASE_SEED = 500
DEFAULT_MAX_FALSE_ALARM_RATE = 0.05
DRIFT_SEVERITIES = ("subtle", "moderate", "severe")


def _validate_seed_splits(
    *, seeds: int, samples_per_seed: int, calibration_base_seed: int,
    validation_base_seed: int, max_false_alarm_rate: float,
) -> None:
    if seeds <= 0 or samples_per_seed <= 0:
        raise ValueError("seeds and samples_per_seed must be positive")
    if calibration_base_seed < 0 or validation_base_seed < 0:
        raise ValueError("seed bases must be non-negative")
    if set(range(calibration_base_seed, calibration_base_seed + seeds)) & set(
        range(validation_base_seed, validation_base_seed + seeds)
    ):
        raise ValueError("calibration and validation seed ranges must not overlap")
    if not math.isfinite(max_false_alarm_rate) or not 0.0 < max_false_alarm_rate < 1.0:
        raise ValueError("max_false_alarm_rate must be between 0 and 1")


def _appearance_sample(rng: np.random.Generator, *, calibration: bool) -> dict:
    if calibration:
        brightness = float(rng.uniform(0.70, 1.30))
        noise = float(rng.uniform(0.0, 4.0))
        blur = int(rng.choice((1, 3)))
    else:
        brightness = float(rng.uniform(0.55, 1.45))
        noise = float(rng.uniform(0.0, 8.0))
        blur = int(rng.choice((1, 3, 5)))
    return {
        "brightness_scale": brightness,
        "image_noise_std_intensity": noise,
        "blur_kernel_px": blur,
        "image_noise_seed": int(rng.integers(0, 2**32 - 1)),
    }


def _drift_sample(
    rng: np.random.Generator, severity: str, sample_index: int
) -> tuple[str, dict]:
    ranges = {
        "subtle": {"fovy": (0.5, 1.5), "position": (0.003, 0.006), "angle": (0.25, 0.75)},
        "moderate": {"fovy": (1.5, 3.0), "position": (0.006, 0.015), "angle": (0.75, 2.0)},
        "severe": {"fovy": (3.0, 5.0), "position": (0.015, 0.025), "angle": (2.0, 4.0)},
    }
    bounds = ranges[severity]
    kind = ("fovy", "position", "orientation", "combined")[sample_index % 4]
    signed = lambda low_high: float(rng.choice((-1.0, 1.0)) * rng.uniform(*low_high))
    fovy_error = 0.0
    position_error = [0.0, 0.0, 0.0]
    rotation_error = [0.0, 0.0, 0.0]
    if kind in ("fovy", "combined"):
        fovy_error = signed(bounds["fovy"])
    if kind == "position":
        axis = int(rng.choice((0, 1)))
        position_error[axis] = signed(bounds["position"])
    if kind == "orientation":
        rotation_error[2] = signed(bounds["angle"])
    if kind == "combined":
        position_error[0] = signed(bounds["position"])
        rotation_error[2] = signed(bounds["angle"])
    return kind, {
        "fovy_error_deg": fovy_error,
        "camera_position_error_m": tuple(position_error),
        "rotation_error_deg": tuple(rotation_error),
    }


def _make_record(
    scene: CalibrationScene,
    *,
    split: str,
    label: str,
    seed: int,
    sample_index: int,
    camera: dict,
    appearance: dict,
) -> dict:
    condition = f"{split}_{label}"
    camera_settings = {
        key: value for key, value in camera.items() if key != "perturbation_kind"
    }
    _, row = scene.render(
        condition_name=condition,
        **camera_settings,
        **appearance,
    )
    return {
        **row,
        "split": split,
        "label": label,
        "seed": seed,
        "sample_index": sample_index,
        "perturbation_kind": camera.get("perturbation_kind"),
    }


def _make_split_records(
    scene: CalibrationScene,
    *,
    seeds: int,
    samples_per_seed: int,
    base_seed: int,
    calibration: bool,
) -> list[dict]:
    split = "calibration" if calibration else "validation"
    records = []
    for seed in range(base_seed, base_seed + seeds):
        rng = np.random.default_rng(seed)
        for sample_index in range(samples_per_seed):
            appearance = _appearance_sample(rng, calibration=calibration)
            camera: dict = {}
            label = "nominal"
            if not calibration and sample_index == 0:
                label = "nominal_dim_probe"
                appearance = {
                    "brightness_scale": 0.25,
                    "image_noise_std_intensity": 0.0,
                    "blur_kernel_px": 1,
                    "image_noise_seed": int(rng.integers(0, 2**32 - 1)),
                }
            records.append(
                _make_record(
                    scene,
                    split=split,
                    label=label,
                    seed=seed,
                    sample_index=sample_index,
                    camera=camera,
                    appearance=appearance,
                )
            )
    return records


def _make_drift_records(
    scene: CalibrationScene, *, seeds: int, samples_per_seed: int,
    base_seed: int, split: str, calibration: bool,
) -> list[dict]:
    records = []
    for severity in DRIFT_SEVERITIES:
        for seed in range(base_seed, base_seed + seeds):
            rng = np.random.default_rng(seed * 1009 + DRIFT_SEVERITIES.index(severity))
            for sample_index in range(samples_per_seed):
                perturbation_kind, camera = _drift_sample(rng, severity, sample_index)
                camera["perturbation_kind"] = perturbation_kind
                records.append(
                    _make_record(
                        scene,
                        split=split,
                        label=f"{severity}_drift",
                        seed=seed,
                        sample_index=sample_index,
                        camera=camera,
                        appearance=_appearance_sample(rng, calibration=calibration),
                    )
                )
    return records


def _summarize(records: list[dict], *, threshold: float, is_drift: bool) -> dict:
    available = [row for row in records if row["reprojection_rms_px"] is not None]
    scores = [float(row["reprojection_rms_px"]) for row in available]
    positive = sum(score > threshold for score in scores)
    negative = sum(score <= threshold for score in scores)
    return {
        "samples": len(records),
        "available_samples": len(available),
        "unavailable_samples": len(records) - len(available),
        "detection_coverage": len(available) / len(records) if records else None,
        "median_reprojection_rms_px": (
            float(np.median(scores)) if scores else None
        ),
        "p95_reprojection_rms_px": (
            float(np.quantile(scores, 0.95)) if scores else None
        ),
        "drift_detected_samples": positive if is_drift else None,
        "missed_drift_samples": negative if is_drift else None,
        "drift_detection_rate_when_available": (
            positive / len(available) if is_drift and available else None
        ),
        "false_alarm_samples": positive if not is_drift else None,
        "false_alarm_rate_when_available": (
            positive / len(available) if not is_drift and available else None
        ),
    }


def _select_threshold(records: list[dict], max_false_alarm_rate: float) -> dict:
    scores = np.asarray(
        [
            row["reprojection_rms_px"]
            for row in records
            if row["reprojection_rms_px"] is not None
        ],
        dtype=float,
    )
    if scores.size == 0:
        raise RuntimeError("calibration split has no available landmark estimates")
    threshold = float(np.quantile(scores, 1.0 - max_false_alarm_rate, method="higher"))
    return {
        "threshold_rms_px": threshold,
        "target_false_alarm_rate": max_false_alarm_rate,
        "calibration_available_samples": int(scores.size),
        "selection_rule": (
            "empirical upper quantile of nominal calibration residuals; alarms use "
            "strictly greater than the frozen threshold"
        ),
    }


def _select_supervised_threshold(
    nominal_records: list[dict],
    drift_records: list[dict],
    max_false_alarm_rate: float,
) -> dict:
    nominal_scores = np.asarray(
        [row["reprojection_rms_px"] for row in nominal_records
         if row["reprojection_rms_px"] is not None],
        dtype=float,
    )
    drift_scores = np.asarray(
        [row["reprojection_rms_px"] for row in drift_records
         if row["reprojection_rms_px"] is not None],
        dtype=float,
    )
    if nominal_scores.size == 0 or drift_scores.size == 0:
        raise RuntimeError("threshold selection needs nominal and drift calibration samples")
    candidates = {float(np.max(nominal_scores))}
    candidates.update(float(np.nextafter(score, -np.inf)) for score in drift_scores)
    feasible = []
    for threshold in candidates:
        false_alarm_rate = float(np.mean(nominal_scores > threshold))
        if false_alarm_rate <= max_false_alarm_rate:
            detection_rate = float(np.mean(drift_scores > threshold))
            feasible.append((detection_rate, threshold, false_alarm_rate))
    if not feasible:
        raise RuntimeError("no residual threshold satisfies the calibration false-alarm cap")
    detection_rate, threshold, false_alarm_rate = max(feasible)
    return {
        "threshold_rms_px": threshold,
        "target_false_alarm_rate": max_false_alarm_rate,
        "calibration_false_alarm_rate": false_alarm_rate,
        "calibration_drift_detection_rate": detection_rate,
        "calibration_nominal_available_samples": int(nominal_scores.size),
        "calibration_drift_available_samples": int(drift_scores.size),
        "selection_rule": (
            "maximize drift detection on calibration drift examples subject to the "
            "nominal false-alarm cap; ties choose the higher threshold"
        ),
    }


def _threshold_policy_report(
    *,
    threshold: float,
    calibration_nominal: list[dict],
    calibration_drift: list[dict],
    validation_nominal: list[dict],
    validation_drift: list[dict],
) -> dict:
    return {
        "threshold_rms_px": threshold,
        "calibration_nominal": _summarize(
            calibration_nominal, threshold=threshold, is_drift=False
        ),
        "calibration_drift_by_severity": {
            severity: _summarize(
                [row for row in calibration_drift if row["label"] == f"{severity}_drift"],
                threshold=threshold,
                is_drift=True,
            )
            for severity in DRIFT_SEVERITIES
        },
        "validation_nominal": _summarize(
            validation_nominal, threshold=threshold, is_drift=False
        ),
        "validation_drift_by_severity": {
            severity: _summarize(
                [row for row in validation_drift if row["label"] == f"{severity}_drift"],
                threshold=threshold,
                is_drift=True,
            )
            for severity in DRIFT_SEVERITIES
        },
    }


def run_camera_calibration_validation(
    *,
    seeds: int = 5,
    samples_per_seed: int = 20,
    calibration_base_seed: int = DEFAULT_CALIBRATION_BASE_SEED,
    validation_base_seed: int = DEFAULT_VALIDATION_BASE_SEED,
    max_false_alarm_rate: float = DEFAULT_MAX_FALSE_ALARM_RATE,
    output_path: Path | None = None,
) -> dict:
    """Compare nominal-only and supervised threshold selection on held-out renders."""
    _validate_seed_splits(
        seeds=seeds,
        samples_per_seed=samples_per_seed,
        calibration_base_seed=calibration_base_seed,
        validation_base_seed=validation_base_seed,
        max_false_alarm_rate=max_false_alarm_rate,
    )
    with CalibrationScene() as scene:
        calibration_records = _make_split_records(
            scene,
            seeds=seeds,
            samples_per_seed=samples_per_seed,
            base_seed=calibration_base_seed,
            calibration=True,
        )
        calibration_drift_records = _make_drift_records(
            scene,
            seeds=seeds,
            samples_per_seed=samples_per_seed,
            base_seed=calibration_base_seed,
            split="calibration",
            calibration=True,
        )
        validation_nominal = _make_split_records(
            scene,
            seeds=seeds,
            samples_per_seed=samples_per_seed,
            base_seed=validation_base_seed,
            calibration=False,
        )
        drift_records = _make_drift_records(
            scene,
            seeds=seeds,
            samples_per_seed=samples_per_seed,
            base_seed=validation_base_seed,
            split="validation",
            calibration=False,
        )

    nominal_only_selection = _select_threshold(calibration_records, max_false_alarm_rate)
    supervised_selection = _select_supervised_threshold(
        calibration_records,
        calibration_drift_records,
        max_false_alarm_rate,
    )
    # The drift examples are labelled development data used only for the
    # supervised operating point; the validation seeds remain frozen and separate.
    supervised_threshold = supervised_selection["threshold_rms_px"]
    nominal_only_threshold = nominal_only_selection["threshold_rms_px"]
    all_records = (
        calibration_records
        + calibration_drift_records
        + validation_nominal
        + drift_records
    )
    for row in all_records:
        score = row["reprojection_rms_px"]
        row["frozen_threshold_rms_px"] = supervised_threshold
        row["frozen_drift_decision"] = (
            score > supervised_threshold if score is not None else None
        )
        row["nominal_only_threshold_rms_px"] = nominal_only_threshold
        row["nominal_only_drift_decision"] = (
            score > nominal_only_threshold if score is not None else None
        )
        row["fixed_3px_drift_decision"] = (
            score > 3.0 if score is not None else None
        )

    nominal_only_policy = _threshold_policy_report(
        threshold=nominal_only_threshold,
        calibration_nominal=calibration_records,
        calibration_drift=calibration_drift_records,
        validation_nominal=validation_nominal,
        validation_drift=drift_records,
    )
    supervised_policy = _threshold_policy_report(
        threshold=supervised_threshold,
        calibration_nominal=calibration_records,
        calibration_drift=calibration_drift_records,
        validation_nominal=validation_nominal,
        validation_drift=drift_records,
    )
    fixed_policy = _threshold_policy_report(
        threshold=3.0,
        calibration_nominal=calibration_records,
        calibration_drift=calibration_drift_records,
        validation_nominal=validation_nominal,
        validation_drift=drift_records,
    )
    dim_probe = [
        row for row in validation_nominal if row["label"] == "nominal_dim_probe"
    ]
    report = {
        "benchmark": "fixed_landmark_camera_calibration_threshold_validation",
        "protocol": {
            "scope": (
                "standalone camera diagnostic; no reaching controller, physical "
                "success, or terminal visual confirmation is measured"
            ),
            "landmarks": "four fixed magenta tabletop cylinders at known coordinates",
            "image_model": "simulator render with brightness, RGB noise, and Gaussian blur",
            "calibration_seed_range": [
                calibration_base_seed,
                calibration_base_seed + seeds - 1,
            ],
            "validation_seed_range": [
                validation_base_seed,
                validation_base_seed + seeds - 1,
            ],
            "samples_per_seed_per_class": samples_per_seed,
            "calibration_classes": ["nominal", *[f"{x}_drift" for x in DRIFT_SEVERITIES]],
            "validation_classes": ["nominal", *[f"{x}_drift" for x in DRIFT_SEVERITIES]],
            "calibration_appearance_ranges": {
                "brightness_scale": [0.70, 1.30],
                "image_noise_std_intensity": [0.0, 4.0],
                "blur_kernel_px": [1, 3],
            },
            "validation_appearance_ranges": {
                "brightness_scale": [0.55, 1.45],
                "image_noise_std_intensity": [0.0, 8.0],
                "blur_kernel_px": [1, 3, 5],
            },
            "drift_ranges": {
                "subtle": "FOV 0.5-1.5 deg, position 3-6 mm, or yaw 0.25-0.75 deg",
                "moderate": "FOV 1.5-3 deg, position 6-15 mm, or yaw 0.75-2 deg",
                "severe": "FOV 3-5 deg, position 15-25 mm, or yaw 2-4 deg",
                "factors": "FOV, camera position, yaw, or a combined perturbation",
            },
            "unavailable_probe": "one dim exposure (x0.25) per validation seed",
            "independence_note": (
                "Images are repeated renders from one MuJoCo scene and camera; seed "
                "groups vary image noise and perturbations, not physical cameras. "
                "Calibration camera-shift types and severity ranges are repeated "
                "in validation, with fresh parameters and broader image nuisance"
            ),
        },
        "selected_threshold_policy": "supervised_fpr_constrained",
        "threshold_selection": supervised_selection,
        "threshold_comparisons": {
            "nominal_only_p95": {
                "selection": nominal_only_selection,
                **nominal_only_policy,
            },
            "supervised_fpr_constrained": {
                "selection": supervised_selection,
                **supervised_policy,
            },
            "fixed_3px_exploratory": {
                "selection": {
                    "threshold_rms_px": 3.0,
                    "selection_rule": "fixed threshold from the initial one-view experiment",
                },
                **fixed_policy,
            },
        },
        "calibration": supervised_policy["calibration_nominal"],
        "calibration_drift_by_severity": supervised_policy[
            "calibration_drift_by_severity"
        ],
        "validation_nominal": supervised_policy["validation_nominal"],
        "validation_drift_by_severity": supervised_policy[
            "validation_drift_by_severity"
        ],
        "validation_dim_probe": _summarize(
            dim_probe, threshold=supervised_threshold, is_drift=False
        ),
        "calibration_per_seed": [
            {
                "seed": seed,
                **_summarize(
                    [row for row in calibration_records if row["seed"] == seed],
                    threshold=supervised_threshold,
                    is_drift=False,
                ),
            }
            for seed in range(calibration_base_seed, calibration_base_seed + seeds)
        ],
        "calibration_drift_per_seed": [
            {
                "seed": seed,
                "by_severity": {
                    severity: _summarize(
                        [
                            row
                            for row in calibration_drift_records
                            if row["seed"] == seed
                            and row["label"] == f"{severity}_drift"
                        ],
                        threshold=supervised_threshold,
                        is_drift=True,
                    )
                    for severity in DRIFT_SEVERITIES
                },
            }
            for seed in range(calibration_base_seed, calibration_base_seed + seeds)
        ],
        "validation_nominal_per_seed": [
            {
                "seed": seed,
                **_summarize(
                    [row for row in validation_nominal if row["seed"] == seed],
                    threshold=supervised_threshold,
                    is_drift=False,
                ),
            }
            for seed in range(validation_base_seed, validation_base_seed + seeds)
        ],
        "validation_drift_per_seed": [
            {
                "seed": seed,
                "by_severity": {
                    severity: _summarize(
                        [
                            row
                            for row in drift_records
                            if row["seed"] == seed
                            and row["label"] == f"{severity}_drift"
                        ],
                        threshold=supervised_threshold,
                        is_drift=True,
                    )
                    for severity in DRIFT_SEVERITIES
                },
            }
            for seed in range(validation_base_seed, validation_base_seed + seeds)
        ],
        "trial_results": all_records,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze a landmark residual threshold and evaluate held-out renders."
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--samples-per-seed", type=int, default=20)
    parser.add_argument("--calibration-base-seed", type=int, default=400)
    parser.add_argument("--validation-base-seed", type=int, default=500)
    parser.add_argument("--max-false-alarm-rate", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/camera-calibration-health-validation.json"),
    )
    args = parser.parse_args()
    report = run_camera_calibration_validation(
        seeds=args.seeds,
        samples_per_seed=args.samples_per_seed,
        calibration_base_seed=args.calibration_base_seed,
        validation_base_seed=args.validation_base_seed,
        max_false_alarm_rate=args.max_false_alarm_rate,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "benchmark": report["benchmark"],
                "threshold_rms_px": report["threshold_selection"]["threshold_rms_px"],
                "calibration": report["calibration"],
                "validation_nominal": report["validation_nominal"],
                "validation_drift_by_severity": report[
                    "validation_drift_by_severity"
                ],
                "validation_dim_probe": report["validation_dim_probe"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
