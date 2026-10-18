"""Apply the frozen East-view health threshold to new camera poses and lenses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .camera_calibration_health import CalibrationScene, CALIBRATION_CAMERA_VIEWS
from .camera_calibration_validation import DRIFT_SEVERITIES

DEFAULT_THRESHOLD_REPORT = Path("artifacts/camera-calibration-health-validation.json")
DEFAULT_VALIDATION_BASE_SEED = 600
DEFAULT_EXCLUDED_SEED_RANGES = ((400, 404), (500, 504))
RADIAL_K1_VALUES = (-0.04, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.04)
TRANSFER_VIEWS = tuple(CALIBRATION_CAMERA_VIEWS)


def _appearance_sample(rng: np.random.Generator) -> dict:
    """Sample the wider held-out exposure/noise/blur range."""
    return {
        "brightness_scale": float(rng.uniform(0.55, 1.45)),
        "image_noise_std_intensity": float(rng.uniform(0.0, 8.0)),
        "blur_kernel_px": int(rng.choice((1, 3, 5))),
        "image_noise_seed": int(rng.integers(0, 2**32 - 1)),
    }


def _camera_drift_sample(
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
        position_error[int(rng.choice((0, 1)))] = signed(bounds["position"])
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


def _record(
    scene: CalibrationScene,
    *,
    threshold_px: float,
    split: str,
    label: str,
    seed: int,
    sample_index: int,
    appearance: dict,
    camera: dict | None = None,
    distortion: dict | None = None,
) -> dict:
    camera = camera or {}
    distortion = distortion or {}
    camera_settings = {
        key: value for key, value in camera.items() if key != "perturbation_kind"
    }
    row = scene.render(
        condition_name=f"{split}_{label}",
        **camera_settings,
        **appearance,
        **distortion,
    )[1]
    score = row["reprojection_rms_px"]
    row.update(
        {
            "split": split,
            "label": label,
            "seed": seed,
            "sample_index": sample_index,
            "frozen_threshold_rms_px": threshold_px,
            "frozen_drift_decision": score > threshold_px if score is not None else None,
            "frozen_diagnostic_status": (
                "unavailable"
                if score is None
                else "drift_detected"
                if score > threshold_px
                else "healthy"
            ),
            "perturbation_kind": camera.get("perturbation_kind"),
        }
    )
    return row


def _summarize(rows: list[dict], *, expected_drift: bool) -> dict:
    available = [row for row in rows if row["reprojection_rms_px"] is not None]
    scores = np.asarray([row["reprojection_rms_px"] for row in available], dtype=float)
    candidate_scores = np.asarray(
        [
            row["candidate_reprojection_rms_px"]
            for row in rows
            if row["candidate_reprojection_rms_px"] is not None
        ],
        dtype=float,
    )
    alarms = sum(bool(row["frozen_drift_decision"]) for row in available)
    return {
        "samples": len(rows),
        "available_samples": len(available),
        "unavailable_samples": len(rows) - len(available),
        "availability_rate": len(available) / len(rows) if rows else None,
        "median_reprojection_rms_px": float(np.median(scores)) if scores.size else None,
        "p95_reprojection_rms_px": float(np.quantile(scores, 0.95)) if scores.size else None,
        "median_candidate_reprojection_rms_px": (
            float(np.median(candidate_scores)) if candidate_scores.size else None
        ),
        "frozen_alarm_samples": alarms,
        "frozen_alarm_rate_when_available": alarms / len(available) if available else None,
        "missed_drift_samples": (
            len(available) - alarms if expected_drift else None
        ),
        "drift_detection_rate_when_available": (
            alarms / len(available) if expected_drift and available else None
        ),
    }


def _validate_protocol(
    *, seeds: int, samples_per_seed: int, distortion_repeats_per_seed: int,
    validation_base_seed: int, threshold_px: float,
    excluded_seed_ranges: tuple[tuple[int, int], ...],
) -> None:
    if seeds <= 0 or samples_per_seed <= 0 or distortion_repeats_per_seed <= 0:
        raise ValueError("seed and sample counts must be positive")
    if validation_base_seed < 0:
        raise ValueError("validation_base_seed must be non-negative")
    evaluation_seeds = set(range(validation_base_seed, validation_base_seed + seeds))
    for first, last in excluded_seed_ranges:
        if first < 0 or last < first:
            raise ValueError("excluded seed ranges must be non-negative and ordered")
        if evaluation_seeds & set(range(first, last + 1)):
            raise ValueError("transfer evaluation seeds overlap threshold development seeds")
    if not math.isfinite(threshold_px) or threshold_px <= 0.0:
        raise ValueError("threshold_px must be finite and positive")


def run_camera_calibration_transfer_validation(
    *,
    threshold_px: float,
    threshold_source: str = "explicit frozen value",
    seeds: int = 5,
    samples_per_seed: int = 20,
    distortion_repeats_per_seed: int = 5,
    validation_base_seed: int = DEFAULT_VALIDATION_BASE_SEED,
    excluded_seed_ranges: tuple[tuple[int, int], ...] = DEFAULT_EXCLUDED_SEED_RANGES,
    output_path: Path | None = None,
) -> dict:
    """Evaluate a fixed East-trained threshold on new poses and radial distortion.

    This function never estimates or adjusts a threshold from the evaluation data.
    """
    _validate_protocol(
        seeds=seeds,
        samples_per_seed=samples_per_seed,
        distortion_repeats_per_seed=distortion_repeats_per_seed,
        validation_base_seed=validation_base_seed,
        threshold_px=threshold_px,
        excluded_seed_ranges=excluded_seed_ranges,
    )

    view_results = {}
    trial_results = []
    for view_index, view_name in enumerate(TRANSFER_VIEWS):
        view_rows = []
        with CalibrationScene(view_name) as scene:
            for seed in range(validation_base_seed, validation_base_seed + seeds):
                rng = np.random.default_rng(seed * 1_000_003 + view_index)
                for sample_index in range(samples_per_seed):
                    appearance = _appearance_sample(rng)
                    label = "nominal"
                    if sample_index == 0:
                        label = "nominal_dim_probe"
                        appearance = {
                            "brightness_scale": 0.25,
                            "image_noise_std_intensity": 0.0,
                            "blur_kernel_px": 1,
                            "image_noise_seed": int(rng.integers(0, 2**32 - 1)),
                        }
                    view_rows.append(
                        _record(
                            scene,
                            threshold_px=threshold_px,
                            split="heldout_view",
                            label=label,
                            seed=seed,
                            sample_index=sample_index,
                            appearance=appearance,
                        )
                    )

            drift_rows = []
            for severity_index, severity in enumerate(DRIFT_SEVERITIES):
                for seed in range(validation_base_seed, validation_base_seed + seeds):
                    rng = np.random.default_rng(
                        seed * 1_000_003 + view_index * 101 + severity_index
                    )
                    for sample_index in range(samples_per_seed):
                        kind, camera = _camera_drift_sample(rng, severity, sample_index)
                        camera["perturbation_kind"] = kind
                        drift_rows.append(
                            _record(
                                scene,
                                threshold_px=threshold_px,
                                split="heldout_view",
                                label=f"{severity}_drift",
                                seed=seed,
                                sample_index=sample_index,
                                appearance=_appearance_sample(rng),
                                camera=camera,
                            )
                        )

            distortion_rows = []
            for coefficient_index, k1 in enumerate(RADIAL_K1_VALUES):
                for seed in range(validation_base_seed, validation_base_seed + seeds):
                    rng = np.random.default_rng(
                        seed * 1_000_003 + view_index * 10_007 + coefficient_index
                    )
                    for sample_index in range(distortion_repeats_per_seed):
                        distortion_rows.append(
                            _record(
                                scene,
                                threshold_px=threshold_px,
                                split="radial_distortion",
                                label=f"k1_{k1:+.4f}",
                                seed=seed,
                                sample_index=sample_index,
                                appearance=_appearance_sample(rng),
                                distortion={"radial_distortion_k1": k1},
                            )
                        )

        nominal = [row for row in view_rows if row["label"] == "nominal"]
        dim_probe = [row for row in view_rows if row["label"] == "nominal_dim_probe"]
        view_results[view_name] = {
            "camera_position_relative_to_targetbody_m": list(
                CALIBRATION_CAMERA_VIEWS[view_name]
            ),
            "nominal": _summarize(nominal, expected_drift=False),
            "dim_probe": _summarize(dim_probe, expected_drift=False),
            "drift_by_severity": {
                severity: _summarize(
                    [row for row in drift_rows if row["label"] == f"{severity}_drift"],
                    expected_drift=True,
                )
                for severity in DRIFT_SEVERITIES
            },
            "radial_distortion_by_k1": {
                f"{k1:+.4f}": _summarize(
                    [row for row in distortion_rows if row["label"] == f"k1_{k1:+.4f}"],
                    expected_drift=True,
                )
                for k1 in RADIAL_K1_VALUES
            },
        }
        trial_results.extend(view_rows)
        trial_results.extend(drift_rows)
        trial_results.extend(distortion_rows)

    report = {
        "benchmark": "frozen_camera_calibration_threshold_pose_and_lens_transfer",
        "protocol": {
            "scope": (
                "standalone diagnostic only; it measures landmark visibility and a "
                "generic camera-model mismatch alarm, not reaching or terminal visual confirmation"
            ),
            "threshold_px": threshold_px,
            "threshold_source": threshold_source,
            "threshold_policy": "frozen without recalibration or tuning on transfer samples",
            "seed_range": [validation_base_seed, validation_base_seed + seeds - 1],
            "excluded_threshold_development_seed_ranges": [
                [first, last] for first, last in excluded_seed_ranges
            ],
            "samples_per_seed_per_view_class": samples_per_seed,
            "distortion_repeats_per_seed_per_coefficient": distortion_repeats_per_seed,
            "views": {
                name: list(CALIBRATION_CAMERA_VIEWS[name]) for name in TRANSFER_VIEWS
            },
            "nominal_appearance_ranges": {
                "brightness_scale": [0.55, 1.45],
                "image_noise_std_intensity": [0.0, 8.0],
                "blur_kernel_px": [1, 3, 5],
            },
            "drift_ranges": {
                "subtle": "FOV 0.5-1.5 deg, position 3-6 mm, or yaw 0.25-0.75 deg",
                "moderate": "FOV 1.5-3 deg, position 6-15 mm, or yaw 0.75-2 deg",
                "severe": "FOV 3-5 deg, position 15-25 mm, or yaw 2-4 deg",
            },
            "radial_distortion": (
                "OpenCV synthetic image warp after MuJoCo rendering; Brown-Conrady "
                "radial k1 coefficients from -0.04 to +0.04 with k2=0; the health "
                "projection stays pinhole. A flag indicates mismatch, not its cause."
            ),
            "independence_and_limits": (
                "Threshold was selected on the East-view calibration split in the "
                "referenced prior report. Evaluation uses separate seeds and altered "
                "synthetic camera poses in the same MuJoCo scene. This is not evidence "
                "from independent physical cameras, a calibrated lens, or reaching trials."
            ),
        },
        "views": view_results,
        "trial_results": trial_results,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _load_reference(path: Path) -> tuple[float, tuple[tuple[int, int], ...]]:
    report = json.loads(path.read_text())
    threshold = report["threshold_selection"]["threshold_rms_px"]
    if not math.isfinite(float(threshold)) or float(threshold) <= 0.0:
        raise ValueError(f"reference report has invalid threshold: {path}")
    protocol = report["protocol"]
    excluded = []
    for key in ("calibration_seed_range", "validation_seed_range"):
        seed_range = protocol.get(key)
        if not isinstance(seed_range, list) or len(seed_range) != 2:
            raise ValueError(f"reference report has no valid {key}: {path}")
        excluded.append((int(seed_range[0]), int(seed_range[1])))
    return float(threshold), tuple(excluded)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transfer a frozen camera-health threshold to new poses and radial distortion."
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--samples-per-seed", type=int, default=20)
    parser.add_argument("--distortion-repeats-per-seed", type=int, default=5)
    parser.add_argument("--validation-base-seed", type=int, default=DEFAULT_VALIDATION_BASE_SEED)
    parser.add_argument("--threshold-report", type=Path, default=DEFAULT_THRESHOLD_REPORT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/camera-calibration-transfer-validation.json"),
    )
    args = parser.parse_args()
    threshold_px, excluded_seed_ranges = _load_reference(args.threshold_report)
    report = run_camera_calibration_transfer_validation(
        threshold_px=threshold_px,
        threshold_source=str(args.threshold_report),
        seeds=args.seeds,
        samples_per_seed=args.samples_per_seed,
        distortion_repeats_per_seed=args.distortion_repeats_per_seed,
        validation_base_seed=args.validation_base_seed,
        excluded_seed_ranges=excluded_seed_ranges,
        output_path=args.output,
    )
    compact_views = []
    for name, result in report["views"].items():
        nominal = result["nominal"]
        drift = result["drift_by_severity"]
        radial = result["radial_distortion_by_k1"]
        compact_views.append(
            f"{name}: nominal alarms {nominal['frozen_alarm_samples']}/"
            f"{nominal['available_samples']} (unavailable {nominal['unavailable_samples']}); "
            "shift alarms "
            + ", ".join(
                f"{severity} {metrics['frozen_alarm_samples']}/"
                f"{metrics['available_samples']} (unavailable {metrics['unavailable_samples']})"
                for severity, metrics in drift.items()
            )
            + "; k1 alarm rates "
            + ", ".join(
                f"{k1} {radial[k1]['frozen_alarm_samples']}/"
                f"{radial[k1]['available_samples']}"
                for k1 in ("-0.0050", "+0.0050", "+0.0100", "+0.0200")
            )
        )
    print(
        json.dumps(
            {
                "benchmark": report["benchmark"],
                "frozen_threshold_px": threshold_px,
                "view_summaries": compact_views,
                "trial_count": len(report["trial_results"]),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
