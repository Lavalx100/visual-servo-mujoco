"""Evaluate a calibrated second view for fresh terminal visual evidence."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from .run import (
    END_EFFECTOR_HEIGHT,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SUCCESS_THRESHOLD_METERS,
    TARGET_HEIGHT,
    detect_green_end_effector,
    detect_red_target,
    run_trials,
)
from .terminal_verification import (
    IMAGE_HEIGHT as VERIFICATION_IMAGE_HEIGHT,
    IMAGE_WIDTH as VERIFICATION_IMAGE_WIDTH,
    VERIFICATION_VIEWS,
    model_with_verification_views,
    pixel_to_xy_at_height,
)


PIXEL_NOISE_LEVELS_PX = (0.0, 20.0)
def _model_with_verification_views() -> mujoco.MjModel:
    return model_with_verification_views()


def _pixel_to_xy_at_height(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_id: int,
    pixel_xy: tuple[float, float],
    plane_height_m: float,
) -> np.ndarray:
    """Back-project a pixel ray to a horizontal plane using camera calibration."""
    return pixel_to_xy_at_height(
        model,
        data,
        camera_id,
        pixel_xy,
        plane_height_m,
        image_width=VERIFICATION_IMAGE_WIDTH,
        image_height=VERIFICATION_IMAGE_HEIGHT,
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _summarize_records(records: list[dict]) -> dict:
    co_visible = [record for record in records if record["both_markers_visible"]]
    true_positive = sum(
        record["visual_success_estimate"] and record["physical_success"]
        for record in co_visible
    )
    false_positive = sum(
        record["visual_success_estimate"] and not record["physical_success"]
        for record in co_visible
    )
    true_negative = sum(
        not record["visual_success_estimate"] and not record["physical_success"]
        for record in co_visible
    )
    false_negative = sum(
        not record["visual_success_estimate"] and record["physical_success"]
        for record in co_visible
    )
    signed_error_deltas = [record["estimated_minus_physical_error_m"] for record in co_visible]
    return {
        "trials": len(records),
        "overhead_natural_visibility_loss_trials": sum(
            record["overhead_natural_visibility_loss"] for record in records
        ),
        "target_visible_trials": sum(record["target_visible"] for record in records),
        "end_effector_visible_trials": sum(
            record["end_effector_visible"] for record in records
        ),
        "both_markers_visible_trials": len(co_visible),
        "visual_success_estimate_trials": true_positive + false_positive,
        "confusion_counts_when_both_visible": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
        "visual_decision_precision": _rate(true_positive, true_positive + false_positive),
        "visual_decision_recall": _rate(true_positive, true_positive + false_negative),
        "estimated_minus_physical_error_median_m": (
            float(np.median(signed_error_deltas)) if signed_error_deltas else None
        ),
        "estimated_minus_physical_error_p95_abs_m": (
            float(np.quantile(np.abs(signed_error_deltas), 0.95))
            if signed_error_deltas
            else None
        ),
    }


def run_verification_benchmark(
    *,
    seeds: int = 5,
    episodes_per_seed: int = 20,
    base_seed: int = 0,
    output_path: Path | None = None,
) -> dict:
    """Test four fresh oblique camera views on terminal arm configurations.

    The overhead camera remains the only view used for control. Each second-view
    estimate is independently calibrated and compared with MuJoCo ground truth;
    no candidate changes the controller or its stop decision.
    """
    if seeds <= 0 or episodes_per_seed <= 0:
        raise ValueError("seeds and episodes_per_seed must be positive")
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")

    model = _model_with_verification_views()
    renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    target_mocap_id = int(model.body_mocapid[target_body_id])
    shoulder_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_joint"
    )
    elbow_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "elbow_joint"
    )
    camera_ids = {
        name: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, f"verification_{name}"
        )
        for name in VERIFICATION_VIEWS
    }
    records_by_condition: dict[tuple[float, str], list[dict]] = {
        (noise, view): []
        for noise in PIXEL_NOISE_LEVELS_PX
        for view in VERIFICATION_VIEWS
    }

    with tempfile.TemporaryDirectory(prefix="visual-servo-verification-") as temp_dir:
        for noise in PIXEL_NOISE_LEVELS_PX:
            for seed in range(base_seed, base_seed + seeds):
                report = run_trials(
                    episodes_per_seed,
                    seed,
                    Path(temp_dir) / f"noise-{noise:g}" / str(seed),
                    controller="image_feedback",
                    pixel_noise_std_px=noise,
                    verify_terminal=True,
                    save_media=False,
                )
                for trial in report["results"]:
                    terminal_event = trial["target_visibility_trace"][-1]
                    target_xy = np.asarray(
                        terminal_event["target_position_xy_m_for_evaluation_only"],
                        dtype=float,
                    )
                    data = mujoco.MjData(model)
                    data.mocap_pos[target_mocap_id] = (
                        float(target_xy[0]),
                        float(target_xy[1]),
                        TARGET_HEIGHT,
                    )
                    final_joint_angles = trial[
                        "final_joint_angles_rad_for_evaluation_only"
                    ]
                    data.qpos[model.jnt_qposadr[shoulder_joint_id]] = (
                        final_joint_angles[0]
                    )
                    data.qpos[model.jnt_qposadr[elbow_joint_id]] = (
                        final_joint_angles[1]
                    )
                    mujoco.mj_forward(model, data)

                    for view_name, camera_id in camera_ids.items():
                        if view_name == "east":
                            runtime_evidence = trial["terminal_visual_verification"]
                            target_pixel = runtime_evidence["target_pixel_xy"]
                            tip_pixel = runtime_evidence["end_effector_pixel_xy"]
                            estimated_error_m = runtime_evidence["estimated_distance_m"]
                            both_visible = runtime_evidence["both_markers_visible"]
                            verification_path = "runtime_terminal_verifier"
                        else:
                            renderer.update_scene(
                                data, camera=f"verification_{view_name}"
                            )
                            rgb = renderer.render()
                            try:
                                target_pixel = detect_red_target(rgb)
                            except RuntimeError:
                                target_pixel = None
                            try:
                                tip_pixel = detect_green_end_effector(rgb)
                            except RuntimeError:
                                tip_pixel = None
                            both_visible = (
                                target_pixel is not None and tip_pixel is not None
                            )
                            estimated_error_m = None
                            if both_visible:
                                estimated_target_xy = _pixel_to_xy_at_height(
                                    model,
                                    data,
                                    camera_id,
                                    target_pixel,
                                    TARGET_HEIGHT,
                                )
                                estimated_tip_xy = _pixel_to_xy_at_height(
                                    model,
                                    data,
                                    camera_id,
                                    tip_pixel,
                                    END_EFFECTOR_HEIGHT,
                                )
                                estimated_error_m = float(
                                    np.linalg.norm(
                                        estimated_target_xy - estimated_tip_xy
                                    )
                                )
                            verification_path = "offline_candidate_render"
                        physical_error_m = float(trial["reaching_error_m"])
                        records_by_condition[(noise, view_name)].append(
                            {
                                "seed": seed,
                                "episode": int(trial["episode"]),
                                "overhead_stop_reason": trial["controller_stop_reason"],
                                "overhead_natural_visibility_loss": (
                                    trial["target_visibility_stop_cause"]
                                    == "natural_target_visibility_loss"
                                ),
                                "target_visible": target_pixel is not None,
                                "end_effector_visible": tip_pixel is not None,
                                "both_markers_visible": both_visible,
                                "verification_path": verification_path,
                                "physical_error_m": physical_error_m,
                                "physical_success": bool(trial["success"]),
                                "estimated_error_m": estimated_error_m,
                                "estimated_minus_physical_error_m": (
                                    estimated_error_m - physical_error_m
                                    if estimated_error_m is not None
                                    else None
                                ),
                                "visual_success_estimate": (
                                    estimated_error_m <= SUCCESS_THRESHOLD_METERS
                                    if estimated_error_m is not None
                                    else None
                                ),
                            }
                        )

    rows = []
    for noise in PIXEL_NOISE_LEVELS_PX:
        for view_name, view_spec in VERIFICATION_VIEWS.items():
            records = records_by_condition[(noise, view_name)]
            summary = _summarize_records(records)
            summary["per_seed"] = [
                {
                    "seed": seed,
                    **_summarize_records(
                        [record for record in records if record["seed"] == seed]
                    ),
                }
                for seed in range(base_seed, base_seed + seeds)
            ]
            summary["trial_results"] = records
            rows.append(
                {
                    "pixel_noise_std_px": noise,
                    "verification_view": view_name,
                    "camera_position_m": list(view_spec["position_m"]),
                    "camera_fovy_deg": view_spec["fovy_deg"],
                    "visual_success_threshold_m": SUCCESS_THRESHOLD_METERS,
                    **summary,
                }
            )

    result = {
        "schema_version": 1,
        "benchmark": "fresh_oblique_verification_view",
        "control_camera": "overhead",
        "control_policy_changed": False,
        "control_camera_observation_delay_observations": 0,
        "control_prediction_mode": "none",
        "control_stale_target_confirmation": False,
        "east_view_runtime_terminal_verifier_used": True,
        "candidate_view_detection": "RGB color segmentation of target and end-effector markers",
        "candidate_view_metric": (
            "back-project target and end-effector pixels to their known horizontal heights "
            "with the calibrated camera pose and vertical field of view"
        ),
        "visual_success_threshold_m": SUCCESS_THRESHOLD_METERS,
        "pixel_noise_std_levels_px": list(PIXEL_NOISE_LEVELS_PX),
        "verification_views": VERIFICATION_VIEWS,
        "seeds": seeds,
        "base_seed": base_seed,
        "episodes_per_seed": episodes_per_seed,
        "results": rows,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes-per-seed", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/verification-view-benchmark.json"),
    )
    args = parser.parse_args()
    try:
        report = run_verification_benchmark(
            seeds=args.seeds,
            episodes_per_seed=args.episodes_per_seed,
            base_seed=args.base_seed,
            output_path=args.output,
        )
    except ValueError as error:
        parser.error(str(error))

    print("Noise  View   Both visible  False +  False -  |error delta| P95")
    for row in report["results"]:
        counts = row["confusion_counts_when_both_visible"]
        p95_cm = row["estimated_minus_physical_error_p95_abs_m"]
        p95_text = "n/a" if p95_cm is None else f"{p95_cm * 100:.2f} cm"
        print(
            f"{row['pixel_noise_std_px']:4.0f}px  {row['verification_view']:5}  "
            f"{row['both_markers_visible_trials']:3}/{row['trials']:<3}  "
            f"{counts['false_positive']:7}  {counts['false_negative']:7}  {p95_text:>12}"
        )
    print(f"Saved verification-view benchmark report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
