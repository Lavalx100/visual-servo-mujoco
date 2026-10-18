"""Paired end-to-end reaching benchmark for radial camera distortion."""

from __future__ import annotations

import argparse
import csv
import json
import math
from statistics import median
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

from .benchmark import _summarize
from .run import SUCCESS_THRESHOLD_METERS, run_trials

ACTUAL_K1_VALUES = (0.0, -0.10, 0.10)
CORRECTION_FACTORS = (0.0, 0.75, 1.0)
CONTROLLERS = ("open_loop", "image_feedback")
DEFAULT_PIXEL_NOISE_STD_PX = 2.0


def _condition_name(actual_k1: float, correction_factor: float) -> str:
    if actual_k1 == 0.0:
        return "clean_zero_distortion"
    sign = "positive" if actual_k1 > 0 else "negative"
    factor_name = {
        0.0: "uncorrected",
        0.75: "underestimated_25pct",
        1.0: "matched_coefficients",
    }[correction_factor]
    return f"{sign}_k1_{abs(actual_k1):.2f}_{factor_name}"


def _paired_comparisons(conditions: list[dict]) -> list[dict]:
    by_key = {
        (
            row["controller"],
            row["pixel_noise_std_px"],
            row["actual_camera_k1"],
            row["estimated_controller_k1"],
        ): row
        for row in conditions
    }
    comparisons = []
    for controller in CONTROLLERS:
        noise_levels = sorted({row["pixel_noise_std_px"] for row in conditions})
        for noise_std_px in noise_levels:
            for actual_k1 in ACTUAL_K1_VALUES:
                if actual_k1 == 0.0:
                    continue
                baseline = by_key[
                    (controller, noise_std_px, actual_k1, 0.0)
                ]["trial_results"]
                corrected_by_id = {
                    (trial["seed"], trial["episode"]): trial for trial in baseline
                }
                baseline_by_id = corrected_by_id
                for factor in CORRECTION_FACTORS[1:]:
                    estimate = round(actual_k1 * factor, 8)
                    corrected = by_key[
                        (controller, noise_std_px, actual_k1, estimate)
                    ]["trial_results"]
                    corrected_by_id = {
                        (trial["seed"], trial["episode"]): trial
                        for trial in corrected
                    }
                    shared_ids = sorted(set(baseline_by_id) & set(corrected_by_id))
                    deltas = [
                        corrected_by_id[key]["reaching_error_m"]
                        - baseline_by_id[key]["reaching_error_m"]
                        for key in shared_ids
                    ]
                    comparisons.append(
                        {
                            "controller": controller,
                            "pixel_noise_std_px": noise_std_px,
                            "actual_camera_k1": actual_k1,
                            "estimated_controller_k1": estimate,
                            "paired_trials": len(shared_ids),
                            "median_reaching_error_delta_m_vs_uncorrected": (
                                float(median(deltas)) if deltas else None
                            ),
                            "correction_lower_error_trials": sum(
                                delta < -1e-9 for delta in deltas
                            ),
                            "tie_error_trials": sum(
                                abs(delta) <= 1e-9 for delta in deltas
                            ),
                            "correction_higher_error_trials": sum(
                                delta > 1e-9 for delta in deltas
                            ),
                            "baseline_successes_recovered": sum(
                                not baseline_by_id[key]["success"]
                                and corrected_by_id[key]["success"]
                                for key in shared_ids
                            ),
                            "baseline_successes_lost": sum(
                                baseline_by_id[key]["success"]
                                and not corrected_by_id[key]["success"]
                                for key in shared_ids
                            ),
                        }
                    )
    return comparisons


def run_distortion_benchmark(
    *,
    seeds: int = 5,
    episodes_per_seed: int = 20,
    base_seed: int = 600,
    output_path: Path | None = None,
    pixel_noise_std_px: float = DEFAULT_PIXEL_NOISE_STD_PX,
    pixel_noise_levels: tuple[float, ...] | None = None,
) -> dict:
    """Compare pinhole, partially corrected, and matched radial models.

    The rendered overhead image receives a known radial warp. Controller
    coefficients are opt-in estimates applied to detected target and arm-marker
    pixels. Every mode uses paired target and pixel-noise seeds.
    """
    if seeds <= 0 or episodes_per_seed <= 0:
        raise ValueError("seeds and episodes_per_seed must be positive")
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    noise_levels = (
        (pixel_noise_std_px,)
        if pixel_noise_levels is None
        else tuple(pixel_noise_levels)
    )
    if not noise_levels or any(
        not math.isfinite(value) or value < 0.0 for value in noise_levels
    ):
        raise ValueError("pixel_noise_std_px must be finite and non-negative")

    settings = []
    for actual_k1 in ACTUAL_K1_VALUES:
        factors = (0.0,) if actual_k1 == 0.0 else CORRECTION_FACTORS
        for factor in factors:
            settings.append(
                {
                    "name": _condition_name(actual_k1, factor),
                    "actual_k1": actual_k1,
                    "correction_factor": factor,
                    "estimated_k1": round(actual_k1 * factor, 8),
                }
            )

    conditions = []
    with tempfile.TemporaryDirectory(prefix="visual-servo-distortion-benchmark-") as root:
        for noise_std_px in noise_levels:
            for controller in CONTROLLERS:
                for setting in settings:
                    trials = []
                    for seed in range(base_seed, base_seed + seeds):
                        report = run_trials(
                            episodes_per_seed,
                            seed,
                            Path(root)
                            / controller
                            / setting["name"]
                            / f"noise-{noise_std_px:g}"
                            / str(seed),
                            controller=controller,
                            pixel_noise_std_px=noise_std_px,
                            camera_radial_distortion_k1=setting["actual_k1"],
                            controller_radial_distortion_k1=setting["estimated_k1"],
                            camera_observation_delay_observations=0,
                            target_motion_amplitude_m=0.0,
                            target_motion_prediction="none",
                            save_media=False,
                        )
                        trials.extend(
                            {"seed": seed, **trial} for trial in report["results"]
                        )
                    conditions.append(
                        {
                            "controller": controller,
                            "condition": setting["name"],
                            "actual_camera_k1": setting["actual_k1"],
                            "estimated_controller_k1": setting["estimated_k1"],
                            "correction_factor": setting["correction_factor"],
                            "pixel_noise_std_px": noise_std_px,
                            **_summarize(trials, SUCCESS_THRESHOLD_METERS),
                            "per_seed": [
                                {
                                    "seed": seed,
                                    **_summarize(
                                        [
                                            trial
                                            for trial in trials
                                            if trial["seed"] == seed
                                        ],
                                        SUCCESS_THRESHOLD_METERS,
                                    ),
                                }
                                for seed in range(base_seed, base_seed + seeds)
                            ],
                            "trial_results": trials,
                        }
                    )

    result = {
        "schema_version": 1,
        "benchmark": "radial_camera_distortion_end_to_end_reaching",
        "distortion_model": "Brown-Conrady radial k1,k2 with k2=0 and no tangential terms",
        "success_threshold_m": SUCCESS_THRESHOLD_METERS,
        "seeds": seeds,
        "base_seed": base_seed,
        "episodes_per_seed": episodes_per_seed,
        "pixel_noise_std_px": noise_levels[0] if len(noise_levels) == 1 else None,
        "pixel_noise_levels_std_px": list(noise_levels),
        "conditions": conditions,
        "paired_comparisons_vs_uncorrected": _paired_comparisons(conditions),
        "protocol": {
            "target_motion": "stationary",
            "camera_observation_delay_observations": 0,
            "predictor": "none",
            "paired_target_and_noise_seeds": True,
            "controllers": list(CONTROLLERS),
            "camera_distortion_is_applied_to_rendered_overhead_rgb": True,
            "correction_is_applied_to_detected_target_and_camera_observed_tip_pixels": True,
            "current_joint_state_tip_projection": False,
            "terminal_visual_confirmation_evaluated": False,
            "physical_success_uses_simulator_ground_truth": True,
        },
        "interpretation": {
            "matched_coefficients_are_oracle_correction": (
                "The matched mode is an upper-bound geometry check using the exact "
                "injected k1; it does not claim that a real calibration estimator "
                "recovers k1 this accurately."
            ),
            "distortion_magnitude": (
                "The +/-0.10 coefficients are controlled simulation stress values, "
                "not a measurement of the public OpenCV camera or a physical lens."
            ),
            "success_definition": (
                "Physical end-effector-to-target error is at most 4.5 cm. Fresh "
                "visual confirmation is not measured by this benchmark."
            ),
        },
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def export_trial_csv(report: dict, output_path: Path) -> None:
    """Write one transparent result row per controller episode."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "controller",
        "condition",
        "actual_camera_k1",
        "estimated_controller_k1",
        "correction_factor",
        "pixel_noise_std_px",
        "seed",
        "episode",
        "success",
        "reaching_error_m",
        "perception_error_m",
        "final_pixel_error_px",
    )
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for condition in report["conditions"]:
            for trial in condition["trial_results"]:
                writer.writerow(
                    {
                        "controller": condition["controller"],
                        "condition": condition["condition"],
                        "actual_camera_k1": condition["actual_camera_k1"],
                        "estimated_controller_k1": condition[
                            "estimated_controller_k1"
                        ],
                        "correction_factor": condition["correction_factor"],
                        "pixel_noise_std_px": condition["pixel_noise_std_px"],
                        "seed": trial["seed"],
                        "episode": trial["episode"],
                        "success": trial["success"],
                        "reaching_error_m": trial["reaching_error_m"],
                        "perception_error_m": trial["perception_error_m"],
                        "final_pixel_error_px": trial["final_pixel_error_px"],
                    }
                )


def render_benchmark_figure(report: dict, output_path: Path) -> None:
    """Write a GitHub-renderable scorecard of paired reaching outcomes."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    noise_levels = report.get("pixel_noise_levels_std_px") or [
        report["pixel_noise_std_px"]
    ]
    lookup = {
        (
            row["controller"],
            row["pixel_noise_std_px"],
            row["actual_camera_k1"],
            row["estimated_controller_k1"],
        ): row
        for row in report["conditions"]
    }
    groups = []
    for controller in CONTROLLERS:
        for noise_std_px in noise_levels:
            for actual_k1 in ACTUAL_K1_VALUES:
                factors = (0.0,) if actual_k1 == 0.0 else CORRECTION_FACTORS
                rows = [
                    lookup[
                        (
                            controller,
                            noise_std_px,
                            actual_k1,
                            round(actual_k1 * factor, 8),
                        )
                    ]
                    for factor in factors
                ]
                groups.append((controller, noise_std_px, actual_k1, rows))
    max_error_cm = max(
        3.0,
        max(
            row["median_reaching_error_m"] * 100.0
            for _, _, _, rows in groups
            for row in rows
        )
        * 1.15,
    )
    width = 1180
    chart_x, chart_width = 394, 545
    y_start, group_height = 156, 55
    height = max(650, y_start + group_height * len(groups) + 190)
    chart_bottom = height - 140
    colors = ("#687789", "#d48a12", "#117a9b")
    factor_labels = ("no correction", "75% coefficient", "matched coefficient")
    noise_label = ", ".join(f"{value:g}" for value in noise_levels)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#17212b}.muted{fill:#526171}'
        '.grid{stroke:#dce2e9;stroke-width:1}.panel{fill:#fff;stroke:#d7dee7;'
        'stroke-width:1.2}</style>',
        '<text x="32" y="40" font-size="25" font-weight="700">'
        'Radial correction under pixel noise</text>',
        '<text x="32" y="68" class="muted" font-size="14">Rendered overhead RGB · '
        f'{report["conditions"][0]["trials"]} paired episodes per row · '
        f'{noise_label} px target-measurement noise · zero camera delay</text>',
        f'<rect class="panel" x="24" y="91" width="1132" '
        f'height="{height - 176}" rx="8"/>',
    ]
    for tick in range(5):
        value = max_error_cm * tick / 4.0
        x = chart_x + chart_width * tick / 4.0
        svg.append(
            f'<line class="grid" x1="{x:.1f}" y1="124" '
            f'x2="{x:.1f}" y2="{chart_bottom}"/>'
        )
        svg.append(
            f'<text x="{x:.1f}" y="119" text-anchor="middle" '
            f'class="muted" font-size="11">{value:.1f} cm</text>'
        )
    for group_index, (controller, noise_std_px, actual_k1, rows) in enumerate(groups):
        base_y = y_start + group_index * group_height
        distortion_label = "clean" if actual_k1 == 0 else f"actual k1 {actual_k1:+.2f}"
        svg.append(
            f'<text x="34" y="{base_y + 13}" font-size="12" font-weight="700">'
            f'{escape(controller)} · {noise_std_px:g} px · {distortion_label}</text>'
        )
        factors = (0.0,) if actual_k1 == 0.0 else CORRECTION_FACTORS
        for offset, (factor, row) in enumerate(zip(factors, rows)):
            y = base_y + 20 + offset * 13
            error_cm = row["median_reaching_error_m"] * 100.0
            bar_width = chart_width * error_cm / max_error_cm
            success_text = f'{row["success_rate"]:.0%} success'
            lane_label = factor_labels[CORRECTION_FACTORS.index(factor)]
            svg.append(
                f'<rect x="{chart_x}" y="{y}" width="{bar_width:.2f}" '
                f'height="10" rx="2" fill="{colors[CORRECTION_FACTORS.index(factor)]}"/>'
            )
            svg.append(
                f'<text x="{chart_x + bar_width + 6:.1f}" y="{y + 9}" '
                f'font-size="10">{error_cm:.2f} cm · {success_text}</text>'
            )
            svg.append(
                f'<text x="{chart_x - 10}" y="{y + 9}" text-anchor="end" '
                f'class="muted" font-size="9">{lane_label}</text>'
            )
    svg.append(
        f'<text x="32" y="{height - 62}" font-size="12" font-weight="700">'
        'Physical reaching is scored in MuJoCo; fresh visual confirmation is not evaluated.</text>'
    )
    svg.append(
        f'<text x="32" y="{height - 38}" class="muted" font-size="11">'
        'Matched coefficients use the injected k1 as an oracle. They test the correction '
        'geometry, not calibration-estimator accuracy.</text>'
    )
    svg.append("</svg>")
    output_path.write_text("\n".join(svg) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes-per-seed", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=600)
    parser.add_argument(
        "--pixel-noise-std-px", type=float, default=DEFAULT_PIXEL_NOISE_STD_PX
    )
    parser.add_argument(
        "--noise-levels",
        nargs="+",
        type=float,
        help="run the same paired distortion settings at each listed noise level",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/radial-distortion-benchmark.json"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/radial-distortion-trials.csv"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("assets/demos/radial_distortion_benchmark.svg"),
    )
    args = parser.parse_args()
    try:
        report = run_distortion_benchmark(
            seeds=args.seeds,
            episodes_per_seed=args.episodes_per_seed,
            base_seed=args.base_seed,
            output_path=args.output,
            pixel_noise_std_px=args.pixel_noise_std_px,
            pixel_noise_levels=(
                tuple(args.noise_levels) if args.noise_levels is not None else None
            ),
        )
    except ValueError as error:
        parser.error(str(error))
    export_trial_csv(report, args.csv)
    render_benchmark_figure(report, args.figure)
    print("Controller        Noise  Actual k1  Estimated k1  Trials  Success  Median error")
    for row in report["conditions"]:
        print(
            f"{row['controller']:16} {row['pixel_noise_std_px']:5.1f} "
            f"{row['actual_camera_k1']:+8.2f} "
            f"{row['estimated_controller_k1']:+13.3f} {row['trials']:7d} "
            f"{row['success_rate']:7.1%} "
            f"{row['median_reaching_error_m'] * 100:8.2f} cm"
        )
    print(f"Saved benchmark to {args.output}, {args.csv}, and {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
