"""Export compact per-trial outcomes from a visual-servo benchmark report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "controller",
    "condition",
    "pixel_noise_std_px",
    "camera_fovy_error_deg",
    "target_dropout_start_observation",
    "target_dropout_duration_observations",
    "seed",
    "episode",
    "physical_success",
    "physical_success_threshold_m",
    "reaching_error_cm",
    "perception_error_cm",
    "final_pixel_error_px",
    "controller_iterations",
    "simulated_motion_time_s",
    "controller_stop_reason",
    "failure_reason",
    "target_visibility_stop_cause",
    "target_reacquisitions",
    "natural_target_visibility_loss_observations",
    "natural_target_visibility_loss_outside_dropout_observations",
    "target_pixel_x_px",
    "target_pixel_y_px",
    "controller_pixel_x_px",
    "controller_pixel_y_px",
    "estimated_target_x_m",
    "estimated_target_y_m",
    "true_target_x_m_evaluation_only",
    "true_target_y_m_evaluation_only",
    "end_effector_x_m",
    "end_effector_y_m",
)


def _coordinate(values: list[float] | None, index: int) -> float | str:
    return "" if values is None else values[index]


def _centimetres(value: float | None) -> float | str:
    return "" if value is None else value * 100.0


def _trial_row(
    condition: dict[str, Any], trial: dict[str, Any], threshold_m: float
) -> dict[str, Any]:
    settings = condition["condition"]
    row: dict[str, Any] = {
        "controller": trial["controller"],
        "condition": settings["name"],
        "pixel_noise_std_px": settings["pixel_noise_std_px"],
        "camera_fovy_error_deg": settings["camera_fovy_error_deg"],
        "target_dropout_start_observation": settings[
            "target_dropout_start_observation"
        ],
        "target_dropout_duration_observations": settings[
            "target_dropout_duration_observations"
        ],
        "seed": trial["seed"],
        "episode": trial["episode"],
        "physical_success": trial["success"],
        "physical_success_threshold_m": threshold_m,
        "reaching_error_cm": _centimetres(trial["reaching_error_m"]),
        "perception_error_cm": _centimetres(trial["perception_error_m"]),
        "final_pixel_error_px": trial["final_pixel_error_px"],
        "controller_iterations": trial["controller_iterations"],
        "simulated_motion_time_s": trial["simulated_motion_time_s"],
        "controller_stop_reason": trial["controller_stop_reason"],
        "failure_reason": trial["failure_reason"],
        "target_visibility_stop_cause": trial["target_visibility_stop_cause"],
        "target_reacquisitions": trial["target_reacquisitions"],
        "natural_target_visibility_loss_observations": trial[
            "natural_target_visibility_loss_observations"
        ],
        "natural_target_visibility_loss_outside_dropout_observations": trial[
            "natural_target_visibility_loss_outside_dropout_observations"
        ],
    }

    for x_column, y_column, source in (
        ("target_pixel_x_px", "target_pixel_y_px", "target_pixel_xy"),
        ("controller_pixel_x_px", "controller_pixel_y_px", "controller_pixel_xy"),
        ("estimated_target_x_m", "estimated_target_y_m", "estimated_target_xy_m"),
        (
            "true_target_x_m_evaluation_only",
            "true_target_y_m_evaluation_only",
            "true_target_xy_m_for_evaluation_only",
        ),
        ("end_effector_x_m", "end_effector_y_m", "end_effector_xy_m"),
    ):
        row[x_column] = _coordinate(trial[source], 0)
        row[y_column] = _coordinate(trial[source], 1)

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/benchmark.json"),
        help="JSON output from visual-servo-benchmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/robustness-baseline-trials.csv"),
        help="CSV destination",
    )
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    threshold_m = report["success_threshold_m"]
    expected_rows = sum(condition["trials"] for condition in report["conditions"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for condition in report["conditions"]:
            trials = condition["trial_results"]
            if len(trials) != condition["trials"]:
                raise ValueError(
                    f"{condition['controller']} / {condition['condition']['name']} "
                    "does not contain the declared number of trial records"
                )
            for trial in trials:
                writer.writerow(_trial_row(condition, trial, threshold_m))
                row_count += 1

    if row_count != expected_rows:
        raise ValueError(f"Wrote {row_count} rows; expected {expected_rows}")
    print(f"Wrote {row_count} trial rows to {args.output}")


if __name__ == "__main__":
    main()
