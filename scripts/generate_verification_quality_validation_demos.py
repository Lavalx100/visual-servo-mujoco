"""Create paired clips for a false-success case from the frozen-seed split."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from visual_servo_mujoco.run import SUCCESS_THRESHOLD_METERS, run_trials


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    report_path = repository / "artifacts" / "verification-quality-calibration-validation.json"
    report = json.loads(report_path.read_text())
    frozen_threshold = float(report["validation_frozen_threshold"])
    fov_condition = next(
        row
        for row in report["validation_benchmark"]["results"]
        if row["condition"] == "fovy_minus_5deg"
    )
    false_successes = [
        record
        for record in fov_condition["trial_results"]
        if not record["physical_success"]
        and record["candidate_estimated_distance_m"] is not None
        and record["candidate_estimated_distance_m"] <= SUCCESS_THRESHOLD_METERS
        and record["detection_quality_score"] < frozen_threshold
    ]
    if not false_successes:
        raise RuntimeError(
            "the frozen validation split contains no FOV -5 degree false-success "
            "case rejected by the selected quality gate"
        )
    selected = max(false_successes, key=lambda record: record["detection_quality_score"])
    seed = int(selected["seed"])
    episode = int(selected["episode"])
    gate_settings = (
        ("terminal_verification_quality_heldout_false_success", 0.0),
        ("terminal_verification_quality_heldout_abstention", frozen_threshold),
    )
    destination_dir = repository / "assets" / "demos"
    destination_dir.mkdir(parents=True, exist_ok=True)
    paired_trials = []

    for stem, quality_gate in gate_settings:
        with tempfile.TemporaryDirectory(prefix="visual-servo-quality-validation-demo-") as temp:
            output_dir = Path(temp)
            run = run_trials(
                episodes=episode + 1,
                seed=seed,
                output_dir=output_dir,
                controller="image_feedback",
                pixel_noise_std_px=float(report["protocol"]["pixel_noise_std_px"]),
                verify_terminal=True,
                terminal_verification_fovy_error_deg=-5.0,
                terminal_verification_min_quality_score=quality_gate,
                video_episode_index=episode,
                save_media=True,
            )
            trial = run["results"][episode]
            source = output_dir / trial["terminal_verification_video"]
            destination = destination_dir / f"{stem}.mp4"
            shutil.copyfile(source, destination)
            paired_trials.append(trial)
            evidence = trial["terminal_visual_verification"]
            print(
                f"{destination.relative_to(repository)}: seed={seed}, episode={episode}, "
                f"quality={evidence['combined_detection_quality_score']}, "
                f"gate={quality_gate:.2f}, "
                f"estimate={evidence['estimated_distance_m']}, "
                f"physical_success={trial['success']}"
            )

    baseline, gated = paired_trials
    for key in (
        "success",
        "reaching_error_m",
        "controller_stop_reason",
        "controller_iterations",
    ):
        if baseline[key] != gated[key]:
            raise RuntimeError(f"quality gate changed reaching outcome field {key}")
    if baseline["terminal_visual_verification"]["visual_success_estimate"] is not True:
        raise RuntimeError("baseline demo must retain the held-out false-success decision")
    if gated["terminal_visual_verification"]["visual_success_estimate"] is not None:
        raise RuntimeError("frozen gate must abstain on the selected held-out case")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
