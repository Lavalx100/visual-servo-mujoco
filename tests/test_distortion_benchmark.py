import csv

from visual_servo_mujoco.distortion_benchmark import (
    export_trial_csv,
    render_benchmark_figure,
    run_distortion_benchmark,
)


def test_distortion_benchmark_pairs_physical_trials_and_exports_csv(tmp_path):
    report = run_distortion_benchmark(
        seeds=1,
        episodes_per_seed=1,
        base_seed=17,
        output_path=tmp_path / "report.json",
    )
    csv_path = tmp_path / "trials.csv"
    export_trial_csv(report, csv_path)

    assert report["protocol"]["paired_target_and_noise_seeds"] is True
    assert report["protocol"]["camera_observation_delay_observations"] == 0
    assert report["protocol"]["current_joint_state_tip_projection"] is False
    assert report["protocol"]["terminal_visual_confirmation_evaluated"] is False
    assert len(report["conditions"]) == 14
    assert len(report["paired_comparisons_vs_uncorrected"]) == 8
    assert all(
        comparison["paired_trials"] == 1
        for comparison in report["paired_comparisons_vs_uncorrected"]
    )
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 14
    assert {row["controller"] for row in rows} == {"open_loop", "image_feedback"}
    assert {row["actual_camera_k1"] for row in rows} == {"0.0", "-0.1", "0.1"}


def test_distortion_benchmark_supports_noise_sweeps_and_renders_figure(tmp_path):
    report = run_distortion_benchmark(
        seeds=1,
        episodes_per_seed=1,
        base_seed=23,
        pixel_noise_levels=(2.0, 20.0),
    )
    figure_path = tmp_path / "noise-sweep.svg"
    render_benchmark_figure(report, figure_path)

    assert report["pixel_noise_levels_std_px"] == [2.0, 20.0]
    assert len(report["conditions"]) == 28
    assert len(report["paired_comparisons_vs_uncorrected"]) == 16
    assert {
        row["pixel_noise_std_px"]
        for row in report["paired_comparisons_vs_uncorrected"]
    } == {2.0, 20.0}
    assert "20 px · actual k1 +0.10" in figure_path.read_text()
