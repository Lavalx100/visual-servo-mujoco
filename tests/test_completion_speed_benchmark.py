import pytest

from visual_servo_mujoco.completion_speed_benchmark import (
    MOTION_CONDITIONS,
    _aggregate,
    _mode_identity,
    _write_figure,
)
from visual_servo_mujoco.timestamped_verification_benchmark import (
    CONTROL_MODES,
    evaluate_arrival_only_policy,
)


@pytest.mark.parametrize(
    ("name", "expected_m_s"),
    (
        ("stationary", 0.0),
        ("sinusoidal_0p25_hz", 2 * 3.141592653589793 * 0.04 * 0.25),
        ("sinusoidal_0p50_hz", 2 * 3.141592653589793 * 0.04 * 0.5),
        ("piecewise_linear_0p50_hz", 4 * 0.04 * 0.5),
    ),
)
def test_motion_conditions_report_analytic_peak_speed(name, expected_m_s):
    condition = next(item for item in MOTION_CONDITIONS if item.name == name)

    assert condition.peak_speed_m_s == pytest.approx(expected_m_s)


def test_speed_summary_keeps_modes_distinct_and_chart_pools_them(tmp_path):
    records = []
    condition = MOTION_CONDITIONS[0]
    for mode, delay, current_tip in CONTROL_MODES:
        trial = {
            "seed": 950,
            "episode": 0,
            "controller_stop_time_simulated_s": 10.0,
            "success": True,
            "reaching_error_m": 0.02,
            "target_motion_profile": condition.profile,
            "target_motion_phase_rad": 0.0,
            "use_current_end_effector_state": current_tip,
            "camera_observation_delay_observations": delay,
            "camera_observation_delay_simulated_s": delay * 0.4,
        }
        event = {
            "frame_index": 0,
            "capture_time_s": 10.0,
            "delivery_time_s": 10.8,
            "scheduled_arrival_time_s": 10.8,
            "transport_dropped": False,
            "both_markers_visible": True,
            "estimated_distance_m": 0.02,
            "visual_success_estimate": True,
            "physical_error_at_capture_m": 0.02,
            "physical_error_at_delivery_m": 0.05,
        }
        records.append(
            {
                "motion_condition": condition.name,
                "target_motion_profile": condition.profile,
                "target_motion_amplitude_m": condition.amplitude_m,
                "target_motion_frequency_hz": condition.frequency_hz,
                "peak_target_speed_m_s": condition.peak_speed_m_s,
                "target_motion_phase_rad": 0.0,
                "seed": 950,
                "episode": 0,
                **_mode_identity(trial),
                "delivery_profile": "high_latency",
                "policy": "arrival_first_visible_frame",
                "max_frame_age_s": None,
                **evaluate_arrival_only_policy(trial, [event]),
            }
        )

    rows = _aggregate(records)

    assert len(rows) == len(CONTROL_MODES)
    assert all(row["trials"] == 1 for row in rows)
    assert all(row["visually_confirmed"] == 1 for row in rows)
    assert all(row["false_confirmations_at_delivery"] == 1 for row in rows)
    assert all(row["delivery_mismatches_from_motion_after_capture"] == 1 for row in rows)
    assert all(row["delivery_mismatches_already_out_at_capture"] == 0 for row in rows)

    figure_path = tmp_path / "speed.svg"
    _write_figure(rows, figure_path)
    figure = figure_path.read_text(encoding="utf-8")
    assert ">100.0%</text>" in figure
    assert ">n=5</text>" in figure
