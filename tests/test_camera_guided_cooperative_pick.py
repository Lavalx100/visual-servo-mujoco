import csv
import math

import mujoco
import pytest

from visual_servo_mujoco.camera_guided_cooperative_pick import (
    DEMO_TARGET_XY,
    TARGET_HALF_SPAN_M,
    _apply_release_profile,
    _finger_pad_payload_contact_indices,
    _sample_positions,
    _unexpected_arm_contact_details,
    run_trial,
    run_validation,
)
from visual_servo_mujoco.cooperative_pick_demo import (
    ARM_COUNT,
    PAYLOAD_INITIAL_Z_M,
    build_model_xml,
    desired_pick_state,
)


def test_camera_guided_scene_adds_yaw_camera_and_visual_only_target_marker():
    model = mujoco.MjModel.from_xml_string(
        build_model_xml(
            include_base_yaw=True,
            include_gripper_yaw=True,
            include_tracking_camera=True,
            include_target_marker=True,
        )
    )

    assert model.njnt == ARM_COUNT * 7 + 1
    assert model.nu == ARM_COUNT * 7
    assert model.neq == 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead") >= 0
    for arm_index in range(ARM_COUNT):
        for joint_suffix in ("base_yaw", "gripper_yaw"):
            assert mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"arm{arm_index}_{joint_suffix}",
            ) >= 0

    marker = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "payload_target_marker"
    )
    assert marker >= 0
    assert model.geom_contype[marker] == 0
    assert model.geom_conaffinity[marker] == 0
    model_without_marker = mujoco.MjModel.from_xml_string(
        build_model_xml(
            include_base_yaw=True,
            include_gripper_yaw=True,
            include_tracking_camera=True,
            include_target_marker=False,
        )
    )
    payload_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload")
    payload_without_marker = mujoco.mj_name2id(
        model_without_marker, mujoco.mjtObj.mjOBJ_BODY, "payload"
    )
    assert model.body_mass[payload_body] == pytest.approx(
        model_without_marker.body_mass[payload_without_marker]
    )


def test_fresh_verification_camera_and_fiducial_are_visual_only():
    without_verification = mujoco.MjModel.from_xml_string(
        build_model_xml(include_base_yaw=True, include_gripper_yaw=True)
    )
    with_verification = mujoco.MjModel.from_xml_string(
        build_model_xml(
            include_base_yaw=True,
            include_gripper_yaw=True,
            include_visual_verification=True,
        )
    )

    camera_id = mujoco.mj_name2id(
        with_verification, mujoco.mjtObj.mjOBJ_CAMERA, "verification_side"
    )
    marker_id = mujoco.mj_name2id(
        with_verification, mujoco.mjtObj.mjOBJ_GEOM, "payload_verification_marker"
    )
    assert camera_id >= 0
    assert marker_id >= 0
    payload_id = mujoco.mj_name2id(
        with_verification, mujoco.mjtObj.mjOBJ_BODY, "payload"
    )
    assert with_verification.geom_bodyid[marker_id] == payload_id
    assert with_verification.geom_contype[marker_id] == 0
    assert with_verification.geom_conaffinity[marker_id] == 0
    assert with_verification.body_mass[payload_id] == pytest.approx(
        without_verification.body_mass[
            mujoco.mj_name2id(without_verification, mujoco.mjtObj.mjOBJ_BODY, "payload")
        ]
    )


def test_orthogonal_camera_markers_and_occluder_do_not_enter_contact_physics():
    base_model = mujoco.MjModel.from_xml_string(
        build_model_xml(
            include_base_yaw=True,
            include_gripper_yaw=True,
            include_visual_verification=True,
        )
    )
    experiment_model = mujoco.MjModel.from_xml_string(
        build_model_xml(
            include_base_yaw=True,
            include_gripper_yaw=True,
            include_visual_verification=True,
            include_orthogonal_verification_camera=True,
            include_verification_occluder=True,
        )
    )

    assert mujoco.mj_name2id(
        experiment_model, mujoco.mjtObj.mjOBJ_CAMERA, "verification_orthogonal"
    ) >= 0
    payload_base = mujoco.mj_name2id(base_model, mujoco.mjtObj.mjOBJ_BODY, "payload")
    payload_experiment = mujoco.mj_name2id(
        experiment_model, mujoco.mjtObj.mjOBJ_BODY, "payload"
    )
    assert experiment_model.body_mass[payload_experiment] == pytest.approx(
        base_model.body_mass[payload_base]
    )
    for geom_name in (
        "payload_verification_marker_orthogonal",
        "verification_occluder_rail",
    ):
        geom_id = mujoco.mj_name2id(
            experiment_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
        )
        assert geom_id >= 0
        assert experiment_model.geom_contype[geom_id] == 0
        assert experiment_model.geom_conaffinity[geom_id] == 0


def test_random_target_split_is_reproducible_and_stays_inside_the_declared_range():
    first = _sample_positions(seeds=3, samples_per_seed=10, base_seed=28100)
    second = _sample_positions(seeds=3, samples_per_seed=10, base_seed=28100)

    assert first == second
    assert len(first) == 30
    for _, _, (x_m, y_m) in first:
        assert abs(x_m) <= TARGET_HALF_SPAN_M
        assert abs(y_m) <= TARGET_HALF_SPAN_M


def test_clearance_release_profile_only_changes_release_and_withdraw_geometry():
    radial, _, jaw, _, phase = desired_pick_state(11.0)
    baseline = _apply_release_profile(radial, jaw, phase, "baseline")
    candidate = _apply_release_profile(radial, jaw, phase, "clearance")
    assert baseline == (radial, None)
    assert candidate[0] >= radial
    assert candidate[1] is not None
    assert min(candidate[1]) >= 0.035

    approach = desired_pick_state(2.0)
    assert _apply_release_profile(approach[0], approach[2], approach[4], "clearance") == (
        approach[0],
        None,
    )


def test_camera_and_oracle_pairs_lift_the_free_payload_with_collision_checks():
    selected_clearance = {
        "release_profile": "clearance",
        "lowest_approach_tool_height_offset_m": 0.020,
    }
    camera = run_trial(
        DEMO_TARGET_XY, controller="camera_estimate", **selected_clearance
    )
    oracle = run_trial(DEMO_TARGET_XY, controller="oracle_xy", **selected_clearance)

    assert camera["target_detected"] is True
    assert camera["initial_target_projection_error_m"] < 0.005
    assert camera["control_target_source"] == "detected_image_position"
    assert oracle["control_target_source"] == "oracle_xy"
    for metrics in (camera, oracle):
        assert metrics["physical_stable_lift_success"] is True
        assert metrics["payload_stable_lift_success"] is True
        assert metrics["full_sequence_collision_safe"] is True
        assert metrics["minimum_payload_z_during_hold_m"] >= PAYLOAD_INITIAL_Z_M + 0.07
        assert metrics["minimum_finger_payload_contact_points_during_hold"] >= 3
        assert metrics["maximum_payload_tilt_during_hold_deg"] <= 5.0
        assert metrics["maximum_inter_arm_penetration_m"] <= metrics[
            "inter_arm_penetration_limit_m"
        ]
        assert metrics["physics_steps_with_arm_table_or_ground_contact"] == 0
        assert metrics["physics_steps_with_nonfinger_arm_payload_contact"] == 0
        assert metrics["unexpected_contact_steps_by_phase"] == {
            "arm_environment": {},
            "nonfinger_payload": {},
        }
        assert metrics["unexpected_contact_pairs"] == {
            "arm_environment": [],
            "nonfinger_payload": [],
        }
        assert metrics["first_unexpected_contact"] == {
            "arm_environment": None,
            "nonfinger_payload": None,
        }
        assert metrics["fresh_visual_confirmation"] == "not measured"
        assert math.isclose(metrics["stable_lift_hold_duration_s"], 1.6)
        contact_trace = metrics["finger_pad_payload_contact_trace"]
        close_tracking = metrics["finger_slide_tracking_during_close"]
        assert len(contact_trace) == ARM_COUNT * 2
        assert len(close_tracking) == ARM_COUNT * 2
        assert any(event["contact_steps"] > 0 for event in contact_trace.values())
        assert all(event["first_contact_time_s"] is not None for event in contact_trace.values() if event["contact_steps"])
        assert all(event["maximum_normal_contact_force_n"] >= 0 for event in contact_trace.values())
        assert all(record["close_phase_samples"] > 0 for record in close_tracking.values())
        assert all(record["maximum_absolute_tracking_error_m"] >= 0 for record in close_tracking.values())


def test_contact_retention_telemetry_is_opt_in_and_does_not_change_the_pick():
    baseline = run_trial(
        DEMO_TARGET_XY,
        controller="camera_estimate",
        finger_pad_sliding_friction=0.5,
        release_profile="clearance",
        lowest_approach_tool_height_offset_m=0.020,
    )
    measured = run_trial(
        DEMO_TARGET_XY,
        controller="camera_estimate",
        finger_pad_sliding_friction=0.5,
        release_profile="clearance",
        lowest_approach_tool_height_offset_m=0.020,
        record_contact_retention_kinematics=True,
        capture_contact_retention_trace=True,
    )

    assert baseline["contact_retention_kinematics"] is None
    assert measured["physical_stable_lift_success"] == baseline["physical_stable_lift_success"]
    assert measured["minimum_finger_payload_contact_points_during_hold"] == baseline[
        "minimum_finger_payload_contact_points_during_hold"
    ]
    assert measured["finger_pad_payload_contact_trace"] == baseline[
        "finger_pad_payload_contact_trace"
    ]
    telemetry = measured["contact_retention_kinematics"]
    assert len(telemetry["samples"]) > 1000
    assert telemetry["sampling"]["sample_period_s"] == 0.02
    assert all(
        record["HOLD UNDER LOAD"]["contact_samples"] == 0
        for record in telemetry["by_finger_pad_and_phase"].values()
    )
    arm_motion = telemetry["payload_relative_motion_by_arm_and_phase"]
    assert max(
        arm["by_phase"]["HOLD UNDER LOAD"]["maximum_downward_payload_drift_m"]
        for arm in arm_motion.values()
    ) > 0.10
    assert all(
        pad["HOLD UNDER LOAD"]["maximum_slide_tracking_error_m"] < 0.001
        for pad in telemetry["by_finger_pad_and_phase"].values()
    )


def test_contact_retention_trace_requires_aggregate_telemetry():
    with pytest.raises(ValueError, match="requires telemetry"):
        run_trial(
            DEMO_TARGET_XY,
            controller="camera_estimate",
            capture_contact_retention_trace=True,
        )


def test_per_finger_contact_indexing_handles_geom_order_and_ignores_other_pairs():
    class Contact:
        def __init__(self, geom1, geom2):
            self.geom1 = geom1
            self.geom2 = geom2

    class Data:
        contact = [Contact(10, 30), Contact(11, 10), Contact(10, 12), Contact(30, 11)]
        ncon = len(contact)

    assert _finger_pad_payload_contact_indices(
        Data(), 10, {"arm0_left_pad": 30, "arm0_right_pad": 11}
    ) == {"arm0_left_pad": [0], "arm0_right_pad": [1]}


def test_orthogonal_candidate_never_confirms_an_unstable_side_view():
    metrics = run_trial(
        (0.0780231361362153, -0.008218048169793555),
        controller="camera_estimate",
        finger_pad_half_size_m=0.014,
        release_profile="clearance",
        visual_verification=True,
        visual_verification_min_upward_displacement_px=50.0,
        visual_verification_max_marker_rotation_from_start_deg=4.0,
        visual_verification_orthogonal_candidate=True,
        visual_verification_occluder=True,
    )

    baseline = metrics["visual_verification"]
    candidate = metrics["visual_verification_orthogonal_candidate"]
    assert baseline["decision"] in {"confirmed", "not_confirmed", "abstain"}
    assert candidate["decision"] in {"not_confirmed", "abstain"}
    assert candidate["evaluation_only_physics"]["raised_and_stable_at_both_frames"] is False
    assert candidate["evaluation_only_physics"]["available_to_visual_decision"] is False


def test_unexpected_contact_attribution_names_geom_pairs_in_either_order(monkeypatch):
    class Contact:
        def __init__(self, geom1, geom2, dist, pos):
            self.geom1 = geom1
            self.geom2 = geom2
            self.dist = dist
            self.pos = pos

    class Data:
        contact = [
            Contact(10, 30, -0.002, (0.1, 0.2, 0.3)),
            Contact(20, 11, -0.0004, (0.4, 0.5, 0.6)),
            Contact(12, 20, -0.001, (0.7, 0.8, 0.9)),
        ]
        ncon = len(contact)

    names = {
        10: "arm0_palm",
        11: "arm1_forearm_visual",
        12: "arm0_left_pad",
        20: "payload_canister",
        30: "table",
    }
    monkeypatch.setattr(
        mujoco,
        "mj_id2name",
        lambda _model, _object_type, geom_id: names[geom_id],
    )

    details = _unexpected_arm_contact_details(
        object(),
        Data(),
        geom_owners={10: 0, 11: 1, 12: 0},
        pad_ids={12},
        payload_geom_id=20,
        table_geom_id=30,
        ground_geom_id=31,
    )

    assert [detail["category"] for detail in details] == [
        "arm_environment",
        "nonfinger_payload",
    ]
    assert details[0]["arm_geom"] == "arm0_palm"
    assert details[0]["other_geom"] == "table"
    assert details[0]["penetration_m"] == 0.002
    assert details[1]["arm_geom"] == "arm1_forearm_visual"
    assert details[1]["other_geom"] == "payload_canister"
    assert details[1]["penetration_m"] == 0.0004


def test_paired_csv_preserves_exact_seeded_target_positions(tmp_path):
    summary = run_validation(
        tmp_path,
        seeds=1,
        samples_per_seed=1,
        base_seed=28100,
        media_dir=None,
        release_profile="clearance",
    )
    expected = _sample_positions(
        seeds=1,
        samples_per_seed=1,
        base_seed=28100,
    )[0][2]
    lines = (tmp_path / summary["trial_data"]).read_text().splitlines()
    assert len(lines) == 3
    values = lines[1].split(",")
    assert float(values[3]) == expected[0]
    assert float(values[4]) == expected[1]
    assert (tmp_path / summary["contact_events"]).is_file()
    assert (tmp_path / summary["phase_outcomes_chart"]).is_file()
    assert set(summary["contact_attribution"]) == {"camera_estimate", "oracle_xy"}
    assert summary["release_profile"] == "clearance"
    with (tmp_path / summary["trial_data"]).open(newline="") as stream:
        trial_rows = list(csv.DictReader(stream))
    for row in trial_rows:
        assert row["release_profile"] == "clearance"
        assert (row["physical_stable_lift_success"] == "True") == (
            row["payload_stable_lift_success"] == "True"
            and row["full_sequence_collision_safe"] == "True"
        )
