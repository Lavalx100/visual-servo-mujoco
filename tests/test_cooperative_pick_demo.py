import json

import mujoco
import pytest

from visual_servo_mujoco.cooperative_pick_demo import (
    ARM_COUNT,
    ARM_TOOL_Z_OFFSETS_M,
    CLOSED_JAW_OFFSET_M,
    DEFAULT_PAYLOAD_BODY_MASS_KG,
    GROUND_BIT,
    OPEN_JAW_OFFSET_M,
    PAYLOAD_BIT,
    PAYLOAD_INITIAL_Z_M,
    TABLE_BIT,
    _arm_collision_masks,
    _tool_height_offsets,
    build_model_xml,
    run_demo,
)


ARM_COLLIDER_SUFFIXES = (
    "base_visual",
    "upper_visual",
    "forearm_visual",
    "wrist_link",
    "palm",
    "left_pad",
    "right_pad",
)


def test_cooperative_pick_scene_has_three_articulated_arms_and_free_payload():
    model = mujoco.MjModel.from_xml_string(build_model_xml())

    assert model.njnt == ARM_COUNT * 5 + 1
    assert model.nu == ARM_COUNT * 5
    assert model.neq == 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "payload_free") >= 0
    for arm_index in range(ARM_COUNT):
        for joint_name in (
            "shoulder",
            "elbow_joint",
            "wrist_joint",
            "left_slide",
            "right_slide",
        ):
            assert mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"arm{arm_index}_{joint_name}",
            ) >= 0


def test_material_parameters_reach_pad_and_free_payload_physics():
    xml = build_model_xml(
        finger_pad_half_size_m=0.016,
        finger_pad_sliding_friction=0.55,
        payload_body_mass_kg=0.99,
    )
    model = mujoco.MjModel.from_xml_string(xml)
    pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "arm0_left_pad")
    payload_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "payload_canister"
    )
    payload_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "payload"
    )

    assert model.geom_friction[pad, 0] == pytest.approx(0.55)
    assert model.geom_friction[payload_geom, 0] == pytest.approx(0.55)
    assert model.body_mass[payload_body] == pytest.approx(0.99)
    for geom_name in (
        "payload_canister",
        "payload_lower_band",
        "payload_upper_band",
        "payload_cap",
    ):
        geom_start = xml.index(f'<geom name="{geom_name}"')
        geom_end = xml.index("/>", geom_start)
        assert 'density="0"' in xml[geom_start:geom_end]

    default_model = mujoco.MjModel.from_xml_string(build_model_xml())
    default_payload = mujoco.mj_name2id(
        default_model, mujoco.mjtObj.mjOBJ_BODY, "payload"
    )
    assert default_model.body_mass[default_payload] == pytest.approx(
        DEFAULT_PAYLOAD_BODY_MASS_KG, rel=1e-12
    )
    assert default_model.body_inertia[default_payload] == pytest.approx(
        (0.0014217686579099003, 0.0014217686579099003, 0.0011211411686258299),
        rel=1e-12,
    )
    assert model.body_inertia[payload_body] == pytest.approx(
        default_model.body_inertia[default_payload] * (0.99 / DEFAULT_PAYLOAD_BODY_MASS_KG),
        rel=1e-9,
    )


def test_arm_collision_masks_separate_arms_and_enable_shared_scene_contacts():
    model = mujoco.MjModel.from_xml_string(build_model_xml())
    arm_geom_ids = {}

    for arm_index in range(ARM_COUNT):
        bit, affinity = _arm_collision_masks(arm_index)
        assert bit == 1 << arm_index
        assert affinity & bit == 0
        for suffix in ARM_COLLIDER_SUFFIXES:
            geom_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"arm{arm_index}_{suffix}"
            )
            assert geom_id >= 0
            arm_geom_ids[arm_index, suffix] = geom_id
            assert model.geom_contype[geom_id] == bit
            assert model.geom_conaffinity[geom_id] == affinity
        wrist_connector = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"arm{arm_index}_wrist_palm_connector",
        )
        assert wrist_connector >= 0
        hand = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"arm{arm_index}_hand",
        )
        assert model.geom_bodyid[wrist_connector] == hand
        assert model.geom_contype[wrist_connector] == 0
        assert model.geom_conaffinity[wrist_connector] == 0
        for side in ("left", "right"):
            finger_rail = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"arm{arm_index}_{side}_finger_rail",
            )
            assert finger_rail >= 0
            assert model.geom_bodyid[finger_rail] == hand
            assert model.geom_contype[finger_rail] == 0
            assert model.geom_conaffinity[finger_rail] == 0

            finger_link = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"arm{arm_index}_{side}_finger_link",
            )
            assert finger_link >= 0
            finger_body = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"arm{arm_index}_{side}_finger",
            )
            assert model.geom_bodyid[finger_link] == finger_body
            # Rails attach to the palm body and the shanks attach to the moving
            # finger bodies. Both are visual-only; pads preserve grasp contact.
            assert model.geom_contype[finger_link] == 0
            assert model.geom_conaffinity[finger_link] == 0

    # MuJoCo enables a pair when either geom's type intersects the other's
    # affinity. Same-arm pairs are excluded; every different-arm pair is enabled.
    for first_arm in range(ARM_COUNT):
        for second_arm in range(ARM_COUNT):
            first_geom = arm_geom_ids[first_arm, "upper_visual"]
            second_geom = arm_geom_ids[second_arm, "upper_visual"]
            compatible = bool(
                model.geom_contype[first_geom] & model.geom_conaffinity[second_geom]
                or model.geom_contype[second_geom] & model.geom_conaffinity[first_geom]
            )
            assert compatible is (first_arm != second_arm)

    payload = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "payload_canister")
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    assert model.geom_contype[payload] == PAYLOAD_BIT
    assert model.geom_contype[table] == TABLE_BIT
    assert model.geom_contype[ground] == GROUND_BIT
    for arm_index in range(ARM_COUNT):
        bit, affinity = _arm_collision_masks(arm_index)
        assert bit & model.geom_conaffinity[payload]
        assert PAYLOAD_BIT & affinity
        assert bit & model.geom_conaffinity[table]
        assert TABLE_BIT & affinity
        assert bit & model.geom_conaffinity[ground]
        assert GROUND_BIT & affinity


def test_open_grippers_stagger_then_align_for_an_equatorial_grasp():
    assert _tool_height_offsets(OPEN_JAW_OFFSET_M) == pytest.approx(
        (0.005, 0.050, 0.095)
    )
    assert _tool_height_offsets(CLOSED_JAW_OFFSET_M) == pytest.approx(
        ARM_TOOL_Z_OFFSETS_M
    )


def test_cooperative_pick_lifts_payload_with_finger_contacts(tmp_path):
    metrics = run_demo(tmp_path, render_video=False)

    assert metrics["stable_lift"] is True
    assert metrics["peak_lift_m"] >= 0.07
    assert (
        metrics["minimum_payload_z_during_hold_m"]
        >= PAYLOAD_INITIAL_Z_M + 0.07
    )
    assert metrics["minimum_finger_payload_contact_points_during_hold"] >= 1
    assert metrics["maximum_payload_tilt_during_hold_deg"] <= 5.0
    assert metrics["maximum_inter_arm_penetration_m"] <= metrics[
        "inter_arm_penetration_limit_m"
    ]
    for phase in (
        "LOWER TO GRASP",
        "CLOSE SIX FINGERS",
        "SECURE CONTACT",
        "COOPERATIVE LIFT",
        "HOLD UNDER LOAD",
    ):
        assert metrics["inter_arm_contact_steps_by_phase"].get(phase, 0) == 0
    assert metrics["physics_steps_with_arm_table_or_ground_contact"] == 0
    assert metrics["physics_steps_with_nonfinger_arm_payload_contact"] == 0
    assert metrics["stable_lift_hold_duration_s"] == pytest.approx(1.6)
    assert metrics["payload_final_z_m"] == pytest.approx(PAYLOAD_INITIAL_Z_M, abs=0.01)
    saved = json.loads((tmp_path / "cooperative_pick_triarm.json").read_text())
    assert saved == metrics
