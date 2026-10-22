"""Reusable contact-based three-arm scene and scripted pick demonstration.

The original demo is separate from the camera-driven single-arm controller:
three independently actuated arms share a Cartesian payload trajectory. Open
grippers approach in staggered height bands, then align six pads at the payload
equator. Inter-arm contacts are enabled; the free payload is lifted through
gripper contact and friction, without a weld or state teleport. The camera-
guided experiment imports this scene and its contact accounting while adding a
one-shot target estimate and yaw-capable arm planning in a separate module.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image


ARM_COUNT = 3
ARM_BITS = (1 << ARM_COUNT) - 1
PAYLOAD_BIT = 1 << ARM_COUNT
TABLE_BIT = 1 << (ARM_COUNT + 1)
GROUND_BIT = 1 << (ARM_COUNT + 2)
LINK_LENGTHS_M = (0.26, 0.24, 0.18)
BASE_RADIUS_M = 0.54
BASE_HEIGHT_M = 0.36
TABLE_TOP_M = 0.18
PAYLOAD_HALF_HEIGHT_M = 0.06
PAYLOAD_INITIAL_Z_M = TABLE_TOP_M + PAYLOAD_HALF_HEIGHT_M
# These inertial values match the original compiled payload. The nominal
# scene keeps its legacy geometry-derived inertia so the scripted demo remains
# bit-for-bit stable; off-nominal mass experiments use these values explicitly.
DEFAULT_PAYLOAD_BODY_MASS_KG = 0.7201166534540158
DEFAULT_PAYLOAD_CANISTER_DENSITY_KG_M3 = 450.0
PAYLOAD_INERTIAL_POSITION_M = (0.0, 0.0, 0.0019206309215699748)
PAYLOAD_INERTIA_DIAGONAL_KGM2 = (
    0.0014217686579099003,
    0.0014217686579099003,
    0.0011211411686258299,
)
# The pads approach in separated height bands, then align into one equatorial
# six-point grasp as the jaws close.
ARM_TOOL_Z_OFFSETS_M = (0.050, 0.050, 0.050)
APPROACH_TOOL_Z_OFFSETS_M = (0.005, 0.050, 0.095)
GRASP_RADIUS_M = 0.084
FINGER_PAD_INWARD_OFFSET_M = 0.022
FINGER_PAD_CENTER_RADIUS_M = GRASP_RADIUS_M - FINGER_PAD_INWARD_OFFSET_M
OPEN_JAW_OFFSET_M = 0.080
CLOSED_JAW_OFFSET_M = FINGER_PAD_CENTER_RADIUS_M * math.tan(math.pi / 6.0)
STAGGER_CONVERGE_JAW_M = 0.060
INTER_ARM_PENETRATION_LIMIT_M = 0.001
TARGET_MARKER_LOCAL_Z_M = 0.067
SIMULATION_TIMESTEP_S = 0.002
BASE_YAW_ACTUATOR_KP = 300.0
BASE_YAW_ACTUATOR_KV = 40.0
BASE_YAW_ACTUATOR_FORCE_NM = 150.0
GRIPPER_YAW_ACTUATOR_KP = 500.0
GRIPPER_YAW_ACTUATOR_KV = 50.0
GRIPPER_YAW_ACTUATOR_FORCE_NM = 500.0
VIDEO_FPS = 30
VIDEO_SIZE = (960, 540)
DEMO_DURATION_S = 13.8

ARM_COLORS = (
    (0.18, 0.58, 0.96, 1),
    (0.98, 0.38, 0.25, 1),
    (0.20, 0.78, 0.58, 1),
)
# The camera-guided scene keeps the default red-marker detector's target color
# distinct from every robot link. The scripted demo retains its original palette.
CAMERA_GUIDED_ARM_COLORS = (
    ARM_COLORS[0],
    (0.96, 0.68, 0.18, 1),
    ARM_COLORS[2],
)
HOLD_START_S = 7.4
HOLD_END_S = 9.0
VIDEO_NAMES = (
    "cooperative_pick_triarm.mp4",
    "cooperative_pick_triarm_top.mp4",
    "cooperative_pick_triarm_closeup.mp4",
)


def _arm_collision_masks(arm_index: int) -> tuple[int, int]:
    """Return masks that allow cross-arm and environment contact, not self-contact."""
    arm_bit = 1 << arm_index
    affinity = (ARM_BITS ^ arm_bit) | PAYLOAD_BIT | TABLE_BIT | GROUND_BIT
    return arm_bit, affinity


def _arm_xml(
    arm_index: int,
    *,
    include_base_yaw: bool = False,
    include_gripper_yaw: bool = False,
    finger_pad_half_size_m: float = 0.014,
    finger_pad_sliding_friction: float = 2.2,
    wrist_center_radial_standoff_m: float = 0.0,
    arm_colors: tuple[tuple[float, float, float, float], ...] = ARM_COLORS,
) -> str:
    angle = 2.0 * math.pi * arm_index / ARM_COUNT
    base_x = BASE_RADIUS_M * math.cos(angle)
    base_y = BASE_RADIUS_M * math.sin(angle)
    heading = angle + math.pi
    color = " ".join(str(value) for value in arm_colors[arm_index])
    length_1, length_2, length_3 = LINK_LENGTHS_M
    finger_pad_inward_offset_m = (
        FINGER_PAD_INWARD_OFFSET_M + wrist_center_radial_standoff_m
    )
    arm_bit, affinity = _arm_collision_masks(arm_index)
    side_specs = (("left", 1.0, -1.0), ("right", -1.0, 1.0))
    fingers = []
    finger_rails = []
    # The rail is fixed to the hand and spans the full slider travel. The
    # carriage shank below follows each finger and terminates at its pad.
    for side, sign, axis_y in side_specs:
        finger_rails.append(
            f'''              <geom name="arm{arm_index}_{side}_finger_rail" type="capsule"
                    fromto="-0.026 {sign * 0.025:.3f} 0 -0.026 {sign * OPEN_JAW_OFFSET_M:.3f} 0"
                    size="0.008" rgba="0.60 0.67 0.75 1"
                    contype="0" conaffinity="0" density="0"/>'''
        )
        fingers.append(
            f"""            <body name="arm{arm_index}_{side}_finger"
                  pos="0 {sign * OPEN_JAW_OFFSET_M:.5f} 0">
              <joint name="arm{arm_index}_{side}_slide" type="slide" axis="0 {axis_y:.0f} 0"
                     range="0 {OPEN_JAW_OFFSET_M - CLOSED_JAW_OFFSET_M:.5f}"/>
              <geom name="arm{arm_index}_{side}_finger_link" type="capsule"
                    fromto="-0.026 0 0 {finger_pad_inward_offset_m:.4f} 0 -0.050" size="0.008"
                    rgba="0.60 0.67 0.75 1"
                    contype="0" conaffinity="0" density="0"/>
              <geom name="arm{arm_index}_{side}_pad" type="box"
                    pos="{finger_pad_inward_offset_m:.4f} 0 -0.050"
                    size="{finger_pad_half_size_m:.4f} {finger_pad_half_size_m:.4f} {finger_pad_half_size_m:.4f}"
                    rgba="0.90 0.93 0.98 1"
                    friction="{finger_pad_sliding_friction:.5f} 0.04 0.004" contype="{arm_bit}"
                    conaffinity="{affinity}"/>
            </body>"""
        )
    yaw_joint = (
        f'''      <joint name="arm{arm_index}_base_yaw" type="hinge" axis="0 0 1"
             range="-0.36 0.36" damping="5" armature="0.05"/>'''
        if include_base_yaw
        else ""
    )
    gripper_yaw_joint = (
        f'''              <joint name="arm{arm_index}_gripper_yaw" type="hinge"
                     axis="0 0 1" range="-0.36 0.36"
                     damping="20" armature="0.12"/>'''
        if include_gripper_yaw
        else ""
    )
    return f"""    <body name="arm{arm_index}_base"
          pos="{base_x:.6f} {base_y:.6f} {BASE_HEIGHT_M:.5f}"
          euler="0 0 {heading:.8f}">
{yaw_joint}
      <geom name="arm{arm_index}_base_visual" type="cylinder" pos="0 0 -0.095"
            size="0.068 0.085" rgba="0.20 0.27 0.36 1" contype="{arm_bit}"
            conaffinity="{affinity}"/>
      <body name="arm{arm_index}_upper" pos="0 0 0">
        <joint name="arm{arm_index}_shoulder" type="hinge" axis="0 -1 0"
               range="-2.65 2.65" damping="3" armature="0.025"/>
        <geom name="arm{arm_index}_upper_visual" type="capsule"
              fromto="0 0 0 {length_1:.4f} 0 0" size="0.027" rgba="{color}"
              contype="{arm_bit}" conaffinity="{affinity}"/>
        <body name="arm{arm_index}_elbow" pos="{length_1:.4f} 0 0">
          <joint name="arm{arm_index}_elbow_joint" type="hinge" axis="0 -1 0"
                 range="-2.65 2.65" damping="2.5" armature="0.02"/>
          <geom name="arm{arm_index}_forearm_visual" type="capsule"
                fromto="0 0 0 {length_2:.4f} 0 0" size="0.023" rgba="{color}"
                contype="{arm_bit}" conaffinity="{affinity}"/>
          <body name="arm{arm_index}_wrist" pos="{length_2:.4f} 0 0">
            <joint name="arm{arm_index}_wrist_joint" type="hinge" axis="0 -1 0"
                   range="-2.65 2.65" damping="1.8" armature="0.015"/>
            <geom name="arm{arm_index}_wrist_link" type="capsule"
                  fromto="0 0 0 {length_3:.4f} 0 0" size="0.019" rgba="{color}"
                  contype="{arm_bit}" conaffinity="{affinity}"/>
            <body name="arm{arm_index}_hand" pos="{length_3:.4f} 0 0">
{gripper_yaw_joint}
              <geom name="arm{arm_index}_palm" type="box" pos="-0.055 0 0"
                    size="0.030 0.027 0.009"
                    rgba="0.16 0.20 0.27 1" contype="{arm_bit}"
                    conaffinity="{affinity}"/>
              <!-- Render-only adapter overlaps the wrist end and palm box. -->
              <geom name="arm{arm_index}_wrist_palm_connector" type="box"
                    pos="-0.013 0 0" size="0.022 0.021 0.017"
                    rgba="0.36 0.43 0.53 1"
                    contype="0" conaffinity="0" density="0"/>
{chr(10).join(finger_rails)}
              <site name="arm{arm_index}_tool_center" type="sphere" size="0.009"
                    rgba="1 1 1 0.5"/>
{chr(10).join(fingers)}
            </body>
          </body>
        </body>
      </body>
    </body>
"""


def build_model_xml(
    *,
    include_base_yaw: bool = False,
    include_gripper_yaw: bool = False,
    finger_pad_half_size_m: float = 0.014,
    finger_pad_sliding_friction: float = 2.2,
    payload_body_mass_kg: float = DEFAULT_PAYLOAD_BODY_MASS_KG,
    wrist_center_radial_standoff_m: float = 0.0,
    include_tracking_camera: bool = False,
    include_target_marker: bool = False,
    include_visual_verification: bool = False,
    include_orthogonal_verification_camera: bool = False,
    include_verification_occluder: bool = False,
) -> str:
    """Build the contact-based scene; optional sensors extend the scripted demo."""
    if include_orthogonal_verification_camera and not include_visual_verification:
        raise ValueError("orthogonal verification camera requires visual verification")
    if include_verification_occluder and not include_orthogonal_verification_camera:
        raise ValueError("the verification occluder requires the orthogonal camera experiment")
    if not math.isfinite(finger_pad_half_size_m) or finger_pad_half_size_m <= 0.0:
        raise ValueError("finger_pad_half_size_m must be finite and positive")
    if (
        not math.isfinite(finger_pad_sliding_friction)
        or finger_pad_sliding_friction <= 0.0
    ):
        raise ValueError("finger_pad_sliding_friction must be finite and positive")
    if not math.isfinite(payload_body_mass_kg) or payload_body_mass_kg <= 0.0:
        raise ValueError("payload_body_mass_kg must be finite and positive")
    if (
        not math.isfinite(wrist_center_radial_standoff_m)
        or wrist_center_radial_standoff_m < 0.0
        or wrist_center_radial_standoff_m > finger_pad_half_size_m
    ):
        raise ValueError(
            "wrist_center_radial_standoff_m must be between zero and the finger-pad half-size"
        )
    arm_colors = (
        CAMERA_GUIDED_ARM_COLORS
        if include_target_marker or include_visual_verification
        else ARM_COLORS
    )
    arms_xml = "\n".join(
        _arm_xml(
            index,
            include_base_yaw=include_base_yaw,
            include_gripper_yaw=include_gripper_yaw,
            finger_pad_half_size_m=finger_pad_half_size_m,
            finger_pad_sliding_friction=finger_pad_sliding_friction,
            wrist_center_radial_standoff_m=wrist_center_radial_standoff_m,
            arm_colors=arm_colors,
        )
        for index in range(ARM_COUNT)
    )
    arm_actuators = "\n".join(
        f"""    <position name="arm{index}_{joint}_position" joint="arm{index}_{joint}"
              kp="210" kv="31" forcerange="-70 70"/>"""
        for index in range(ARM_COUNT)
        for joint in ("shoulder", "elbow_joint", "wrist_joint")
    )
    jaw_actuators = "\n".join(
        f"""    <position name="arm{index}_{side}_grip"
          joint="arm{index}_{side}_slide" kp="520" kv="26"
          forcerange="-38 38" ctrlrange="0 {OPEN_JAW_OFFSET_M - CLOSED_JAW_OFFSET_M:.5f}"/>"""
        for index in range(ARM_COUNT)
        for side in ("left", "right")
    )
    yaw_actuators = (
        "\n".join(
            f'''    <position name="arm{index}_base_yaw_position"
              joint="arm{index}_base_yaw" kp="{BASE_YAW_ACTUATOR_KP:g}"
              kv="{BASE_YAW_ACTUATOR_KV:g}"
              forcerange="-{BASE_YAW_ACTUATOR_FORCE_NM:g} {BASE_YAW_ACTUATOR_FORCE_NM:g}"/>'''
            for index in range(ARM_COUNT)
        )
        if include_base_yaw
        else ""
    )
    gripper_yaw_actuators = (
        "\n".join(
            f'''    <position name="arm{index}_gripper_yaw_position"
              joint="arm{index}_gripper_yaw" kp="{GRIPPER_YAW_ACTUATOR_KP:g}"
              kv="{GRIPPER_YAW_ACTUATOR_KV:g}"
              forcerange="-{GRIPPER_YAW_ACTUATOR_FORCE_NM:g} {GRIPPER_YAW_ACTUATOR_FORCE_NM:g}"/>'''
            for index in range(ARM_COUNT)
        )
        if include_gripper_yaw
        else ""
    )
    tracking_camera = (
        '''    <camera name="overhead" pos="0 0 2.0"
            xyaxes="1 0 0 0 1 0" fovy="40"/>'''
        if include_tracking_camera
        else ""
    )
    verification_camera = (
        '''    <camera name="verification_side" pos="0 -1.2 0.35"
            xyaxes="1 0 0 0 0 1" fovy="32"/>'''
        if include_visual_verification
        else ""
    )
    orthogonal_verification_camera = (
        '''    <camera name="verification_orthogonal" pos="-1.2 0 0.35"
            xyaxes="0 -1 0 0 0 1" fovy="32"/>'''
        if include_orthogonal_verification_camera
        else ""
    )
    target_marker = (
        f'''      <geom name="payload_target_marker" type="cylinder"
            pos="0 0 {TARGET_MARKER_LOCAL_Z_M:.4f}" size="0.034 0.0015"
            rgba="0.96 0.08 0.06 1" contype="0" conaffinity="0" density="0"/>'''
        if include_target_marker
        else ""
    )
    verification_marker = (
        '''      <geom name="payload_verification_marker" type="box"
            pos="0 -0.056 0" size="0.010 0.0015 0.028"
            rgba="0.92 0.08 0.78 1" contype="0" conaffinity="0" density="0"/>'''
        if include_visual_verification
        else ""
    )
    orthogonal_verification_marker = (
        '''      <geom name="payload_verification_marker_orthogonal" type="box"
            pos="-0.056 0 0" size="0.0015 0.010 0.028"
            rgba="0.92 0.08 0.78 1" contype="0" conaffinity="0" density="0"/>'''
        if include_orthogonal_verification_camera
        else ""
    )
    # The fixed bar is a rendered optical fixture in the sight line of the
    # original side camera. Collision is disabled so this paired image test
    # cannot change arm contact dynamics or the physical outcome labels.
    verification_occluder = (
        '''    <geom name="verification_occluder_rail" type="box"
            pos="0 -0.62 0.347" size="0.12 0.008 0.006"
            rgba="0.10 0.13 0.17 1" contype="0" conaffinity="0" density="0"/>'''
        if include_verification_occluder
        else ""
    )
    uses_legacy_nominal_inertial = math.isclose(
        payload_body_mass_kg,
        DEFAULT_PAYLOAD_BODY_MASS_KG,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    if uses_legacy_nominal_inertial:
        # Retain the original geometry-derived inertial path at the default.
        # Although an explicit inertial compiles to the same values, tiny
        # solver differences changed contact outcomes in the long scripted run.
        payload_inertial = ""
        payload_canister_density = f"{DEFAULT_PAYLOAD_CANISTER_DENSITY_KG_M3:.5f}"
        payload_trim_density = ""
    else:
        inertia_scale = payload_body_mass_kg / DEFAULT_PAYLOAD_BODY_MASS_KG
        payload_inertial = (
            f'''      <inertial pos="{' '.join(f'{value:.12g}' for value in PAYLOAD_INERTIAL_POSITION_M)}"
                mass="{payload_body_mass_kg:.12g}"
                diaginertia="{' '.join(f'{value * inertia_scale:.12g}' for value in PAYLOAD_INERTIA_DIAGONAL_KGM2)}"
                quat="1 0 0 0"/>'''
        )
        payload_canister_density = "0"
        payload_trim_density = ' density="0"'
    # Each arm has a separate contact bit. Its affinity excludes its own bit
    # (avoiding self-collision) but includes both other arms and the environment.
    return f"""<mujoco model="three_arm_cooperative_pick">
  <compiler angle="radian"/>
  <option timestep="{SIMULATION_TIMESTEP_S}" gravity="0 0 -9.81" integrator="RK4"
          cone="elliptic" impratio="8"/>
  <size njmax="2000" nconmax="400"/>
  <visual>
    <global offwidth="{VIDEO_SIZE[0]}" offheight="{VIDEO_SIZE[1]}"/>
    <quality shadowsize="2048"/>
    <map znear="0.02" zfar="10"/>
  </visual>
  <default>
    <joint damping="1.5" armature="0.01" limited="true"/>
    <geom density="700" friction="0.8 0.08 0.008" solref="0.006 1" solimp="0.94 0.99 0.001"/>
  </default>
  <worldbody>
    <light name="key_light" pos="0 -0.7 2.1" directional="false"
           diffuse="0.82 0.84 0.90" specular="0.35 0.35 0.40"/>
    <light name="rim_light" pos="-1.1 0.6 1.6" directional="true"
           diffuse="0.32 0.43 0.62" specular="0.16 0.18 0.24"/>
    <geom name="ground" type="plane" size="2 2 0.1"
          rgba="0.055 0.075 0.10 1" contype="{GROUND_BIT}"
          conaffinity="{ARM_BITS | PAYLOAD_BIT | TABLE_BIT}"/>
    <geom name="table" type="cylinder" pos="0 0 0.12" size="0.78 0.06"
          rgba="0.39 0.45 0.52 1" friction="1.0 0.1 0.01"
          contype="{TABLE_BIT}"
          conaffinity="{ARM_BITS | PAYLOAD_BIT | GROUND_BIT}"/>
    <geom name="table_top_inlay" type="cylinder" pos="0 0 0.181" size="0.43 0.001"
          rgba="0.51 0.58 0.64 1" contype="0" conaffinity="0"/>
{tracking_camera}
{verification_camera}
{orthogonal_verification_camera}
{verification_occluder}
{arms_xml}
    <body name="payload" pos="0 0 {PAYLOAD_INITIAL_Z_M:.5f}">
      <freejoint name="payload_free"/>
{payload_inertial}
      <geom name="payload_canister" type="cylinder"
            size="0.056 {PAYLOAD_HALF_HEIGHT_M:.3f}"
            rgba="0.98 0.49 0.12 1" friction="{finger_pad_sliding_friction:.5f} 0.06 0.006"
            density="{payload_canister_density}" contype="{PAYLOAD_BIT}"
            conaffinity="{ARM_BITS | TABLE_BIT | GROUND_BIT}"/>
      <geom name="payload_lower_band" type="cylinder" pos="0 0 -0.027"
            size="0.0575 0.006" rgba="0.18 0.24 0.31 1"
            contype="0" conaffinity="0"{payload_trim_density}/>
      <geom name="payload_upper_band" type="cylinder" pos="0 0 0.028"
            size="0.0575 0.005" rgba="0.18 0.24 0.31 1"
            contype="0" conaffinity="0"{payload_trim_density}/>
      <geom name="payload_cap" type="cylinder"
            pos="0 0 {PAYLOAD_HALF_HEIGHT_M + 0.0005:.4f}"
            size="0.040 0.004" rgba="0.96 0.75 0.31 1"
            contype="0" conaffinity="0"{payload_trim_density}/>
{target_marker}
{verification_marker}
{orthogonal_verification_marker}
    </body>
  </worldbody>
  <actuator>
{arm_actuators}
{yaw_actuators}
{gripper_yaw_actuators}
{jaw_actuators}
  </actuator>
</mujoco>"""


KEYFRAMES = (
    (0.0, 0.20, 0.50, OPEN_JAW_OFFSET_M, "READY"),
    (1.5, 0.14, 0.40, OPEN_JAW_OFFSET_M, "APPROACH"),
    (3.2, GRASP_RADIUS_M, 0.40, OPEN_JAW_OFFSET_M, "POSITION AROUND PAYLOAD"),
    (4.2, GRASP_RADIUS_M, PAYLOAD_INITIAL_Z_M,
     OPEN_JAW_OFFSET_M, "LOWER TO GRASP"),
    (4.95, GRASP_RADIUS_M, PAYLOAD_INITIAL_Z_M,
     CLOSED_JAW_OFFSET_M, "CLOSE SIX FINGERS"),
    (5.6, GRASP_RADIUS_M, PAYLOAD_INITIAL_Z_M,
     CLOSED_JAW_OFFSET_M, "SECURE CONTACT"),
    (HOLD_START_S, GRASP_RADIUS_M, 0.38,
     CLOSED_JAW_OFFSET_M, "COOPERATIVE LIFT"),
    (HOLD_END_S, GRASP_RADIUS_M, 0.38,
     CLOSED_JAW_OFFSET_M, "HOLD UNDER LOAD"),
    (10.5, GRASP_RADIUS_M, PAYLOAD_INITIAL_Z_M,
     CLOSED_JAW_OFFSET_M, "LOWER TO TABLE"),
    (11.25, GRASP_RADIUS_M, PAYLOAD_INITIAL_Z_M,
     OPEN_JAW_OFFSET_M, "RELEASE"),
    (12.6, 0.16, 0.35,
     OPEN_JAW_OFFSET_M, "WITHDRAW"),
    (13.8, 0.16, 0.35,
     OPEN_JAW_OFFSET_M, "PICK COMPLETE"),
)


def _smoothstep(fraction: float) -> float:
    clipped = min(1.0, max(0.0, fraction))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _tool_height_offsets(
    jaw_offset_m: float,
    *,
    lowest_approach_tool_height_offset_m: float = APPROACH_TOOL_Z_OFFSETS_M[0],
) -> tuple[float, float, float]:
    """Stagger open fingers and align all three grippers as the jaws close."""
    if (
        not math.isfinite(lowest_approach_tool_height_offset_m)
        or not APPROACH_TOOL_Z_OFFSETS_M[0]
        <= lowest_approach_tool_height_offset_m
        <= ARM_TOOL_Z_OFFSETS_M[0]
    ):
        raise ValueError(
            "lowest approach tool-height offset must be between the default and aligned offsets"
        )
    approach_offsets = (
        lowest_approach_tool_height_offset_m,
        *APPROACH_TOOL_Z_OFFSETS_M[1:],
    )
    stagger_fraction = _smoothstep(
        (jaw_offset_m - CLOSED_JAW_OFFSET_M)
        / (STAGGER_CONVERGE_JAW_M - CLOSED_JAW_OFFSET_M)
    )
    return tuple(
        aligned + stagger_fraction * (approach - aligned)
        for aligned, approach in zip(
            ARM_TOOL_Z_OFFSETS_M, approach_offsets, strict=True
        )
    )


def desired_pick_state(time_s: float) -> tuple[float, float, float, float, str]:
    """Return payload-center height, wrist radius, jaw offset, blend, and phase."""
    if time_s <= KEYFRAMES[0][0]:
        _, radial, height, jaw, phase = KEYFRAMES[0]
        return radial, height, jaw, 0.0, phase
    for start, end in zip(KEYFRAMES[:-1], KEYFRAMES[1:], strict=True):
        if time_s <= end[0]:
            span = end[0] - start[0]
            blend = _smoothstep((time_s - start[0]) / span)
            radial = start[1] + blend * (end[1] - start[1])
            height = start[2] + blend * (end[2] - start[2])
            jaw = start[3] + blend * (end[3] - start[3])
            return radial, height, jaw, blend, end[4]
    _, radial, height, jaw, phase = KEYFRAMES[-1]
    return radial, height, jaw, 1.0, phase


def _solve_three_link_pose(
    radial_from_base_m: float,
    vertical_from_base_m: float,
    *,
    lengths: tuple[float, float, float] = LINK_LENGTHS_M,
) -> tuple[float, float, float]:
    """Solve a planar 3R pose while keeping the tool frame level."""
    length_1, length_2, length_3 = lengths
    wrist_x = radial_from_base_m - length_3
    wrist_z = vertical_from_base_m
    radius_squared = wrist_x * wrist_x + wrist_z * wrist_z
    cosine_elbow = (radius_squared - length_1**2 - length_2**2) / (
        2.0 * length_1 * length_2
    )
    if not -1.0 <= cosine_elbow <= 1.0:
        raise ValueError("shared grasp pose is outside a three-arm workspace")
    # The negative-elbow branch keeps the first link above the tabletop while
    # the forearm descends to the grasp pose. The positive branch cuts through
    # the table in this radial workcell.
    elbow = -math.acos(cosine_elbow)
    shoulder = math.atan2(wrist_z, wrist_x) - math.atan2(
        length_2 * math.sin(elbow), length_1 + length_2 * math.cos(elbow)
    )
    wrist = -shoulder - elbow
    return shoulder, elbow, wrist


def _set_arm_controls(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    radial: float,
    payload_center_height: float,
    jaw: float,
) -> None:
    local_radius = BASE_RADIUS_M - radial
    tool_height_offsets = _tool_height_offsets(jaw)
    for arm_index in range(ARM_COUNT):
        tool_height = payload_center_height + tool_height_offsets[arm_index]
        vertical = tool_height - BASE_HEIGHT_M
        pose = _solve_three_link_pose(local_radius, vertical)
        for joint, value in zip(
            ("shoulder", "elbow_joint", "wrist_joint"), pose, strict=True
        ):
            actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"arm{arm_index}_{joint}_position"
            )
            data.ctrl[actuator_id] = value
        slide_target = OPEN_JAW_OFFSET_M - jaw
        for side in ("left", "right"):
            actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"arm{arm_index}_{side}_grip"
            )
            data.ctrl[actuator_id] = slide_target


def _contact_count(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    payload_geom_id: int,
) -> int:
    pad_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"arm{i}_{side}_pad")
        for i in range(ARM_COUNT)
        for side in ("left", "right")
    }
    return sum(
        1
        for contact_index in range(data.ncon)
        if (
            (data.contact[contact_index].geom1 == payload_geom_id
             and data.contact[contact_index].geom2 in pad_ids)
            or (data.contact[contact_index].geom2 == payload_geom_id
                and data.contact[contact_index].geom1 in pad_ids)
        )
    )


def _payload_tilt_degrees(data: mujoco.MjData, payload_body_id: int) -> float:
    """Return the free payload's angle away from upright, using its simulated pose."""
    vertical_alignment = min(1.0, max(-1.0, float(data.xmat[payload_body_id, 8])))
    return math.degrees(math.acos(vertical_alignment))


def _arm_geom_owners(model: mujoco.MjModel) -> dict[int, int]:
    """Map every arm collision geom to its owning arm index."""
    suffixes = (
        "base_visual",
        "upper_visual",
        "forearm_visual",
        "wrist_link",
        "palm",
        "left_pad",
        "right_pad",
    )
    return {
        geom_id: arm_index
        for arm_index in range(ARM_COUNT)
        for suffix in suffixes
        if (
            geom_id := mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"arm{arm_index}_{suffix}"
            )
        ) >= 0
    }


def _inter_arm_contact_stats(
    data: mujoco.MjData,
    geom_owners: dict[int, int],
) -> tuple[int, float]:
    """Count cross-arm contact pairs and report their deepest penetration."""
    count = 0
    deepest_penetration_m = 0.0
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        arm_1 = geom_owners.get(contact.geom1)
        arm_2 = geom_owners.get(contact.geom2)
        if arm_1 is not None and arm_2 is not None and arm_1 != arm_2:
            count += 1
            deepest_penetration_m = max(deepest_penetration_m, -float(contact.dist))
    return count, deepest_penetration_m


def _unexpected_arm_contact_flags(
    data: mujoco.MjData,
    geom_owners: dict[int, int],
    pad_ids: set[int],
    payload_geom_id: int,
    table_geom_id: int,
    ground_geom_id: int,
) -> tuple[bool, bool]:
    """Report arm-table/ground and non-pad arm-payload contacts."""
    environment_contact = False
    nonfinger_payload_contact = False
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        owner_1 = geom_owners.get(contact.geom1)
        owner_2 = geom_owners.get(contact.geom2)
        if (
            owner_1 is not None
            and contact.geom2 in {table_geom_id, ground_geom_id}
        ) or (
            owner_2 is not None
            and contact.geom1 in {table_geom_id, ground_geom_id}
        ):
            environment_contact = True
        if (
            owner_1 is not None
            and contact.geom2 == payload_geom_id
            and contact.geom1 not in pad_ids
        ) or (
            owner_2 is not None
            and contact.geom1 == payload_geom_id
            and contact.geom2 not in pad_ids
        ):
            nonfinger_payload_contact = True
    return environment_contact, nonfinger_payload_contact


def _camera(view: str) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.29 if view == "closeup" else 0.245)
    camera.distance = {
        "oblique": 1.58,
        "overhead": 2.42,
        "closeup": 0.72,
    }[view]
    camera.azimuth = 138.0
    camera.elevation = {"oblique": -33.0, "overhead": -88.0, "closeup": -24.0}[view]
    return camera


def _annotate_frame(
    rgb: np.ndarray,
    *,
    title: str,
    phase: str,
    time_s: float,
    payload_z: float,
    contact_count: int,
    inter_arm_contact_count: int,
    camera_label: str,
) -> np.ndarray:
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    shade = frame.copy()
    cv2.rectangle(shade, (0, 0), (VIDEO_SIZE[0], 76), (18, 24, 34), -1)
    cv2.rectangle(
        shade,
        (0, VIDEO_SIZE[1] - 58),
        (VIDEO_SIZE[0], VIDEO_SIZE[1]),
        (18, 24, 34),
        -1,
    )
    frame = cv2.addWeighted(shade, 0.90, frame, 0.10, 0)
    cv2.putText(
        frame,
        title,
        (26, 31),
        cv2.FONT_HERSHEY_DUPLEX,
        0.68,
        (242, 246, 250),
        1,
        cv2.LINE_AA,
    )
    subheading = "3 collidable arms | 9 joints | 6 fingers | friction grasp | MuJoCo simulation"
    cv2.putText(
        frame,
        subheading,
        (27, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (174, 192, 211),
        1,
        cv2.LINE_AA,
    )
    phase_color = (
        (103, 227, 178)
        if phase in {"HOLD UNDER LOAD", "PICK COMPLETE"}
        else (90, 208, 247)
    )
    cv2.putText(
        frame,
        phase,
        (VIDEO_SIZE[0] - 322, 45),
        cv2.FONT_HERSHEY_DUPLEX,
        0.53,
        phase_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        camera_label,
        (26, VIDEO_SIZE[1] - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (174, 192, 211),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"SIM {time_s:04.1f}s | payload z {payload_z:.3f} m | "
        f"finger contacts {contact_count} | arm contacts {inter_arm_contact_count}",
        (298, VIDEO_SIZE[1] - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (238, 242, 246),
        1,
        cv2.LINE_AA,
    )
    return frame


def run_demo(output_dir: Path, *, render_video: bool = True) -> dict[str, object]:
    """Run the deterministic pick scene and optionally write three MP4 views."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_string(build_model_xml())
    data = mujoco.MjData(model)
    payload_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload")
    payload_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "payload_canister")
    payload_free_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "payload_free")
    payload_qpos = int(model.jnt_qposadr[payload_free_joint])
    arm_geom_owners = _arm_geom_owners(model)
    pad_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"arm{i}_{side}_pad")
        for i in range(ARM_COUNT)
        for side in ("left", "right")
    }
    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    ground_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    data.qpos[payload_qpos : payload_qpos + 3] = (0.0, 0.0, PAYLOAD_INITIAL_Z_M)
    radial, height, jaw, _, _ = desired_pick_state(0.0)
    _set_arm_controls(model, data, radial, height, jaw)
    for arm_index in range(ARM_COUNT):
        tool_height = height + _tool_height_offsets(jaw)[arm_index]
        pose = _solve_three_link_pose(
            BASE_RADIUS_M - radial, tool_height - BASE_HEIGHT_M
        )
        for joint_name, value in zip(
            (
                f"arm{arm_index}_shoulder",
                f"arm{arm_index}_elbow_joint",
                f"arm{arm_index}_wrist_joint",
            ),
            pose,
            strict=True,
        ):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            data.qpos[int(model.jnt_qposadr[joint_id])] = value
    mujoco.mj_forward(model, data)

    writers: list[cv2.VideoWriter] = []
    renderers: list[mujoco.Renderer] = []
    preview_frames: list[Image.Image] = []
    if render_video:
        for filename in VIDEO_NAMES:
            writer = cv2.VideoWriter(
                str(output_dir / filename),
                cv2.VideoWriter_fourcc(*"mp4v"),
                VIDEO_FPS,
                VIDEO_SIZE,
            )
            if not writer.isOpened():
                writer.release()
                for opened in writers:
                    opened.release()
                raise RuntimeError("OpenCV could not open an MP4 writer")
            writers.append(writer)
        renderers = [
            mujoco.Renderer(model, height=VIDEO_SIZE[1], width=VIDEO_SIZE[0])
            for _ in VIDEO_NAMES
        ]
    cameras = [_camera("oblique"), _camera("overhead"), _camera("closeup")]

    frame_count = int(round(DEMO_DURATION_S * VIDEO_FPS)) + 1
    frame_time_s = 1.0 / VIDEO_FPS
    simulated_steps = 0
    held_z_values: list[float] = []
    held_contact_values: list[int] = []
    contact_frames = 0
    max_contacts = 0
    arm_contact_steps = 0
    arm_contact_steps_by_phase: dict[str, int] = {}
    max_arm_contacts_per_step = 0
    max_arm_penetration_m = 0.0
    arm_environment_contact_steps = 0
    nonfinger_payload_contact_steps = 0
    peak_z = PAYLOAD_INITIAL_Z_M
    min_z_during_hold = float("inf")
    max_tilt_during_hold_deg = 0.0
    for frame_index in range(frame_count):
        time_s = min(DEMO_DURATION_S, frame_index * frame_time_s)
        radial, target_z, jaw, _, phase = desired_pick_state(time_s)
        _set_arm_controls(model, data, radial, target_z, jaw)
        target_steps = int(round(time_s / SIMULATION_TIMESTEP_S))
        steps_this_frame = target_steps - simulated_steps
        for _ in range(steps_this_frame):
            mujoco.mj_step(model, data)
            if HOLD_START_S <= data.time <= HOLD_END_S:
                max_tilt_during_hold_deg = max(
                    max_tilt_during_hold_deg,
                    _payload_tilt_degrees(data, payload_body),
                )
            arm_contacts, arm_penetration_m = _inter_arm_contact_stats(
                data, arm_geom_owners
            )
            if arm_contacts:
                arm_contact_steps += 1
                arm_contact_steps_by_phase[phase] = (
                    arm_contact_steps_by_phase.get(phase, 0) + 1
                )
                max_arm_contacts_per_step = max(max_arm_contacts_per_step, arm_contacts)
            max_arm_penetration_m = max(max_arm_penetration_m, arm_penetration_m)
            environment_contact, nonfinger_payload_contact = (
                _unexpected_arm_contact_flags(
                    data,
                    arm_geom_owners,
                    pad_ids,
                    payload_geom,
                    table_geom,
                    ground_geom,
                )
            )
            if environment_contact:
                arm_environment_contact_steps += 1
            if nonfinger_payload_contact:
                nonfinger_payload_contact_steps += 1
        simulated_steps = target_steps
        payload_z = float(data.xpos[payload_body, 2])
        contacts = _contact_count(model, data, payload_geom)
        arm_contacts, _ = _inter_arm_contact_stats(data, arm_geom_owners)
        peak_z = max(peak_z, payload_z)
        max_contacts = max(max_contacts, contacts)
        if contacts:
            contact_frames += 1
        if HOLD_START_S <= time_s <= HOLD_END_S:
            held_z_values.append(payload_z)
            held_contact_values.append(contacts)
            min_z_during_hold = min(min_z_during_hold, payload_z)

        if render_video:
            for view_index, (renderer, camera, writer, filename) in enumerate(
                zip(renderers, cameras, writers, VIDEO_NAMES, strict=True)
            ):
                if view_index == 0:
                    camera.azimuth = 138.0 + 12.0 * math.sin(
                        math.pi * time_s / DEMO_DURATION_S
                    )
                renderer.update_scene(data, camera=camera)
                rgb = renderer.render()
                frame = _annotate_frame(
                    rgb,
                    title="TRI-ARM COOPERATIVE PICK",
                    phase=phase,
                    time_s=time_s,
                    payload_z=payload_z,
                    contact_count=contacts,
                    inter_arm_contact_count=arm_contacts,
                    camera_label={
                        "cooperative_pick_triarm.mp4": "OBLIQUE CAMERA",
                        "cooperative_pick_triarm_top.mp4": "OVERHEAD CAMERA",
                        "cooperative_pick_triarm_closeup.mp4": "GRIPPER CLOSE-UP",
                    }[filename],
                )
                writer.write(frame)
                if view_index == 0 and frame_index % 6 == 0:
                    small = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
                    preview_frames.append(
                        Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                    )

    for writer in writers:
        writer.release()
    for renderer in renderers:
        renderer.close()

    if not held_z_values:
        raise RuntimeError("the cooperative lift phase did not produce samples")
    stable_lift = bool(
        min_z_during_hold >= PAYLOAD_INITIAL_Z_M + 0.07
        and min(held_contact_values) > 0
        and max_tilt_during_hold_deg <= 5.0
        and max_arm_penetration_m <= INTER_ARM_PENETRATION_LIMIT_M
        and arm_environment_contact_steps == 0
        and nonfinger_payload_contact_steps == 0
    )
    final_z = float(data.xpos[payload_body, 2])
    final_tilt_deg = _payload_tilt_degrees(data, payload_body)
    metrics: dict[str, object] = {
        "experiment": "three-arm cooperative pick and lift",
        "evidence_type": "deterministic MuJoCo simulation demo; not a benchmark",
        "control_method": "shared scripted Cartesian trajectory with closed-form 3R IK",
        "grasp_method": "six position-actuated fingers; payload lifted by contact and friction",
        "arms": ARM_COUNT,
        "revolute_joints": ARM_COUNT * 3,
        "actuated_gripper_slides": ARM_COUNT * 2,
        "collision_model": (
            "per-arm contact bits; cross-arm and arm-environment contacts enabled; "
            "same-arm self-collision suppressed"
        ),
        "payload_radius_m": 0.056,
        "payload_half_height_m": PAYLOAD_HALF_HEIGHT_M,
        "grasp_tool_radius_m": GRASP_RADIUS_M,
        "finger_pad_inward_offset_m": FINGER_PAD_INWARD_OFFSET_M,
        "finger_pad_center_radius_m": FINGER_PAD_CENTER_RADIUS_M,
        "approach_tool_height_offsets_m": list(APPROACH_TOOL_Z_OFFSETS_M),
        "grasp_tool_height_offsets_m": list(ARM_TOOL_Z_OFFSETS_M),
        "closed_jaw_center_offset_m": CLOSED_JAW_OFFSET_M,
        "finger_pad_half_sizes_m": [0.014, 0.014, 0.014],
        "simulation_timestep_s": SIMULATION_TIMESTEP_S,
        "video_fps": VIDEO_FPS,
        "stable_lift_hold_duration_s": round(HOLD_END_S - HOLD_START_S, 2),
        "simulated_duration_s": round(float(data.time), 3),
        "payload_initial_z_m": round(PAYLOAD_INITIAL_Z_M, 4),
        "payload_peak_z_m": round(peak_z, 4),
        "payload_final_z_m": round(final_z, 4),
        "peak_lift_m": round(peak_z - PAYLOAD_INITIAL_Z_M, 4),
        "minimum_payload_z_during_hold_m": round(min_z_during_hold, 4),
        "maximum_payload_tilt_during_hold_deg": round(max_tilt_during_hold_deg, 2),
        "payload_final_tilt_deg": round(final_tilt_deg, 2),
        "minimum_finger_payload_contact_points_during_hold": min(held_contact_values),
        "maximum_finger_payload_contact_points": max_contacts,
        "frames_with_finger_payload_contact": contact_frames,
        "physics_steps_with_inter_arm_contact": arm_contact_steps,
        "maximum_inter_arm_contact_pairs_per_step": max_arm_contacts_per_step,
        "maximum_inter_arm_penetration_m": round(max_arm_penetration_m, 6),
        "inter_arm_penetration_limit_m": INTER_ARM_PENETRATION_LIMIT_M,
        "inter_arm_contact_steps_by_phase": arm_contact_steps_by_phase,
        "physics_steps_with_arm_table_or_ground_contact": arm_environment_contact_steps,
        "physics_steps_with_nonfinger_arm_payload_contact": nonfinger_payload_contact_steps,
        "stable_lift": stable_lift,
        "videos": list(VIDEO_NAMES) if render_video else [],
    }
    (output_dir / "cooperative_pick_triarm.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    if render_video and preview_frames:
        preview_frames[0].save(
            output_dir / "cooperative_pick_triarm.gif",
            save_all=True,
            append_images=preview_frames[1:],
            duration=1000 // (VIDEO_FPS // 6),
            loop=0,
            optimize=True,
        )
    return metrics


def main() -> int:
    repository = Path(__file__).resolve().parents[2]
    output = repository / "assets" / "demos"
    metrics = run_demo(output)
    print(json.dumps(metrics, indent=2))
    if not metrics["stable_lift"]:
        raise SystemExit("three-arm contact-based lift did not remain stable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
