"""Camera-guided three-arm pick with paired oracle-position diagnostics.

This experiment joins the default red-marker detector and pinhole image
projection from the single-arm visual-servo pipeline to the contact-based
three-arm MuJoCo scene. A camera estimate selects the object pose for a planned
cooperative grasp; the paired oracle condition substitutes ground-truth XY to
isolate the effect of target localization. Neither condition measures fresh
post-stop visual confirmation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
from PIL import Image

from .controller import pixel_to_table_xy
from .cooperative_pick_demo import (
    APPROACH_TOOL_Z_OFFSETS_M,
    ARM_COUNT,
    ARM_TOOL_Z_OFFSETS_M,
    BASE_HEIGHT_M,
    BASE_RADIUS_M,
    BASE_YAW_ACTUATOR_FORCE_NM,
    BASE_YAW_ACTUATOR_KP,
    BASE_YAW_ACTUATOR_KV,
    DEFAULT_PAYLOAD_BODY_MASS_KG,
    DEMO_DURATION_S,
    FINGER_PAD_INWARD_OFFSET_M,
    GRIPPER_YAW_ACTUATOR_FORCE_NM,
    GRIPPER_YAW_ACTUATOR_KP,
    GRIPPER_YAW_ACTUATOR_KV,
    HOLD_END_S,
    HOLD_START_S,
    INTER_ARM_PENETRATION_LIMIT_M,
    LINK_LENGTHS_M,
    OPEN_JAW_OFFSET_M,
    PAYLOAD_INITIAL_Z_M,
    RELEASE_CLEARANCE_TOOL_RADIUS_M,
    RELEASE_MIN_TOOL_HEIGHT_OFFSET_M,
    SIMULATION_TIMESTEP_S,
    TARGET_MARKER_LOCAL_Z_M,
    VIDEO_FPS,
    _arm_geom_owners,
    _contact_count,
    _inter_arm_contact_stats,
    _payload_tilt_degrees,
    _solve_three_link_pose,
    _tool_height_offsets,
    build_model_xml,
    desired_pick_state,
)
from .cooperative_pick_visual_verification import (
    DEFAULT_MAX_MARKER_ROTATION_FROM_START_DEG,
    MIN_UPWARD_DISPLACEMENT_PX,
    V3_MAX_MARKER_ROTATION_FROM_START_DEG,
    VERIFICATION_CAMERA_NAME,
    VERIFICATION_CAPTURE_TIMES_S,
    verify_raised_payload_from_images,
    verify_raised_payload_from_orthogonal_images,
)
from .run import detect_red_target
from .visual_image_faults import (
    apply_visual_image_fault,
    validate_visual_image_fault,
)

CAMERA_WIDTH = 640
CAMERA_HEIGHT_PX = 480
CAMERA_HEIGHT_M = 2.0
CAMERA_FOVY_DEG = 40.0
TARGET_MARKER_HEIGHT_M = PAYLOAD_INITIAL_Z_M + TARGET_MARKER_LOCAL_Z_M
TARGET_HALF_SPAN_M = 0.09
DEMO_TARGET_XY = (0.03965371929578002, -0.0835736804661705)
FINGER_PAD_HALF_SIZE_M = 0.016
BASE_SEED = 38100
DEFAULT_SEEDS = 3
DEFAULT_SAMPLES_PER_SEED = 10
VIDEO_WIDTH = CAMERA_WIDTH * 2
VIDEO_HEADER_HEIGHT = 64
VIDEO_FOOTER_HEIGHT = 80
VIDEO_HEIGHT = CAMERA_HEIGHT_PX + VIDEO_HEADER_HEIGHT + VIDEO_FOOTER_HEIGHT
GIF_PREVIEW_FRAME_STRIDE = 8
GIF_PREVIEW_SIZE = (720, 351)
VIDEO_NAME = "camera_guided_cooperative_pick.mp4"
GIF_NAME = "camera_guided_cooperative_pick.gif"
CSV_NAME = "cooperative-pick-pose-validation-trials.csv"
JSON_NAME = "cooperative-pick-pose-validation-summary.json"
SVG_NAME = "cooperative-pick-pose-validation.svg"
CONTACT_EVENTS_CSV_NAME = "cooperative-pick-contact-events.csv"
PHASE_OUTCOMES_SVG_NAME = "cooperative-pick-phase-outcomes.svg"
RETENTION_KINEMATICS_SAMPLE_STRIDE = 10
RETENTION_KINEMATICS_PHASES = ("COOPERATIVE LIFT", "HOLD UNDER LOAD")


def _target_world_position_from_image(
    rgb_image: np.ndarray,
    *,
    camera_height_m: float = CAMERA_HEIGHT_M,
    target_height_m: float = TARGET_MARKER_HEIGHT_M,
    vertical_fov_deg: float = CAMERA_FOVY_DEG,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Use the project's unchanged red detector and pinhole pixel projection."""
    pixel_xy = detect_red_target(rgb_image)
    image_height, image_width = rgb_image.shape[:2]
    world_xy = pixel_to_table_xy(
        pixel_xy,
        (image_width, image_height),
        camera_height=camera_height_m,
        target_height=target_height_m,
        vertical_fov_degrees=vertical_fov_deg,
    )
    return pixel_xy, world_xy


def _arm_target_pose(
    arm_index: int,
    target_xy: tuple[float, float],
    tool_radius_m: float,
    tool_height_m: float,
) -> tuple[float, tuple[float, float, float], float]:
    """Return base yaw, elbow-up 3R pose, and independent gripper yaw."""
    arm_angle = 2.0 * math.pi * arm_index / ARM_COUNT
    base_xy = (
        BASE_RADIUS_M * math.cos(arm_angle),
        BASE_RADIUS_M * math.sin(arm_angle),
    )
    heading = arm_angle + math.pi
    forward_axis = (math.cos(heading), math.sin(heading))
    left_axis = (-math.sin(heading), math.cos(heading))
    # Keep each wrist at the same radial grasp location relative to the object.
    # Base yaw aims at that wrist; the independent gripper yaw keeps paired
    # finger contacts symmetric when the object moves away from the table center.
    outward = (math.cos(arm_angle), math.sin(arm_angle))
    tool_xy = (
        target_xy[0] + tool_radius_m * outward[0],
        target_xy[1] + tool_radius_m * outward[1],
    )
    to_tool = (tool_xy[0] - base_xy[0], tool_xy[1] - base_xy[1])
    local_forward = to_tool[0] * forward_axis[0] + to_tool[1] * forward_axis[1]
    local_left = to_tool[0] * left_axis[0] + to_tool[1] * left_axis[1]
    base_yaw = math.atan2(local_left, local_forward)
    radial_to_tool = math.hypot(*to_tool)
    pose = _solve_three_link_pose(
        radial_to_tool,
        tool_height_m - BASE_HEIGHT_M,
        lengths=LINK_LENGTHS_M,
    )
    return base_yaw, pose, -base_yaw


def _set_camera_guided_arm_controls(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_xy: tuple[float, float],
    payload_center_height_m: float,
    tool_radius_m: float,
    jaw_offset_m: float,
    tool_height_offsets_m: tuple[float, ...] | None = None,
) -> None:
    """Plan all three yaw-plus-3R arm poses from one target position estimate."""
    tool_height_offsets = tool_height_offsets_m or _tool_height_offsets(jaw_offset_m)
    for arm_index in range(ARM_COUNT):
        tool_height = payload_center_height_m + tool_height_offsets[arm_index]
        base_yaw, pose, gripper_yaw = _arm_target_pose(
            arm_index,
            target_xy,
            tool_radius_m,
            tool_height,
        )
        references = (
            ("base_yaw", base_yaw),
            ("shoulder", pose[0]),
            ("elbow_joint", pose[1]),
            ("wrist_joint", pose[2]),
            ("gripper_yaw", gripper_yaw),
        )
        for joint_suffix, value in references:
            actuator_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"arm{arm_index}_{joint_suffix}_position",
            )
            data.ctrl[actuator_id] = value
        slide_target = OPEN_JAW_OFFSET_M - jaw_offset_m
        for side in ("left", "right"):
            actuator_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"arm{arm_index}_{side}_grip",
            )
            data.ctrl[actuator_id] = slide_target


def _apply_release_profile(
    radial: float,
    jaw_offset_m: float,
    phase: str,
    release_profile: str,
) -> tuple[float, tuple[float, ...] | None]:
    """Return the wrist radius and optional per-arm height offsets for a phase."""
    if release_profile == "baseline" or phase not in {"RELEASE", "WITHDRAW"}:
        return radial, None
    if release_profile != "clearance":
        raise ValueError("release_profile must be 'baseline' or 'clearance'")
    heights = tuple(
        max(offset, RELEASE_MIN_TOOL_HEIGHT_OFFSET_M)
        for offset in _tool_height_offsets(jaw_offset_m)
    )
    return max(radial, RELEASE_CLEARANCE_TOOL_RADIUS_M), heights


def _initialize_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_xy: tuple[float, float],
    wrist_center_radial_standoff_m: float = 0.0,
    lowest_approach_tool_height_offset_m: float = APPROACH_TOOL_Z_OFFSETS_M[0],
) -> None:
    """Place the free payload and initialize the actuated arms at their ready pose."""
    payload_free_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "payload_free"
    )
    qpos_address = int(model.jnt_qposadr[payload_free_joint])
    data.qpos[qpos_address : qpos_address + 3] = (
        target_xy[0], target_xy[1], PAYLOAD_INITIAL_Z_M
    )
    data.qpos[qpos_address + 3 : qpos_address + 7] = (1.0, 0.0, 0.0, 0.0)
    # Start from the same fixed centered ready pose for every randomized target.
    # Ground-truth XY is used only to place the payload and is not leaked into
    # the robot's initial joint configuration.
    ready_target_xy = (0.0, 0.0)
    radius, center_height, jaw, _, _ = desired_pick_state(0.0)
    radius += wrist_center_radial_standoff_m
    _set_camera_guided_arm_controls(
        model, data, ready_target_xy, center_height, radius, jaw
    )
    for arm_index in range(ARM_COUNT):
        base_yaw, pose, gripper_yaw = _arm_target_pose(
            arm_index,
            ready_target_xy,
            radius,
            center_height
            + _tool_height_offsets(
                jaw,
                lowest_approach_tool_height_offset_m=lowest_approach_tool_height_offset_m,
            )[arm_index],
        )
        for joint_suffix, value in (
            ("base_yaw", base_yaw),
            ("shoulder", pose[0]),
            ("elbow_joint", pose[1]),
            ("wrist_joint", pose[2]),
            ("gripper_yaw", gripper_yaw),
        ):
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"arm{arm_index}_{joint_suffix}",
            )
            data.qpos[int(model.jnt_qposadr[joint_id])] = value
    mujoco.mj_forward(model, data)


def _render_overhead_rgb(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
) -> np.ndarray:
    """Render the current scene through the overhead camera for video display."""
    renderer.update_scene(data, camera="overhead")
    return renderer.render()


def _write_overlay(
    writer: cv2.VideoWriter,
    detected_pixel: tuple[float, float],
    estimated_xy: tuple[float, float],
    estimate_error_m: float,
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    view: mujoco.MjvCamera,
    *,
    time_s: float,
    phase: str,
    payload_z_m: float,
    contact_count: int,
    arm_contact_count: int,
    unexpected_contact_details: list[dict[str, Any]],
    release_profile: str,
    wrist_center_radial_standoff_m: float,
    lowest_approach_tool_height_offset_m: float,
    preview_frames: list[Image.Image],
) -> None:
    """Compose live overhead RGB beside the three-arm physical response."""
    camera_panel = cv2.cvtColor(_render_overhead_rgb(renderer, data), cv2.COLOR_RGB2BGR)

    renderer.update_scene(data, camera=view)
    scene_panel = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
    scene_panel = cv2.resize(scene_panel, (CAMERA_WIDTH, CAMERA_HEIGHT_PX))
    if unexpected_contact_details:
        first_contact = unexpected_contact_details[0]
        contact_label = (
            f"CONTACT: {first_contact['arm_geom']} -> {first_contact['other_geom']}"
        )
        badge_color = (45, 55, 220)
    else:
        contact_label = "NO UNEXPECTED ARM CONTACT"
        badge_color = (48, 125, 82)
    cv2.rectangle(scene_panel, (12, 49), (CAMERA_WIDTH - 12, 78), badge_color, -1)
    cv2.putText(
        scene_panel,
        contact_label,
        (20, 69),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (250, 252, 255),
        1,
        cv2.LINE_AA,
    )
    frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
    frame[:] = (22, 28, 39)
    frame[
        VIDEO_HEADER_HEIGHT : VIDEO_HEADER_HEIGHT + CAMERA_HEIGHT_PX,
        :CAMERA_WIDTH,
    ] = camera_panel
    frame[
        VIDEO_HEADER_HEIGHT : VIDEO_HEADER_HEIGHT + CAMERA_HEIGHT_PX,
        CAMERA_WIDTH:,
    ] = scene_panel
    cv2.putText(
        frame,
        "OVERHEAD CAMERA | LIVE RENDER",
        (18, 23),
        cv2.FONT_HERSHEY_DUPLEX,
        0.52,
        (234, 240, 247),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "First image plans motion; later frames are display-only",
        (18, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (179, 194, 210),
        1,
        cv2.LINE_AA,
    )
    standoff_label = (
        f"+{wrist_center_radial_standoff_m * 1000:.0f}MM"
        if wrist_center_radial_standoff_m > 0.0
        else "BASELINE"
    )
    cv2.putText(
        frame,
        "THREE-ARM MUJOCO | CONTACT PHYSICS",
        (CAMERA_WIDTH + 18, 23),
        cv2.FONT_HERSHEY_DUPLEX,
        0.52,
        (234, 240, 247),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"APPROACH {lowest_approach_tool_height_offset_m * 1000:.0f} MM  |  "
        f"WRIST {standoff_label}  |  RELEASE {release_profile.upper()}",
        (CAMERA_WIDTH + 18, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (179, 194, 210),
        1,
        cv2.LINE_AA,
    )

    phase_steps = {
        "READY": 0,
        "APPROACH": 0,
        "POSITION AROUND PAYLOAD": 0,
        "LOWER TO GRASP": 0,
        "CLOSE SIX FINGERS": 1,
        "SECURE CONTACT": 1,
        "COOPERATIVE LIFT": 2,
        "HOLD UNDER LOAD": 2,
        "LOWER TO TABLE": 2,
        "RELEASE": 3,
        "WITHDRAW": 3,
        "PICK COMPLETE": 3,
    }
    active_step = phase_steps[phase]
    stage_labels = ("APPROACH", "GRASP", "LIFT", "RELEASE")
    stage_width = 305
    stage_gap = 8
    stage_top = VIDEO_HEADER_HEIGHT + CAMERA_HEIGHT_PX + 7
    for index, label in enumerate(stage_labels):
        x0 = 18 + index * (stage_width + stage_gap)
        if index < active_step:
            color = (52, 83, 77)
            text_color = (185, 222, 207)
        elif index == active_step:
            color = (58, 132, 111)
            text_color = (248, 252, 250)
        else:
            color = (43, 52, 65)
            text_color = (148, 163, 180)
        cv2.rectangle(
            frame,
            (x0, stage_top),
            (x0 + stage_width, stage_top + 28),
            color,
            -1,
        )
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
        )[0]
        text_x = x0 + (stage_width - text_size[0]) // 2
        cv2.putText(
            frame,
            label,
            (text_x, stage_top + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            text_color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        f"INITIAL PIXEL ({detected_pixel[0]:.0f}, {detected_pixel[1]:.0f})"
        f"  ->  TABLE XY ({estimated_xy[0]:+.3f}, {estimated_xy[1]:+.3f}) m"
        f"  | SIM XY ERROR {estimate_error_m * 100:.1f} cm",
        (18, VIDEO_HEIGHT - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (205, 218, 231),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{phase}  |  {time_s:04.1f} s  |  PAYLOAD {payload_z_m:.3f} m"
        f"  |  FINGERS {contact_count}  |  INTER-ARM {arm_contact_count}",
        (CAMERA_WIDTH + 18, VIDEO_HEIGHT - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (205, 218, 231),
        1,
        cv2.LINE_AA,
    )
    writer.write(frame)
    if (
        len(preview_frames) == 0
        or round(time_s * VIDEO_FPS) % GIF_PREVIEW_FRAME_STRIDE == 0
    ):
        preview = cv2.resize(frame, GIF_PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
        preview_frames.append(Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)))


def _write_visual_verification_endcard(
    writer: cv2.VideoWriter,
    initial_rgb: np.ndarray | None,
    post_lift_frames: list[tuple[float, np.ndarray]],
    visual_record: dict[str, Any],
    physics_evaluation: dict[str, Any],
    preview_frames: list[Image.Image],
    translation_only_record: dict[str, Any] | None = None,
    orientation_only_record: dict[str, Any] | None = None,
    rise_orientation_record: dict[str, Any] | None = None,
    gap_tolerant_records: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Append fresh RGB observations and a separate post-hoc physics audit."""
    frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
    frame[:] = (22, 28, 39)
    cv2.putText(
        frame,
        "FRESH SIDE-CAMERA VERIFICATION",
        (22, 25),
        cv2.FONT_HERSHEY_DUPLEX,
        0.58,
        (234, 240, 247),
        1,
        cv2.LINE_AA,
    )
    subtitle = "Three captured RGB frames; only image measurements decide the vision result"
    if visual_record.get("input_image_fault") is not None:
        fault_label = visual_record["input_image_fault"].get("label", "injected")
        subtitle += f" | IMAGE FAULT: {fault_label.upper()}"
    cv2.putText(
        frame,
        subtitle,
        (22, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (179, 194, 210),
        1,
        cv2.LINE_AA,
    )

    observations = [visual_record.get("initial_marker")]
    observations.extend(visual_record.get("post_lift_markers", []))
    images: list[np.ndarray | None] = [initial_rgb]
    images.extend(image for _, image in post_lift_frames)
    labels = ["START | 0.0 s"] + [f"FRESH | {time_s:.1f} s" for time_s, _ in post_lift_frames]
    tile_width, tile_height = 400, 300
    tile_gap = 16
    tile_start_x = 24
    tile_label_y = 85
    tile_top = 101
    for index in range(3):
        x0 = tile_start_x + index * (tile_width + tile_gap)
        label = labels[index] if index < len(labels) else "FRESH FRAME | MISSING"
        cv2.putText(
            frame,
            label,
            (x0 + 4, tile_label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (205, 218, 231),
            1,
            cv2.LINE_AA,
        )
        image = images[index] if index < len(images) else None
        if image is None:
            tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
            tile[:] = (32, 40, 52)
            cv2.putText(
                tile,
                "CAMERA FRAME UNAVAILABLE",
                (25, tile_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (185, 195, 205),
                1,
                cv2.LINE_AA,
            )
        else:
            marked = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            observation = observations[index] if index < len(observations) else None
            gap_tolerance = (
                max(gap_tolerant_records, key=int) if gap_tolerant_records else None
            )
            candidate_record = (
                gap_tolerant_records[gap_tolerance]
                if gap_tolerance is not None
                else None
            )
            candidate_observations = (
                [candidate_record.get("initial_marker")]
                + candidate_record.get("post_lift_markers", [])
                if candidate_record is not None
                else []
            )
            candidate_observation = (
                candidate_observations[index]
                if index < len(candidate_observations)
                else None
            )
            if candidate_observation is not None:
                x, y, width, height = candidate_observation["bbox_xywh_px"]
                center = tuple(round(value) for value in candidate_observation["center_px"])
                cv2.rectangle(marked, (x, y), (x + width, y + height), (83, 255, 100), 1)
                cv2.drawMarker(
                    marked,
                    center,
                    (83, 255, 100),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=7,
                    thickness=1,
                    line_type=cv2.LINE_AA,
                )
            if observation is not None:
                x, y, width, height = observation["bbox_xywh_px"]
                center = tuple(round(value) for value in observation["center_px"])
                cv2.rectangle(marked, (x, y), (x + width, y + height), (24, 194, 255), 1)
                cv2.circle(marked, center, 2, (24, 194, 255), -1, cv2.LINE_AA)
            if candidate_observation is not None:
                center_x, center_y = candidate_observation["center_px"]
                _, _, _, marker_height = candidate_observation["bbox_xywh_px"]
                crop_height = min(
                    image.shape[0], max(96, round(marker_height * 2.5))
                )
                crop_width = min(image.shape[1], round(crop_height * 4 / 3))
                crop_x0 = max(
                    0,
                    min(image.shape[1] - crop_width, round(center_x - crop_width / 2)),
                )
                crop_y0 = max(
                    0,
                    min(image.shape[0] - crop_height, round(center_y - crop_height / 2)),
                )
                focused = marked[
                    crop_y0 : crop_y0 + crop_height,
                    crop_x0 : crop_x0 + crop_width,
                ]
            else:
                focused = marked
            tile = cv2.resize(focused, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
            if gap_tolerance is not None:
                cv2.rectangle(tile, (0, 0), (tile_width, 47), (20, 26, 35), -1)
                cv2.putText(
                    tile,
                    "V3 component box",
                    (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (24, 194, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    tile,
                    f"V4 union, {gap_tolerance}px bound",
                    (8, 39),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (83, 255, 100),
                    1,
                    cv2.LINE_AA,
                )
        frame[tile_top : tile_top + tile_height, x0 : x0 + tile_width] = tile

    decision = visual_record.get("decision", "abstain")
    verdict_color = {
        "confirmed": (60, 148, 112),
        "not_confirmed": (65, 96, 135),
        "abstain": (67, 78, 94),
    }.get(decision, (67, 78, 94))
    cv2.rectangle(frame, (24, 426), (624, 584), (33, 43, 57), -1)
    cv2.rectangle(frame, (644, 426), (1256, 584), (33, 43, 57), -1)
    cv2.putText(
        frame,
        "IMAGE-ONLY DECISION",
        (42, 453),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (179, 194, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(frame, (42, 466), (238, 499), verdict_color, -1)
    cv2.putText(
        frame,
        decision.replace("_", " ").upper(),
        (52, 489),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (248, 252, 255),
        1,
        cv2.LINE_AA,
    )
    displacements = visual_record.get("upward_displacement_px")
    displacement_text = (
        f"Marker rise: {displacements[0]:.1f}, {displacements[1]:.1f} px"
        if displacements and len(displacements) == 2
        else "Marker rise: unavailable"
    )
    drift = visual_record.get("interframe_drift_px")
    drift_text = f"Inter-frame drift: {drift:.1f} px" if drift is not None else "Inter-frame drift: unavailable"
    cv2.putText(
        frame,
        displacement_text,
        (254, 480),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (218, 227, 236),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        drift_text,
        (42, 516),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (179, 194, 210),
        1,
        cv2.LINE_AA,
    )
    rotations = visual_record.get("marker_rotation_from_start_deg")
    rotation_limit = visual_record.get("thresholds_deg", {}).get(
        "maximum_marker_rotation_from_start"
    )
    rotation_text = (
        f"Marker rotation: {rotations[0]:.1f}, {rotations[1]:.1f} deg"
        if rotations and len(rotations) == 2
        else "Marker rotation: unavailable"
    )
    if rotation_limit is not None:
        rotation_text += f" (limit {rotation_limit:.0f} deg)"
    cv2.putText(
        frame,
        rotation_text,
        (42, 537),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (179, 194, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        visual_record.get("reason", "unknown").replace("_", " "),
        (42, 558),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (179, 194, 210),
        1,
        cv2.LINE_AA,
    )
    if (
        translation_only_record is not None
        and orientation_only_record is not None
        and rise_orientation_record is not None
    ):
        variant_text = (
            "SAME FRAMES  V0 "
            f"{translation_only_record['decision'].upper()}  |  V1 "
            f"{orientation_only_record['decision'].upper()}  |  V2 "
            f"{rise_orientation_record['decision'].upper()}  |  V3 "
            f"{visual_record['decision'].upper()}"
        )
        cv2.putText(
            frame,
            variant_text,
            (42, 580),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (218, 227, 236),
            1,
            cv2.LINE_AA,
        )
    if gap_tolerant_records:
        candidate_text = "V4 opt-in gap candidates: " + " | ".join(
            f"{tolerance}px {record['decision'].upper()}"
            for tolerance, record in sorted(
                gap_tolerant_records.items(), key=lambda item: int(item[0])
            )
        )
        cv2.putText(
            frame,
            candidate_text,
            (42, 601),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (195, 174, 230),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        "SIMULATOR EVALUATION - NOT AN INPUT TO THE VISION DECISION",
        (662, 453),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (179, 194, 210),
        1,
        cv2.LINE_AA,
    )
    raised = physics_evaluation.get("raised_and_stable_at_both_frames")
    physical_text = "raised/stable at both frames: " + ("YES" if raised else "NO")
    hold_text = "full 1.6 s payload hold: " + (
        "PASS" if physics_evaluation.get("full_hold_payload_stable_lift_success") else "FAIL"
    )
    strict_text = "full pick cycle collision-safe: " + (
        "YES" if physics_evaluation.get("strict_physical_stable_lift_success") else "NO"
    )
    for line_index, value in enumerate((physical_text, hold_text, strict_text)):
        cv2.putText(
            frame,
            value,
            (662, 486 + line_index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (83, 209, 157) if value.endswith(("YES", "PASS")) else (205, 168, 120),
            1,
            cv2.LINE_AA,
        )
    note = "Side fiducial: visual-only, non-colliding; camera fixed in world"
    cv2.putText(
        frame,
        note,
        (662, 564),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (179, 194, 210),
        1,
        cv2.LINE_AA,
    )

    for _ in range(60):
        writer.write(frame)
    preview = cv2.resize(frame, GIF_PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
    preview_image = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
    preview_frames.extend([preview_image.copy() for _ in range(8)])


def _write_orthogonal_visual_verification_endcard(
    writer: cv2.VideoWriter,
    initial_side_rgb: np.ndarray | None,
    side_frames: list[tuple[float, np.ndarray]],
    initial_orthogonal_rgb: np.ndarray | None,
    orthogonal_frames: list[tuple[float, np.ndarray]],
    candidate_record: dict[str, Any],
    physics_evaluation: dict[str, Any],
    preview_frames: list[Image.Image],
) -> None:
    """Append a two-camera evidence card for the opt-in verifier candidate."""
    frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
    frame[:] = (22, 28, 39)
    cv2.putText(
        frame,
        "ORTHOGONAL-VIEW VERIFICATION | OPT-IN CANDIDATE",
        (22, 25),
        cv2.FONT_HERSHEY_DUPLEX,
        0.56,
        (234, 240, 247),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Two perpendicular side views; a static MuJoCo rail occludes the original view only",
        (22, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (179, 194, 210),
        1,
        cv2.LINE_AA,
    )

    records = candidate_record.get("view_records", {})
    view_specs = (
        ("SIDE VIEW", initial_side_rgb, side_frames, records.get("side", {})),
        (
            "ORTHOGONAL VIEW",
            initial_orthogonal_rgb,
            orthogonal_frames,
            records.get("orthogonal", {}),
        ),
    )
    tile_width, tile_height = 392, 180
    tile_xs = (24, 444, 864)
    row_y_positions = (80, 300)
    for row_index, (label, initial_rgb, post_frames, record) in enumerate(view_specs):
        row_y = row_y_positions[row_index]
        cv2.putText(
            frame,
            label,
            (tile_xs[0], row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 191, 111) if row_index == 0 else (112, 206, 190),
            1,
            cv2.LINE_AA,
        )
        images = [initial_rgb] + [image for _, image in post_frames]
        times = ["START | 0.0 s"] + [f"FRESH | {time_s:.1f} s" for time_s, _ in post_frames]
        observations = [record.get("initial_marker")]
        observations.extend(record.get("post_lift_markers", []))
        for column, x0 in enumerate(tile_xs):
            cv2.putText(
                frame,
                times[column] if column < len(times) else "CAPTURE MISSING",
                (x0 + 2, row_y + 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (205, 218, 231),
                1,
                cv2.LINE_AA,
            )
            image = images[column] if column < len(images) else None
            if image is None:
                tile = np.full((tile_height, tile_width, 3), (32, 40, 52), dtype=np.uint8)
                cv2.putText(
                    tile,
                    "CAMERA FRAME UNAVAILABLE",
                    (24, tile_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (185, 195, 205),
                    1,
                    cv2.LINE_AA,
                )
            else:
                marked = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                observation = observations[column] if column < len(observations) else None
                if observation is not None:
                    x, y, width, height = observation["bbox_xywh_px"]
                    cv2.rectangle(
                        marked,
                        (x, y),
                        (x + width, y + height),
                        (80, 240, 170),
                        2,
                    )
                tile = cv2.resize(
                    marked, (tile_width, tile_height), interpolation=cv2.INTER_AREA
                )
            y0 = row_y + 25
            frame[y0 : y0 + tile_height, x0 : x0 + tile_width] = tile

    decision = candidate_record.get("decision", "abstain")
    decision_color = {
        "confirmed": (60, 148, 112),
        "not_confirmed": (65, 96, 135),
        "abstain": (67, 78, 94),
    }.get(decision, (67, 78, 94))
    cv2.rectangle(frame, (24, 525), (505, 558), decision_color, -1)
    cv2.putText(
        frame,
        f"STEREO IMAGE DECISION: {decision.replace('_', ' ').upper()}",
        (34, 547),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (248, 252, 255),
        1,
        cv2.LINE_AA,
    )
    side_decision = records.get("side", {}).get("decision", "missing").upper()
    orthogonal_decision = records.get("orthogonal", {}).get("decision", "missing").upper()
    cv2.putText(
        frame,
        f"side: {side_decision} | orthogonal: {orthogonal_decision} | {candidate_record.get('reason', 'unknown').replace('_', ' ')}",
        (24, 582),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (205, 218, 231),
        1,
        cv2.LINE_AA,
    )
    truth = physics_evaluation.get("raised_and_stable_at_both_frames")
    evaluation = (
        "SIMULATOR LABEL (post-hoc): raised/stable at both frames "
        + ("YES" if truth else "NO")
        + " | full hold "
        + ("PASS" if physics_evaluation.get("full_hold_payload_stable_lift_success") else "FAIL")
        + " | strict cycle "
        + ("PASS" if physics_evaluation.get("strict_physical_stable_lift_success") else "FAIL")
    )
    cv2.putText(
        frame,
        evaluation,
        (24, 606),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (83, 209, 157) if truth else (205, 168, 120),
        1,
        cv2.LINE_AA,
    )
    for _ in range(60):
        writer.write(frame)
    preview = cv2.resize(frame, GIF_PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
    preview_image = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
    preview_frames.extend([preview_image.copy() for _ in range(45)])


def _failure_reason(metrics: dict[str, Any]) -> str:
    if metrics["physical_stable_lift_success"]:
        return "none"
    if (
        metrics["control_target_source"] == "detected_image_position"
        and not metrics["target_detected"]
    ):
        return "target_not_detected"
    if metrics["minimum_finger_payload_contact_points_during_hold"] == 0:
        return "no_grasp_contact"
    if metrics["minimum_payload_z_during_hold_m"] < PAYLOAD_INITIAL_Z_M + 0.07:
        return "insufficient_lift"
    if metrics["maximum_payload_tilt_during_hold_deg"] > 5.0:
        return "payload_tilt"
    if metrics["maximum_inter_arm_penetration_m"] > INTER_ARM_PENETRATION_LIMIT_M:
        return "inter_arm_collision"
    if metrics["physics_steps_with_arm_table_or_ground_contact"]:
        return "arm_environment_contact"
    if metrics["physics_steps_with_nonfinger_arm_payload_contact"]:
        return "nonfinger_payload_contact"
    return "unstable_lift"


def _unexpected_arm_contact_details(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_owners: dict[int, int],
    pad_ids: set[int],
    payload_geom_id: int,
    table_geom_id: int,
    ground_geom_id: int,
) -> list[dict[str, Any]]:
    """Describe non-pad payload and arm/table/ground contacts in this step."""
    environment_ids = {table_geom_id, ground_geom_id}
    details = []
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom_1, geom_2 = int(contact.geom1), int(contact.geom2)
        owner_1 = geom_owners.get(geom_1)
        owner_2 = geom_owners.get(geom_2)
        category = None
        arm_geom = None
        other_geom = None
        arm_index = None
        if owner_1 is not None and geom_2 in environment_ids:
            category, arm_geom, other_geom, arm_index = (
                "arm_environment",
                geom_1,
                geom_2,
                owner_1,
            )
        elif owner_2 is not None and geom_1 in environment_ids:
            category, arm_geom, other_geom, arm_index = (
                "arm_environment",
                geom_2,
                geom_1,
                owner_2,
            )
        elif (
            owner_1 is not None
            and geom_2 == payload_geom_id
            and geom_1 not in pad_ids
        ):
            category, arm_geom, other_geom, arm_index = (
                "nonfinger_payload",
                geom_1,
                geom_2,
                owner_1,
            )
        elif (
            owner_2 is not None
            and geom_1 == payload_geom_id
            and geom_2 not in pad_ids
        ):
            category, arm_geom, other_geom, arm_index = (
                "nonfinger_payload",
                geom_2,
                geom_1,
                owner_2,
            )
        if category is None or arm_geom is None or other_geom is None:
            continue
        arm_geom_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, arm_geom
        )
        other_geom_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
        )
        details.append(
            {
                "category": category,
                "arm_index": arm_index,
                "arm_geom": arm_geom_name or f"geom_{arm_geom}",
                "other_geom": other_geom_name or f"geom_{other_geom}",
                "penetration_m": round(max(0.0, -float(contact.dist)), 6),
                "position_m": [round(float(value), 6) for value in contact.pos],
            }
        )
    return sorted(
        details,
        key=lambda detail: (
            detail["category"],
            detail["arm_geom"],
            detail["other_geom"],
            -detail["penetration_m"],
        ),
    )


def _finger_pad_payload_contact_indices(
    data: mujoco.MjData,
    payload_geom_id: int,
    finger_pad_geom_ids: Mapping[str, int],
) -> dict[str, list[int]]:
    """Map payload contacts to their individual pad geometry ids."""
    pad_names_by_id = {geom_id: name for name, geom_id in finger_pad_geom_ids.items()}
    indices = {name: [] for name in finger_pad_geom_ids}
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if contact.geom1 == payload_geom_id:
            pad_name = pad_names_by_id.get(contact.geom2)
        elif contact.geom2 == payload_geom_id:
            pad_name = pad_names_by_id.get(contact.geom1)
        else:
            pad_name = None
        if pad_name is not None:
            indices[pad_name].append(contact_index)
    return indices


def _contact_force_components(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contact_indices: list[int],
) -> tuple[float, float]:
    """Sum normal and tangential force magnitudes in MuJoCo contact frames."""
    normal_force_n = 0.0
    tangential_force_n = 0.0
    for contact_index in contact_indices:
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_index, wrench)
        normal_force_n += abs(float(wrench[0]))
        tangential_force_n += float(np.linalg.norm(wrench[1:3]))
    return normal_force_n, tangential_force_n


def _payload_position_in_hand_frame(
    data: mujoco.MjData,
    payload_body_id: int,
    hand_body_id: int,
) -> np.ndarray:
    """Return the free payload center in one gripper's current hand frame."""
    hand_rotation_world = np.asarray(data.xmat[hand_body_id], dtype=np.float64).reshape(3, 3)
    payload_delta_world = (
        np.asarray(data.xpos[payload_body_id], dtype=np.float64)
        - np.asarray(data.xpos[hand_body_id], dtype=np.float64)
    )
    return hand_rotation_world.T @ payload_delta_world


def run_trial(
    target_xy: tuple[float, float],
    *,
    controller: str,
    video_path: Path | None = None,
    finger_pad_half_size_m: float = FINGER_PAD_HALF_SIZE_M,
    finger_pad_sliding_friction: float = 2.2,
    payload_body_mass_kg: float = DEFAULT_PAYLOAD_BODY_MASS_KG,
    release_profile: str = "baseline",
    wrist_center_radial_standoff_m: float = 0.0,
    lowest_approach_tool_height_offset_m: float = APPROACH_TOOL_Z_OFFSETS_M[0],
    preview_path: Path | None = None,
    visual_verification: bool = False,
    visual_verification_min_upward_displacement_px: float | None = None,
    visual_verification_max_marker_rotation_from_start_deg: float | None = None,
    visual_verification_image_fault: dict[str, Any] | None = None,
    visual_verification_image_fault_candidates: Mapping[
        str, Mapping[str, Any] | None
    ] | None = None,
    visual_verification_gap_tolerance_candidates_px: tuple[int, ...] = (),
    visual_verification_orthogonal_candidate: bool = False,
    visual_verification_occluder: bool = False,
    visual_verification_orthogonal_gap_tolerance_px: int = 28,
    record_contact_retention_kinematics: bool = False,
    capture_contact_retention_trace: bool = False,
) -> dict[str, Any]:
    """Run a pick; optional fresh-frame verification is image-only.

    Named image-fault candidates are applied after capture to the same raw
    verifier frames, allowing paired perception studies without duplicate
    physics trajectories. Optional contact-retention telemetry is evaluation-
    only and does not change the control trajectory.
    """
    if controller not in {"camera_estimate", "oracle_xy"}:
        raise ValueError("controller must be 'camera_estimate' or 'oracle_xy'")
    if not isinstance(record_contact_retention_kinematics, bool) or not isinstance(
        capture_contact_retention_trace, bool
    ):
        raise TypeError("contact-retention telemetry options must be booleans")
    if capture_contact_retention_trace and not record_contact_retention_kinematics:
        raise ValueError("capturing contact-retention samples requires telemetry")
    if release_profile not in {"baseline", "clearance"}:
        raise ValueError("release_profile must be 'baseline' or 'clearance'")
    if (
        not math.isfinite(wrist_center_radial_standoff_m)
        or wrist_center_radial_standoff_m < 0.0
        or wrist_center_radial_standoff_m > finger_pad_half_size_m
    ):
        raise ValueError(
            "wrist center radial standoff must be between zero and the finger-pad half-size"
        )
    if (
        not math.isfinite(finger_pad_sliding_friction)
        or finger_pad_sliding_friction <= 0.0
    ):
        raise ValueError("finger-pad sliding friction must be finite and positive")
    if not math.isfinite(payload_body_mass_kg) or payload_body_mass_kg <= 0.0:
        raise ValueError("payload body mass must be finite and positive")
    if (
        not math.isfinite(lowest_approach_tool_height_offset_m)
        or not APPROACH_TOOL_Z_OFFSETS_M[0]
        <= lowest_approach_tool_height_offset_m
        <= ARM_TOOL_Z_OFFSETS_M[0]
    ):
        raise ValueError(
            "lowest approach tool-height offset must be between the default and aligned offsets"
        )
    if (
        visual_verification_min_upward_displacement_px is not None
        and (
            not math.isfinite(visual_verification_min_upward_displacement_px)
            or visual_verification_min_upward_displacement_px < 0.0
        )
    ):
        raise ValueError("visual-verification rise threshold must be finite and nonnegative")
    if (
        visual_verification_max_marker_rotation_from_start_deg is not None
        and (
            not math.isfinite(visual_verification_max_marker_rotation_from_start_deg)
            or not 0.0 <= visual_verification_max_marker_rotation_from_start_deg <= 90.0
        )
    ):
        raise ValueError("visual-verification angle threshold must be between 0 and 90 degrees")
    if (
        any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 1 <= value <= 100
            for value in visual_verification_gap_tolerance_candidates_px
        )
        or len(set(visual_verification_gap_tolerance_candidates_px))
        != len(visual_verification_gap_tolerance_candidates_px)
    ):
        raise ValueError("gap-tolerance candidates must be unique integers from 1 to 100")
    if visual_verification_gap_tolerance_candidates_px and not visual_verification:
        raise ValueError("gap-tolerance candidates require visual verification")
    if (visual_verification_orthogonal_candidate or visual_verification_occluder) and not visual_verification:
        raise ValueError("orthogonal verification options require visual verification")
    if visual_verification_occluder and not visual_verification_orthogonal_candidate:
        raise ValueError("the in-scene occluder experiment requires the orthogonal candidate")
    if (
        not isinstance(visual_verification_orthogonal_gap_tolerance_px, int)
        or isinstance(visual_verification_orthogonal_gap_tolerance_px, bool)
        or not 0 <= visual_verification_orthogonal_gap_tolerance_px <= 100
    ):
        raise ValueError("orthogonal candidate gap tolerance must be an integer from 0 to 100")
    visual_verification_image_fault = validate_visual_image_fault(
        visual_verification_image_fault
    )
    normalized_fault_candidates: dict[str, dict[str, Any] | None] = {}
    if visual_verification_image_fault_candidates is not None:
        if not visual_verification:
            raise ValueError("image-fault candidates require visual verification")
        if visual_verification_image_fault is not None:
            raise ValueError(
                "use either one primary image fault or image-fault candidates, not both"
            )
        if not isinstance(visual_verification_image_fault_candidates, Mapping) or not visual_verification_image_fault_candidates:
            raise ValueError("image-fault candidates must be a non-empty name-to-fault mapping")
        for name, fault in visual_verification_image_fault_candidates.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("image-fault candidate names must be non-empty strings")
            normalized_fault_candidates[name] = validate_visual_image_fault(fault)
    model = mujoco.MjModel.from_xml_string(
        build_model_xml(
            include_base_yaw=True,
            include_gripper_yaw=True,
            finger_pad_half_size_m=finger_pad_half_size_m,
            finger_pad_sliding_friction=finger_pad_sliding_friction,
            payload_body_mass_kg=payload_body_mass_kg,
            wrist_center_radial_standoff_m=wrist_center_radial_standoff_m,
            include_tracking_camera=True,
            include_target_marker=True,
            include_visual_verification=visual_verification,
            include_orthogonal_verification_camera=visual_verification_orthogonal_candidate,
            include_verification_occluder=visual_verification_occluder,
        )
    )
    data = mujoco.MjData(model)
    _initialize_state(
        model,
        data,
        target_xy,
        wrist_center_radial_standoff_m,
        lowest_approach_tool_height_offset_m,
    )
    renderer = mujoco.Renderer(
        model,
        height=CAMERA_HEIGHT_PX,
        width=CAMERA_WIDTH,
    )
    renderer.update_scene(data, camera="overhead")
    initial_rgb = renderer.render()
    target_detected = True
    detected_pixel: tuple[float, float] | None = None
    measured_xy: tuple[float, float] | None = None
    try:
        detected_pixel, measured_xy = _target_world_position_from_image(initial_rgb)
    except RuntimeError:
        target_detected = False
    initial_verification_rgb: np.ndarray | None = None
    initial_verification_raw_rgb: np.ndarray | None = None
    initial_orthogonal_verification_rgb: np.ndarray | None = None
    initial_orthogonal_verification_raw_rgb: np.ndarray | None = None
    verification_frames: dict[float, np.ndarray] = {}
    verification_raw_frames: dict[float, np.ndarray] = {}
    orthogonal_verification_frames: dict[float, np.ndarray] = {}
    verification_physics: dict[float, dict[str, float]] = {}
    verification_frame_indices = {
        round(time_s * VIDEO_FPS): time_s
        for time_s in VERIFICATION_CAPTURE_TIMES_S
    }
    if visual_verification:
        renderer.update_scene(data, camera=VERIFICATION_CAMERA_NAME)
        initial_verification_raw_rgb = renderer.render()
        initial_verification_rgb = apply_visual_image_fault(
            initial_verification_raw_rgb,
            visual_verification_image_fault,
            is_initial_frame=True,
        )
        if visual_verification_orthogonal_candidate:
            renderer.update_scene(data, camera="verification_orthogonal")
            initial_orthogonal_verification_raw_rgb = renderer.render()
            initial_orthogonal_verification_rgb = apply_visual_image_fault(
                initial_orthogonal_verification_raw_rgb,
                visual_verification_image_fault,
                is_initial_frame=True,
            )
    control_xy = (
        measured_xy
        if controller == "camera_estimate" and measured_xy is not None
        else target_xy
    )
    estimate_error_m = (
        math.dist(measured_xy, target_xy) if measured_xy is not None else math.nan
    )
    pixel_for_video = detected_pixel or (CAMERA_WIDTH / 2, CAMERA_HEIGHT_PX / 2)
    estimate_for_video = measured_xy or (float("nan"), float("nan"))
    view = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(view)
    view.type = mujoco.mjtCamera.mjCAMERA_FREE
    view.lookat[:] = (target_xy[0], target_xy[1], 0.245)
    view.distance = 1.58
    view.azimuth = 138.0
    view.elevation = -33.0

    writer = None
    preview_frames: list[Image.Image] = []
    if video_path is not None and target_detected:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            VIDEO_FPS,
            (VIDEO_WIDTH, VIDEO_HEIGHT),
        )
        if not writer.isOpened():
            writer.release()
            renderer.close()
            raise RuntimeError("OpenCV could not open the camera-guided MP4 writer")

    payload_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload")
    payload_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "payload_canister"
    )
    arm_owners = _arm_geom_owners(model)
    finger_pad_geom_ids = {
        f"arm{i}_{side}_pad": mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"arm{i}_{side}_pad"
        )
        for i in range(ARM_COUNT)
        for side in ("left", "right")
    }
    pad_ids = set(finger_pad_geom_ids.values())
    finger_slide_qpos_addresses = {
        pad_name: int(
            model.jnt_qposadr[
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    pad_name.removesuffix("_pad") + "_slide",
                )
            ]
        )
        for pad_name in finger_pad_geom_ids
    }
    finger_grip_actuator_ids = {
        pad_name: mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            pad_name.removesuffix("_pad") + "_grip",
        )
        for pad_name in finger_pad_geom_ids
    }
    hand_body_ids = {
        arm_index: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"arm{arm_index}_hand"
        )
        for arm_index in range(ARM_COUNT)
    }
    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    ground_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")

    hold_z: list[float] = []
    hold_contacts: list[int] = []
    max_tilt_deg = 0.0
    peak_z = PAYLOAD_INITIAL_Z_M
    inter_arm_steps = 0
    max_inter_arm_penetration_m = 0.0
    inter_arm_contact_steps_by_phase: dict[str, int] = {}
    arm_environment_steps = 0
    nonfinger_payload_steps = 0
    arm_environment_steps_during_hold = 0
    nonfinger_payload_steps_during_hold = 0
    max_inter_arm_penetration_during_hold_m = 0.0
    unexpected_categories = ("arm_environment", "nonfinger_payload")
    unexpected_contact_steps_by_phase: dict[str, dict[str, int]] = {
        category: {} for category in unexpected_categories
    }
    unexpected_pair_stats: dict[
        str, dict[tuple[str, str, str], dict[str, Any]]
    ] = {category: {} for category in unexpected_categories}
    first_unexpected_contact: dict[str, dict[str, Any] | None] = {
        category: None for category in unexpected_categories
    }
    max_finger_contacts = 0
    finger_pad_payload_contact_trace: dict[str, dict[str, Any]] = {
        pad_name: {
            "contact_steps": 0,
            "first_contact_time_s": None,
            "first_contact_phase": None,
            "last_contact_time_s": None,
            "maximum_contact_points": 0,
            "maximum_normal_contact_force_n": 0.0,
            "normal_contact_impulse_ns": 0.0,
            "contact_steps_by_phase": {},
            "maximum_normal_force_by_phase_n": {},
            "normal_contact_impulse_by_phase_ns": {},
        }
        for pad_name in finger_pad_geom_ids
    }
    finger_slide_tracking_during_close: dict[str, dict[str, Any]] = {
        pad_name: {
            "close_phase_samples": 0,
            "commanded_slide_start_m": None,
            "commanded_slide_end_m": None,
            "measured_slide_start_m": None,
            "measured_slide_end_m": None,
            "maximum_absolute_tracking_error_m": 0.0,
            "mean_absolute_tracking_error_m": 0.0,
            "_absolute_error_sum_m": 0.0,
        }
        for pad_name in finger_pad_geom_ids
    }
    retention_force_tracking: dict[str, dict[str, dict[str, Any]]] = {
        pad_name: {
            phase: {
                "samples": 0,
                "contact_samples": 0,
                "normal_force_sum_n": 0.0,
                "maximum_normal_force_n": 0.0,
                "tangential_force_sum_n": 0.0,
                "maximum_tangential_force_n": 0.0,
                "tangential_to_normal_ratio_sum": 0.0,
                "maximum_tangential_to_normal_ratio": 0.0,
                "maximum_slide_tracking_error_m": 0.0,
                "slide_tracking_error_sum_m": 0.0,
            }
            for phase in RETENTION_KINEMATICS_PHASES
        }
        for pad_name in finger_pad_geom_ids
    }
    payload_hand_motion: dict[int, dict[str, Any]] = {
        arm_index: {
            "lift_start_payload_position_in_hand_m": None,
            "by_phase": {
                phase: {
                    "samples": 0,
                    "maximum_downward_payload_drift_m": 0.0,
                    "maximum_horizontal_payload_drift_m": 0.0,
                    "last_downward_payload_drift_m": 0.0,
                    "last_horizontal_payload_drift_m": 0.0,
                }
                for phase in RETENTION_KINEMATICS_PHASES
            },
        }
        for arm_index in range(ARM_COUNT)
    }
    contact_retention_samples: list[dict[str, Any]] = []
    simulated_steps = 0
    failure_to_plan: str | None = None
    if not target_detected and controller == "camera_estimate":
        failure_to_plan = "target_not_detected"

    frame_count = round(DEMO_DURATION_S * VIDEO_FPS) + 1
    for frame_index in range(frame_count):
        time_s = min(DEMO_DURATION_S, frame_index / VIDEO_FPS)
        radial, target_z, jaw, _, phase = desired_pick_state(time_s)
        tool_height_offsets = _tool_height_offsets(
            jaw,
            lowest_approach_tool_height_offset_m=lowest_approach_tool_height_offset_m,
        )
        # Keep the lowest open pad above the table and move the wrist outward
        # while the jaws release. This opt-in profile leaves the historical
        # trajectory and score reproducible.
        radial, release_height_offsets = _apply_release_profile(
            radial, jaw, phase, release_profile
        )
        if release_height_offsets is not None:
            tool_height_offsets = release_height_offsets
        radial += wrist_center_radial_standoff_m
        if failure_to_plan is None:
            _set_camera_guided_arm_controls(
                model,
                data,
                control_xy,
                target_z,
                radial,
                jaw,
                tool_height_offsets,
            )
        target_steps = round(time_s / SIMULATION_TIMESTEP_S)
        for _ in range(target_steps - simulated_steps):
            mujoco.mj_step(model, data)
            step_phase = desired_pick_state(float(data.time))[4]
            pad_contact_indices = _finger_pad_payload_contact_indices(
                data, payload_geom, finger_pad_geom_ids
            )
            for pad_name, contact_indices in pad_contact_indices.items():
                trace = finger_pad_payload_contact_trace[pad_name]
                if contact_indices:
                    normal_force_n = 0.0
                    for contact_index in contact_indices:
                        contact_wrench = np.zeros(6, dtype=np.float64)
                        mujoco.mj_contactForce(
                            model, data, contact_index, contact_wrench
                        )
                        normal_force_n += abs(float(contact_wrench[0]))
                    if trace["first_contact_time_s"] is None:
                        trace["first_contact_time_s"] = round(float(data.time), 6)
                        trace["first_contact_phase"] = step_phase
                    trace["last_contact_time_s"] = round(float(data.time), 6)
                    trace["contact_steps"] += 1
                    trace["maximum_contact_points"] = max(
                        trace["maximum_contact_points"], len(contact_indices)
                    )
                    trace["maximum_normal_contact_force_n"] = max(
                        trace["maximum_normal_contact_force_n"], normal_force_n
                    )
                    trace["normal_contact_impulse_ns"] += (
                        normal_force_n * SIMULATION_TIMESTEP_S
                    )
                    trace["contact_steps_by_phase"][step_phase] = (
                        trace["contact_steps_by_phase"].get(step_phase, 0) + 1
                    )
                    trace["maximum_normal_force_by_phase_n"][step_phase] = max(
                        trace["maximum_normal_force_by_phase_n"].get(step_phase, 0.0),
                        normal_force_n,
                    )
                    trace["normal_contact_impulse_by_phase_ns"][step_phase] = (
                        trace["normal_contact_impulse_by_phase_ns"].get(step_phase, 0.0)
                        + normal_force_n * SIMULATION_TIMESTEP_S
                    )
                if step_phase == "CLOSE SIX FINGERS":
                    tracking = finger_slide_tracking_during_close[pad_name]
                    actuator_id = finger_grip_actuator_ids[pad_name]
                    measured = float(
                        data.qpos[finger_slide_qpos_addresses[pad_name]]
                    )
                    commanded = float(data.ctrl[actuator_id])
                    absolute_error = abs(commanded - measured)
                    if tracking["close_phase_samples"] == 0:
                        tracking["commanded_slide_start_m"] = commanded
                        tracking["measured_slide_start_m"] = measured
                    tracking["commanded_slide_end_m"] = commanded
                    tracking["measured_slide_end_m"] = measured
                    tracking["close_phase_samples"] += 1
                    tracking["_absolute_error_sum_m"] += absolute_error
                    tracking["maximum_absolute_tracking_error_m"] = max(
                        tracking["maximum_absolute_tracking_error_m"],
                        absolute_error,
                    )
                    tracking["mean_absolute_tracking_error_m"] = (
                        tracking["_absolute_error_sum_m"]
                        / tracking["close_phase_samples"]
                    )
            retention_sample_step = round(float(data.time) / SIMULATION_TIMESTEP_S)
            sample_retention = (
                record_contact_retention_kinematics
                and step_phase in RETENTION_KINEMATICS_PHASES
                and retention_sample_step % RETENTION_KINEMATICS_SAMPLE_STRIDE == 0
            )
            if sample_retention:
                payload_relative_by_arm: dict[int, np.ndarray] = {}
                for arm_index, hand_body_id in hand_body_ids.items():
                    relative_position = _payload_position_in_hand_frame(
                        data, payload_body, hand_body_id
                    )
                    payload_relative_by_arm[arm_index] = relative_position
                    arm_motion = payload_hand_motion[arm_index]
                    if (
                        arm_motion["lift_start_payload_position_in_hand_m"] is None
                        and step_phase == "COOPERATIVE LIFT"
                    ):
                        arm_motion["lift_start_payload_position_in_hand_m"] = (
                            relative_position.copy()
                        )
                    lift_start = arm_motion[
                        "lift_start_payload_position_in_hand_m"
                    ]
                    if lift_start is not None:
                        displacement = relative_position - lift_start
                        downward_drift_m = max(0.0, -float(displacement[2]))
                        horizontal_drift_m = float(
                            np.linalg.norm(displacement[:2])
                        )
                        phase_motion = arm_motion["by_phase"][step_phase]
                        phase_motion["samples"] += 1
                        phase_motion["maximum_downward_payload_drift_m"] = max(
                            phase_motion["maximum_downward_payload_drift_m"],
                            downward_drift_m,
                        )
                        phase_motion["maximum_horizontal_payload_drift_m"] = max(
                            phase_motion["maximum_horizontal_payload_drift_m"],
                            horizontal_drift_m,
                        )
                        phase_motion["last_downward_payload_drift_m"] = (
                            downward_drift_m
                        )
                        phase_motion["last_horizontal_payload_drift_m"] = (
                            horizontal_drift_m
                        )

                for pad_name, contact_indices in pad_contact_indices.items():
                    arm_index = int(pad_name.split("_", maxsplit=1)[0][3:])
                    phase_stats = retention_force_tracking[pad_name][step_phase]
                    normal_force_n, tangential_force_n = _contact_force_components(
                        model, data, contact_indices
                    )
                    ratio = (
                        tangential_force_n / normal_force_n
                        if normal_force_n > 1e-8
                        else 0.0
                    )
                    actuator_id = finger_grip_actuator_ids[pad_name]
                    slide_position_m = float(
                        data.qpos[finger_slide_qpos_addresses[pad_name]]
                    )
                    slide_command_m = float(data.ctrl[actuator_id])
                    slide_error_m = abs(slide_command_m - slide_position_m)
                    phase_stats["samples"] += 1
                    phase_stats["normal_force_sum_n"] += normal_force_n
                    phase_stats["tangential_force_sum_n"] += tangential_force_n
                    phase_stats["slide_tracking_error_sum_m"] += slide_error_m
                    phase_stats["maximum_slide_tracking_error_m"] = max(
                        phase_stats["maximum_slide_tracking_error_m"],
                        slide_error_m,
                    )
                    if contact_indices:
                        phase_stats["contact_samples"] += 1
                        phase_stats["maximum_normal_force_n"] = max(
                            phase_stats["maximum_normal_force_n"], normal_force_n
                        )
                        phase_stats["maximum_tangential_force_n"] = max(
                            phase_stats["maximum_tangential_force_n"],
                            tangential_force_n,
                        )
                        phase_stats["tangential_to_normal_ratio_sum"] += ratio
                        phase_stats["maximum_tangential_to_normal_ratio"] = max(
                            phase_stats["maximum_tangential_to_normal_ratio"], ratio
                        )
                    if capture_contact_retention_trace:
                        hand_relative = payload_relative_by_arm[arm_index]
                        lift_start = payload_hand_motion[arm_index][
                            "lift_start_payload_position_in_hand_m"
                        ]
                        downward_drift_m = None
                        horizontal_drift_m = None
                        if lift_start is not None:
                            delta = hand_relative - lift_start
                            downward_drift_m = max(0.0, -float(delta[2]))
                            horizontal_drift_m = float(np.linalg.norm(delta[:2]))
                        contact_retention_samples.append(
                            {
                                "time_s": round(float(data.time), 4),
                                "phase": step_phase,
                                "finger_pad_geom": pad_name,
                                "contact_points": len(contact_indices),
                                "normal_force_n": round(normal_force_n, 6),
                                "tangential_force_n": round(tangential_force_n, 6),
                                "tangential_to_normal_force_ratio": round(ratio, 6),
                                "slide_command_m": round(slide_command_m, 8),
                                "slide_position_m": round(slide_position_m, 8),
                                "slide_tracking_error_m": round(slide_error_m, 8),
                                "payload_x_in_hand_m": round(
                                    float(hand_relative[0]), 8
                                ),
                                "payload_y_in_hand_m": round(
                                    float(hand_relative[1]), 8
                                ),
                                "payload_z_in_hand_m": round(
                                    float(hand_relative[2]), 8
                                ),
                                "downward_payload_drift_in_hand_m": (
                                    round(downward_drift_m, 8)
                                    if downward_drift_m is not None
                                    else None
                                ),
                                "horizontal_payload_drift_in_hand_m": (
                                    round(horizontal_drift_m, 8)
                                    if horizontal_drift_m is not None
                                    else None
                                ),
                            }
                        )
            arm_contacts, penetration = _inter_arm_contact_stats(data, arm_owners)
            if arm_contacts:
                inter_arm_steps += 1
                inter_arm_contact_steps_by_phase[step_phase] = (
                    inter_arm_contact_steps_by_phase.get(step_phase, 0) + 1
                )
            max_inter_arm_penetration_m = max(max_inter_arm_penetration_m, penetration)
            in_hold_window = HOLD_START_S <= data.time <= HOLD_END_S
            if in_hold_window:
                max_inter_arm_penetration_during_hold_m = max(
                    max_inter_arm_penetration_during_hold_m,
                    penetration,
                )
            unexpected_contacts = _unexpected_arm_contact_details(
                model,
                data,
                arm_owners,
                pad_ids,
                payload_geom,
                table_geom,
                ground_geom,
            )
            pairs_this_step: dict[
                str, dict[tuple[str, str], dict[str, Any]]
            ] = {category: {} for category in unexpected_categories}
            for detail in unexpected_contacts:
                category = detail["category"]
                if first_unexpected_contact[category] is None:
                    first_unexpected_contact[category] = {
                        "time_s": round(float(data.time), 6),
                        "phase": step_phase,
                        **detail,
                    }
                pair = (detail["arm_geom"], detail["other_geom"])
                existing = pairs_this_step[category].get(pair)
                if (
                    existing is None
                    or detail["penetration_m"] > existing["penetration_m"]
                ):
                    pairs_this_step[category][pair] = detail

            for category in unexpected_categories:
                pair_details = pairs_this_step[category]
                if not pair_details:
                    continue
                unexpected_contact_steps_by_phase[category][step_phase] = (
                    unexpected_contact_steps_by_phase[category].get(step_phase, 0)
                    + 1
                )
                for (arm_geom, other_geom), detail in pair_details.items():
                    key = (step_phase, arm_geom, other_geom)
                    stats = unexpected_pair_stats[category].setdefault(
                        key,
                        {
                            "phase": step_phase,
                            "arm_geom": arm_geom,
                            "other_geom": other_geom,
                            "contact_steps": 0,
                            "maximum_penetration_m": 0.0,
                            "first_contact_time_s": round(float(data.time), 6),
                            "first_contact_position_m": detail["position_m"],
                        },
                    )
                    stats["contact_steps"] += 1
                    stats["maximum_penetration_m"] = max(
                        stats["maximum_penetration_m"], detail["penetration_m"]
                    )
            arm_environment_steps += int(bool(pairs_this_step["arm_environment"]))
            nonfinger_payload_steps += int(bool(pairs_this_step["nonfinger_payload"]))
            if in_hold_window:
                arm_environment_steps_during_hold += int(
                    bool(pairs_this_step["arm_environment"])
                )
                nonfinger_payload_steps_during_hold += int(
                    bool(pairs_this_step["nonfinger_payload"])
                )
            if HOLD_START_S <= data.time <= HOLD_END_S:
                hold_z.append(float(data.xpos[payload_body, 2]))
                hold_contacts.append(_contact_count(model, data, payload_geom))
                max_tilt_deg = max(
                    max_tilt_deg,
                    _payload_tilt_degrees(data, payload_body),
                )
        simulated_steps = target_steps
        payload_z = float(data.xpos[payload_body, 2])
        peak_z = max(peak_z, payload_z)
        finger_contacts = _contact_count(model, data, payload_geom)
        max_finger_contacts = max(max_finger_contacts, finger_contacts)
        if visual_verification and frame_index in verification_frame_indices:
            capture_time_s = verification_frame_indices[frame_index]
            renderer.update_scene(data, camera=VERIFICATION_CAMERA_NAME)
            raw_rgb = renderer.render()
            verification_raw_frames[capture_time_s] = raw_rgb
            verification_frames[capture_time_s] = apply_visual_image_fault(
                raw_rgb,
                visual_verification_image_fault,
                is_initial_frame=False,
            )
            verification_physics[capture_time_s] = {
                "capture_time_s": float(data.time),
                "payload_center_z_m": payload_z,
                "payload_tilt_deg": _payload_tilt_degrees(data, payload_body),
            }
            if visual_verification_orthogonal_candidate:
                renderer.update_scene(data, camera="verification_orthogonal")
                orthogonal_raw_rgb = renderer.render()
                orthogonal_verification_frames[capture_time_s] = apply_visual_image_fault(
                    orthogonal_raw_rgb,
                    visual_verification_image_fault,
                    is_initial_frame=False,
                )
        if writer is not None and target_detected and measured_xy is not None:
            active_unexpected_contacts = _unexpected_arm_contact_details(
                model,
                data,
                arm_owners,
                pad_ids,
                payload_geom,
                table_geom,
                ground_geom,
            )
            _write_overlay(
                writer,
                pixel_for_video,
                estimate_for_video,
                estimate_error_m,
                renderer,
                data,
                view,
                time_s=time_s,
                phase=phase,
                payload_z_m=payload_z,
                contact_count=finger_contacts,
                arm_contact_count=_inter_arm_contact_stats(data, arm_owners)[0],
                unexpected_contact_details=active_unexpected_contacts,
                release_profile=release_profile,
                wrist_center_radial_standoff_m=wrist_center_radial_standoff_m,
                lowest_approach_tool_height_offset_m=lowest_approach_tool_height_offset_m,
                preview_frames=preview_frames,
            )

    visual_record: dict[str, Any] | None = None
    translation_only_visual_record: dict[str, Any] | None = None
    orientation_only_visual_record: dict[str, Any] | None = None
    rise_orientation_visual_record: dict[str, Any] | None = None
    gap_tolerant_visual_records: dict[str, dict[str, Any]] = {}
    fault_candidate_visual_records: dict[str, dict[str, Any]] = {}
    orthogonal_visual_record: dict[str, Any] | None = None
    if visual_verification:
        orthogonal_rotation_limit = (
            visual_verification_max_marker_rotation_from_start_deg
            if visual_verification_max_marker_rotation_from_start_deg is not None
            else V3_MAX_MARKER_ROTATION_FROM_START_DEG
        )
        post_lift_observations = [
            (time_s, verification_frames[time_s])
            for time_s in VERIFICATION_CAPTURE_TIMES_S
            if time_s in verification_frames
        ]
        if len(verification_frames) == len(VERIFICATION_CAPTURE_TIMES_S):
            visual_record = verify_raised_payload_from_images(
                initial_verification_rgb,
                post_lift_observations,
                min_upward_displacement_px=(
                    visual_verification_min_upward_displacement_px
                    if visual_verification_min_upward_displacement_px is not None
                    else MIN_UPWARD_DISPLACEMENT_PX
                ),
                max_marker_rotation_from_start_deg=(
                    visual_verification_max_marker_rotation_from_start_deg
                ),
            )
            for gap_tolerance_px in visual_verification_gap_tolerance_candidates_px:
                gap_tolerant_visual_records[str(gap_tolerance_px)] = (
                    verify_raised_payload_from_images(
                        initial_verification_rgb,
                        post_lift_observations,
                        min_upward_displacement_px=(
                            visual_verification_min_upward_displacement_px
                            if visual_verification_min_upward_displacement_px is not None
                            else MIN_UPWARD_DISPLACEMENT_PX
                        ),
                        max_marker_rotation_from_start_deg=(
                            visual_verification_max_marker_rotation_from_start_deg
                        ),
                        marker_vertical_gap_tolerance_px=gap_tolerance_px,
                    )
                )
            if visual_verification_image_fault is not None:
                visual_record["input_image_fault"] = dict(
                    visual_verification_image_fault
                )
            if visual_verification_max_marker_rotation_from_start_deg is not None:
                translation_only_visual_record = verify_raised_payload_from_images(
                    initial_verification_rgb,
                    post_lift_observations,
                )
                orientation_only_visual_record = verify_raised_payload_from_images(
                    initial_verification_rgb,
                    post_lift_observations,
                    max_marker_rotation_from_start_deg=(
                        DEFAULT_MAX_MARKER_ROTATION_FROM_START_DEG
                    ),
                )
                rise_orientation_visual_record = verify_raised_payload_from_images(
                    initial_verification_rgb,
                    post_lift_observations,
                    min_upward_displacement_px=50.0,
                    max_marker_rotation_from_start_deg=(
                        DEFAULT_MAX_MARKER_ROTATION_FROM_START_DEG
                    ),
                )
            else:
                orientation_only_visual_record = None
        else:
            missing_record = {
                "decision": "abstain",
                "reason": "post_lift_capture_missing",
                "runtime_inputs": "initial and two fresh side-camera RGB frames only",
                "capture_times_s": [round(time_s, 4) for time_s in verification_frames],
                "initial_marker": None,
                "post_lift_markers": [],
                "upward_displacement_px": None,
                "interframe_drift_px": None,
                "marker_rotation_from_start_deg": None,
                "thresholds_px": {},
                "thresholds_deg": {},
            }
            visual_record = missing_record
            if visual_verification_max_marker_rotation_from_start_deg is not None:
                translation_only_visual_record = dict(missing_record)
                orientation_only_visual_record = dict(missing_record)
                rise_orientation_visual_record = dict(missing_record)
            else:
                orientation_only_visual_record = None
            gap_tolerant_visual_records = {
                str(gap_tolerance_px): dict(missing_record)
                for gap_tolerance_px in visual_verification_gap_tolerance_candidates_px
            }

        if visual_verification_orthogonal_candidate:
            if (
                len(verification_frames) == len(VERIFICATION_CAPTURE_TIMES_S)
                and len(orthogonal_verification_frames) == len(VERIFICATION_CAPTURE_TIMES_S)
                and initial_verification_rgb is not None
                and initial_orthogonal_verification_rgb is not None
            ):
                orthogonal_visual_record = verify_raised_payload_from_orthogonal_images(
                    initial_verification_rgb,
                    initial_orthogonal_verification_rgb,
                    [
                        (
                            time_s,
                            verification_frames[time_s],
                            orthogonal_verification_frames[time_s],
                        )
                        for time_s in VERIFICATION_CAPTURE_TIMES_S
                    ],
                    min_upward_displacement_px=(
                        visual_verification_min_upward_displacement_px
                        if visual_verification_min_upward_displacement_px is not None
                        else MIN_UPWARD_DISPLACEMENT_PX
                    ),
                    max_marker_rotation_from_start_deg=(
                        orthogonal_rotation_limit
                    ),
                    marker_vertical_gap_tolerance_px=(
                        visual_verification_orthogonal_gap_tolerance_px
                    ),
                )
            else:
                orthogonal_visual_record = {
                    "decision": "abstain",
                    "reason": "orthogonal_view_capture_missing",
                    "runtime_inputs": "initial and two fresh RGB frames from each of two perpendicular side cameras only",
                    "camera_views": ["verification_side", "verification_orthogonal"],
                    "capture_times_s": [round(time_s, 4) for time_s in verification_frames],
                    "view_records": {},
                    "all_views_confirmed": False,
                    "simulator_truth_is_input": False,
                }

        if normalized_fault_candidates and len(verification_raw_frames) == len(
            VERIFICATION_CAPTURE_TIMES_S
        ) and initial_verification_raw_rgb is not None:
            for name, fault in normalized_fault_candidates.items():
                candidate_initial = apply_visual_image_fault(
                    initial_verification_raw_rgb,
                    fault,
                    is_initial_frame=True,
                )
                candidate_observations = [
                    (
                        time_s,
                        apply_visual_image_fault(
                            verification_raw_frames[time_s],
                            fault,
                            is_initial_frame=False,
                        ),
                    )
                    for time_s in VERIFICATION_CAPTURE_TIMES_S
                ]
                v3_record = verify_raised_payload_from_images(
                    candidate_initial,
                    candidate_observations,
                    min_upward_displacement_px=(
                        visual_verification_min_upward_displacement_px
                        if visual_verification_min_upward_displacement_px is not None
                        else MIN_UPWARD_DISPLACEMENT_PX
                    ),
                    max_marker_rotation_from_start_deg=(
                        visual_verification_max_marker_rotation_from_start_deg
                    ),
                )
                gap_records = {
                    str(tolerance): verify_raised_payload_from_images(
                        candidate_initial,
                        candidate_observations,
                        min_upward_displacement_px=(
                            visual_verification_min_upward_displacement_px
                            if visual_verification_min_upward_displacement_px is not None
                            else MIN_UPWARD_DISPLACEMENT_PX
                        ),
                        max_marker_rotation_from_start_deg=(
                            visual_verification_max_marker_rotation_from_start_deg
                        ),
                        marker_vertical_gap_tolerance_px=tolerance,
                    )
                    for tolerance in visual_verification_gap_tolerance_candidates_px
                }
                fault_candidate_visual_records[name] = {
                    "input_image_fault": fault,
                    "v3": v3_record,
                    "gap_tolerant_candidates": gap_records,
                }
        elif normalized_fault_candidates:
            fault_candidate_visual_records = {
                name: {
                    "input_image_fault": fault,
                    "v3": {
                        "decision": "abstain",
                        "reason": "post_lift_capture_missing",
                    },
                    "gap_tolerant_candidates": {
                        str(tolerance): {
                            "decision": "abstain",
                            "reason": "post_lift_capture_missing",
                        }
                        for tolerance in visual_verification_gap_tolerance_candidates_px
                    },
                }
                for name, fault in normalized_fault_candidates.items()
            }

    minimum_hold_z = min(hold_z) if hold_z else PAYLOAD_INITIAL_Z_M
    minimum_hold_contacts = min(hold_contacts) if hold_contacts else 0
    final_z = float(data.xpos[payload_body, 2])
    payload_stable_lift = bool(
        failure_to_plan is None
        and minimum_hold_z >= PAYLOAD_INITIAL_Z_M + 0.07
        and minimum_hold_contacts >= 3
        and max_tilt_deg <= 5.0
        and max_inter_arm_penetration_during_hold_m
        <= INTER_ARM_PENETRATION_LIMIT_M
        and arm_environment_steps_during_hold == 0
        and nonfinger_payload_steps_during_hold == 0
    )
    full_sequence_collision_safe = bool(
        max_inter_arm_penetration_m <= INTER_ARM_PENETRATION_LIMIT_M
        and arm_environment_steps == 0
        and nonfinger_payload_steps == 0
    )
    stable_lift = payload_stable_lift and full_sequence_collision_safe
    contact_retention_kinematics: dict[str, Any] | None = None
    if record_contact_retention_kinematics:
        contact_retention_kinematics = {
            "sampling": {
                "physics_step_stride": RETENTION_KINEMATICS_SAMPLE_STRIDE,
                "sample_period_s": round(
                    RETENTION_KINEMATICS_SAMPLE_STRIDE * SIMULATION_TIMESTEP_S, 6
                ),
                "phases": list(RETENTION_KINEMATICS_PHASES),
                "force_units": "MuJoCo simulated contact-solver forces in newtons; not calibrated sensor measurements",
                "payload_motion_frame": "each arm hand body frame, relative to that arm's first sampled cooperative-lift pose",
            },
            "by_finger_pad_and_phase": {},
            "payload_relative_motion_by_arm_and_phase": {},
        }
        for pad_name, phases in retention_force_tracking.items():
            contact_retention_kinematics["by_finger_pad_and_phase"][pad_name] = {}
            for phase, stats in phases.items():
                sample_count = int(stats["samples"])
                contact_sample_count = int(stats["contact_samples"])
                contact_retention_kinematics["by_finger_pad_and_phase"][pad_name][
                    phase
                ] = {
                    "samples": sample_count,
                    "contact_samples": contact_sample_count,
                    "contact_sample_fraction": round(
                        contact_sample_count / sample_count if sample_count else 0.0, 6
                    ),
                    "mean_normal_force_n": round(
                        stats["normal_force_sum_n"] / sample_count
                        if sample_count else 0.0,
                        6,
                    ),
                    "maximum_normal_force_n": round(
                        stats["maximum_normal_force_n"], 6
                    ),
                    "mean_tangential_force_n": round(
                        stats["tangential_force_sum_n"] / sample_count
                        if sample_count else 0.0,
                        6,
                    ),
                    "maximum_tangential_force_n": round(
                        stats["maximum_tangential_force_n"], 6
                    ),
                    "mean_tangential_to_normal_force_ratio_when_contacting": round(
                        stats["tangential_to_normal_ratio_sum"] / contact_sample_count
                        if contact_sample_count else 0.0,
                        6,
                    ),
                    "maximum_tangential_to_normal_force_ratio_when_contacting": round(
                        stats["maximum_tangential_to_normal_ratio"], 6
                    ),
                    "mean_slide_tracking_error_m": round(
                        stats["slide_tracking_error_sum_m"] / sample_count
                        if sample_count else 0.0,
                        8,
                    ),
                    "maximum_slide_tracking_error_m": round(
                        stats["maximum_slide_tracking_error_m"], 8
                    ),
                }
        for arm_index, motion in payload_hand_motion.items():
            lift_start = motion["lift_start_payload_position_in_hand_m"]
            phases: dict[str, Any] = {}
            for phase, stats in motion["by_phase"].items():
                phases[phase] = {
                    **stats,
                    "maximum_downward_payload_drift_m": round(
                        stats["maximum_downward_payload_drift_m"], 8
                    ),
                    "maximum_horizontal_payload_drift_m": round(
                        stats["maximum_horizontal_payload_drift_m"], 8
                    ),
                    "last_downward_payload_drift_m": round(
                        stats["last_downward_payload_drift_m"], 8
                    ),
                    "last_horizontal_payload_drift_m": round(
                        stats["last_horizontal_payload_drift_m"], 8
                    ),
                }
            contact_retention_kinematics[
                "payload_relative_motion_by_arm_and_phase"
            ][f"arm{arm_index}"] = {
                "lift_start_payload_position_in_hand_m": (
                    [round(float(value), 8) for value in lift_start]
                    if lift_start is not None else None
                ),
                "by_phase": phases,
            }
        if capture_contact_retention_trace:
            contact_retention_kinematics["samples"] = contact_retention_samples
    metrics: dict[str, Any] = {
        "controller": controller,
        "release_profile": release_profile,
        "finger_pad_sliding_friction": finger_pad_sliding_friction,
        "payload_body_mass_kg": float(model.body_mass[payload_body]),
        "wrist_center_radial_standoff_m": wrist_center_radial_standoff_m,
        "lowest_approach_tool_height_offset_m": lowest_approach_tool_height_offset_m,
        "finger_pad_inward_offset_m": (
            FINGER_PAD_INWARD_OFFSET_M + wrist_center_radial_standoff_m
        ),
        "control_target_source": (
            "detected_image_position" if controller == "camera_estimate" else "oracle_xy"
        ),
        "target_true_xy_m": [float(target_xy[0]), float(target_xy[1])],
        "target_detected": target_detected,
        "detected_target_pixel_xy": (
            [float(value) for value in detected_pixel]
            if detected_pixel is not None
            else None
        ),
        "camera_estimated_target_xy_m": (
            [float(value) for value in measured_xy]
            if measured_xy is not None
            else None
        ),
        "initial_target_projection_error_m": (
            float(estimate_error_m)
            if math.isfinite(estimate_error_m)
            else None
        ),
        "physical_stable_lift_success": stable_lift,
        "payload_stable_lift_success": payload_stable_lift,
        "full_sequence_collision_safe": full_sequence_collision_safe,
        "fresh_visual_confirmation": (
            visual_record["decision"] if visual_record is not None else "not measured"
        ),
        "visual_verification": visual_record,
        "visual_verification_gap_tolerant_candidates": gap_tolerant_visual_records,
        "visual_verification_fault_candidate_records": fault_candidate_visual_records,
        "visual_verification_orthogonal_candidate": orthogonal_visual_record,
        "failure_reason": failure_to_plan or "",
        "payload_initial_z_m": round(PAYLOAD_INITIAL_Z_M, 4),
        "finger_pad_half_size_m": round(finger_pad_half_size_m, 4),
        "payload_peak_z_m": round(peak_z, 4),
        "payload_final_z_m": round(final_z, 4),
        "peak_lift_m": round(peak_z - PAYLOAD_INITIAL_Z_M, 4),
        "minimum_payload_z_during_hold_m": round(minimum_hold_z, 4),
        "minimum_finger_payload_contact_points_during_hold": minimum_hold_contacts,
        "maximum_finger_payload_contact_points": max_finger_contacts,
        "finger_pad_payload_contact_trace": {
            pad_name: {
                **trace,
                "maximum_normal_contact_force_n": round(
                    trace["maximum_normal_contact_force_n"], 6
                ),
                "normal_contact_impulse_ns": round(
                    trace["normal_contact_impulse_ns"], 8
                ),
                "maximum_normal_force_by_phase_n": {
                    phase: round(force_n, 6)
                    for phase, force_n in trace[
                        "maximum_normal_force_by_phase_n"
                    ].items()
                },
                "normal_contact_impulse_by_phase_ns": {
                    phase: round(impulse_ns, 8)
                    for phase, impulse_ns in trace[
                        "normal_contact_impulse_by_phase_ns"
                    ].items()
                },
            }
            for pad_name, trace in finger_pad_payload_contact_trace.items()
        },
        "finger_slide_tracking_during_close": {
            pad_name: {
                key: (
                    round(value, 8)
                    if isinstance(value, float)
                    else value
                )
                for key, value in tracking.items()
                if not key.startswith("_")
            }
            for pad_name, tracking in finger_slide_tracking_during_close.items()
        },
        "contact_retention_kinematics": contact_retention_kinematics,
        "maximum_payload_tilt_during_hold_deg": round(max_tilt_deg, 2),
        "physics_steps_with_inter_arm_contact": inter_arm_steps,
        "maximum_inter_arm_penetration_m": round(max_inter_arm_penetration_m, 6),
        "maximum_inter_arm_penetration_during_hold_m": round(
            max_inter_arm_penetration_during_hold_m, 6
        ),
        "inter_arm_penetration_limit_m": INTER_ARM_PENETRATION_LIMIT_M,
        "inter_arm_contact_steps_by_phase": inter_arm_contact_steps_by_phase,
        "physics_steps_with_arm_table_or_ground_contact": arm_environment_steps,
        "physics_steps_with_arm_table_or_ground_contact_during_hold": (
            arm_environment_steps_during_hold
        ),
        "physics_steps_with_nonfinger_arm_payload_contact": nonfinger_payload_steps,
        "physics_steps_with_nonfinger_arm_payload_contact_during_hold": (
            nonfinger_payload_steps_during_hold
        ),
        "unexpected_contact_steps_by_phase": unexpected_contact_steps_by_phase,
        "unexpected_contact_pairs": {
            category: sorted(
                stats.values(),
                key=lambda item: (
                    item["phase"],
                    item["arm_geom"],
                    item["other_geom"],
                ),
            )
            for category, stats in unexpected_pair_stats.items()
        },
        "first_unexpected_contact": first_unexpected_contact,
        "stable_lift_hold_duration_s": round(HOLD_END_S - HOLD_START_S, 2),
        "videos": [video_path.name] if video_path is not None and target_detected else [],
    }
    if translation_only_visual_record is not None:
        metrics["visual_verification_translation_only_baseline"] = (
            translation_only_visual_record
        )
    if orientation_only_visual_record is not None:
        metrics["visual_verification_orientation_only_baseline"] = (
            orientation_only_visual_record
        )
    if rise_orientation_visual_record is not None:
        metrics["visual_verification_rise_orientation_baseline"] = (
            rise_orientation_visual_record
        )
    if visual_record is not None:
        capture_states = [
            verification_physics[time_s]
            for time_s in VERIFICATION_CAPTURE_TIMES_S
            if time_s in verification_physics
        ]
        heights = [
            item["payload_center_z_m"] - PAYLOAD_INITIAL_Z_M
            for item in capture_states
        ]
        max_capture_tilt = max(
            (item["payload_tilt_deg"] for item in capture_states), default=math.inf
        )
        raised_at_both_frames = len(heights) == 2 and min(heights) >= 0.07
        stable_between_frames = (
            len(heights) == 2 and abs(heights[1] - heights[0]) <= 0.015
        )
        visual_record["evaluation_only_physics"] = {
            "payload_center_z_m_at_capture": [
                round(item["payload_center_z_m"], 6) for item in capture_states
            ],
            "payload_lift_m_at_capture": [round(value, 6) for value in heights],
            "payload_tilt_deg_at_capture": [
                round(item["payload_tilt_deg"], 3) for item in capture_states
            ],
            "raised_at_both_frames_7cm": raised_at_both_frames,
            "stable_between_frames_15mm": stable_between_frames,
            "raised_and_stable_at_both_frames": (
                raised_at_both_frames and stable_between_frames and max_capture_tilt <= 5.0
            ),
            "full_hold_payload_stable_lift_success": payload_stable_lift,
            "full_sequence_collision_safe": full_sequence_collision_safe,
            "strict_physical_stable_lift_success": stable_lift,
            "available_to_visual_decision": False,
        }
        if translation_only_visual_record is not None:
            translation_only_visual_record["evaluation_only_physics"] = dict(
                visual_record["evaluation_only_physics"]
            )
        if orientation_only_visual_record is not None:
            orientation_only_visual_record["evaluation_only_physics"] = dict(
                visual_record["evaluation_only_physics"]
            )
        if rise_orientation_visual_record is not None:
            rise_orientation_visual_record["evaluation_only_physics"] = dict(
                visual_record["evaluation_only_physics"]
            )
        if orthogonal_visual_record is not None:
            orthogonal_visual_record["evaluation_only_physics"] = dict(
                visual_record["evaluation_only_physics"]
            )
        for candidate_record in gap_tolerant_visual_records.values():
            candidate_record["evaluation_only_physics"] = dict(
                visual_record["evaluation_only_physics"]
            )
        for fault_record in fault_candidate_visual_records.values():
            fault_record["evaluation_only_physics"] = dict(
                visual_record["evaluation_only_physics"]
            )
            for candidate_record in fault_record["gap_tolerant_candidates"].values():
                candidate_record["evaluation_only_physics"] = dict(
                    visual_record["evaluation_only_physics"]
                )
    metrics["failure_reason"] = _failure_reason(metrics)
    if writer is not None:
        if visual_record is not None:
            _write_visual_verification_endcard(
                writer,
                initial_verification_rgb,
                [
                    (time_s, verification_frames[time_s])
                    for time_s in VERIFICATION_CAPTURE_TIMES_S
                    if time_s in verification_frames
                ],
                visual_record,
                metrics["visual_verification"]["evaluation_only_physics"],
                preview_frames,
                translation_only_visual_record,
                orientation_only_visual_record,
                rise_orientation_visual_record,
                gap_tolerant_visual_records,
            )
            if orthogonal_visual_record is not None:
                _write_orthogonal_visual_verification_endcard(
                    writer,
                    initial_verification_rgb,
                    [
                        (time_s, verification_frames[time_s])
                        for time_s in VERIFICATION_CAPTURE_TIMES_S
                        if time_s in verification_frames
                    ],
                    initial_orthogonal_verification_rgb,
                    [
                        (time_s, orthogonal_verification_frames[time_s])
                        for time_s in VERIFICATION_CAPTURE_TIMES_S
                        if time_s in orthogonal_verification_frames
                    ],
                    orthogonal_visual_record,
                    metrics["visual_verification"]["evaluation_only_physics"],
                    preview_frames,
                )
        writer.release()
        if preview_frames:
            preview_frames[0].save(
                preview_path or video_path.with_name(GIF_NAME),
                save_all=True,
                append_images=preview_frames[1:],
                duration=round(GIF_PREVIEW_FRAME_STRIDE * 1000 / VIDEO_FPS),
                loop=0,
                optimize=True,
            )
    renderer.close()
    return metrics


def _sample_positions(
    *, seeds: int, samples_per_seed: int, base_seed: int
) -> list[tuple[int, int, tuple[float, float]]]:
    """Draw deterministic target positions from the prespecified XY square."""
    if seeds <= 0 or samples_per_seed <= 0:
        raise ValueError("seeds and samples_per_seed must both be positive")
    samples = []
    for seed_index in range(seeds):
        rng = np.random.default_rng(base_seed + seed_index)
        for sample_index in range(samples_per_seed):
            samples.append(
                (
                    base_seed + seed_index,
                    sample_index,
                    (
                        float(rng.uniform(-TARGET_HALF_SPAN_M, TARGET_HALF_SPAN_M)),
                        float(rng.uniform(-TARGET_HALF_SPAN_M, TARGET_HALF_SPAN_M)),
                    ),
                )
            )
    return samples


def _summary_for_condition(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = [row for row in rows if row["controller"] == condition]
    successes = sum(bool(row["physical_stable_lift_success"]) for row in selected)
    valid_errors = [
        row["initial_target_projection_error_m"]
        for row in selected
        if row["initial_target_projection_error_m"] is not None
    ]
    payload_hold_successes = sum(
        bool(row["payload_stable_lift_success"]) for row in selected
    )
    collision_safe_trials = sum(
        bool(row["full_sequence_collision_safe"]) for row in selected
    )
    return {
        "controller": condition,
        "trials": len(selected),
        "physical_stable_lift_successes": successes,
        "physical_stable_lift_success_rate": round(
            successes / len(selected) if selected else 0.0, 4
        ),
        "payload_stable_lift_successes": payload_hold_successes,
        "payload_stable_lift_success_rate": round(
            payload_hold_successes / len(selected) if selected else 0.0, 4
        ),
        "full_sequence_collision_safe_trials": collision_safe_trials,
        "full_sequence_collision_safe_rate": round(
            collision_safe_trials / len(selected) if selected else 0.0, 4
        ),
        "target_detection_rate": round(
            sum(bool(row["target_detected"]) for row in selected) / len(selected)
            if selected
            else 0.0,
            4,
        ),
        "median_initial_target_projection_error_cm": (
            round(float(np.median(valid_errors)) * 100, 3)
            if valid_errors
            else None
        ),
        "failure_reasons": {
            reason: sum(row["failure_reason"] == reason for row in selected)
            for reason in sorted({row["failure_reason"] for row in selected})
            if reason != "none"
        },
    }


def _unexpected_contact_attribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate unexpected contacts by controller, phase, and geom pair."""
    attribution: dict[str, Any] = {}
    for controller in ("camera_estimate", "oracle_xy"):
        selected = [row for row in rows if row["controller"] == controller]
        controller_summary: dict[str, Any] = {}
        for category in ("arm_environment", "nonfinger_payload"):
            trials_with_contact = 0
            phase_stats: dict[str, dict[str, int]] = {}
            pair_stats: dict[tuple[str, str, str], dict[str, Any]] = {}
            first_pair_counts: dict[tuple[str, str, str], int] = {}
            for row in selected:
                category_steps = row["unexpected_contact_steps_by_phase"][category]
                if sum(category_steps.values()) > 0:
                    trials_with_contact += 1
                for phase, steps in category_steps.items():
                    stats = phase_stats.setdefault(
                        phase, {"trials_with_contact": 0, "contact_steps": 0}
                    )
                    stats["trials_with_contact"] += int(steps > 0)
                    stats["contact_steps"] += steps
                for event in row["unexpected_contact_pairs"][category]:
                    key = (event["phase"], event["arm_geom"], event["other_geom"])
                    stats = pair_stats.setdefault(
                        key,
                        {
                            "phase": event["phase"],
                            "arm_geom": event["arm_geom"],
                            "other_geom": event["other_geom"],
                            "trials_with_contact": 0,
                            "contact_steps": 0,
                            "maximum_penetration_m": 0.0,
                        },
                    )
                    stats["trials_with_contact"] += 1
                    stats["contact_steps"] += event["contact_steps"]
                    stats["maximum_penetration_m"] = max(
                        stats["maximum_penetration_m"],
                        event["maximum_penetration_m"],
                    )
                first = row["first_unexpected_contact"][category]
                if first is not None:
                    key = (first["phase"], first["arm_geom"], first["other_geom"])
                    first_pair_counts[key] = first_pair_counts.get(key, 0) + 1

            controller_summary[category] = {
                "trials_with_contact": trials_with_contact,
                "phase_counts": [
                    {"phase": phase, **stats}
                    for phase, stats in sorted(phase_stats.items())
                ],
                "geometry_pairs": sorted(
                    pair_stats.values(),
                    key=lambda item: (
                        -item["trials_with_contact"],
                        -item["contact_steps"],
                        item["phase"],
                        item["arm_geom"],
                        item["other_geom"],
                    ),
                ),
                "first_contact_pairs": [
                    {
                        "phase": phase,
                        "arm_geom": arm_geom,
                        "other_geom": other_geom,
                        "trials": count,
                    }
                    for (phase, arm_geom, other_geom), count in sorted(
                        first_pair_counts.items(),
                        key=lambda item: (-item[1], *item[0]),
                    )
                ],
            }
        attribution[controller] = controller_summary
    return attribution


def _write_trials_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "pair_id",
        "seed",
        "sample_index",
        "target_x_m",
        "target_y_m",
        "controller",
        "release_profile",
        "wrist_center_radial_standoff_m",
        "lowest_approach_tool_height_offset_m",
        "finger_pad_inward_offset_m",
        "control_target_source",
        "target_detected",
        "detected_pixel_x_px",
        "detected_pixel_y_px",
        "estimated_target_x_m",
        "estimated_target_y_m",
        "initial_target_projection_error_m",
        "physical_stable_lift_success",
        "payload_stable_lift_success",
        "full_sequence_collision_safe",
        "fresh_visual_confirmation",
        "failure_reason",
        "peak_lift_m",
        "minimum_payload_z_during_hold_m",
        "minimum_finger_payload_contact_points_during_hold",
        "maximum_payload_tilt_during_hold_deg",
        "finger_pad_half_size_m",
        "physics_steps_with_inter_arm_contact",
        "maximum_inter_arm_penetration_m",
        "maximum_inter_arm_penetration_during_hold_m",
        "physics_steps_with_arm_table_or_ground_contact",
        "physics_steps_with_arm_table_or_ground_contact_during_hold",
        "physics_steps_with_nonfinger_arm_payload_contact",
        "physics_steps_with_nonfinger_arm_payload_contact_during_hold",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            pixel = row["detected_target_pixel_xy"] or [None, None]
            estimate = row["camera_estimated_target_xy_m"] or [None, None]
            writer.writerow(
                {
                    "pair_id": row["pair_id"],
                    "seed": row["seed"],
                    "sample_index": row["sample_index"],
                    "target_x_m": row["target_true_xy_m"][0],
                    "target_y_m": row["target_true_xy_m"][1],
                    "controller": row["controller"],
                    "release_profile": row["release_profile"],
                    "wrist_center_radial_standoff_m": row[
                        "wrist_center_radial_standoff_m"
                    ],
                    "lowest_approach_tool_height_offset_m": row[
                        "lowest_approach_tool_height_offset_m"
                    ],
                    "finger_pad_inward_offset_m": row[
                        "finger_pad_inward_offset_m"
                    ],
                    "control_target_source": row["control_target_source"],
                    "target_detected": row["target_detected"],
                    "detected_pixel_x_px": pixel[0],
                    "detected_pixel_y_px": pixel[1],
                    "estimated_target_x_m": estimate[0],
                    "estimated_target_y_m": estimate[1],
                    "initial_target_projection_error_m": row[
                        "initial_target_projection_error_m"
                    ],
                    "physical_stable_lift_success": row[
                        "physical_stable_lift_success"
                    ],
                    "payload_stable_lift_success": row[
                        "payload_stable_lift_success"
                    ],
                    "full_sequence_collision_safe": row[
                        "full_sequence_collision_safe"
                    ],
                    "fresh_visual_confirmation": row[
                        "fresh_visual_confirmation"
                    ],
                    "failure_reason": row["failure_reason"],
                    "peak_lift_m": row["peak_lift_m"],
                    "minimum_payload_z_during_hold_m": row[
                        "minimum_payload_z_during_hold_m"
                    ],
                    "minimum_finger_payload_contact_points_during_hold": row[
                        "minimum_finger_payload_contact_points_during_hold"
                    ],
                    "maximum_payload_tilt_during_hold_deg": row[
                        "maximum_payload_tilt_during_hold_deg"
                    ],
                    "finger_pad_half_size_m": row["finger_pad_half_size_m"],
                    "physics_steps_with_inter_arm_contact": row[
                        "physics_steps_with_inter_arm_contact"
                    ],
                    "maximum_inter_arm_penetration_m": row[
                        "maximum_inter_arm_penetration_m"
                    ],
                    "maximum_inter_arm_penetration_during_hold_m": row[
                        "maximum_inter_arm_penetration_during_hold_m"
                    ],
                    "physics_steps_with_arm_table_or_ground_contact": row[
                        "physics_steps_with_arm_table_or_ground_contact"
                    ],
                    "physics_steps_with_arm_table_or_ground_contact_during_hold": row[
                        "physics_steps_with_arm_table_or_ground_contact_during_hold"
                    ],
                    "physics_steps_with_nonfinger_arm_payload_contact": row[
                        "physics_steps_with_nonfinger_arm_payload_contact"
                    ],
                    "physics_steps_with_nonfinger_arm_payload_contact_during_hold": row[
                        "physics_steps_with_nonfinger_arm_payload_contact_during_hold"
                    ],
                }
            )


def _write_contact_events_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one row per trial, phase, and unexpected contact geometry pair."""
    fields = (
        "pair_id",
        "seed",
        "sample_index",
        "controller",
        "target_x_m",
        "target_y_m",
        "category",
        "phase",
        "arm_geom",
        "other_geom",
        "contact_steps",
        "maximum_penetration_m",
        "first_contact_time_s",
        "first_contact_x_m",
        "first_contact_y_m",
        "first_contact_z_m",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for category, events in row["unexpected_contact_pairs"].items():
                for event in events:
                    position = event["first_contact_position_m"]
                    writer.writerow(
                        {
                            "pair_id": row.get("pair_id", ""),
                            "seed": row.get("seed", ""),
                            "sample_index": row.get("sample_index", ""),
                            "controller": row["controller"],
                            "target_x_m": row["target_true_xy_m"][0],
                            "target_y_m": row["target_true_xy_m"][1],
                            "category": category,
                            "phase": event["phase"],
                            "arm_geom": event["arm_geom"],
                            "other_geom": event["other_geom"],
                            "contact_steps": event["contact_steps"],
                            "maximum_penetration_m": event[
                                "maximum_penetration_m"
                            ],
                            "first_contact_time_s": event[
                                "first_contact_time_s"
                            ],
                            "first_contact_x_m": position[0],
                            "first_contact_y_m": position[1],
                            "first_contact_z_m": position[2],
                        }
                    )


def _write_phase_outcomes_svg(
    path: Path,
    camera_summary: dict[str, Any],
    oracle_summary: dict[str, Any],
    count: int,
) -> None:
    """Compare payload hold quality with contact safety over the full motion."""
    width, height = 1040, 490
    chart_x, chart_width, bar_height = 280, 620, 34
    rows = (
        (
            "Full-sequence collision-safe",
            "full_sequence_collision_safe_trials",
            "#f3a849",
            160,
        ),
        (
            "Payload stable lift",
            "payload_stable_lift_successes",
            "#45c8a6",
            312,
        ),
    )
    summaries = (camera_summary, oracle_summary)
    fragments = [
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" rx="18" fill="#111923"/>
<text x="42" y="52" fill="#f2f6fa" font-family="sans-serif" font-size="27" font-weight="700">Separate the payload hold from full-cycle safety</text>
<text x="42" y="83" fill="#b6c4d3" font-family="sans-serif" font-size="15">Seed 38100 tuning set · 30 poses · 16 mm pad half-size · physical simulation outcomes</text>
<text x="{chart_x}" y="119" fill="#899bad" font-family="sans-serif" font-size="13">0</text>
<text x="{chart_x + chart_width - 28}" y="119" fill="#899bad" font-family="sans-serif" font-size="13">30 trials</text>'''
    ]
    for label, metric, color, section_y in rows:
        fragments.append(
            f'<text x="42" y="{section_y + 11}" fill="{color}" font-family="sans-serif" font-size="17" font-weight="700">{label}</text>'
        )
        for index, summary in enumerate(summaries):
            y = section_y + 28 + index * 42
            value = int(summary[metric])
            bar_width = chart_width * value / count if count else 0
            name = "Camera estimate" if index == 0 else "Oracle XY diagnostic"
            fragments.extend(
                [
                    f'<text x="42" y="{y + 23}" fill="#eaf0f5" font-family="sans-serif" font-size="15">{name}</text>',
                    f'<rect x="{chart_x}" y="{y}" width="{chart_width}" height="{bar_height}" rx="7" fill="#253343"/>',
                    f'<rect x="{chart_x}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" rx="7" fill="{color}"/>',
                    f'<text x="{chart_x + chart_width + 16}" y="{y + 23}" fill="#f2f6fa" font-family="sans-serif" font-size="16" font-weight="700">{value}/{count}</text>',
                ]
            )
    fragments.append(
        '<text x="42" y="462" fill="#b6c4d3" font-family="sans-serif" font-size="14">The original strict score requires both outcomes. Neither measure is fresh visual confirmation.</text></svg>'
    )
    path.write_text("\n".join(fragments), encoding="utf-8")


def _write_scorecard(path: Path, camera_summary: dict[str, Any], oracle_summary: dict[str, Any], count: int) -> None:
    width, height = 960, 500
    chart_x, chart_y, chart_width, chart_height = 130, 150, 690, 200
    camera_rate = camera_summary["physical_stable_lift_success_rate"]
    oracle_rate = oracle_summary["physical_stable_lift_success_rate"]
    camera_width = round(chart_width * camera_rate)
    oracle_width = round(chart_width * oracle_rate)
    camera_error = camera_summary["median_initial_target_projection_error_cm"]
    camera_error_label = f"{camera_error:.2f} cm" if camera_error is not None else "unavailable"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#111923"/>
<text x="54" y="56" fill="#f2f6fa" font-family="sans-serif" font-size="27" font-weight="700">Camera-guided cooperative pick</text>
<text x="54" y="91" fill="#b6c4d3" font-family="sans-serif" font-size="16">Paired randomized target poses · {count} locations · three collidable arms · free payload</text>
<text x="54" y="132" fill="#dce6ef" font-family="sans-serif" font-size="17" font-weight="700">Physical stable-lift success rate</text>
<text x="54" y="{chart_y + 28}" fill="#eaf0f5" font-family="sans-serif" font-size="16">Camera estimate</text>
<rect x="{chart_x}" y="{chart_y}" width="{chart_width}" height="50" rx="8" fill="#253343"/>
<rect x="{chart_x}" y="{chart_y}" width="{camera_width}" height="50" rx="8" fill="#49c9ab"/>
<text x="{chart_x + chart_width + 20}" y="{chart_y + 31}" fill="#f2f6fa" font-family="sans-serif" font-size="18" font-weight="700">{camera_summary['physical_stable_lift_successes']}/{count}</text>
<text x="54" y="{chart_y + 103}" fill="#eaf0f5" font-family="sans-serif" font-size="16">Oracle XY diagnostic</text>
<rect x="{chart_x}" y="{chart_y + 75}" width="{chart_width}" height="50" rx="8" fill="#253343"/>
<rect x="{chart_x}" y="{chart_y + 75}" width="{oracle_width}" height="50" rx="8" fill="#63a9f5"/>
<text x="{chart_x + chart_width + 20}" y="{chart_y + 106}" fill="#f2f6fa" font-family="sans-serif" font-size="18" font-weight="700">{oracle_summary['physical_stable_lift_successes']}/{count}</text>
<text x="54" y="405" fill="#b6c4d3" font-family="sans-serif" font-size="16">Median initial camera-to-world position error: {camera_error_label}</text>
<text x="54" y="435" fill="#b6c4d3" font-family="sans-serif" font-size="15">Position estimate uses the existing HSV red detector and pinhole projection. Oracle XY is a diagnostic baseline.</text>
<text x="54" y="464" fill="#f4c976" font-family="sans-serif" font-size="15">Stable lift is a physics outcome; fresh visual confirmation was not measured.</text>
<text x="{chart_x}" y="{chart_y + chart_height + 20}" fill="#8e9cab" font-family="sans-serif" font-size="12">0</text>
<text x="{chart_x + chart_width - 15}" y="{chart_y + chart_height + 20}" fill="#8e9cab" font-family="sans-serif" font-size="12">100%</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def run_validation(
    output_dir: Path,
    *,
    seeds: int = DEFAULT_SEEDS,
    samples_per_seed: int = DEFAULT_SAMPLES_PER_SEED,
    base_seed: int = BASE_SEED,
    media_dir: Path | None = None,
    finger_pad_half_size_m: float = FINGER_PAD_HALF_SIZE_M,
    release_profile: str = "baseline",
    wrist_center_radial_standoff_m: float = 0.0,
    lowest_approach_tool_height_offset_m: float = APPROACH_TOOL_Z_OFFSETS_M[0],
) -> dict[str, Any]:
    """Run paired randomized camera and oracle trials, then write linked evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if media_dir is not None:
        media_dir.mkdir(parents=True, exist_ok=True)
    positions = _sample_positions(
        seeds=seeds,
        samples_per_seed=samples_per_seed,
        base_seed=base_seed,
    )
    rows = []
    demo_record = None
    demo_oracle_record = None
    for seed, sample_index, target_xy in positions:
        pair_id = f"{seed}-{sample_index:03d}"
        for condition in ("camera_estimate", "oracle_xy"):
            metrics = run_trial(
                target_xy,
                controller=condition,
                finger_pad_half_size_m=finger_pad_half_size_m,
                release_profile=release_profile,
                wrist_center_radial_standoff_m=wrist_center_radial_standoff_m,
                lowest_approach_tool_height_offset_m=lowest_approach_tool_height_offset_m,
            )
            rows.append(
                {
                    "pair_id": pair_id,
                    "seed": seed,
                    "sample_index": sample_index,
                    **metrics,
                }
            )

    if media_dir is not None:
        demo_record = run_trial(
            DEMO_TARGET_XY,
            controller="camera_estimate",
            video_path=media_dir / VIDEO_NAME,
            finger_pad_half_size_m=finger_pad_half_size_m,
            release_profile=release_profile,
            wrist_center_radial_standoff_m=wrist_center_radial_standoff_m,
            lowest_approach_tool_height_offset_m=lowest_approach_tool_height_offset_m,
        )
        demo_oracle_record = run_trial(
            DEMO_TARGET_XY,
            controller="oracle_xy",
            finger_pad_half_size_m=finger_pad_half_size_m,
            release_profile=release_profile,
            wrist_center_radial_standoff_m=wrist_center_radial_standoff_m,
            lowest_approach_tool_height_offset_m=lowest_approach_tool_height_offset_m,
        )

    camera_summary = _summary_for_condition(rows, "camera_estimate")
    oracle_summary = _summary_for_condition(rows, "oracle_xy")
    paired_success_differences = {
        "camera_success_oracle_failure": 0,
        "camera_failure_oracle_success": 0,
        "both_success": 0,
        "both_failure": 0,
    }
    by_pair: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], {})[row["controller"]] = row
    for pair in by_pair.values():
        camera_success = bool(pair["camera_estimate"]["physical_stable_lift_success"])
        oracle_success = bool(pair["oracle_xy"]["physical_stable_lift_success"])
        if camera_success and oracle_success:
            paired_success_differences["both_success"] += 1
        elif not camera_success and not oracle_success:
            paired_success_differences["both_failure"] += 1
        elif camera_success:
            paired_success_differences["camera_success_oracle_failure"] += 1
        else:
            paired_success_differences["camera_failure_oracle_success"] += 1

    summary: dict[str, Any] = {
        "experiment": "camera-guided cooperative pick pose validation",
        "evidence_type": "paired randomized MuJoCo simulation; not hardware evidence",
        "target_sampling": {
            "distribution": "independent uniform XY samples in a square",
            "x_range_m": [-TARGET_HALF_SPAN_M, TARGET_HALF_SPAN_M],
            "y_range_m": [-TARGET_HALF_SPAN_M, TARGET_HALF_SPAN_M],
            "seeds": seeds,
            "samples_per_seed": samples_per_seed,
            "base_seed": base_seed,
            "unique_target_positions": len(positions),
        },
        "camera_pipeline": {
            "detector": "unchanged default red HSV contour detector from run.py",
            "projection": "shared pixel_to_table_xy pinhole helper from controller.py",
            "camera_height_m": CAMERA_HEIGHT_M,
            "target_marker_height_m": TARGET_MARKER_HEIGHT_M,
            "vertical_fov_deg": CAMERA_FOVY_DEG,
            "image_size_px": [CAMERA_WIDTH, CAMERA_HEIGHT_PX],
            "recalibration_or_distortion": "none; ideal fixed top-down calibration",
        },
        "gripper_pad_half_size_m": finger_pad_half_size_m,
        "release_profile": release_profile,
        "wrist_center_radial_standoff_m": wrist_center_radial_standoff_m,
        "lowest_approach_tool_height_offset_m": lowest_approach_tool_height_offset_m,
        "finger_pad_inward_offset_m": (
            FINGER_PAD_INWARD_OFFSET_M + wrist_center_radial_standoff_m
        ),
        "yaw_actuators": {
            "base_yaw": {
                "kp": BASE_YAW_ACTUATOR_KP,
                "kv": BASE_YAW_ACTUATOR_KV,
                "absolute_force_limit_nm": BASE_YAW_ACTUATOR_FORCE_NM,
            },
            "gripper_yaw": {
                "kp": GRIPPER_YAW_ACTUATOR_KP,
                "kv": GRIPPER_YAW_ACTUATOR_KV,
                "absolute_force_limit_nm": GRIPPER_YAW_ACTUATOR_FORCE_NM,
            },
        },
        "task_success_definition": {
            "physical_stable_lift_success": (
                "payload_stable_lift_success plus full_sequence_collision_safe; "
                "retained for direct comparison with the original strict score"
            ),
            "payload_stable_lift_success": (
                "during the 1.6 s hold, payload stays at least 7 cm above start, "
                "has at least three finger-pad contacts, stays within 5 degrees tilt, "
                "respects the 1 mm inter-arm penetration limit, and has no unexpected arm contacts"
            ),
            "full_sequence_collision_safe": (
                "throughout the complete pick, lower, release, and withdraw sequence, "
                "there is no arm-table/ground or nonfinger arm-payload contact and "
                "inter-arm penetration remains within 1 mm"
            ),
            "fresh_visual_confirmation": "not measured by this experiment",
        },
        "conditions": [camera_summary, oracle_summary],
        "paired_physical_success_outcomes": paired_success_differences,
        "contact_attribution": _unexpected_contact_attribution(rows),
        "illustrative_demo_target_xy_m": (
            [round(float(DEMO_TARGET_XY[0]), 5), round(float(DEMO_TARGET_XY[1]), 5)]
            if media_dir is not None
            else None
        ),
        "illustrative_demo_selection": (
            "Development-selected successful camera-guided run; excluded from randomized aggregate statistics."
            if media_dir is not None
            else None
        ),
        "demo_metrics": demo_record,
        "demo_oracle_diagnostic_metrics": demo_oracle_record,
        "limitations": [
            "The controller estimates a static target pose once at planning time; it does not servo on later images.",
            "The camera uses ideal intrinsics and no lens distortion, noise, lighting shift, clutter, or occlusion.",
            "The default detector is used; the opt-in circle-prior detector remains experimental and is not promoted.",
            "The oracle condition isolates position-estimation effects and is not a deployable controller.",
            "Physical stable-lift success is not fresh visual confirmation or real-robot performance.",
            "Same-arm self-collision remains suppressed by the shared scene masks.",
        ],
        "trial_data": CSV_NAME,
        "contact_events": CONTACT_EVENTS_CSV_NAME,
        "scorecard": SVG_NAME,
        "phase_outcomes_chart": PHASE_OUTCOMES_SVG_NAME,
        "video": VIDEO_NAME if media_dir is not None else None,
        "preview": GIF_NAME if media_dir is not None else None,
    }
    _write_trials_csv(output_dir / CSV_NAME, rows)
    _write_contact_events_csv(output_dir / CONTACT_EVENTS_CSV_NAME, rows)
    (output_dir / JSON_NAME).write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_scorecard(output_dir / SVG_NAME, camera_summary, oracle_summary, len(positions))
    _write_phase_outcomes_svg(
        output_dir / PHASE_OUTCOMES_SVG_NAME,
        camera_summary,
        oracle_summary,
        len(positions),
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--samples-per-seed", type=int, default=DEFAULT_SAMPLES_PER_SEED)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument(
        "--finger-pad-half-size-m",
        type=float,
        default=FINGER_PAD_HALF_SIZE_M,
    )
    parser.add_argument(
        "--release-profile",
        choices=("baseline", "clearance"),
        default="baseline",
        help="select the historical or experimental release/withdraw trajectory",
    )
    parser.add_argument(
        "--wrist-center-radial-standoff-m",
        type=float,
        default=0.0,
        help=(
            "optional whole-trajectory wrist setback with a matching inward finger-pad "
            "extension, capped at the finger-pad half-size"
        ),
    )
    parser.add_argument(
        "--lowest-approach-tool-height-offset-m",
        type=float,
        default=APPROACH_TOOL_Z_OFFSETS_M[0],
        help="minimum Z offset of the first open gripper during its staggered approach",
    )
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--media-dir", type=Path)
    parser.add_argument(
        "--no-media",
        action="store_true",
        help="skip the illustrative MP4 and GIF; useful for parameter sweeps",
    )
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    results_dir = args.results_dir or repository / "results"
    media_dir = None if args.no_media else (args.media_dir or repository / "assets" / "demos")
    summary = run_validation(
        results_dir,
        seeds=args.seeds,
        samples_per_seed=args.samples_per_seed,
        base_seed=args.base_seed,
        media_dir=media_dir,
        finger_pad_half_size_m=args.finger_pad_half_size_m,
        release_profile=args.release_profile,
        wrist_center_radial_standoff_m=args.wrist_center_radial_standoff_m,
        lowest_approach_tool_height_offset_m=args.lowest_approach_tool_height_offset_m,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
