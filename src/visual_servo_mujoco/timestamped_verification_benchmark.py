"""Benchmark delayed, timestamp-aware visual task completion after reaching."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .model import SIMULATION_TIMESTEP_S
from .run import (
    END_EFFECTOR_HEIGHT,
    FEEDBACK_STEPS_PER_UPDATE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SUCCESS_THRESHOLD_METERS,
    TARGET_HEIGHT,
    _detect_marker_with_quality,
    _step_with_target_motion,
    run_trials,
)
from .terminal_verification import (
    model_with_verification_views,
    pixel_to_xy_at_height,
)

MOTION_AMPLITUDE_M = 0.04
MOTION_FREQUENCY_HZ = 0.25
MOTION_PROFILES = ("sinusoidal", "piecewise_linear")
MOTION_PHASES_RAD = (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)
CONTROL_DELAYS = (1, 2)
CAMERA_FRAME_PERIOD_S = 0.2
SETTLE_SPEED_THRESHOLD_RAD_S = 0.02
SETTLE_DWELL_S = 0.1
MAX_SETTLE_S = 0.8
MAX_CONFIRMATION_WAIT_S = 2.0
VERIFICATION_PIXEL_NOISE_STD_PX = 2.0
AGE_BUDGETS_S = (0.5, 1.0)
BOOTSTRAP_SEED = 20261018


@dataclass(frozen=True)
class DeliveryProfile:
    """Transport behavior for captured frames; jitter is uniform and seeded."""

    name: str
    latency_s: float
    jitter_s: float
    frame_drop_probability: float


DELIVERY_PROFILES = (
    DeliveryProfile("near_real_time", 0.05, 0.02, 0.05),
    DeliveryProfile("moderate_latency", 0.40, 0.10, 0.10),
    DeliveryProfile("high_latency", 0.80, 0.10, 0.10),
)

CONTROL_MODES = (
    ("fresh_rgb", 0, False),
    ("fully_delayed_rgb", 1, False),
    ("fully_delayed_rgb", 2, False),
    ("delayed_target_current_joint_tip", 1, True),
    ("delayed_target_current_joint_tip", 2, True),
)

CSV_FIELDS = (
    "seed",
    "episode",
    "target_motion_profile",
    "target_motion_phase_rad",
    "control_mode",
    "control_delay_observations",
    "control_delay_s",
    "physical_success_at_controller_stop",
    "reaching_error_at_controller_stop_m",
    "delivery_profile",
    "policy",
    "max_frame_age_s",
    "status",
    "visual_confirmed",
    "physical_success_at_decision_delivery",
    "false_confirmation_at_delivery",
    "missed_controller_success",
    "estimated_distance_at_decision_m",
    "physical_success_at_decision_capture",
    "physical_error_at_decision_capture_m",
    "physical_error_at_decision_delivery_m",
    "decision_latency_after_stop_s",
    "usable_frames",
    "dropped_frames",
    "pre_settle_rejections",
    "stale_rejections",
    "out_of_order_rejections",
)
SUMMARY_CSV_FIELDS = (
    "target_motion_profile",
    "target_motion_phases_rad",
    "control_mode",
    "control_delay_observations",
    "control_delay_s",
    "delivery_profile",
    "policy",
    "max_frame_age_s",
    "trials",
    "controller_stop_physical_successes",
    "controller_stop_physical_success_rate",
    "visually_confirmed",
    "visual_confirmation_rate",
    "true_confirmations_at_delivery",
    "false_confirmations_at_delivery",
    "delivery_mismatches_from_motion_after_capture",
    "delivery_mismatches_already_out_at_capture",
    "visual_confirmation_precision_at_delivery",
    "missed_controller_successes",
    "not_confirmed",
    "unverified",
    "usable_frames_total",
    "dropped_frames_total",
    "pre_settle_rejections_total",
    "stale_rejections_total",
    "out_of_order_rejections_total",
    "confirmation_latency_median_s",
    "confirmation_latency_p95_s",
)


def _motion_fraction(
    time_s: float,
    profile: str,
    phase_rad: float,
    frequency_hz: float = MOTION_FREQUENCY_HZ,
) -> float:
    phase = 2.0 * math.pi * frequency_hz * time_s + phase_rad
    if profile == "sinusoidal":
        return math.sin(phase)
    if profile == "piecewise_linear":
        return 2.0 / math.pi * math.asin(math.sin(phase))
    raise ValueError("target_motion_profile must be sinusoidal or piecewise_linear")


def _physical_success(error_m: float | None) -> bool | None:
    return None if error_m is None else error_m <= SUCCESS_THRESHOLD_METERS


def _outcome(
    *,
    status: str,
    event: dict | None,
    trial: dict,
    counts: dict,
) -> dict:
    confirmed = status == "visually_confirmed"
    physical_at_delivery = (
        _physical_success(event.get("physical_error_at_delivery_m"))
        if event is not None
        else None
    )
    estimated_distance = event.get("estimated_distance_m") if event else None
    return {
        "status": status,
        "visual_confirmed": confirmed,
        "physical_success_at_controller_stop": bool(trial["success"]),
        "reaching_error_at_controller_stop_m": float(trial["reaching_error_m"]),
        "physical_success_at_decision_delivery": physical_at_delivery,
        "false_confirmation_at_delivery": bool(
            confirmed and physical_at_delivery is False
        ),
        "missed_controller_success": bool(trial["success"] and not confirmed),
        "estimated_distance_at_decision_m": estimated_distance,
        "physical_success_at_decision_capture": (
            _physical_success(event.get("physical_error_at_capture_m"))
            if event is not None
            else None
        ),
        "physical_error_at_decision_capture_m": (
            event.get("physical_error_at_capture_m") if event else None
        ),
        "physical_error_at_decision_delivery_m": (
            event.get("physical_error_at_delivery_m") if event else None
        ),
        "decision_latency_after_stop_s": (
            round(
                float(event["delivery_time_s"])
                - float(trial["controller_stop_time_simulated_s"]),
                4,
            )
            if event is not None
            else None
        ),
        **counts,
    }


def evaluate_arrival_only_policy(trial: dict, events: list[dict]) -> dict:
    """A deliberately naive comparison: use the first delivered visible frame."""
    stop_time = float(trial["controller_stop_time_simulated_s"])
    deadline = stop_time + MAX_CONFIRMATION_WAIT_S
    delivered = sorted(
        (
            event
            for event in events
            if not event["transport_dropped"]
            and event.get("delivery_time_s") is not None
            and float(event["delivery_time_s"]) <= deadline + 1e-9
        ),
        key=lambda event: (float(event["delivery_time_s"]), event["frame_index"]),
    )
    dropped_count = sum(
        event["transport_dropped"]
        and float(event["scheduled_arrival_time_s"]) <= deadline + 1e-9
        for event in events
    )
    for event in delivered:
        if event["both_markers_visible"] and event["estimated_distance_m"] is not None:
            status = (
                "visually_confirmed"
                if event["visual_success_estimate"]
                else "not_confirmed"
            )
            return _outcome(
                status=status,
                event=event,
                trial=trial,
                counts={
                    "usable_frames": 1,
                    "dropped_frames": dropped_count,
                    "pre_settle_rejections": 0,
                    "stale_rejections": 0,
                    "out_of_order_rejections": 0,
                },
            )
    return _outcome(
        status="unverified",
        event=None,
        trial=trial,
        counts={
            "usable_frames": 0,
            "dropped_frames": dropped_count,
            "pre_settle_rejections": 0,
            "stale_rejections": 0,
            "out_of_order_rejections": 0,
        },
    )


def evaluate_timestamp_guarded_policy(
    trial: dict,
    events: list[dict],
    *,
    settled_time_s: float | None,
    max_frame_age_s: float,
) -> dict:
    """Confirm only from two consecutive, post-settle frames inside an age bound."""
    if not math.isfinite(max_frame_age_s) or max_frame_age_s < 0.0:
        raise ValueError("max_frame_age_s must be finite and non-negative")
    stop_time = float(trial["controller_stop_time_simulated_s"])
    deadline = stop_time + MAX_CONFIRMATION_WAIT_S
    delivered = sorted(
        (
            event
            for event in events
            if not event["transport_dropped"]
            and event.get("delivery_time_s") is not None
            and float(event["delivery_time_s"]) <= deadline + 1e-9
        ),
        key=lambda event: (float(event["delivery_time_s"]), event["frame_index"]),
    )
    dropped_count = sum(
        event["transport_dropped"]
        and float(event["scheduled_arrival_time_s"]) <= deadline + 1e-9
        for event in events
    )
    pre_settle_rejections = 0
    stale_rejections = 0
    out_of_order_rejections = 0
    usable_frames = 0
    positive_run: list[dict] = []
    newest_capture_index = -1
    last_usable_event: dict | None = None

    for event in delivered:
        frame_index = int(event["frame_index"])
        if frame_index <= newest_capture_index:
            out_of_order_rejections += 1
            positive_run.clear()
            continue
        newest_capture_index = frame_index
        capture_time = float(event["capture_time_s"])
        if settled_time_s is None or capture_time + 1e-9 < settled_time_s:
            pre_settle_rejections += 1
            positive_run.clear()
            continue
        age_s = float(event["camera_age_at_delivery_s"])
        if age_s > max_frame_age_s + 1e-9:
            stale_rejections += 1
            positive_run.clear()
            continue
        if not event["both_markers_visible"] or event["estimated_distance_m"] is None:
            positive_run.clear()
            continue
        usable_frames += 1
        last_usable_event = event
        if not event["visual_success_estimate"]:
            positive_run.clear()
            continue
        if positive_run and frame_index == int(positive_run[-1]["frame_index"]) + 1:
            positive_run.append(event)
        else:
            positive_run = [event]
        if len(positive_run) == 2:
            return _outcome(
                status="visually_confirmed",
                event=event,
                trial=trial,
                counts={
                    "usable_frames": usable_frames,
                    "dropped_frames": dropped_count,
                    "pre_settle_rejections": pre_settle_rejections,
                    "stale_rejections": stale_rejections,
                    "out_of_order_rejections": out_of_order_rejections,
                },
            )

    status = "not_confirmed" if usable_frames else "unverified"
    return _outcome(
        status=status,
        event=last_usable_event,
        trial=trial,
        counts={
            "usable_frames": usable_frames,
            "dropped_frames": dropped_count,
            "pre_settle_rejections": pre_settle_rejections,
            "stale_rejections": stale_rejections,
            "out_of_order_rejections": out_of_order_rejections,
        },
    )


class PostStopCameraReplay:
    """Replay a held robot pose while an asynchronous verification camera streams."""

    def __init__(self, *, save_images: bool = False) -> None:
        self.model = model_with_verification_views(("east",))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(
            self.model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH
        )
        self.target_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self.target_mocap_id = int(self.model.body_mocapid[self.target_id])
        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector"
        )
        self.shoulder_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_joint"
        )
        self.elbow_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "elbow_joint"
        )
        self.shoulder_qpos = int(self.model.jnt_qposadr[self.shoulder_id])
        self.elbow_qpos = int(self.model.jnt_qposadr[self.elbow_id])
        self.shoulder_dof = int(self.model.jnt_dofadr[self.shoulder_id])
        self.elbow_dof = int(self.model.jnt_dofadr[self.elbow_id])
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "verification_east"
        )
        self.save_images = save_images

    def close(self) -> None:
        self.renderer.close()

    def _capture(
        self,
        *,
        frame_index: int,
        measurement_rng: np.random.Generator,
    ) -> tuple[dict, np.ndarray | None]:
        self.renderer.update_scene(self.data, camera="verification_east")
        rgb = self.renderer.render()
        target_pixel = tip_pixel = None
        target_quality = tip_quality = None
        try:
            target_pixel, target_quality = _detect_marker_with_quality(
                rgb, marker="target"
            )
        except RuntimeError:
            pass
        try:
            tip_pixel, tip_quality = _detect_marker_with_quality(
                rgb, marker="end_effector"
            )
        except RuntimeError:
            pass

        estimated_distance = None
        quality_score = None
        noisy_target_pixel = noisy_tip_pixel = None
        pixel_noise_distance_sigma_m = None
        quality_adjusted_pixel_noise_scale_m = None
        if target_pixel is not None and tip_pixel is not None:
            target_noise = measurement_rng.normal(
                0.0, VERIFICATION_PIXEL_NOISE_STD_PX, size=2
            )
            tip_noise = measurement_rng.normal(
                0.0, VERIFICATION_PIXEL_NOISE_STD_PX, size=2
            )
            noisy_target_pixel = tuple(
                float(value + noise)
                for value, noise in zip(target_pixel, target_noise, strict=True)
            )
            noisy_tip_pixel = tuple(
                float(value + noise)
                for value, noise in zip(tip_pixel, tip_noise, strict=True)
            )
            try:
                target_xy = pixel_to_xy_at_height(
                    self.model,
                    self.data,
                    self.camera_id,
                    noisy_target_pixel,
                    TARGET_HEIGHT,
                )
                tip_xy = pixel_to_xy_at_height(
                    self.model,
                    self.data,
                    self.camera_id,
                    noisy_tip_pixel,
                    END_EFFECTOR_HEIGHT,
                )
            except ValueError:
                estimated_distance = None
            else:
                estimated_distance = float(np.linalg.norm(target_xy - tip_xy))
                quality_score = min(
                    target_quality["score"], tip_quality["score"]
                )

                # Propagate the configured pixel noise through the local
                # calibrated image-to-world map. This is an image/calibration
                # feature; no simulator ground-truth coordinates enter it.
                def distance_for_pixels(
                    target_xy_px: tuple[float, float],
                    tip_xy_px: tuple[float, float],
                ) -> float:
                    projected_target = pixel_to_xy_at_height(
                        self.model,
                        self.data,
                        self.camera_id,
                        target_xy_px,
                        TARGET_HEIGHT,
                    )
                    projected_tip = pixel_to_xy_at_height(
                        self.model,
                        self.data,
                        self.camera_id,
                        tip_xy_px,
                        END_EFFECTOR_HEIGHT,
                    )
                    return float(np.linalg.norm(projected_target - projected_tip))

                pixel_gradients = []
                epsilon_px = 1.0
                for marker_index in range(2):
                    for axis in range(2):
                        target_plus = list(noisy_target_pixel)
                        target_minus = list(noisy_target_pixel)
                        tip_plus = list(noisy_tip_pixel)
                        tip_minus = list(noisy_tip_pixel)
                        (target_plus if marker_index == 0 else tip_plus)[axis] += epsilon_px
                        (target_minus if marker_index == 0 else tip_minus)[axis] -= epsilon_px
                        pixel_gradients.append(
                            (
                                distance_for_pixels(tuple(target_plus), tuple(tip_plus))
                                - distance_for_pixels(tuple(target_minus), tuple(tip_minus))
                            )
                            / (2.0 * epsilon_px)
                        )
                pixel_noise_distance_sigma_m = (
                    VERIFICATION_PIXEL_NOISE_STD_PX
                    * float(np.linalg.norm(pixel_gradients))
                )
                quality_adjusted_pixel_noise_scale_m = (
                    pixel_noise_distance_sigma_m / max(float(quality_score), 0.25)
                )
        target_xy_true = self.data.xpos[self.target_id][:2]
        tip_xy_true = self.data.site_xpos[self.site_id][:2]
        event = {
            "frame_index": frame_index,
            "capture_time_s": round(float(self.data.time), 6),
            "target_visible": target_pixel is not None,
            "end_effector_visible": tip_pixel is not None,
            "both_markers_visible": target_pixel is not None and tip_pixel is not None,
            "target_detection_quality": target_quality,
            "end_effector_detection_quality": tip_quality,
            "combined_detection_quality_score": quality_score,
            "target_pixel_xy": (
                [round(float(value), 3) for value in noisy_target_pixel]
                if noisy_target_pixel is not None
                else None
            ),
            "end_effector_pixel_xy": (
                [round(float(value), 3) for value in noisy_tip_pixel]
                if noisy_tip_pixel is not None
                else None
            ),
            "pixel_noise_distance_sigma_m": (
                round(pixel_noise_distance_sigma_m, 8)
                if pixel_noise_distance_sigma_m is not None
                else None
            ),
            "quality_adjusted_pixel_noise_scale_m": (
                round(quality_adjusted_pixel_noise_scale_m, 8)
                if quality_adjusted_pixel_noise_scale_m is not None
                else None
            ),
            "estimated_distance_m": (
                round(estimated_distance, 6)
                if estimated_distance is not None
                else None
            ),
            "visual_success_estimate": (
                estimated_distance <= SUCCESS_THRESHOLD_METERS
                if estimated_distance is not None
                else None
            ),
            "physical_error_at_capture_m": round(
                float(np.linalg.norm(target_xy_true - tip_xy_true)), 6
            ),
        }
        return event, rgb.copy() if self.save_images else None

    def replay(
        self,
        trial: dict,
        *,
        delivery_profiles: tuple[DeliveryProfile, ...] = DELIVERY_PROFILES,
        seed: int,
    ) -> dict:
        if "final_joint_velocities_rad_s_for_evaluation_only" not in trial:
            raise ValueError("trial is missing final joint velocity replay state")
        stop_time = float(trial["controller_stop_time_simulated_s"])
        base_xy = tuple(
            float(value)
            for value in trial["target_start_xy_m_for_evaluation_only"]
        )
        amplitude = float(trial["target_motion_amplitude_m"])
        frequency = float(trial["target_motion_frequency_hz"])
        path_profile = str(trial["target_motion_profile"])
        phase = float(trial.get("target_motion_phase_rad", 0.0))
        joint_angles = trial["final_joint_angles_rad_for_evaluation_only"]
        joint_velocities = trial["final_joint_velocities_rad_s_for_evaluation_only"]

        mujoco.mj_resetData(self.model, self.data)
        self.data.time = stop_time
        self.data.qpos[self.shoulder_qpos] = float(joint_angles[0])
        self.data.qpos[self.elbow_qpos] = float(joint_angles[1])
        self.data.qvel[self.shoulder_dof] = float(joint_velocities[0])
        self.data.qvel[self.elbow_dof] = float(joint_velocities[1])
        # Hold the measured stop pose while the arm damps to rest.
        self.data.ctrl[0] = float(joint_angles[0])
        self.data.ctrl[1] = float(joint_angles[1])
        self.data.mocap_pos[self.target_mocap_id] = (
            base_xy[0]
            + amplitude
            * _motion_fraction(stop_time, path_profile, phase, frequency),
            base_xy[1],
            TARGET_HEIGHT,
        )
        mujoco.mj_forward(self.model, self.data)

        stream_rngs = {
            profile.name: np.random.default_rng(
                np.random.SeedSequence([seed, int(trial["episode"]), index, 0xCAFE])
            )
            for index, profile in enumerate(delivery_profiles)
        }
        measurement_rng = np.random.default_rng(
            np.random.SeedSequence([seed, int(trial["episode"]), 0xBEEF])
        )
        pending: dict[str, list[dict]] = {
            profile.name: [] for profile in delivery_profiles
        }
        events_by_profile: dict[str, list[dict]] = {
            profile.name: [] for profile in delivery_profiles
        }
        images: dict[int, np.ndarray] = {}
        next_capture_time = stop_time
        capture_index = 0
        settled_time_s: float | None = None
        settle_steps = 0
        timestep = float(self.model.opt.timestep)
        settle_dwell_steps = max(1, math.ceil(SETTLE_DWELL_S / timestep))
        settle_deadline = stop_time + MAX_SETTLE_S
        max_latency = max(
            profile.latency_s + profile.jitter_s for profile in delivery_profiles
        )
        simulation_end = stop_time + MAX_CONFIRMATION_WAIT_S + max_latency + timestep

        def capture_and_schedule() -> None:
            nonlocal capture_index
            base_event, image = self._capture(
                frame_index=capture_index,
                measurement_rng=measurement_rng,
            )
            if image is not None:
                images[capture_index] = image
            for profile in delivery_profiles:
                rng = stream_rngs[profile.name]
                dropped = bool(rng.random() < profile.frame_drop_probability)
                latency = max(
                    0.0,
                    profile.latency_s
                    + float(rng.uniform(-profile.jitter_s, profile.jitter_s)),
                )
                event = {
                    **base_event,
                    "delivery_profile": profile.name,
                    "transport_dropped": dropped,
                    "scheduled_latency_s": round(latency, 6),
                    "scheduled_arrival_time_s": round(
                        float(base_event["capture_time_s"]) + latency, 6
                    ),
                    "delivery_time_s": None,
                    "camera_age_at_delivery_s": None,
                    "physical_error_at_delivery_m": None,
                }
                if dropped:
                    events_by_profile[profile.name].append(event)
                else:
                    pending[profile.name].append(event)
            capture_index += 1

        def deliver_due_frames() -> None:
            for profile in delivery_profiles:
                still_pending = []
                for event in pending[profile.name]:
                    if event["scheduled_arrival_time_s"] <= self.data.time + 1e-9:
                        event["delivery_time_s"] = round(float(self.data.time), 6)
                        event["camera_age_at_delivery_s"] = round(
                            float(self.data.time) - float(event["capture_time_s"]), 6
                        )
                        target_xy = self.data.xpos[self.target_id][:2]
                        tip_xy = self.data.site_xpos[self.site_id][:2]
                        event["physical_error_at_delivery_m"] = round(
                            float(np.linalg.norm(target_xy - tip_xy)), 6
                        )
                        events_by_profile[profile.name].append(event)
                    else:
                        still_pending.append(event)
                pending[profile.name] = still_pending

        capture_and_schedule()
        deliver_due_frames()
        while self.data.time < simulation_end - timestep / 2:
            _step_with_target_motion(
                self.model,
                self.data,
                self.target_mocap_id,
                base_xy,
                amplitude,
                frequency,
                path_profile,
                phase,
            )
            joint_speed = max(
                abs(float(self.data.qvel[self.shoulder_dof])),
                abs(float(self.data.qvel[self.elbow_dof])),
            )
            if (
                settled_time_s is None
                and self.data.time <= settle_deadline + 1e-9
                and joint_speed <= SETTLE_SPEED_THRESHOLD_RAD_S
            ):
                settle_steps += 1
                if settle_steps >= settle_dwell_steps:
                    settled_time_s = round(float(self.data.time), 6)
            elif settled_time_s is None:
                settle_steps = 0

            if self.data.time + 1e-9 >= next_capture_time:
                capture_and_schedule()
                next_capture_time += CAMERA_FRAME_PERIOD_S
            deliver_due_frames()

        for profile in delivery_profiles:
            for event in events_by_profile[profile.name]:
                event["capture_after_settle"] = (
                    settled_time_s is not None
                    and float(event["capture_time_s"]) + 1e-9 >= settled_time_s
                )
        return {
            "controller_stop_time_s": stop_time,
            "settled_time_s": settled_time_s,
            "settle_duration_s": (
                round(settled_time_s - stop_time, 6)
                if settled_time_s is not None
                else None
            ),
            "events_by_profile": events_by_profile,
            "images_by_frame": images,
        }


def _evaluate_trial_policies(
    trial: dict,
    replay: dict,
    delivery_profiles: tuple[DeliveryProfile, ...] = DELIVERY_PROFILES,
) -> list[dict]:
    records = []
    for profile in delivery_profiles:
        events = replay["events_by_profile"][profile.name]
        baseline = evaluate_arrival_only_policy(trial, events)
        records.append(
            {
                **_trial_identity(trial),
                "delivery_profile": profile.name,
                "policy": "arrival_first_visible_frame",
                "max_frame_age_s": None,
                **baseline,
            }
        )
        for max_age in AGE_BUDGETS_S:
            guarded = evaluate_timestamp_guarded_policy(
                trial,
                events,
                settled_time_s=replay["settled_time_s"],
                max_frame_age_s=max_age,
            )
            records.append(
                {
                    **_trial_identity(trial),
                    "delivery_profile": profile.name,
                    "policy": "two_post_settle_frames",
                    "max_frame_age_s": max_age,
                    **guarded,
                }
            )
    return records


def _trial_identity(trial: dict) -> dict:
    return {
        "seed": int(trial["seed"]),
        "episode": int(trial["episode"]),
        "target_motion_profile": trial["target_motion_profile"],
        "target_motion_phase_rad": float(trial.get("target_motion_phase_rad", 0.0)),
        "control_mode": (
            "delayed_target_current_joint_tip"
            if trial["use_current_end_effector_state"]
            else "fully_delayed_rgb"
            if trial["camera_observation_delay_observations"]
            else "fresh_rgb"
        ),
        "control_delay_observations": int(
            trial["camera_observation_delay_observations"]
        ),
        "control_delay_s": float(trial["camera_observation_delay_simulated_s"]),
    }


def _summarize_policy_records(records: list[dict]) -> dict:
    trials = len(records)
    confirmed = [record for record in records if record["visual_confirmed"]]
    true_positive = sum(
        record["physical_success_at_decision_delivery"] is True
        for record in confirmed
    )
    false_positive = sum(
        record["false_confirmation_at_delivery"] for record in confirmed
    )
    mismatches = [
        record
        for record in confirmed
        if record["physical_success_at_decision_delivery"] is False
    ]
    stop_successes = sum(
        record["physical_success_at_controller_stop"] for record in records
    )
    latencies = [
        float(record["decision_latency_after_stop_s"])
        for record in confirmed
        if record["decision_latency_after_stop_s"] is not None
    ]
    return {
        "trials": trials,
        "controller_stop_physical_successes": stop_successes,
        "controller_stop_physical_success_rate": stop_successes / trials if trials else None,
        "visually_confirmed": len(confirmed),
        "visual_confirmation_rate": len(confirmed) / trials if trials else None,
        "true_confirmations_at_delivery": true_positive,
        "false_confirmations_at_delivery": false_positive,
        "delivery_mismatches_from_motion_after_capture": sum(
            record.get("physical_success_at_decision_capture") is True
            for record in mismatches
        ),
        "delivery_mismatches_already_out_at_capture": sum(
            record.get("physical_success_at_decision_capture") is False
            for record in mismatches
        ),
        "visual_confirmation_precision_at_delivery": (
            true_positive / len(confirmed) if confirmed else None
        ),
        "missed_controller_successes": sum(
            record["missed_controller_success"] for record in records
        ),
        "not_confirmed": sum(record["status"] == "not_confirmed" for record in records),
        "unverified": sum(record["status"] == "unverified" for record in records),
        "usable_frames_total": sum(record["usable_frames"] for record in records),
        "dropped_frames_total": sum(record["dropped_frames"] for record in records),
        "pre_settle_rejections_total": sum(
            record["pre_settle_rejections"] for record in records
        ),
        "stale_rejections_total": sum(
            record["stale_rejections"] for record in records
        ),
        "out_of_order_rejections_total": sum(
            record["out_of_order_rejections"] for record in records
        ),
        "confirmation_latency_median_s": (
            float(np.median(latencies)) if latencies else None
        ),
        "confirmation_latency_p95_s": (
            float(np.quantile(latencies, 0.95)) if latencies else None
        ),
    }


def _aggregate_results(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        key = (
            record["target_motion_profile"],
            record["control_mode"],
            record["control_delay_observations"],
            record["delivery_profile"],
            record["policy"],
            record["max_frame_age_s"],
        )
        grouped[key].append(record)
    rows = []
    for key, group in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        motion_profile, control_mode, delay, delivery, policy, max_age = key
        per_seed = []
        for seed in sorted({record["seed"] for record in group}):
            seed_records = [record for record in group if record["seed"] == seed]
            per_seed.append({"seed": seed, **_summarize_policy_records(seed_records)})
        rows.append(
            {
                "target_motion_profile": motion_profile,
                "target_motion_phases_rad": sorted(
                    {record["target_motion_phase_rad"] for record in group}
                ),
                "control_mode": control_mode,
                "control_delay_observations": delay,
                "control_delay_s": round(
                    delay * FEEDBACK_STEPS_PER_UPDATE * SIMULATION_TIMESTEP_S, 4
                ),
                "delivery_profile": delivery,
                "policy": policy,
                "max_frame_age_s": max_age,
                **_summarize_policy_records(group),
                "per_seed": per_seed,
            }
        )
    return rows


def _write_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, separators=(",", ":"))
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in record.items()
                }
            )


def _write_summary_csv(rows: list[dict], path: Path) -> None:
    """Write the compact, GitHub-friendly aggregate table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=SUMMARY_CSV_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, separators=(",", ":"))
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in row.items()
                    if key in SUMMARY_CSV_FIELDS
                }
            )


def _write_figure(rows: list[dict], path: Path) -> None:
    """Write a dependency-free SVG comparing confirmation and stale decisions."""
    width, height = 1100, 610
    panel_y = 96
    panel_h = 400
    plot_left, plot_right = 105, 1040
    plot_top, plot_bottom = panel_y, panel_y + panel_h
    colors = {
        "arrival_first_visible_frame": "#c44536",
        "two_post_settle_frames|0.5": "#1f77b4",
        "two_post_settle_frames|1.0": "#2a9d65",
    }
    profiles = [profile.name for profile in DELIVERY_PROFILES]
    policy_specs = (
        ("arrival_first_visible_frame", None, "arrival-only: one visible frame"),
        ("two_post_settle_frames", 0.5, "two post-settle frames; age ≤ 0.5 s"),
        ("two_post_settle_frames", 1.0, "two post-settle frames; age ≤ 1.0 s"),
    )
    aggregate = {}
    for profile_name in profiles:
        for policy, age, _ in policy_specs:
            selected = [
                row for row in rows
                if row["delivery_profile"] == profile_name
                and row["policy"] == policy
                and row["max_frame_age_s"] == age
            ]
            trials = sum(row["trials"] for row in selected)
            confirmed = sum(row["visually_confirmed"] for row in selected)
            false = sum(row["false_confirmations_at_delivery"] for row in selected)
            aggregate[(profile_name, policy, age)] = (
                confirmed / trials if trials else 0.0,
                100.0 * false / trials if trials else 0.0,
            )
    false_rate_max = max(
        (values[1] for values in aggregate.values()), default=0.0
    )
    false_rate_scale_max = max(5.0, math.ceil(false_rate_max / 5.0) * 5.0)

    svg = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        (
            '<text x="550" y="38" text-anchor="middle" font-family="Arial" '
            'font-size="24" font-weight="bold" fill="#15202b">'
            'Timestamped post-stop visual confirmation</text>'
        ),
        (
            '<text x="550" y="64" text-anchor="middle" font-family="Arial" '
            'font-size="13" fill="#4a5560">Four target-motion phases • '
            '2 px pixel noise • delayed/jittered frames • 2 required positive frames</text>'
        ),
    ]
    panel_titles = (
        "Visual confirmation rate",
        "False confirmations at delivery per 100 trials",
    )
    for panel_index, title in enumerate(panel_titles):
        x_offset = panel_index * 550
        left = 72 + x_offset
        right = 490 + x_offset
        top, bottom = plot_top, plot_bottom
        svg.append(
            f'<text x="{(left + right) / 2}" y="{top - 16}" '
            'text-anchor="middle" font-family="Arial" font-size="16" '
            f'font-weight="bold" fill="#263746">{title}</text>'
        )
        svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#5d6b75"/>')
        svg.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#5d6b75"/>')
        scale_max = 1.0 if panel_index == 0 else false_rate_scale_max
        for tick in range(0, 6):
            value = scale_max * tick / 5
            y = bottom - (bottom - top) * tick / 5
            label = (
                f"{100 * value:.0f}%"
                if panel_index == 0
                else f"{scale_max * tick / 5:.1f}"
            )
            svg.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{right}" '
                f'y2="{y:.1f}" stroke="#e4e9ed"/>'
            )
            svg.append(
                f'<text x="{left - 9}" y="{y + 4:.1f}" text-anchor="end" '
                'font-family="Arial" font-size="11" fill="#53626d">'
                f'{label}</text>'
            )
        for x_index, profile_name in enumerate(profiles):
            x = left + 36 + x_index * ((right - left - 72) / max(1, len(profiles) - 1))
            short = {
                "near_real_time": "~0.05 s",
                "moderate_latency": "~0.40 s",
                "high_latency": "~0.80 s",
            }[profile_name]
            svg.append(
                f'<text x="{x:.1f}" y="{bottom + 24}" text-anchor="middle" '
                'font-family="Arial" font-size="11" fill="#53626d">'
                f'{short}</text>'
            )
        for policy, age, _label in policy_specs:
            key = (policy, age)
            color_key = f"{policy}|{age}" if age is not None else policy
            color = colors[color_key]
            points = []
            for x_index, profile_name in enumerate(profiles):
                value = aggregate[(profile_name, policy, age)][panel_index]
                if panel_index == 1:
                    value = min(1.0, value / scale_max)
                x = left + 36 + x_index * (
                    (right - left - 72) / max(1, len(profiles) - 1)
                )
                y = bottom - (bottom - top) * value
                points.append((x, y))
            point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            svg.append(
                f'<polyline points="{point_text}" fill="none" '
                f'stroke="{color}" stroke-width="3"/>'
            )
            for x, y in points:
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
        if panel_index == 0:
            svg.append(
                f'<text transform="translate({left - 49},{(top + bottom) / 2}) '
                'rotate(-90)" text-anchor="middle" font-family="Arial" '
                'font-size="12" fill="#53626d">confirmed / all trials</text>'
            )
        else:
            svg.append(
                f'<text transform="translate({left - 49},{(top + bottom) / 2}) '
                'rotate(-90)" text-anchor="middle" font-family="Arial" '
                'font-size="12" fill="#53626d">'
                'false confirmations / 100 trials</text>'
            )
    legend_y = 548
    for index, (policy, age, label) in enumerate(policy_specs):
        color_key = f"{policy}|{age}" if age is not None else policy
        x = 90 + index * 330
        svg.append(
            f'<line x1="{x}" y1="{legend_y}" x2="{x + 28}" '
            f'y2="{legend_y}" stroke="{colors[color_key]}" stroke-width="3"/>'
        )
        svg.append(
            f'<text x="{x + 36}" y="{legend_y + 4}" font-family="Arial" '
            f'font-size="11" fill="#35434e">{label}</text>'
        )
    svg.append(
        '<text x="550" y="590" text-anchor="middle" font-family="Arial" '
        'font-size="11" fill="#65737e">Controller-stop reaching and '
        'camera-delivery verification are reported separately in the trial CSV.</text>'
    )
    svg.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def run_timestamped_verification_benchmark(
    *,
    seeds: int = 5,
    episodes_per_seed: int = 5,
    base_seed: int = 900,
    output_path: Path | None = None,
    csv_path: Path | None = None,
    summary_csv_path: Path | None = None,
    figure_path: Path | None = None,
) -> dict:
    if seeds <= 0 or episodes_per_seed <= 0:
        raise ValueError("seeds and episodes_per_seed must be positive")
    all_records: list[dict] = []
    settle_summaries = []
    replay = PostStopCameraReplay()
    try:
        with tempfile.TemporaryDirectory(prefix="visual-servo-post-stop-") as temp_dir:
            for motion_profile in MOTION_PROFILES:
                for phase_rad in MOTION_PHASES_RAD:
                    for mode, delay_updates, use_current_tip in CONTROL_MODES:
                        for seed in range(base_seed, base_seed + seeds):
                            report = run_trials(
                                episodes_per_seed,
                                seed,
                                (
                                    Path(temp_dir)
                                    / motion_profile
                                    / f"phase-{phase_rad:.5f}"
                                    / mode
                                    / str(delay_updates)
                                    / str(seed)
                                ),
                                controller="image_feedback",
                                pixel_noise_std_px=2.0,
                                target_motion_amplitude_m=MOTION_AMPLITUDE_M,
                                target_motion_frequency_hz=MOTION_FREQUENCY_HZ,
                                target_motion_profile=motion_profile,
                                target_motion_phase_rad=phase_rad,
                                camera_observation_delay_observations=delay_updates,
                                target_motion_prediction="none",
                                use_current_end_effector_state=use_current_tip,
                                feedback_allow_stale_target_confirmation=False,
                                verify_terminal=False,
                                save_media=False,
                            )
                            for raw_trial in report["results"]:
                                trial = {"seed": seed, **raw_trial}
                                stream = replay.replay(trial, seed=seed)
                                all_records.extend(_evaluate_trial_policies(trial, stream))
                                settle_summaries.append(
                                    {
                                        **_trial_identity(trial),
                                        "settled_time_s": stream["settled_time_s"],
                                        "settle_duration_s": stream["settle_duration_s"],
                                    }
                                )
    finally:
        replay.close()

    rows = _aggregate_results(all_records)
    report = {
        "schema_version": 1,
        "benchmark": "timestamped_post_stop_visual_confirmation",
        "protocol": {
            "controller": "existing image feedback; controller gains unchanged",
            "target_motion_amplitude_m": MOTION_AMPLITUDE_M,
            "target_motion_frequency_hz": MOTION_FREQUENCY_HZ,
            "target_motion_profiles": list(MOTION_PROFILES),
            "target_motion_phases_rad": list(MOTION_PHASES_RAD),
            "target_motion_phase_purpose": (
                "spread evidence across the periodic cycle because the prior 0.4 s versus 0.8 s "
                "comparison was phase-dependent"
            ),
            "pixel_noise_std_px": 2.0,
            "control_modes": {
                "fresh_rgb": "current RGB target and tip",
                "fully_delayed_rgb": "delayed RGB target and tip",
                "delayed_target_current_joint_tip": (
                    "delayed RGB target with current joint-state projection of the tip"
                ),
            },
            "verification_frame_period_s": CAMERA_FRAME_PERIOD_S,
            "verification_pixel_centroid_noise_std_px": VERIFICATION_PIXEL_NOISE_STD_PX,
            "camera_delivery_profiles": [asdict(profile) for profile in DELIVERY_PROFILES],
            "maximum_confirmation_wait_s": MAX_CONFIRMATION_WAIT_S,
            "arm_settle_rule": {
                "joint_speed_threshold_rad_s": SETTLE_SPEED_THRESHOLD_RAD_S,
                "continuous_dwell_s": SETTLE_DWELL_S,
                "maximum_wait_s": MAX_SETTLE_S,
                "hold_command": "measured joint pose at controller stop",
            },
            "policies": {
                "arrival_first_visible_frame": (
                    "ablation: first delivered frame with both markers, ignoring "
                    "capture time, settling, and age"
                ),
                "two_post_settle_frames": (
                    "two consecutive captured frames, both captured after arm settling, "
                    "monotonic capture sequence, within the configured receive-age bound"
                ),
            },
            "age_budgets_s": list(AGE_BUDGETS_S),
            "physical_success_threshold_m": SUCCESS_THRESHOLD_METERS,
            "success_ground_truth_use": (
                "evaluation only; excluded from control and visual decisions"
            ),
            "camera_delay_type": (
                "continuous-time capture-to-delivery with seeded jitter and frame loss"
            ),
            "seeds": seeds,
            "base_seed": base_seed,
            "episodes_per_seed": episodes_per_seed,
            "control_trial_count": (
                seeds
                * episodes_per_seed
                * len(MOTION_PROFILES)
                * len(MOTION_PHASES_RAD)
                * len(CONTROL_MODES)
            ),
            "policy_trial_count": len(all_records),
        },
        "results": rows,
        "settling_results": settle_summaries,
        "trial_results": all_records,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n")
    if csv_path is not None:
        _write_csv(all_records, csv_path)
    if summary_csv_path is not None:
        _write_summary_csv(rows, summary_csv_path)
    if figure_path is not None:
        _write_figure(rows, figure_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes-per-seed", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=900)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/timestamped-verification.json"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/timestamped-verification-trials.csv"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("assets/demos/timestamped_verification_benchmark.svg"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results/timestamped-verification-summary.csv"),
    )
    args = parser.parse_args()
    try:
        report = run_timestamped_verification_benchmark(
            seeds=args.seeds,
            episodes_per_seed=args.episodes_per_seed,
            base_seed=args.base_seed,
            output_path=args.output,
            csv_path=args.csv,
            summary_csv_path=args.summary_csv,
            figure_path=args.figure,
        )
    except ValueError as error:
        parser.error(str(error))
    print(
        f"Timestamped verification: {report['protocol']['control_trial_count']} controller trials; "
        f"{report['protocol']['policy_trial_count']} policy outcomes; saved {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
