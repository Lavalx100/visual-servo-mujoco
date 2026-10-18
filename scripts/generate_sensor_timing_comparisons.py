"""Render a labeled video for every delayed-feedback benchmark condition."""

from __future__ import annotations

import tempfile
from pathlib import Path

from generate_filter_comparison import compose_paired_videos
from visual_servo_mujoco.run import run_trials


SEED = 7
TARGET_PROFILES = ("sinusoidal", "piecewise_linear")
PIXEL_NOISE_LEVELS_PX = (0.0, 2.0)
CAMERA_DELAYS = (1, 2)
MOTION_AMPLITUDE_M = 0.04
MOTION_FREQUENCY_HZ = 0.25


def _fresh_visual_confirmation(trial: dict) -> bool:
    if trial["controller_stop_reason"] != "pixel_tolerance_reached":
        return False
    terminal_event = trial["target_visibility_trace"][-1]
    return (
        terminal_event.get("target_measurement_used", False)
        and terminal_event.get("camera_observation_age_observations", 0) == 0
        and terminal_event.get("camera_observation_age_simulated_s", 0.0) <= 1e-9
    )


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="visual-servo-sensor-timing-demo-") as temporary:
        temporary_root = Path(temporary)
        for profile in TARGET_PROFILES:
            for noise in PIXEL_NOISE_LEVELS_PX:
                for delay in CAMERA_DELAYS:
                    sources = []
                    modes = (
                        ("fresh_camera", 0, False, "Fresh RGB: target + tip"),
                        (
                            "fully_delayed_camera",
                            delay,
                            False,
                            "Fully delayed RGB: target + tip",
                        ),
                        (
                            "delayed_target_current_joint_state_tip",
                            delay,
                            True,
                            "Delayed target + current joint-state tip",
                        ),
                    )
                    for mode, camera_delay, use_current_tip, mode_label in modes:
                        output_dir = (
                            temporary_root
                            / profile
                            / f"noise-{noise:g}"
                            / f"delay-{delay}"
                            / mode
                        )
                        report = run_trials(
                            episodes=1,
                            seed=SEED,
                            output_dir=output_dir,
                            controller="image_feedback",
                            pixel_noise_std_px=noise,
                            target_motion_amplitude_m=MOTION_AMPLITUDE_M,
                            target_motion_frequency_hz=MOTION_FREQUENCY_HZ,
                            target_motion_profile=profile,
                            camera_observation_delay_observations=camera_delay,
                            target_motion_prediction="none",
                            use_current_end_effector_state=use_current_tip,
                            feedback_allow_stale_target_confirmation=False,
                            save_media=True,
                        )
                        trial = report["results"][0]
                        outcome = "REACHED" if trial["success"] else "MISS"
                        fresh = "fresh vision yes" if _fresh_visual_confirmation(trial) else "fresh vision no"
                        label = (
                            f"{mode_label} | {outcome} | {fresh} | "
                            f"{trial['reaching_error_m'] * 100:.1f}cm"
                        )
                        color = (80, 220, 110) if trial["success"] else (90, 130, 255)
                        sources.append((output_dir / "reaching_demo.mp4", label, color))

                    destination = (
                        repository
                        / "assets"
                        / "demos"
                        / (
                            f"sensor_timing_{profile}_noise-{int(noise)}px_"
                            f"delay-{delay}.mp4"
                        )
                    )
                    compose_paired_videos(
                        sources,
                        destination,
                        title=(
                            f"{profile.replace('_', ' ')} target | {noise:.0f} px control noise | "
                            f"{delay * 0.4:.1f} s delay | seed {SEED} | one illustrative trial"
                        ),
                        label_scale=0.4,
                    )
                    if not destination.is_file() or destination.stat().st_size == 0:
                        raise RuntimeError(f"comparison video was not created: {destination}")
                    print(
                        f"Wrote {destination.relative_to(repository)} "
                        f"({destination.stat().st_size / 1024:.0f} KiB)"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
