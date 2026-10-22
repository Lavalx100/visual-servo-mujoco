"""Render the selected camera-guided three-arm demo and its run metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from visual_servo_mujoco.camera_guided_cooperative_pick import (
    DEMO_TARGET_XY,
    run_trial,
)


VIDEO_NAME = "camera_guided_cooperative_pick.mp4"
GIF_NAME = "camera_guided_cooperative_pick.gif"
METADATA_NAME = "camera_guided_cooperative_pick_demo.json"


def generate_demo(output_dir: Path) -> dict[str, Any]:
    """Render one camera-estimate replay at the documented example pose."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = run_trial(
        DEMO_TARGET_XY,
        controller="camera_estimate",
        video_path=output_dir / VIDEO_NAME,
        preview_path=output_dir / GIF_NAME,
        visual_verification=True,
        visual_verification_min_upward_displacement_px=50.0,
        visual_verification_max_marker_rotation_from_start_deg=4.0,
    )
    metadata = {
        "evidence_type": "illustrative development-selected replay",
        "target_xy_m": list(DEMO_TARGET_XY),
        "controller": "camera_estimate",
        "release_profile": "baseline",
        "control_image_policy": (
            "The target is detected once from the initial overhead image. Later "
            "overhead frames are rendered for video display only and are not "
            "fed back into control."
        ),
        "fresh_visual_verification": {
            "camera": "fixed verification_side view with visual-only magenta payload fiducial",
            "capture_times_s": [8.4, 8.8],
            "decision": metrics["fresh_visual_confirmation"],
            "minimum_upward_displacement_px": 50.0,
            "maximum_interframe_drift_px": 12.0,
            "maximum_marker_rotation_from_start_deg": 4.0,
            "physics_truth_is_evaluation_only": True,
            "note": "Two fresh frames test raised-and-stable at those samples; they do not certify the full hold or release trajectory.",
        },
        "videos": [VIDEO_NAME],
        "previews": [GIF_NAME],
        "aggregate_counting": (
            "This selected demo pose is excluded from the randomized aggregate."
        ),
        "metrics": metrics,
    }
    (output_dir / METADATA_NAME).write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/demos"))
    args = parser.parse_args()
    result = generate_demo(args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
