# Camera-based reaching in MuJoCo

A small robotics demo with a clear perception-to-action loop:

1. MuJoCo renders an overhead RGB camera view of a tabletop and a red target.
2. OpenCV detects the target centroid from pixels.
3. A pinhole camera model projects that pixel onto the table.
4. Two-link inverse kinematics turns the estimated point into joint targets.
5. MuJoCo runs the arm, and ground truth is used only to score the final error.

![MuJoCo tabletop scene with the camera-detected target marked](assets/camera_view.png)

[Watch the reaching demo](assets/reaching_demo.mp4)

The scene and target positions are generated locally. The project downloads no
robot datasets, pretrained models, or external simulator assets.

## Run

```bash
uv venv --python 3.12
uv pip install -e '.[dev]'
uv run pytest
uv run visual-servo-demo --episodes 20 --seed 7
```

The command writes `artifacts/results.json`, `artifacts/camera_view.png`, and
`artifacts/reaching_demo.mp4` (when the local OpenCV build supports MP4 output).
The JSON records every target's camera pixel, estimated table position, final
end-effector error, and success. Ground-truth target coordinates are included
only for evaluation; the controller estimates the target from the rendered
image.

On the included deterministic setup (`--episodes 20 --seed 7`), the current run
reached all 20 targets within 4.5 cm. Median end-effector error was 0.9 cm; the
largest was 3.68 cm. These are simulation results, not real-robot performance.

## Model and assumptions

- A top-down pinhole camera with no lens distortion.
- A known target height and camera field of view.
- A planar two-joint arm with position actuators.
- Color segmentation is used instead of a learned detector, keeping the demo
  deterministic and easy to inspect.
- The target is localized from one camera frame at the start of each trial; the
  controller does not use image feedback during the arm motion.

The demo tests visual reaching, not grasping or contact-rich manipulation. A
next extension could add a gripper and expose the scene as a LeRobot EnvHub
environment.
