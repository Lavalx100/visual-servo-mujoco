# System overview

[Project overview](../README.md)

## Model and assumptions

- A top-down pinhole camera with no lens distortion.
- A known target height and camera field of view.
- A planar two-joint arm with position actuators.
- Color segmentation estimates the circle centers of a red target and green
  end-effector marker; it is deterministic and easy to inspect.
- The open-loop baseline localizes the target once before moving. The feedback
  controller observes the target and end effector again after each joint update.
  It filters only target detections. The separate prediction benchmark uses a
  current joint-state projection for the end effector while target pixels are
  delayed.
- If the arm briefly covers the target, image feedback reuses its last observed
  target position for up to five updates before stopping.

The feedback controller uses a local image Jacobian derived from the two-link
arm geometry and pinhole camera model. It therefore tests closed-loop correction
under calibration error, while still depending on a reasonable local camera and
arm model.

At each feedback update, the controller measures the pixel difference
`e = target_pixel - end_effector_pixel`. The Jacobian `J` predicts how those
pixels move when the two joint angles change. It computes
`Δq = 0.5 Jᵀ (J Jᵀ + I)⁻¹ e`, limits the joint change to 0.30 radians, advances
the simulation for 0.4 seconds, and measures again. The identity term keeps the
inverse stable near arm singularities. Ground-truth target position is read
only after the motion to score the trial.

## Code map

- `src/visual_servo_mujoco/model.py` defines the arm, tabletop, target, camera,
  and actuators as an inline MuJoCo scene.
- `src/visual_servo_mujoco/controller.py` contains the pixel projection, inverse
  and forward kinematics, image Jacobian, and damped joint update.
- `src/visual_servo_mujoco/run.py` implements color detection, both controllers,
  randomized trials, deterministic target-dropout injection, and per-trial
  scoring, including a per-observation target-visibility trace.
- `src/visual_servo_mujoco/benchmark.py` runs paired seeds across controllers
  and stress conditions, then writes the JSON summary.
- `src/visual_servo_mujoco/perception_appearance_benchmark.py` measures the
  existing red-target detector under rendered lighting, clutter, and occlusion
  changes, and writes trial data, a scorecard, and a paired demo.
- `scripts/generate_demo_gallery.py` regenerates the deterministic clips in
  `assets/demos/`, including a selected high-noise failure episode.
- `scripts/export_public_trial_data.py` exports compact per-trial outcomes from
  the paired robustness report to `results/robustness-baseline-trials.csv`.
- `scripts/generate_filter_comparison.py` creates the paired 3-vs-5 sample
  filter clip shown above.
- `scripts/generate_stale_confirmation_comparison.py` creates the strict-versus-
  experimental arrival-stop clip from one seeded target sequence.
- `src/visual_servo_mujoco/dynamic_benchmark.py` compares arrival policies under
  moving-target and camera-delay conditions.
- `scripts/generate_latency_comparison.py` renders a paired moving-target video
  with live and delayed camera feedback.
- `src/visual_servo_mujoco/prediction_benchmark.py` compares measured target
  positions with constant-velocity and constant-acceleration image estimates.
- `scripts/generate_prediction_comparison.py` renders the three-way delayed-
  target prediction demo.
- `src/visual_servo_mujoco/sensor_timing_benchmark.py` compares fully delayed
  RGB feedback with delayed target observations and current joint-state tip
  projection.
- `src/visual_servo_mujoco/timestamped_verification_benchmark.py` replays the
  stopped arm with timestamped camera delivery, settling checks, frame-age
  guards, and separate capture-time/delivery-time physical labels.
- `scripts/generate_timestamped_verification_demo.py` renders the annotated
  post-stop camera and transport demo.
- `src/visual_servo_mujoco/completion_speed_benchmark.py` repeats that replay
  across stationary and moving target paths while keeping the control modes
  separately labeled.
- `scripts/generate_timestamped_speed_demo.py` creates the paired
  stationary-versus-fast target video and README preview.
- `src/visual_servo_mujoco/motion_envelope_benchmark.py` calibrates a one-sided
  image-distance margin on disjoint seeds, then evaluates a speed-bounded
  delivery envelope on held-out camera replays.
- `scripts/generate_motion_envelope_demo.py` selects a paired held-out policy
  outcome and renders the same replay under the fixed-age and envelope rules.
- `src/visual_servo_mujoco/conditional_uncertainty_benchmark.py` calibrates
  q99 margins from pixel-projection sensitivity and heuristic marker quality
  on training seeds, then compares global and conditional rules on held-out
  captures.
- `scripts/generate_conditional_uncertainty_demo.py` replays one held-out case
  with the prior global and conditional margins on the same delivered frames.
- `src/visual_servo_mujoco/verification_benchmark.py` tests four calibrated
  oblique views as independent terminal visual-evidence candidates; it does
  not change the controller or its stop decision.
- `src/visual_servo_mujoco/terminal_verification.py` defines calibrated oblique
  camera poses and pixel-ray back-projection to known marker heights, with
  estimator-only camera calibration and image stress hooks.
- `src/visual_servo_mujoco/verification_robustness_benchmark.py` pairs held-out
  controller trials across nine terminal-verifier conditions and reports
  per-marker image features, estimate availability, false/missed decisions,
  and distance error.
- `src/visual_servo_mujoco/verification_quality_validation.py` selects an
  abstention threshold on disjoint calibration seeds, freezes it, and measures
  coverage and decision errors on a separate seed block.
- `src/visual_servo_mujoco/camera_calibration_health.py` compares four fixed
  tabletop landmarks with their nominal pinhole projections under rendered
  camera perturbations; it is a standalone diagnostic and does not control the arm.
- `src/visual_servo_mujoco/camera_calibration_validation.py` tunes residual
  thresholds on separate nominal and labelled-drift seeds, then measures false
  alarms, drift detection, and landmark availability on held-out renders.
- `src/visual_servo_mujoco/camera_calibration_transfer.py` applies the frozen
  East-view threshold to held-out camera poses and synthetic radial distortion;
  it never adjusts the threshold on transfer samples.
- `src/visual_servo_mujoco/camera_calibration_planar.py` renders a known planar
  checkerboard, associates its 49 corners, and reports residuals by image radius
  beside a separately rendered legacy four-marker reference.
- `scripts/generate_camera_calibration_health_demo.py` writes the landmark
  residual report and the paired nominal-versus-drift video.
- `scripts/generate_camera_calibration_validation_demo.py` visualizes the
  nominal-only, false-alarm-constrained, and fixed 3 px threshold rules on
  representative held-out images.
- `scripts/generate_camera_calibration_transfer_demo.py` replays representative
  pose, edge-visibility, radial-distortion, and shift transfer trials.
- `scripts/generate_camera_calibration_planar_demo.py` creates the annotated
  radius-profile video and image preview across camera poses and lens warps.
- `scripts/generate_camera_calibration_planar_design_demo.py` preserves the
  larger-target availability failure beside the selected layout.
- `run.py --verify-terminal` takes a fresh east-view observation after control
  stops and reports its image-based distance separately from physical reach.
- `scripts/generate_sensor_timing_comparisons.py` renders a labeled three-mode
  video for each of the eight delayed-feedback benchmark conditions.
- `scripts/generate_verification_view_demos.py` renders synchronized overhead
  and oblique camera views for clean and high-noise terminal examples.
- `scripts/generate_verification_stress_demos.py` records one two-view terminal
  clip for each calibration, exposure, and marker-occlusion stress condition.
- `scripts/generate_verification_quality_demos.py` records a paired
  false-success and abstention example with identical reaching outcomes.
- `scripts/generate_verification_quality_validation_demos.py` finds a false
  success in the independent validation split and records the baseline and
  frozen-gate views for the same target and arm pose.
- `scripts/generate_terminal_verification_demos.py` records the actual opt-in
  terminal verifier in clean and high control-noise conditions.
