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
- `src/visual_servo_mujoco/cooperative_pick_demo.py` builds the reusable
  three-arm MuJoCo scene. Per-arm collision bits enable contact between
  different arms and with the payload/table while suppressing same-arm
  self-contact. Elbow-up IK keeps the links above the table; staggered open
  grippers align six pads into an equatorial grasp. Render-only wrist adapters
  close the visible seam between wrist links and palm boxes; palm-mounted rails
  and carriage-mounted finger shanks connect the moving pads. These additions
  are non-colliding visuals, so the pads remain the modeled grasp surfaces.
  Contact and friction lift a free payload. The original
  [scripted experiment](experiments/cooperative-pick.md) preserves its centered
  baseline and collision evidence.
- `src/visual_servo_mujoco/camera_guided_cooperative_pick.py` extends that scene
  with base yaw, independently controlled gripper yaw, a top-down camera and a
  visual-only target marker. It imports the same red detector from `run.py` and
  `pixel_to_table_xy` from `controller.py`, then plans camera-estimated or
  oracle diagnostic target poses for the shared contact-based pick trajectory.
  Paired randomized outcomes and a camera-plus-physics video are linked from
  the [camera-guided experiment](experiments/camera-guided-cooperative-pick.md)
  and [video gallery](video-gallery.md).
  The rendered demo pairs a live overhead and oblique view, labels the
  approach/grasp/lift/release stages, and keeps the initial image estimate
  distinct from current simulation-state and contact readouts.
- `src/visual_servo_mujoco/cooperative_pick_visual_verification.py` adds an
  opt-in post-lift image-only check. A fixed side camera observes a visual-only
  fiducial in the initial frame and two fresh frames during the hold. It reports
  confirmed, not confirmed, or abstain from pixel displacement and stability;
  it receives no MuJoCo state.
- `src/visual_servo_mujoco/cooperative_pick_visual_validation.py` evaluates that
  frozen image rule on matched randomized target poses under the 5 mm baseline
  and opt-in 20 mm approach. Its CSV keeps capture-time simulator truth,
  full-hold stability, and full-sequence collision-safe success as separate
  columns. See the [visual-confirmation experiment](experiments/cooperative-pick-visual-confirmation.md).
- `src/visual_servo_mujoco/visual_image_faults.py` applies controlled exposure,
  fiducial occlusion, horizontal blur, and image-plane camera-shift faults to
  side-camera RGB after capture. These changes do not reach the overhead
  controller image or MuJoCo state.
- `src/visual_servo_mujoco/cooperative_pick_visual_fault_validation.py` runs
  frozen V3 on the same held-out poses under clean and faulted side-camera
  images, then compares image decisions while checking physical outcomes stay
  identical across each pose's image-only conditions.
- `src/visual_servo_mujoco/cooperative_pick_occlusion_validation.py` tunes a
  bounded fragment-gap tolerance on seed 39700 and evaluates the frozen value
  on seed 39800. The optional V4 reuses V3 thresholds and frames; the default
  V3 detector and motion path are unchanged. Its paired CSV, scorecards, and
  illustrative clips are linked from the
  [occlusion experiment](experiments/cooperative-pick-occlusion.md).
- `src/visual_servo_mujoco/cooperative_pick_orthogonal_validation.py` compares
  side-camera V3, frozen 28 px V4, and an opt-in two-view V4 candidate on a
  fresh held-out seed block. It renders an opaque MuJoCo rail in the original
  camera's sight line, keeps the rail and markers out of collision dynamics,
  and audits image decisions separately from capture-time truth, hold success,
  and full-cycle success. V3 remains the default; see the
  [orthogonal-view experiment](experiments/cooperative-pick-orthogonal-verification.md).
- `src/visual_servo_mujoco/cooperative_pick_material_validation.py` runs the
  same 12 target poses across three explicit payload-body masses and three
  pad/payload friction settings, paired with both the camera-estimate and
  oracle target controllers. Its CSV, scorecard, and matched demos report
  physical stable-lift, hold, and full-cycle collision outcomes separately;
  it does not score fresh visual confirmation. The nominal mass retains the
  original geometry-derived inertial model; off-nominal mass conditions use
  explicit inertials and zero-density visual trim.
  See the [mass/friction experiment](experiments/cooperative-pick-material-robustness.md).
- `src/visual_servo_mujoco/cooperative_pick_pad_friction_validation.py` runs a
  fresh-seed 96-trial comparison of 16 mm and 18 mm pad half-sizes at low and
  high friction, keeping target poses, nominal payload, approach, release, and
  controllers matched. The larger pad did not rescue any low-friction
  `no_grasp_contact` trial; it stays experimental. Paired outcome counts,
  collision-safety components, scorecard, and replay demos are linked from the
  [pad/friction experiment](experiments/cooperative-pick-pad-friction-validation.md).
- `src/visual_servo_mujoco/cooperative_pick_contact_diagnostics.py` evaluates
  12 new poses at low/high friction with the 16 mm baseline pads and both
  camera-estimate/oracle controllers. The trial runner attributes payload-pad
  contacts to each finger and phase, reports simulated normal force as a
  model-only diagnostic, and compares commanded versus measured slide position
  during closure. The 0/12 low-friction hold score hides transient contact
  during closing/early lift; follow-up evidence finds no pad contact in the
  loaded hold. See the [contact-retention experiment](experiments/cooperative-pick-contact-diagnostics.md).
- `scripts/generate_cooperative_pick_orthogonal_demos.py` replays one counted
  occlusion-recovery example and one counted false-confirmation rejection as
  paired-camera MP4/GIF demonstrations; replays add no benchmark trials.
- `scripts/compare_cooperative_pick_geometry.py` validates a matched-pose,
  one-parameter geometry comparison and writes paired outcomes with Wilson
  intervals and contact attribution. The current lowest-pad clearance probe
  raises only the open approach target. The first held-out 30-pose split
  improved camera-estimate results but lost one net oracle success; an
  independent 60-pose split improved both conditions (52/60 to 56/60 camera,
  53/60 to 58/60 oracle). Confidence intervals still overlap, so it remains
  opt-in. `scripts/generate_cooperative_pick_approach_clearance_demos.py`
  replays a logged fail/pass pair as synchronized individual and side-by-side
  clips. The overhead stream is live for display; only its first frame guides
  the motion.
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
