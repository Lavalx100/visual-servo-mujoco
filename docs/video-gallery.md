# Demo gallery

[Project overview](../README.md)

## Video demos

The gallery includes paired controller examples, noisy and miscalibrated
runs, a transparent failure case, filter-window and stale-target comparisons,
two target-dropout cases, and moving-target camera-latency comparisons.
Single-view clips are rendered at 640×480 and 30 fps; side-by-side videos label
each condition, outcome, and error. The eight sensor-timing videos cover every
profile/noise/delay row, and the two verification videos show current-time
overhead and oblique views. The terminal-verification clips demonstrate the
new opt-in east-view check after the controller has stopped.

| Demo | What it shows |
| --- | --- |
| [Clean open loop](../assets/demos/open_loop_clean.mp4) | One target projection followed by one IK motion command. |
| [Clean image feedback](../assets/demos/feedback_clean.mp4) | Repeated pixel-error corrections with no injected sensor error. |
| [20 px target noise](../assets/demos/feedback_noise_20px.mp4) | Image feedback reaching under strong Gaussian pixel noise. |
| [Noise and +5° FOV error](../assets/demos/feedback_noise_and_fov_error.mp4) | Feedback with both sensor noise and a camera-model calibration error. |
| [High-noise failure](../assets/demos/feedback_high_noise_failure.mp4) | A recorded benchmark failure that reaches the motion-update limit. |
| [3 vs. 5 observation filter](../assets/demos/feedback_filter_window_comparison.mp4) | Same seed and noise: three samples miss by 4.8 cm; five samples reach 0.1 cm. One illustrative trial, not an aggregate result. |
| [Target dropout and recovery](../assets/demos/feedback_target_dropout_recovery.mp4) | Target detection is suppressed for three feedback observations; the arm reuses its last target estimate, then reacquires vision. |
| [Target dropout timeout](../assets/demos/feedback_target_dropout_timeout.mp4) | A six-observation blackout exceeds the five-observation grace period, so visual confirmation stops safely. |
| [Strict vs. stale-target stop](../assets/demos/feedback_stale_confirmation_comparison.mp4) | Same target and camera sequence; strict mode stops on target loss, while the experimental rule confirms arrival from recent target history. |
| [Moving target and camera delay](../assets/demos/feedback_moving_target_latency.mp4) | Same seed and sinusoidal target: live feedback reaches within 1.4 cm, while a one-observation (0.4 s) delay misses by 4.95 cm. One illustrative trial. |
| [Delayed-target prediction](../assets/demos/feedback_target_prediction.mp4) | Same delayed target stream and arm state with measured, constant-velocity, and constant-acceleration estimates. One illustrative trial. |
| [Sinusoidal reversal comparison](../assets/demos/feedback_prediction_sinusoidal_reversal.mp4) | Measured target, constant velocity, and reversal-aware estimates with 0.4 s delay and 2 px noise. One illustrative trial. |
| [Piecewise-linear reversal comparison](../assets/demos/feedback_prediction_piecewise_linear_reversal.mp4) | The same comparison on a piecewise-linear path. One illustrative trial. |
| [Clean terminal verification view](../assets/demos/verification_view_clean.mp4) | Synchronized fresh renders show the target hidden in the overhead control view and visible with the end-effector from an east oblique view. |
| [20 px-noise terminal verification view](../assets/demos/verification_view_20px-noise.mp4) | The same two current-time views after a noisy control run. The oblique view is observational and does not alter the controller. |
| [Opt-in terminal estimate, clean](../assets/demos/terminal_verification_clean.mp4) | A fresh east-view render after control stops; reports its pixel-based distance beside the separate MuJoCo physical reach score. |
| [Opt-in terminal estimate, 20 px control noise](../assets/demos/terminal_verification_20px-noise.mp4) | The controller sees 20 px target-measurement noise; the terminal verifier uses a fresh clean simulator render and remains outside the control loop. |

The held-out verifier stress suite has one illustrative clip per injected
condition. Calibration offsets affect only the verifier's back-projection;
exposure and marker occlusion affect only its terminal image. Each clip uses
seed 7 and is an example, not an aggregate benchmark result.

| Verifier condition | Demo |
| --- | --- |
| FOV estimate +5° | [Watch](../assets/demos/terminal_verification_stress_fovy_plus_5deg.mp4) |
| FOV estimate −5° | [Watch](../assets/demos/terminal_verification_stress_fovy_minus_5deg.mp4) |
| Estimated camera world X +2 cm | [Watch](../assets/demos/terminal_verification_stress_camera_x_plus_2cm.mp4) |
| Estimated camera yaw +3° | [Watch](../assets/demos/terminal_verification_stress_camera_yaw_plus_3deg.mp4) |
| Uniform image exposure ×0.25 | [Watch](../assets/demos/terminal_verification_stress_dim_exposure_x0_25.mp4) |
| Uniform image exposure ×1.5 | [Watch](../assets/demos/terminal_verification_stress_bright_exposure_x1_5.mp4) |
| Target marker occluded | [Watch](../assets/demos/terminal_verification_stress_target_occluded.mp4) |
| End-effector marker occluded | [Watch](../assets/demos/terminal_verification_stress_end_effector_occluded.mp4) |

The quality-gate pair shows a held-out false-success case at a −5° FOV
estimate. A 0.50 gate withholds the image result while preserving the same
physical reach outcome. It is one selected example, not an aggregate result.

| Demo | What it shows |
| --- | --- |
| [False visual success, quality 0.47](../assets/demos/terminal_verification_quality_false_positive.mp4) | The camera estimate is within 4.5 cm while MuJoCo says the arm missed. |
| [Quality gate abstains](../assets/demos/terminal_verification_quality_abstention.mp4) | The same seed, target, and arm pose with a 0.50 gate; the visual result is unavailable. |

The seed-separated run selected a 0.60 gate on seeds 200–204, then evaluated
that unchanged threshold on seeds 300–304. These clips show seed 302, episode
17, from the independent validation split. The image estimate is 4.494 cm, but
the physical reaching error is 4.6 cm:

| Held-out validation demo | What it shows |
| --- | --- |
| [Ungated false visual success](../assets/demos/terminal_verification_quality_heldout_false_success.mp4) | The verifier reports within 4.5 cm without a quality gate. |
| [Frozen gate abstains](../assets/demos/terminal_verification_quality_heldout_abstention.mp4) | The same target and physical result; the 0.60 gate withholds the estimate. |

The fixed-landmark demo compares nominal camera calibration with a combined
renderer-side FOV, position, and yaw shift. White crosses show nominal
projections, green rings show detected landmark centers, and connecting lines
show image residuals:

| Demo | What it shows |
| --- | --- |
| [Fixed-landmark calibration health](../assets/demos/camera_calibration_health.mp4) | Four independent tabletop landmarks remain within 0.51 px RMS nominally and move to 32.02 px RMS under the combined rendered camera shift. |
| [Frozen threshold validation](../assets/demos/camera_calibration_validation.mp4) | Same held-out nominal image under two frozen thresholds, then examples of subtle/severe drift and unavailable dim imagery. |
| [Pose and lens transfer](../assets/demos/camera_calibration_transfer.mp4) | Applies the frozen East-view threshold to changed camera poses, image-edge clipping, positive/negative radial warps, and held-out shifts. |
| [Planar feature-radius diagnostic](../assets/demos/camera_calibration_planar.mp4) | A 49-corner tabletop target across five views and radial warps; overlays show reprojection vectors and error by feature radius beside the paired legacy four-marker result. |
| [Planar target-size design probe](../assets/demos/camera_calibration_planar_target_design.mp4) | Side-by-side 0.15 m and 0.17 m checker squares under the same noisy image settings, showing the edge-coverage versus detector-availability tradeoff. |

The eight delayed-feedback conditions each have a three-panel demo (fresh RGB,
fully delayed RGB, and delayed target with current joint-state tip projection):

| Target path | Noise | Delay | Video |
| --- | ---: | ---: | --- |
| Sinusoidal | 0 px | 0.4 s | [Watch](../assets/demos/sensor_timing_sinusoidal_noise-0px_delay-1.mp4) |
| Sinusoidal | 0 px | 0.8 s | [Watch](../assets/demos/sensor_timing_sinusoidal_noise-0px_delay-2.mp4) |
| Sinusoidal | 2 px | 0.4 s | [Watch](../assets/demos/sensor_timing_sinusoidal_noise-2px_delay-1.mp4) |
| Sinusoidal | 2 px | 0.8 s | [Watch](../assets/demos/sensor_timing_sinusoidal_noise-2px_delay-2.mp4) |
| Piecewise linear | 0 px | 0.4 s | [Watch](../assets/demos/sensor_timing_piecewise_linear_noise-0px_delay-1.mp4) |
| Piecewise linear | 0 px | 0.8 s | [Watch](../assets/demos/sensor_timing_piecewise_linear_noise-0px_delay-2.mp4) |
| Piecewise linear | 2 px | 0.4 s | [Watch](../assets/demos/sensor_timing_piecewise_linear_noise-2px_delay-1.mp4) |
| Piecewise linear | 2 px | 0.8 s | [Watch](../assets/demos/sensor_timing_piecewise_linear_noise-2px_delay-2.mp4) |

Regenerate the gallery with:

```bash
uv run python scripts/generate_demo_gallery.py
uv run python scripts/generate_filter_comparison.py
uv run python scripts/generate_stale_confirmation_comparison.py
uv run python scripts/generate_latency_comparison.py
uv run python scripts/generate_prediction_comparison.py
uv run python scripts/generate_reversal_prediction_comparison.py
uv run python scripts/generate_sensor_timing_comparisons.py
uv run python scripts/generate_verification_view_demos.py
uv run python scripts/generate_terminal_verification_demos.py
uv run python scripts/generate_verification_stress_demos.py
uv run python scripts/generate_verification_quality_demos.py
uv run visual-servo-camera-calibration-health --output artifacts/camera-calibration-health.json
uv run python scripts/generate_camera_calibration_health_demo.py
uv run visual-servo-camera-calibration-validation --output artifacts/camera-calibration-health-validation.json
uv run python scripts/generate_camera_calibration_validation_demo.py
uv run visual-servo-camera-calibration-transfer --output artifacts/camera-calibration-transfer-validation.json
uv run python scripts/generate_camera_calibration_transfer_demo.py
uv run visual-servo-camera-calibration-planar --output artifacts/camera-calibration-planar-profile.json
uv run python scripts/generate_camera_calibration_planar_demo.py
uv run python scripts/generate_camera_calibration_planar_design_demo.py
uv run --extra pdf python scripts/generate_project_explainer.py
```

To record a specific episode from a seeded batch, select it with
`--video-episode-index`. For example, this saves seed 3, episode 5 from the
20 px-noise condition:

```bash
uv run visual-servo-demo --controller image_feedback --pixel-noise-std-px 20 \
  --episodes 20 --seed 3 --video-episode-index 5 \
  --output-dir artifacts/high-noise-case
```

The benchmark compares two controllers on identical seeded target sequences:
`open_loop` projects one target detection into table coordinates and solves
inverse kinematics; `image_feedback` repeatedly compares the red target and
green end-effector marker in the image and applies a bounded damped-Jacobian
joint update. It requires three consecutive image-space checks inside an 8 px
tolerance before stopping, or stops after 20 motion updates. If the target
detector temporarily loses the target, feedback can reuse its last measured
position for up to five observations. By default, a stale estimate may guide
motion but cannot confirm arrival; the controller requires fresh vision for
its consecutive-tolerance stop. The opt-in stale-target experiment tests a
different policy for this stationary-target scene. The controller estimates
the target center from up to three recent detections using a coordinate-wise
median, which suppresses isolated pixel outliers. Both controllers start from
the same joint pose.

Seven paired conditions test clean sensing, Gaussian target-pixel noise, and
camera field-of-view calibration errors. Four additional feedback-only
conditions suppress target detection for 1, 3, 5, or 6 observations. Pixel
noise is injected after target detection at each observation; the calibration
perturbation changes the controller's camera model, not the simulated camera.
Each standard controller/condition pair and each dropout condition has 100
trials across five seeds, for 1,800 trials total. The report
includes success rate, error percentiles, failure reasons, update counts,
simulated motion time, and per-seed/per-trial details in
`artifacts/benchmark.json`. Simulated motion time is robot-task time, not wall
clock runtime. Benchmark trials do not save videos. These controlled tests
isolate specific errors; they do not claim to reproduce real camera noise or
lighting.

Target dropout is software fault injection: the simulated scene and physics
continue normally, but selected camera observations are withheld from the red
target detector. During the gap, the arm reuses the last target location to
keep moving. The default policy suspends arrival confirmation until the
detector sees the target again; the separate stale-target option tests
confirmation from recent target history. The benchmark tests gaps up to the
five-observation grace limit and one longer gap that stops with
`target_not_visible`. Run a recovery example locally with:

```bash
uv run visual-servo-demo --controller image_feedback --seed 7 --episodes 1 \
  --target-dropout-start-observation 1 \
  --target-dropout-duration-observations 3 \
  --output-dir artifacts/dropout-recovery
```

Every image-feedback trial now records a visibility trace: whether the target
was detectable in the rendered image, whether the test deliberately suppressed
that observation, whether the controller used a target measurement, the pixel
error, and the number of completed motion updates. This distinguishes an
injected detector fault from a target that the arm itself hides.

On the included deterministic setup (`--episodes 20 --seed 7`), the current run
reached all 20 targets within 4.5 cm. Median end-effector error was 0.9 cm; the
largest was 3.04 cm. These are simulation results, not real-robot performance.
