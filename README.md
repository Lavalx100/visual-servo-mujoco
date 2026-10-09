# Camera-based reaching in MuJoCo

A small robotics demo with a clear perception-to-action loop:

1. MuJoCo renders an overhead RGB camera view of a tabletop and a red target.
2. OpenCV detects the target centroid from pixels.
3. A pinhole camera model projects that pixel onto the table.
4. Two-link inverse kinematics turns the estimated point into joint targets.
5. MuJoCo runs the arm, and ground truth is used only to score the final error.

![MuJoCo tabletop scene with the camera-detected target marked](assets/camera_view.png)

[Watch the clean open-loop baseline](assets/reaching_demo.mp4)

The scene and target positions are generated locally. The project downloads no
robot datasets, pretrained models, or external simulator assets.

## Run

```bash
uv venv --python 3.12
uv pip install -e '.[dev]'
uv run pytest
uv run visual-servo-demo --episodes 20 --seed 7
uv run visual-servo-demo --controller image_feedback --episodes 20 --seed 7
uv run visual-servo-benchmark --seeds 5 --episodes-per-seed 20
uv run visual-servo-benchmark --feedback-target-filter-window 5 --output artifacts/filter-window-5.json
uv run visual-servo-benchmark --feedback-required-tolerance-checks 2 --output artifacts/two-check-stop.json
uv run visual-servo-benchmark --feedback-max-joint-step-radians 0.25 --output artifacts/step-cap-025.json
uv run visual-servo-benchmark --feedback-allow-stale-target-confirmation \
  --output artifacts/benchmark-stale-confirmation.json
uv run visual-servo-dynamic-benchmark --output artifacts/dynamic-target-benchmark.json
uv run visual-servo-prediction-benchmark --output artifacts/prediction-benchmark.json
uv run visual-servo-reversal-benchmark --output artifacts/reversal-benchmark.json
```

The command writes `artifacts/results.json`, `artifacts/camera_view.png`, and
`artifacts/reaching_demo.mp4` (when the local OpenCV build supports MP4 output).
The JSON records every target's camera pixel, estimated table position, final
end-effector error, and success. Ground-truth target coordinates are included
only for evaluation; the controller estimates the target from the rendered
image.

## Video demos

The gallery includes paired controller examples, noisy and miscalibrated
runs, a transparent failure case, filter-window and stale-target comparisons,
two target-dropout cases, and a moving-target camera-latency comparison.
Individual clips are rendered at 640×480 and 30 fps with the controller,
sensor condition, pixel error, and final reaching result overlaid. The filter
comparison is a labeled side-by-side video.

| Demo | What it shows |
| --- | --- |
| [Clean open loop](assets/demos/open_loop_clean.mp4) | One target projection followed by one IK motion command. |
| [Clean image feedback](assets/demos/feedback_clean.mp4) | Repeated pixel-error corrections with no injected sensor error. |
| [20 px target noise](assets/demos/feedback_noise_20px.mp4) | Image feedback reaching under strong Gaussian pixel noise. |
| [Noise and +5° FOV error](assets/demos/feedback_noise_and_fov_error.mp4) | Feedback with both sensor noise and a camera-model calibration error. |
| [High-noise failure](assets/demos/feedback_high_noise_failure.mp4) | A recorded benchmark failure that reaches the motion-update limit. |
| [3 vs. 5 observation filter](assets/demos/feedback_filter_window_comparison.mp4) | Same seed and noise: three samples miss by 4.8 cm; five samples reach 0.1 cm. One illustrative trial, not an aggregate result. |
| [Target dropout and recovery](assets/demos/feedback_target_dropout_recovery.mp4) | Target detection is suppressed for three feedback observations; the arm reuses its last target estimate, then reacquires vision. |
| [Target dropout timeout](assets/demos/feedback_target_dropout_timeout.mp4) | A six-observation blackout exceeds the five-observation grace period, so visual confirmation stops safely. |
| [Strict vs. stale-target stop](assets/demos/feedback_stale_confirmation_comparison.mp4) | Same target and camera sequence; strict mode stops on target loss, while the experimental rule confirms arrival from recent target history. |
| [Moving target and camera delay](assets/demos/feedback_moving_target_latency.mp4) | Same seed and sinusoidal target: live feedback reaches within 1.4 cm, while a one-observation (0.4 s) delay misses by 4.95 cm. One illustrative trial. |
| [Delayed-target prediction](assets/demos/feedback_target_prediction.mp4) | Same delayed target stream and arm state with measured, constant-velocity, and constant-acceleration estimates. One illustrative trial. |
| [Sinusoidal reversal comparison](assets/demos/feedback_prediction_sinusoidal_reversal.mp4) | Measured target, constant velocity, and reversal-aware estimates with 0.4 s delay and 2 px noise. One illustrative trial. |
| [Piecewise-linear reversal comparison](assets/demos/feedback_prediction_piecewise_linear_reversal.mp4) | The same comparison on a piecewise-linear path. One illustrative trial. |

Regenerate the gallery with:

```bash
uv run python scripts/generate_demo_gallery.py
uv run python scripts/generate_filter_comparison.py
uv run python scripts/generate_stale_confirmation_comparison.py
uv run python scripts/generate_latency_comparison.py
uv run python scripts/generate_prediction_comparison.py
uv run python scripts/generate_reversal_prediction_comparison.py
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

### Code map

- `src/visual_servo_mujoco/model.py` defines the arm, tabletop, target, camera,
  and actuators as an inline MuJoCo scene.
- `src/visual_servo_mujoco/controller.py` contains the pixel projection, inverse
  and forward kinematics, image Jacobian, and damped joint update.
- `src/visual_servo_mujoco/run.py` implements color detection, both controllers,
  randomized trials, deterministic target-dropout injection, and per-trial
  scoring, including a per-observation target-visibility trace.
- `src/visual_servo_mujoco/benchmark.py` runs paired seeds across controllers
  and stress conditions, then writes the JSON summary.
- `scripts/generate_demo_gallery.py` regenerates the deterministic clips in
  `assets/demos/`, including a selected high-noise failure episode.
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

### First paired robustness baseline

The first full paired run used five seeds and 20 targets per seed for every
controller/condition pair (100 trials per row). Results:

| Controller | Condition | Success | Median error | 95th percentile | Updates | Simulated motion |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Open loop | Clean | 100% | 0.9 cm | 2.1 cm | 1 | 2.4 s |
| Open loop | Pixel noise, 2 px std. dev. | 100% | 0.9 cm | 2.4 cm | 1 | 2.4 s |
| Open loop | Pixel noise, 8 px std. dev. | 92% | 2.6 cm | 5.1 cm | 1 | 2.4 s |
| Open loop | Pixel noise, 20 px std. dev. | 34% | 6.8 cm | 11.2 cm | 1 | 2.4 s |
| Open loop | Camera FOV error, -5° | 0% | 6.8 cm | 8.2 cm | 1 | 2.4 s |
| Open loop | Camera FOV error, +5° | 0% | 8.4 cm | 30.3 cm | 1 | 2.4 s |
| Open loop | 8 px noise and +5° FOV error | 7% | 8.2 cm | 30.3 cm | 1 | 2.4 s |
| Image feedback | Clean | 100% | 0.8 cm | 1.2 cm | 14 | 5.6 s |
| Image feedback | Pixel noise, 2 px std. dev. | 100% | 0.8 cm | 1.8 cm | 15 | 6.0 s |
| Image feedback | Pixel noise, 8 px std. dev. | 96% | 1.4 cm | 3.9 cm | 15 | 6.0 s |
| Image feedback | Pixel noise, 20 px std. dev. | 72% | 3.4 cm | 7.1 cm | 17 | 6.8 s |
| Image feedback | Camera FOV error, -5° | 100% | 0.7 cm | 1.3 cm | 15 | 6.0 s |
| Image feedback | Camera FOV error, +5° | 100% | 0.8 cm | 1.1 cm | 13.5 | 5.4 s |
| Image feedback | 8 px noise and +5° FOV error | 96% | 1.4 cm | 3.8 cm | 14 | 5.6 s |

In this scene, image feedback recovers from the tested calibration errors and
reduces failures under pixel noise. The initial feedback controller still
failed 28% of trials at 20 px noise, which motivated the target-measurement
filter experiment below.

### Three-observation median filter experiment

I reran the same five seeds and target sequences with a coordinate-wise median
over the latest three target detections. Each result below uses 100 trials per
condition; the paired comparison holds the controller and benchmark settings
fixed and changes only the filter.

| Condition | Success, no filter → median | Median error, no filter → median | 95th percentile, no filter → median |
| --- | ---: | ---: | ---: |
| Clean | 100% → 100% | 0.8 → 0.7 cm | 1.2 → 1.9 cm |
| Pixel noise, 2 px | 100% → 100% | 0.8 → 0.7 cm | 1.8 → 1.7 cm |
| Pixel noise, 8 px | 96% → 99% | 1.4 → 1.3 cm | 3.9 → 3.3 cm |
| Pixel noise, 20 px | 72% → 78% | 3.4 → 2.9 cm | 7.1 → 7.1 cm |
| FOV error, -5° | 100% → 100% | 0.7 → 0.7 cm | 1.3 → 1.8 cm |
| FOV error, +5° | 100% → 100% | 0.8 → 0.7 cm | 1.1 → 2.1 cm |
| Pixel noise 8 px and FOV error +5° | 96% → 99% | 1.4 → 1.4 cm | 3.8 → 3.8 cm |

The filter gained 3 percentage points at 8 px noise and 6 points at 20 px,
while preserving success on clean and calibration-only trials. Its 95th
percentile error grew in clean and calibration-only runs, so it is a targeted
noise-robustness improvement rather than a universal accuracy improvement. A
five-observation window reached 79% success at 20 px noise but increased the
clean median error to 1.0 cm; the three-observation window is the better
overall compromise in these runs. These are deterministic simulation results,
not a claim about real-camera noise.

### Consecutive-tolerance confirmation experiment

With the three-observation filter fixed, I changed only the stopping rule: the
controller now requires two consecutive image-space checks inside the 8 px
tolerance. This rejects a single noisy reading that would have caused a false
stop. Both runs use a 0.20 rad joint-step cap; the paired results compare this
rule with the median-only run:

| Condition | Success, median only → confirmed | Median error, median only → confirmed | 95th percentile, median only → confirmed | Median motion time |
| --- | ---: | ---: | ---: | ---: |
| Clean | 100% → 100% | 0.7 → 0.6 cm | 1.9 → 1.2 cm | 5.2 → 5.6 s |
| Pixel noise, 8 px | 99% → 100% | 1.3 → 1.0 cm | 3.3 → 2.8 cm | 5.2 → 5.6 s |
| Pixel noise, 20 px | 78% → 82% | 2.9 → 2.7 cm | 7.1 → 6.2 cm | 5.6 → 8.0 s |
| Pixel noise 8 px and FOV error +5° | 99% → 100% | 1.4 → 1.0 cm | 3.8 → 3.6 cm | 4.8 → 5.2 s |

Success stayed at 100% for the ±5° FOV-only conditions. At 20 px noise, the
confirmation rule improved success by another 4 points and reduced the 95th
percentile error, at the cost of more motion: median simulated motion time rose
from 5.6 s to 8.0 s. Eighteen of 100 high-noise trials still fail, with most
reaching the 20-update limit.

### Joint-step cap experiment

With the three-observation filter and two-check stop fixed, I raised the maximum
joint change per update from 0.20 to 0.25 rad. This tests whether larger steps
reduce the remaining update-limit failures. The paired results use the same
five seeds and 100 trials per condition:

| Condition | Success, 0.20 → 0.25 rad | Median error, 0.20 → 0.25 rad | 95th percentile, 0.20 → 0.25 rad | Median motion time |
| --- | ---: | ---: | ---: | ---: |
| Clean | 100% → 100% | 0.6 → 0.7 cm | 1.2 → 1.3 cm | 5.6 → 5.2 s |
| Pixel noise, 8 px | 100% → 100% | 1.0 → 1.0 cm | 2.8 → 3.1 cm | 5.6 → 5.2 s |
| Pixel noise, 20 px | 82% → 85% | 2.7 → 2.5 cm | 6.2 → 6.0 cm | 8.0 → 7.4 s |
| Pixel noise 8 px and FOV error +5° | 100% → 100% | 1.0 → 1.2 cm | 3.6 → 2.8 cm | 5.2 → 4.8 s |

The larger cap raised 20 px-noise success by 3 points and reduced its median
motion time by 0.6 s. Clean and 8 px-noise success stayed at 100%, though the
8 px 95th-percentile error increased slightly. At 20 px noise, 15 trials still
fail: 8 hit the update limit and 7 stop inside pixel tolerance while exceeding
the position-error threshold. Further tuning should focus on adaptive steps
and these remaining failure cases, while retaining the paired-seed benchmark.

### Rechecking the filter window with the final controller

The benchmark and demo commands accept
`--feedback-target-filter-window` so filter sizes can be compared without
changing controller code. I compared windows 3 and 5 with the then-current
two-check stop and 0.25 rad step cap held fixed. The first group is the benchmark's original
seed range; the second uses ten additional seeds and 200 high-noise trials.

| Seeds, 20 px noise | Window | Success | Median error | 95th percentile | Worst error | Median updates | False pixel-tolerance stops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–4 (100 trials) | 3 | 85% | 2.52 cm | 6.03 cm | 6.91 cm | 18.5 | 7 |
| 0–4 (100 trials) | 5 | 88% | 2.75 cm | 5.61 cm | 10.35 cm | 14 | 6 |
| 5–14 (200 trials) | 3 | 81.5% | 2.60 cm | 6.20 cm | 9.33 cm | 20 | 14 |
| 5–14 (200 trials) | 5 | 81.5% | 2.79 cm | 6.11 cm | 8.98 cm | 14 | 24 |

The five-sample window reduced median motion updates, but did not improve
success on the additional seeds and produced more false pixel-tolerance stops.
Its small 95th-percentile change does not justify replacing the three-sample
filter default. The side-by-side clip shows one paired episode where the
outcomes differ; the table is the evidence for the overall comparison.

### Three-check stopping rule

I tested three consecutive image-space checks instead of two, leaving the
three-sample median and 0.25 rad joint-step cap unchanged. The original 100
high-noise trials kept the same 85% success rate, while the 95th-percentile
error improved from 6.03 to 5.92 cm and false pixel-tolerance stops fell from
7 to 3. Update-limit failures rose from 8 to 12, and median motion time rose
from 7.4 to 8.0 s.

On ten additional seeds (200 high-noise trials), success rose from 163/200
(81.5%) to 169/200 (84.5%). Median reaching error fell from 2.60 to 2.29 cm,
95th-percentile error from 6.20 to 5.94 cm, and false pixel-tolerance stops
from 14 to 5. Clean success and the ±5° FOV-only results were unchanged; the
8 px-noise and combined-noise median and tail errors also improved. Three
checks are now the default, while `--feedback-required-tolerance-checks` keeps
the ablation reproducible. A tighter 6 px tolerance with two checks cut false
stops but reduced success to 161/200 and increased update-limit failures from
23 to 32, so I kept the 8 px threshold and added another confirmation instead.

### Rechecking the joint-step cap

With three checks and the three-sample median held fixed, I compared 0.25 and
0.30 rad caps. In the original 100 high-noise trials, 0.30 raised success from
85% to 88% and lowered median error from 2.34 to 2.12 cm. The 95th-percentile
error rose from 5.92 to 6.39 cm, while the worst error fell from 8.63 to
7.32 cm; update-limit failures fell from 12 to 9.

On a separate 200-trial group (seeds 15–24), success rose from 158/200 to
164/200, the 95th-percentile error fell from 6.72 to 6.22 cm, and the worst
error fell from 10.16 to 9.26 cm. Other benchmark conditions remained at 100%
success with either cap. I chose 0.30 for the higher high-noise recovery rate
and lower worst errors, while documenting that the 95th-percentile result
varies by seed group. `--feedback-max-joint-step-radians` makes this comparison
reproducible.

### Current default benchmark

The current defaults are a three-observation median, three consecutive checks
within 8 px, and a 0.30 rad joint-step cap. This table summarizes the 100 paired
image-feedback trials per condition in `artifacts/benchmark.json`.

| Condition | Success | Median error | 95th percentile | Median updates | Simulated motion |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 100% | 0.63 cm | 1.29 cm | 12 | 4.8 s |
| Pixel noise, 2 px | 100% | 0.71 cm | 1.19 cm | 12 | 4.8 s |
| Pixel noise, 8 px | 100% | 0.88 cm | 2.62 cm | 13 | 5.2 s |
| Pixel noise, 20 px | 88% | 2.12 cm | 6.39 cm | 20 | 8.0 s |
| Camera FOV error, -5° | 100% | 0.52 cm | 1.12 cm | 14 | 5.6 s |
| Camera FOV error, +5° | 100% | 0.69 cm | 1.39 cm | 11 | 4.4 s |
| 8 px noise and +5° FOV error | 100% | 1.05 cm | 2.56 cm | 12 | 4.8 s |

### Target-detector dropout benchmark

I injected a blackout beginning at feedback observation 1 and varied its
duration over 100 paired trials per case. The detector recovers from each gap
at or below the five-observation grace limit in all trials. A six-observation
gap stops on the visibility timeout in all trials.

| Injected gap | Physical success | Reacquired after injected gap | Median error | 95th percentile |
| ---: | ---: | ---: | ---: | ---: |
| 1 observation | 100% | 100/100 | 0.64 cm | 1.09 cm |
| 3 observations | 100% | 100/100 | 0.63 cm | 1.05 cm |
| 5 observations | 100% | 100/100 | 0.65 cm | 1.18 cm |
| 6 observations | 33% | 0/100 | 6.88 cm | 29.23 cm |

“Physical success” means the final end-effector position is within 4.5 cm of
the target, measured from MuJoCo ground truth after control stops. In 92–94%
of the 1-, 3-, and 5-observation trials, the controller later stopped because
the arm naturally hid the target as it converged; the final position was still
within the success radius. The six-observation condition shows the difference
between physical proximity and visual confirmation: 33 trials were close
enough at stop, but the controller correctly reported that it had lost the
target before it could confirm arrival. This is a conservative stopping rule,
not a claim that the simulated camera models all real occlusions.

### Natural target loss and stale-target confirmation experiment

The clean, no-injected-fault image-feedback benchmark still had 555 rendered
observations where the target detector could not see the target, across 92 of
100 trials. In 91 trials the strict controller reached the physical success
radius but stopped with `natural_target_visibility_loss` before it could make
three fresh visual confirmations. The per-trial visibility trace showed that
this is a separate self-occlusion problem, not a failure of the injected
dropout test.

I tested an opt-in alternative with
`--feedback-allow-stale-target-confirmation`: when the target is temporarily
missing, three consecutive end-effector observations within 8 px of the last
target estimate can confirm arrival during the five-observation grace window.
This assumes the target stays still, so the strict fresh-vision rule remains
the default.

| Policy and condition | Physical success | Stale-estimate arrival stops | Stops outside 4.5 cm |
| --- | ---: | ---: | ---: |
| Strict, clean | 100% | 0/100 | 0 |
| Stale confirmation, clean | 100% | 91/100 | 0 |
| Strict, 20 px noise | 88% | 0/100 | 0 |
| Stale confirmation, 20 px noise | 88% | 17/100 | 0 |
| Strict, 5-observation blackout | 100% | 0/100 | 0 |
| Stale confirmation, 5-observation blackout | 100% | 95/100 | 0 |
| Strict, 6-observation blackout | 33% | 0/100 | 0 |
| Stale confirmation, 6-observation blackout | 33% | 7/100 | 0 |

Across the full 1,800-trial stale-confirmation run, none of the stops based on
the last target estimate ended outside the physical success radius. The policy
did not improve physical success under 20 px noise, and seven 6-observation
blackout trials stopped on the estimate before the detector returned. For
5-observation blackouts, the detector reacquired in 93/100 trials with stale
confirmation, compared with 100/100 under the strict policy; seven trials
stopped on the estimate before vision returned. The moving-target and latency
tests below also evaluate stale confirmation in this simulator; they do not
model real-camera timing or detector behavior. The report is written to
`artifacts/benchmark-stale-confirmation.json`.

### Moving-target and camera-delay benchmark

Run the separate paired stress suite with:

```bash
uv run visual-servo-dynamic-benchmark --output artifacts/dynamic-target-benchmark.json
```

The target follows `x(t) = x₀ + 0.04 sin(2π × 0.25 t)` metres, with a peak
speed of 0.063 m/s. A feedback update advances 0.4 simulated seconds. The
benchmark compares stationary and moving targets with 0, 0.4, and 0.8 seconds
of camera-observation delay, under both strict fresh-vision and experimental
stale-target stopping policies. Each row has 100 trials across five seeds.

| Strict policy condition | Success | Median error | 95th percentile |
| --- | ---: | ---: | ---: |
| Stationary, no delay | 100% | 0.63 cm | 1.29 cm |
| Stationary, 0.8 s delay | 92% | 1.75 cm | 5.47 cm |
| Moving, no delay | 94% | 1.50 cm | 4.98 cm |
| Moving, 0.4 s delay | 10% | 5.13 cm | 6.76 cm |
| Moving, 0.8 s delay | 68% | 3.60 cm | 7.01 cm |

The stale-confirmation policy had the same physical success rate in every
condition; it produced 91 stale-estimate stops in the stationary/no-delay case
and 32 in the moving/no-delay case, with none outside the 4.5 cm success
radius. It did not correct delayed feedback. The 0.8 s result is better than
the 0.4 s result because this test uses a periodic trajectory: each delay
samples a different phase. That does not establish that larger delays are
safer. The per-observation report records live target positions, the camera
frame's target position, visibility, and frame age so the effect can be
inspected directly.

The [paired latency demo](assets/demos/feedback_moving_target_latency.mp4)
shows the same seed and moving target with live camera feedback and a 0.4 s
delay. This benchmark delays both target and end-effector pixels, modeling a
fully delayed camera stream.

### Target-motion prediction experiment

Run a paired comparison of the rolling-median baseline, a constant-velocity
fit, and a constant-acceleration fit with:

```bash
uv run visual-servo-prediction-benchmark --output artifacts/prediction-benchmark.json
```

It uses five seeds and 20 targets per row (1,500 trials total), with noise-free
target pixels. All three modes use the same current joint-state projection for
the arm tip, so this isolates delayed target estimation from stale arm-pose
feedback. This hybrid setup is distinct from the fully delayed-camera
benchmark above. Prediction fits the last four distinct camera-capture times;
the acceleration model falls back to a velocity fit until three samples exist.

| Target and delay | Median baseline | Median velocity | Median acceleration | Success baseline / velocity / acceleration |
| --- | ---: | ---: | ---: | ---: |
| Stationary, 0 s | 0.65 cm | 0.61 cm | 0.81 cm | 100% / 100% / 100% |
| Stationary, 0.8 s | 0.83 cm | 0.83 cm | 1.13 cm | 100% / 100% / 93% |
| Moving, 0 s | 2.16 cm | 3.11 cm | 2.33 cm | 92% / 83% / 92% |
| Moving, 0.4 s | 2.70 cm | 6.48 cm | 2.26 cm | 85% / 10% / 76% |
| Moving, 0.8 s | 2.60 cm | 6.37 cm | 8.13 cm | 85% / 5% / 20% |

The constant-velocity fit performs poorly when the sinusoidal target reverses
direction within the prediction horizon. At 0.4 s delay, the acceleration fit
improves median error (2.26 cm vs. 2.70 cm baseline) but has a worse P95
(8.61 cm vs. 5.20 cm) and lower success (77% vs. 85%). Both predictors degrade
success at 0.8 s. They remain experimental and off by default. The
[three-way demo](assets/demos/feedback_target_prediction.mp4)
shows one seed, not an aggregate result. The experiment below adds reversal
detection and evaluates it on both smooth and piecewise-linear target paths.
Grasping, contact-rich manipulation, and real-robot performance remain future
work.

### Reversal-aware prediction benchmark

Run the paired experiment with:

```bash
uv run visual-servo-reversal-benchmark --output artifacts/reversal-benchmark.json
```

The benchmark compares the rolling-median baseline, constant-velocity
prediction, and a conservative reversal-aware predictor. The new predictor
detects a turn when adjacent displacement vectors differ in direction by at
least 120 degrees and both moves are at least 4 px; it then extrapolates from
the newest segment. A smaller ambiguous turn causes it to hold the latest
measurement. It uses five seeds and 20 targets per row, for 2,400 trials across
sinusoidal and piecewise-linear
target paths, 0.4 s and 0.8 s camera delays, and pixel-noise standard deviations
of 0 and 2 px. Each mode is paired by target seed and episode, and its
measurement-noise generator starts from the same seed. The robot's current pose
is projected from joint state to isolate delayed target estimation. The report
includes paired per-trial outcomes and seed-cluster bootstrap intervals.

| Target path | Delay | Noise | Baseline success / median | Velocity success / median | Reversal-aware success / median | Paired median error Δ vs baseline (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sinusoidal | 0.4 s | 0 px | 85% / 2.70 cm | 10% / 6.48 cm | 33% / 6.44 cm | +3.19 cm [2.55, 4.00] |
| Sinusoidal | 0.4 s | 2 px | 81% / 2.89 cm | 22% / 5.73 cm | 52% / 4.32 cm | +1.08 cm [0.66, 1.97] |
| Sinusoidal | 0.8 s | 0 px | 85% / 2.60 cm | 5% / 6.37 cm | 5% / 6.40 cm | +3.59 cm [3.01, 3.94] |
| Sinusoidal | 0.8 s | 2 px | 85% / 2.44 cm | 5% / 6.35 cm | 12% / 6.37 cm | +4.01 cm [3.56, 4.53] |
| Piecewise linear | 0.4 s | 0 px | 98% / 2.29 cm | 33% / 4.96 cm | 69% / 2.72 cm | +0.23 cm [0.04, 0.42] |
| Piecewise linear | 0.4 s | 2 px | 88% / 2.40 cm | 52% / 4.33 cm | 77% / 2.98 cm | +0.56 cm [0.07, 1.03] |
| Piecewise linear | 0.8 s | 0 px | 92% / 1.78 cm | 18% / 5.27 cm | 31% / 5.21 cm | +2.84 cm [2.71, 2.97] |
| Piecewise linear | 0.8 s | 2 px | 90% / 1.46 cm | 21% / 5.42 cm | 35% / 5.45 cm | +3.37 cm [3.15, 3.49] |

The paired error delta is the median of per-trial predictor-minus-baseline
reaching errors, so it need not equal the difference between the two marginal
medians. Reversal detection usually improves on plain velocity prediction, but
it still increases median error in all eight settings and reduces success in
every setting. The paired 95% bootstrap intervals, resampling the five seed
groups, are above zero in all eight settings. Keep all prediction methods
opt-in and retain the measured-target baseline. The benchmark rules out this
simple reversal guard as a default latency fix; any
follow-up should improve the estimator before adding more control complexity.
