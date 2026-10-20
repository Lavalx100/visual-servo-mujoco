# Reaching and motion experiments

[Project overview](../../README.md)

## First paired robustness baseline

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

## Three-observation median filter experiment

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

## Consecutive-tolerance confirmation experiment

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

## Joint-step cap experiment

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

## Rechecking the filter window with the final controller

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

## Three-check stopping rule

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

## Rechecking the joint-step cap

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

## Current default benchmark

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

## Target-detector dropout benchmark

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

## Natural target loss and stale-target confirmation experiment

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

## Moving-target and camera-delay benchmark

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

The [paired latency demo](../../assets/demos/feedback_moving_target_latency.mp4)
shows the same seed and moving target with live camera feedback and a 0.4 s
delay. This benchmark delays both target and end-effector pixels, modeling a
fully delayed camera stream.

## Sensor timing and end-effector source benchmark

Run the paired comparison with:

```bash
uv run visual-servo-sensor-timing-benchmark \
  --output artifacts/sensor-timing-benchmark.json
```

This experiment tests whether end-effector observation age contributes to the
moving-target failures. It compares a current camera view, a fully delayed RGB
view, and a delayed target observation paired with the current joint-state
projection of the arm tip. The last mode is a vision/proprioception hybrid; it
does not represent fully delayed vision. Motion prediction and stale-target
arrival confirmation are disabled for all modes.

The default run covers sinusoidal and piecewise-linear targets, 0 and 2 px
target noise, and 0, 0.4, and 0.8 simulated seconds of camera delay. Each row
uses five seeds and 20 targets per seed. Delayed modes are paired with each
other and with the same profile/noise fresh-camera reference. The report
includes paired error deltas with seed-cluster bootstrap intervals, physical
success, stop reasons, camera age at the terminal observation, and fresh visual
confirmation counts. Physical success remains the final MuJoCo distance at or
below 4.5 cm. A `pixel_tolerance_reached` stop counts as fresh visual
confirmation only when the consumed target measurement has zero camera-age
observations and zero simulated-time age; delayed-frame tolerance stops do not
count as fresh confirmation. The two outcomes are reported separately.

The measurement-noise generator restarts from the same seed for each mode.
When different arm trajectories cause different natural occlusions, later
noise draws may no longer correspond to identical physical timestamps. Trials
are paired by seed and episode, with bootstrap resampling over seed groups; the
report records this simulator limitation rather than claiming identical
sensor noise after trajectories diverge.

The first run uses seeds 0–4; a second run with seeds 100–104 is held out from
the earlier dynamic and prediction experiments. Each reports 100 trials per
mode/condition. The table gives physical successes for fully delayed RGB →
delayed target/current-joint-state tip, followed by the held-out paired median
error difference (hybrid minus fully delayed) and its seed-cluster bootstrap
interval.

| Target path | Noise | Delay | Seeds 0–4 success | Seeds 100–104 success | Held-out paired median error Δ (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sinusoidal | 0 px | 0.4 s | 10% → 85% | 4% → 84% | -2.46 cm [-2.70, -2.31] |
| Sinusoidal | 0 px | 0.8 s | 68% → 85% | 63% → 81% | -1.54 cm [-1.85, -0.58] |
| Sinusoidal | 2 px | 0.4 s | 15% → 81% | 11% → 68% | -1.94 cm [-2.29, -1.61] |
| Sinusoidal | 2 px | 0.8 s | 66% → 85% | 56% → 85% | -1.65 cm [-2.07, -0.95] |
| Piecewise linear | 0 px | 0.4 s | 46% → 98% | 34% → 88% | -2.52 cm [-2.62, -2.42] |
| Piecewise linear | 0 px | 0.8 s | 69% → 92% | 58% → 93% | -2.11 cm [-2.42, -1.85] |
| Piecewise linear | 2 px | 0.4 s | 40% → 88% | 37% → 83% | -2.27 cm [-2.60, -1.47] |
| Piecewise linear | 2 px | 0.8 s | 65% → 90% | 56% → 95% | -2.17 cm [-2.45, -1.90] |

Current-joint-state tip projection improved physical success in all eight
settings on both seed groups. The paired median error differences were lower
in all held-out settings. The bootstrap resamples only five seed groups, so
the intervals summarize these runs and do not establish broad generalization.
Held-out P95 errors for the hybrid ranged from 4.92 to 6.71 cm. It is a
promising opt-in hybrid for physical reaching; it does not provide current-time
visual confirmation under delayed target frames. In delayed modes, current-time
visual confirmations were 0/100 even when the arm met the physical success
threshold. At zero delay, physical success was 93–100/100 across the two seed
groups, while current-time visual confirmations were 37–53/100. These separate
counts retain the project's distinction between reaching and visual task
completion.
