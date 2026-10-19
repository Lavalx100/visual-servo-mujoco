# Motion prediction experiments

[Project overview](../../README.md)

## Target-motion prediction experiment

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
| Moving, 0.4 s | 2.70 cm | 6.48 cm | 2.26 cm | 85% / 10% / 77% |
| Moving, 0.8 s | 2.60 cm | 6.37 cm | 8.13 cm | 85% / 5% / 20% |

The constant-velocity fit performs poorly when the sinusoidal target reverses
direction within the prediction horizon. At 0.4 s delay, the acceleration fit
improves median error (2.26 cm vs. 2.70 cm baseline) but has a worse P95
(8.61 cm vs. 5.20 cm) and lower success (77% vs. 85%). Both predictors degrade
success at 0.8 s. They remain experimental and off by default. The
[three-way demo](../../assets/demos/feedback_target_prediction.mp4)
shows one seed, not an aggregate result. The experiment below adds reversal
detection and evaluates it on both smooth and piecewise-linear target paths.
Grasping, contact-rich manipulation, and real-robot performance remain future
work.

## Reversal-aware prediction benchmark

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
