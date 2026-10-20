# Reproduction and demo commands

[Project overview](../README.md)

## Run

```bash
uv sync --locked --extra dev --extra demo
uv run --locked pytest
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
uv run visual-servo-sensor-timing-benchmark --output artifacts/sensor-timing-benchmark.json
uv run visual-servo-sensor-timing-benchmark --base-seed 100 \
  --output artifacts/sensor-timing-benchmark-heldout.json
uv run visual-servo-verification-benchmark --output artifacts/verification-view-benchmark.json
uv run visual-servo-verification-benchmark --base-seed 100 \
  --output artifacts/verification-view-benchmark-heldout.json
uv run --locked --extra demo visual-servo-perception-appearance-benchmark
uv run --locked visual-servo-perception-candidate-validation \
  --split-name validation --seeds 5 --samples-per-seed 10 --base-seed 9400
```

The candidate-detector command evaluates the unchanged default and an
experimental circle prior on the same rendered frames. It writes paired trial
and summary data plus an SVG scorecard, MP4, and GIF. Its held-out lighting,
distractor size/distance, and occlusion-width ranges are disjoint from the
development ranges. See the [candidate validation report](experiments/perception-candidate-validation.md)
before interpreting its high abstention rate or remaining false selections.

The command writes `artifacts/results.json`, `artifacts/camera_view.png`, and
`artifacts/reaching_demo.mp4` (when the local OpenCV build supports MP4 output).
The JSON records every target's camera pixel, estimated table position, final
end-effector error, and success. Ground-truth target coordinates are included
only for evaluation; the controller estimates the target from the rendered
image.
