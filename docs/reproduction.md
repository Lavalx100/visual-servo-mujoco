# Reproduction and demo commands

[Project overview](../README.md)

## Run

```bash
uv sync --locked --extra dev --extra demo
uv run --locked pytest
uv run visual-servo-demo --episodes 20 --seed 7
uv run visual-servo-demo --controller image_feedback --episodes 20 --seed 7
uv run visual-servo-benchmark --seeds 5 --episodes-per-seed 20
uv run --locked visual-servo-cooperative-pick
uv run --locked visual-servo-camera-guided-cooperative-pick
uv run --locked visual-servo-camera-guided-cooperative-pick \
  --base-seed 38100 --finger-pad-half-size-m 0.014 --no-media \
  --results-dir results/cooperative-pick-pad-014-heldout-control
uv run --locked python scripts/compare_cooperative_pick_pad_sensitivity.py \
  --baseline-dir results/cooperative-pick-pad-014-heldout-control \
  --candidate-dir results \
  --json-output results/cooperative-pick-pad-size-comparison.json \
  --svg-output results/cooperative-pick-pad-size-comparison.svg
uv run --locked visual-servo-camera-guided-cooperative-pick \
  --base-seed 38200 --release-profile baseline --no-media \
  --results-dir results/cooperative-pick-release-heldout-baseline
uv run --locked visual-servo-camera-guided-cooperative-pick \
  --base-seed 38200 --release-profile clearance --no-media \
  --results-dir results/cooperative-pick-release-heldout-clearance
uv run --locked python scripts/compare_cooperative_pick_release_profiles.py \
  --baseline-dir results/cooperative-pick-release-heldout-baseline \
  --candidate-dir results/cooperative-pick-release-heldout-clearance \
  --json-output results/cooperative-pick-release-heldout-comparison.json \
  --svg-output results/cooperative-pick-release-heldout-comparison.svg \
  --paired-csv-output results/cooperative-pick-release-heldout-paired.csv
uv run --locked python scripts/generate_cooperative_pick_release_demos.py \
  --baseline-dir results/cooperative-pick-release-heldout-baseline \
  --candidate-dir results/cooperative-pick-release-heldout-clearance \
  --pair-id 38201-005 --output-dir assets/demos
uv run --locked visual-servo-camera-guided-cooperative-pick \
  --base-seed 38300 --release-profile clearance \
  --lowest-approach-tool-height-offset-m 0.005 --no-media \
  --results-dir results/cooperative-pick-approach-clearance-heldout-baseline
uv run --locked visual-servo-camera-guided-cooperative-pick \
  --base-seed 38300 --release-profile clearance \
  --lowest-approach-tool-height-offset-m 0.020 --no-media \
  --results-dir results/cooperative-pick-approach-clearance-heldout-candidate
uv run --locked python scripts/compare_cooperative_pick_geometry.py \
  --baseline-dir results/cooperative-pick-approach-clearance-heldout-baseline \
  --candidate-dir results/cooperative-pick-approach-clearance-heldout-candidate \
  --parameter lowest_approach_tool_height_offset_m \
  --json-output results/cooperative-pick-approach-clearance-comparison.json \
  --svg-output results/cooperative-pick-approach-clearance-comparison.svg \
  --paired-csv-output results/cooperative-pick-approach-clearance-paired.csv
uv run --locked --extra demo python scripts/generate_cooperative_pick_approach_clearance_demos.py \
  --baseline-dir results/cooperative-pick-approach-clearance-heldout-baseline \
  --candidate-dir results/cooperative-pick-approach-clearance-heldout-candidate
uv run --locked --extra demo python scripts/generate_camera_guided_cooperative_pick_demo.py
uv run --locked visual-servo-cooperative-pick-visual-validation \
  --base-seed 39400 --seeds 3 --samples-per-seed 20
uv run --locked --extra demo python scripts/generate_cooperative_pick_visual_verification_demos.py
uv run --locked visual-servo-cooperative-pick-visual-fault-validation \
  --base-seed 39600 --seeds 3 --samples-per-seed 20
uv run --locked --extra demo python scripts/generate_cooperative_pick_visual_fault_demos.py
uv run --locked visual-servo-cooperative-pick-occlusion-validation \
  --stage development --base-seed 39700 --seeds 3 --samples-per-seed 10
uv run --locked visual-servo-cooperative-pick-occlusion-validation \
  --stage heldout --base-seed 39800 --seeds 3 --samples-per-seed 20 \
  --development-base-seed 39700 --selected-gap-tolerance 28
uv run --locked --extra demo python scripts/generate_cooperative_pick_occlusion_demos.py
uv run --locked visual-servo-cooperative-pick-orthogonal-validation
uv run --locked --extra demo python scripts/generate_cooperative_pick_orthogonal_demos.py
uv run --locked visual-servo-cooperative-pick-material-validation
uv run --locked visual-servo-cooperative-pick-pad-friction-validation
uv run --locked visual-servo-cooperative-pick-contact-diagnostics
uv run --locked visual-servo-cooperative-pick-retention-diagnostics
uv run --locked python -m visual_servo_mujoco.evidence_audit
uv run --locked python scripts/optimize_demo_media.py --untracked-only
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

The scripted three-arm command writes three synchronized camera views, a GIF
preview, and a JSON run record to `assets/demos/`. Its elbows follow a shared
trajectory and its free payload is lifted through finger contact and friction.
The separate camera-guided command connects the existing default red detector
and pinhole projection to the same contact scene, then writes paired randomized
camera-estimate/oracle trials to `results/`, a scorecard, a camera-plus-physics
video, and a GIF preview. The target is detected once for planning; the camera
does not provide feedback during the lift. The extra commands replay the exact
30 held-out target poses with the 14 mm pad baseline and regenerate a paired
pad-size comparison against the default 16 mm result. See the
[scripted experiment](experiments/cooperative-pick.md) and the
[camera-guided extension](experiments/camera-guided-cooperative-pick.md) for
their separate protocols and limits. The optional clearance release profile
is an experimental actuator trajectory; the baseline remains the default.
The camera-guided demo command rerenders the overhead view at each recorded
step for display and writes a JSON sidecar; this does not add image feedback to
the controller.
The opt-in occlusion study evaluates V3 and three gap tolerances on the same
clean and half-occluded side-camera frames from development seed 39700. The
held-out command uses the gap tolerance frozen by development (28 px in the
saved run); it does not tune on seed 39800. The two video demos replay one
development pose already counted in the CSV. V3 remains the runtime default.
Its reproduction commands above use the same 30 seed-38200 poses for both
profiles, then render a held-out paired example already counted in those data.
The approach-height commands compare exact paired poses while preserving the
clearance release profile in both arms. The seed-38300 30-pose block showed an
oracle regression; a separate seed-38400 block evaluates 60 new poses and
improves both camera-estimate and oracle scores. The candidate's 20 mm offset
is still opt-in because the samples are simulated and confidence intervals
overlap; the 5 mm default remains unchanged.

The material stress command uses 12 shared target poses across three explicit
payload-body masses and three pad/payload sliding-friction settings, then runs
both the camera-estimate and oracle target controllers. It writes a 216-row
CSV, summary JSON, scorecard, and matched nominal/stress MP4 and GIF demos.
The command's illustrative video replays are not extra trials. The nominal
mass keeps the original geometry-derived inertial path to preserve the
established contact trajectory; off-nominal masses use explicit body
inertials with zero-density decorative trim. Values are synthetic stresses,
and the experiment does not measure fresh visual confirmation. Its full settings,
results, and limitations are in the
[material robustness experiment](experiments/cooperative-pick-material-robustness.md).

The pad/friction command compares 16 mm and 18 mm pad half-sizes on the same
12 new poses under low and high synthetic friction, with nominal payload mass,
approach clearance, and release motion fixed. It writes paired trial rows, a
scorecard, four single-condition replay videos, and a four-panel MP4/GIF
preview. The 18 mm candidate did not improve the low-friction hold score; its
legacy failure label does not establish whether the fingers touched the object
during closing. See the [pad/friction experiment](experiments/cooperative-pick-pad-friction-validation.md).

The contact-diagnostics command follows six named pad colliders at each physics
step and records contact by grasp phase plus slide-joint position tracking
during closure. On a new 12-pose block, low-friction trials made contact during
closing and early lift but none retained pad contact into the loaded hold. The
scorecard keeps these phases separate; force outputs are MuJoCo solver values,
not real sensor readings. See the
[contact diagnostic](experiments/cooperative-pick-contact-diagnostics.md).

The retention-diagnostics command adds 20 ms samples of pad normal/tangential
contact forces, each finger's commanded-versus-measured slide position, and
payload motion in each gripper-hand frame during the cooperative lift and
loaded hold. It runs 12 additional paired poses over two friction settings and
both target controllers, writes trial and per-pad/phase CSVs, and rerenders an
instrumented low/high-friction replay. The replay is an example from an already
scored pose. This experiment describes simulated mechanics, not sensor forces
or hardware reliability; see the [retention diagnosis](experiments/cooperative-pick-retention-diagnostics.md).

## Frozen occlusion-geometry challenge

The expanded V4 study runs one physical trajectory per target pose, then
evaluates clean RGB plus eight marker-mask shapes and placements from the same
captured side-camera frames. It uses disjoint seed blocks for nominal 16 mm
pads and a 14 mm hard-negative geometry. The detector tolerance is fixed at
the prior development-selected 28 px value throughout.

```bash
uv run --locked visual-servo-cooperative-pick-occlusion-challenge
uv run --locked --extra demo python scripts/generate_cooperative_pick_occlusion_challenge_demos.py
uv run --locked python -m visual_servo_mujoco.evidence_audit
```

The demo script replays selected challenge rows and checks that its V3, V4,
and capture-time simulator decisions match the saved trial table. Those clips
are illustrative replays and do not add benchmark trials.
