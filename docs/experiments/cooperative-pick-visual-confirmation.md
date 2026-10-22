# Fresh visual confirmation after the cooperative pick

[Project overview](../../README.md) · [Camera-guided pick experiment](camera-guided-cooperative-pick.md) · [Video gallery](../video-gallery.md)

## Question and scope

The camera-guided three-arm planner detects the target from one initial
overhead image, then executes a planned contact trajectory. Physical lifting
does not itself provide fresh visual evidence. This experiment adds an optional
fixed side-camera view and asks a narrower question: do two fresh images show a
payload fiducial that is raised and visually stable at those two sample times?

The verifier is image-only. MuJoCo height, tilt, contacts, and collision state
are logged afterward as evaluation labels. The two image samples do not certify
continuous holding, release quality, or full-cycle collision safety.

![Initial image planning, fresh side-camera decision, and separate post-hoc physics evaluation](../../assets/demos/camera_guided_cooperative_pick_verification_pipeline.svg)

## Implementation

The scene adds a fixed `verification_side` camera. A magenta box is attached to
the free payload as a visual fiducial; it has zero collision type, zero
collision affinity, and zero density, so it does not change contact geometry
or payload mass. The camera takes an initial frame at 0.0 s and fresh frames at
8.4 s and 8.8 s during the 1.6 s hold.

[`cooperative_pick_visual_verification.py`](../../src/visual_servo_mujoco/cooperative_pick_visual_verification.py)
segments the magenta component, measures its center and principal-axis angle,
and returns `confirmed`, `not_confirmed`, or `abstain`. The classifier receives
only the initial side RGB and the two fresh side RGB frames. It does not accept
simulator state.

The held-out benchmark compares four frozen rules on the exact same captured
images:

| Version | Image rule |
| --- | --- |
| V0 | At least 35 px upward displacement in both frames; at most 12 px drift. |
| V1 | V0 plus at most 12° fiducial rotation relative to the initial frame. |
| V2 | V1 plus at least 50 px upward displacement. |
| V3 | V2 with a tighter 4° rotation limit. This is the selected rule for the demo, not a claim that it is superior on every condition. |

Threshold development preserved two distinct failures. On the first held-out
V0 split, two no-grasp trials were falsely confirmed while the payload was
badly tilted. The orientation gate rejected those examples. The seed-39200
development split then exposed an under-lift false confirmation with little
visible tilt; raising the rise threshold to 50 px rejected it. A later V2
validation split exposed a 5.23° payload tilt that passed the 12° image-angle
rule, motivating V3's tighter 4° limit. The archived results are retained in
the [V0 report](../../results/cooperative-pick-visual-confirmation-v0-heldout/cooperative-pick-visual-confirmation-summary.json),
[seed-39200 development report](../../results/cooperative-pick-visual-confirmation-development/cooperative-pick-visual-confirmation-summary.json),
and [V2 held-out report](../../results/cooperative-pick-visual-confirmation-v2-heldout/cooperative-pick-visual-confirmation-summary.json).

## Independent V3 validation

The thresholds were frozen before base seed 39400. The held-out run uses three
seeds with 20 randomized target positions each, for 60 matched poses and 120
simulations. Every pose is replayed with a 5 mm baseline approach and the
opt-in 20 mm approach; both use 16 mm pads and the same clearance release
profile. The 20 mm approach remains experimental.

| Measure | 5 mm baseline | 20 mm experimental |
| --- | ---: | ---: |
| V3 capture-time physical labels positive / negative | 53 / 7 | 59 / 1 |
| V3 true positive / false confirmation | 53 / 0 | 59 / 0 |
| V3 true negative / false non-confirmation | 7 / 0 | 1 / 0 |
| Abstention / coverage | 0 / 100% | 0 / 100% |
| Payload stable through the 1.6 s hold | 53 / 60 | 59 / 60 |
| Strict full-sequence physical success | 52 / 60 | 59 / 60 |

All four rules produced the same decisions on this held-out set: 112 true
positives, 8 true negatives, no false confirmations, no false
non-confirmations, and no abstentions. This is a clean result on the sampled
poses, but it does not show an incremental advantage for V3 over V0. There were
only eight negative capture-time labels, so the absence of a false confirmation
does not establish a low false-confirmation rate under new camera conditions.
For the aggregate 8/8 negative class, the Wilson 95% interval for specificity
is approximately 68%-100%.

The separate matched physical comparison had 8 strict-success poses won only
by the 20 mm condition and 1 won only by the 5 mm baseline; both conditions
passed 51 poses. For the 1.6 s hold, the paired counts were 7 candidate-only,
1 baseline-only, and 52 both-success. This strengthens the prior simulated
approach-clearance evidence, but the strict Wilson intervals still overlap
narrowly (5 mm: 76%-93%; 20 mm: 91%-100%). Keep 20 mm opt-in.

The [V3 scorecard](../../results/cooperative-pick-visual-confirmation/cooperative-pick-visual-confirmation-scorecard.svg),
[summary JSON](../../results/cooperative-pick-visual-confirmation/cooperative-pick-visual-confirmation-summary.json),
and [120-row paired trial CSV](../../results/cooperative-pick-visual-confirmation/cooperative-pick-visual-confirmation-trials.csv)
preserve the exact outcomes. The CSV includes the V0/V1/V2/V3 decision for
each image pair alongside capture-time truth, hold success, full-sequence
collision safety, and failure reason.

## Demos

The main clip shows the one-shot overhead plan, contact-based lift, two fresh
side-camera frames, image-only V3 decision, and separate post-hoc physics
labels. Two selected replays preserve the earlier false confirmations and
show how later rules classify those same images. The selected poses remain
counted only in their original data splits.

| Demo | Evidence |
| --- | --- |
| [Camera-guided pick with fresh side-camera end card](../../assets/demos/camera_guided_cooperative_pick.mp4) | [GIF preview](../../assets/demos/camera_guided_cooperative_pick.gif) |
| [Tilted no-grasp false-confirmation replay](../../assets/demos/cooperative_pick_verification_false_tilt.mp4) | [GIF preview](../../assets/demos/cooperative_pick_verification_false_tilt.gif) |
| [Under-lift false-confirmation replay](../../assets/demos/cooperative_pick_verification_false_underlift.mp4) | [GIF preview](../../assets/demos/cooperative_pick_verification_false_underlift.gif) |
| [Replay metadata](../../assets/demos/cooperative_pick_visual_verification_failure_demos.json) | Original pair IDs, V0-V3 decisions, and separate physical evaluation. |

## Reproduction

```bash
uv run --locked visual-servo-cooperative-pick-visual-validation \
  --base-seed 39400 --seeds 3 --samples-per-seed 20
uv run --locked --extra demo python scripts/generate_camera_guided_cooperative_pick_demo.py
uv run --locked --extra demo python scripts/generate_cooperative_pick_visual_verification_demos.py
```

The classifier and scene checks are covered by
[`test_cooperative_pick_visual_verification.py`](../../tests/test_cooperative_pick_visual_verification.py)
and [`test_camera_guided_cooperative_pick.py`](../../tests/test_camera_guided_cooperative_pick.py).

## Limits and next experiment

- The side camera is a fixed, ideal MuJoCo render. No physical camera, lens
  distortion, exposure variation, motion blur, real occluder, or camera movement
  was tested.
- The payload carries an artificial color fiducial. This does not establish
  object recognition or tracking from natural appearance.
- Two frames only label raised-and-stable at the sample times. They cannot
  prove continuous holding or safe release.
- The approach comparison is simulation evidence. The 20 mm setting remains
  opt-in despite its 8-to-1 paired strict-success result on this split.
- V0-V3 were identical on seed 39400. More deliberately difficult negatives
  are needed before claiming the tighter gates improve generalization.

The next useful test is to preserve these thresholds and introduce held-out
image faults specific to the side view: partial fiducial occlusion, exposure
changes, camera translation/rotation, and blur between samples. Record image
abstention and false confirmation separately from MuJoCo hold and collision
outcomes. Do not retune on the new validation split.
