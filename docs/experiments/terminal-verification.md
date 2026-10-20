# Terminal verification experiments

[Project overview](../../README.md)

## Independent-view terminal verification experiment

Natural target loss motivates testing a second camera view: 91 of 100 clean
default trials ended after the overhead camera lost the target, even though
all 100 were physically within 4.5 cm. The verification benchmark leaves that
controller unchanged, renders four oblique views at terminal arm poses,
detects the red target and green tip, back-projects their pixels to the known
target and tip heights, and compares the estimated distance with MuJoCo ground
truth. It runs clean and 20 px controller-noise conditions on seeds 0–4, then
repeats them on held-out seeds 100–104. The 20 px noise is applied to the
controller's target measurements; verification camera images are rendered
directly from the simulator. The east-view rows now exercise the same runtime
`--verify-terminal` branch shown in the demo; north, west, and south remain
offline camera candidates.

The table pools both seed groups (200 trials per noise level). “False
success” means the camera estimate was within the 4.5 cm physical threshold
while MuJoCo ground truth was outside it. “Missed success” means the reverse.
Rows with a missing marker cannot make a visual estimate.

| View | Noise | Both markers visible | False success | Missed success | P95 absolute distance-estimate error |
| --- | ---: | ---: | ---: | ---: | ---: |
| East oblique | 0 px | 200/200 | 0 | 1 | 2.27 cm |
| East oblique | 20 px | 198/200 | 0 | 14 | 1.71 cm |
| North oblique | 0 px | 200/200 | 0 | 0 | 1.87 cm |
| North oblique | 20 px | 200/200 | 6 | 11 | 2.28 cm |

East is the conservative candidate in these runs: it produced no false
success decisions across either noise level or seed group. Under 20 px control
noise, it failed to recognize 14 physically successful trials, so this result
does not establish a production-ready stop rule. The north view had lower
clean-condition estimation error but labeled six physically failed high-noise
trials as successful. This is a simulator camera experiment, not evidence of
real-camera calibration or illumination robustness.

The [clean two-view demo](../../assets/demos/verification_view_clean.mp4) and
[20 px-noise two-view demo](../../assets/demos/verification_view_20px-noise.mp4)
show current-time renders at feedback updates and hold the terminal state. The
control policy continues to use only its overhead camera. The additional view
is a separate terminal observation; it does not count toward current
controller confirmation.

The optional `--verify-terminal` mode now renders the calibrated east camera
after the reaching controller has stopped. It segments both markers,
back-projects each pixel to the marker's known height, and records the
estimated distance, visibility, camera age, and visual threshold estimate.
The report keeps the existing MuJoCo physical-reach score in `success` and
stores the camera result under `terminal_visual_verification`. Missing markers
make the visual estimate unavailable. The verifier never changes the motion,
stop reason, or physical score; it is off by default. Run one example with:

```bash
uv run visual-servo-demo --controller image_feedback --episodes 1 --seed 7 \
  --verify-terminal --output-dir artifacts/terminal-verification-demo
```

This run writes a terminal two-view video alongside the usual control demo.
The verification image is a current simulator render. The 20 px condition
perturbs controller target measurements only; the stress options below alter
only the separate terminal verifier.

## Terminal verifier calibration and image-stress benchmark

The verifier robustness benchmark pairs the same 100 held-out reaching trials
(seeds 100–104, 20 targets per seed) across nine one-factor conditions. The
reaching controller receives 20 px target-measurement noise. `False +` counts
visual success estimates on physically failed trials; `Missed` counts visual
failure estimates on physically successful trials. Counts exclude trials
where either marker is unavailable. Physical reaching succeeded on 85/100
trials in every condition. The stress settings did not change any controller
outcome.

| Verifier condition | Markers visible / available | False + | Missed | P95 absolute distance error |
| --- | ---: | ---: | ---: | ---: |
| Nominal | 99/100 | 0 | 7 | 1.74 cm |
| Estimated FOV +5° | 99/100 | 0 | 15 | 2.12 cm |
| Estimated FOV −5° | 99/100 | 1 | 3 | 1.66 cm |
| Estimated camera world X +2 cm | 99/100 | 0 | 7 | 1.74 cm |
| Estimated camera yaw +3° | 99/100 | 0 | 5 | 1.84 cm |
| Uniform image exposure ×0.25 | 0/100 | — | — | unavailable |
| Uniform image exposure ×1.5 | 99/100 | 0 | 10 | 2.52 cm |
| Target marker occluded | 0/100 | — | — | unavailable |
| End-effector marker occluded | 0/100 | — | — | unavailable |

The ±5° FOV cases show asymmetric classification risk: +5° raised missed
successes from 7 to 15, while −5° produced one false success. The camera X
offset changed neither pooled decisions nor rounded P95 error in this setup.
A shared translation may cancel when the verifier compares two reconstructed
points, but that should not be generalized to other camera poses or marker
heights. The bright exposure condition increased the P95 distance error. When
the image was dimmed or a marker was occluded, the verifier withheld all
estimates. Those unavailable cases have no false/missed counts; they are not
scored as correct decisions.

### Marker-quality threshold sweep

The terminal verifier now reports a per-marker quality score from normalized
contour area, circularity, median saturation, and median brightness. The
combined score is the lower score of the target and end-effector markers. It
is a unitless heuristic, not a probability, and it cannot diagnose incorrect
camera calibration from marker appearance alone. A positive
`--terminal-verification-min-quality-score` makes the opt-in verifier abstain
below that threshold. The default is 0.0, which preserves the estimate
behavior.

| Minimum score | Nominal available | False + | Missed | FOV −5° available | False + | Missed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 99/100 | 0 | 7 | 99/100 | 1 | 3 |
| 0.15 | 93/100 | 0 | 7 | 93/100 | 1 | 3 |
| 0.20 | 39/100 | 0 | 5 | 39/100 | 1 | 3 |
| 0.50 | 9/100 | 0 | 1 | 9/100 | 0 | 1 |

The 0.50 gate removed the one false-success decision in the FOV −5° condition,
but it left only 9% of estimates available. The lower gates kept that false
success. The score exposes a coverage tradeoff but does not yet provide a
useful safety setting. Fully dimmed and fully occluded images remain
unavailable at every threshold. These curves are exploratory: thresholds were
applied retrospectively to the same held-out trials, and none is selected as a
default.

The paired [false-success](../../assets/demos/terminal_verification_quality_false_positive.mp4)
and [abstention](../../assets/demos/terminal_verification_quality_abstention.mp4)
clips use seed 101, episode 16, the −5° FOV estimate, and the same reaching
trajectory. Run the opt-in gate manually with:

```bash
uv run visual-servo-demo --controller image_feedback --episodes 1 --seed 7 \
  --verify-terminal --terminal-verification-min-quality-score 0.50
```

When the gate abstains, `estimated_distance_m` and
`visual_success_estimate` are null. The report retains a clearly named
evaluation-only candidate distance for the offline threshold sweep; it is not
used as a verifier decision.

This is controlled simulator evidence, not real-camera robustness evidence.
Exposure is a uniform post-render brightness multiplier, not a lighting or
sensor model; marker occlusion removes marker-colored pixels and inpaints the
region. Camera offsets are applied to estimator calibration while rendering
remains nominal. The benchmark therefore isolates verifier sensitivity and
does not measure real calibration drift, shadows, blur, clipping, or partial
occlusion. The verifier remains opt-in and cannot change the physical success
score or control stop.

Reproduce the held-out report and all eight illustrative condition videos:

```bash
uv run visual-servo-verification-robustness-benchmark --seeds 5 \
  --episodes-per-seed 20 --base-seed 100 \
  --output artifacts/verification-robustness-benchmark-heldout.json
uv run python scripts/generate_verification_stress_demos.py
uv run python scripts/generate_verification_quality_demos.py
```

The JSON is retained at
`artifacts/verification-robustness-benchmark-heldout.json`. Each stress video
uses seed 7 and demonstrates one case; the 100-trial table above is the
aggregate held-out evidence.

### Seed-separated threshold calibration and validation

The calibration split uses seeds 200–204 (100 target trials), repeated across
the same nine paired camera/image conditions. It chooses the highest-coverage
threshold with zero observed false visual success decisions; a gate must also
retain at least 50% of candidate estimates to count as usable. The selected
0.60 gate retained 21/595 calibration candidates (3.5%), so it failed that
coverage criterion.

The frozen threshold was then evaluated without retuning on seeds 300–304
(another 100 target trials, paired across nine conditions):

| Validation setting | Accepted / candidate estimates | Candidate coverage | False visual successes | Missed success estimates |
| --- | ---: | ---: | ---: | ---: |
| Ungated baseline (0.00) | 600/600 | 100% | 10 | 48 |
| Frozen threshold (0.60) | 23/600 | 3.8% | 0 | 3 |

All 23 accepted frozen-gate estimates came from the bright-exposure condition;
the gate accepted none in nominal, FOV-shifted, or camera-pose-shifted images.
The 23/600 figure is coverage among images where both markers already produced
a candidate estimate; it is 23/900 (2.6%) across all validation trials. Dim and
occluded conditions had no candidate estimates before applying the gate. The
controller outcomes matched the nominal controller in both seed blocks.

This is a negative result: the threshold removed all observed false visual
success decisions on this split, but it abstained on 96.2% of otherwise
candidate estimates and behaved as a brightness filter. The 10 false successes
and 48 missed estimates in the ungated row are counts across paired conditions;
each target episode is repeated across conditions, so those counts are not 900
independent target trials. Zero observed false successes is not a calibrated
risk guarantee. The gate remains opt-in and no threshold is recommended.

Reproduce both seed blocks and generate the paired held-out clips with:

```bash
uv run visual-servo-verification-quality-validation --seeds 5 \
  --episodes-per-seed 20 --calibration-base-seed 200 \
  --validation-base-seed 300 \
  --output artifacts/verification-quality-calibration-validation.json
uv run python scripts/generate_verification_quality_validation_demos.py
```

The report stores the component-level marker features and simulator labels for
both splits. The labels are MuJoCo physical outcomes, while the verifier
estimates distance from a fresh rendered RGB view. This is controlled
simulation evidence, not a recorded-camera dataset.

### Fixed-landmark camera calibration-health experiment

The appearance-only threshold did not diagnose camera-model bias, so this
separate diagnostic adds four fixed magenta landmarks at known tabletop
coordinates. It matches their detected image centers to pinhole projections
from the nominal camera pose and field of view, then measures reprojection
residual. Here the rendering camera itself is perturbed while the diagnostic
continues to use nominal calibration. This is different from the earlier
verifier stress benchmark, which rendered nominal images and perturbed only
the estimator's camera parameters. No reaching controller runs in this test.

| Render condition | Detected | Reprojection RMS | Diagnostic result |
| --- | ---: | ---: | --- |
| Nominal camera | 4/4 | 0.51 px | Healthy |
| FOV +5° | 4/4 | 20.23 px | Drift detected |
| FOV −5° | 4/4 | 24.20 px | Drift detected |
| Camera world X +2 cm | 4/4 | 6.84 px | Drift detected |
| Camera yaw +3° | 4/4 | 21.43 px | Drift detected |
| Combined FOV, position, yaw shift | 4/4 | 32.02 px | Drift detected |
| Bright exposure ×1.5 | 4/4 | 0.45 px | Healthy |
| Dim exposure ×0.25 | 0/4 | unavailable | Unavailable |

The exploratory alarm is 3 px RMS: all five rendered camera shifts exceeded it,
while nominal and bright exposure stayed below it. Dim exposure made the color
landmarks unavailable, so the diagnostic reports no calibration decision. This
first run did not calibrate the threshold on an independent camera dataset.
The scene uses ideal pinhole projections and simulator renders; it does not
model lens distortion or real-world landmark localization. It diagnoses a
camera-to-model mismatch in this controlled setup and does not update the
calibration or alter reaching.

Reproduce the report and paired visual demo with:

```bash
uv run visual-servo-camera-calibration-health \
  --output artifacts/camera-calibration-health.json
uv run python scripts/generate_camera_calibration_health_demo.py
```

The demo shows nominal and combined-drift images side by side. White crosses
mark expected projections, green rings mark detected centers, and orange lines
visualize residuals. The JSON retains per-landmark errors and all eight
conditions.

### Frozen threshold calibration and held-out validation

The fixed 3 px threshold was still an arbitrary margin. This follow-up uses
seeds 400–404 to tune on 100 nominal calibration images and 300 labelled camera
shift images, then freezes the selected threshold. Seeds 500–504 provide 100
nominal validation images (including five dim-image availability probes) and
300 fresh camera-shift images. The validation shifts use new parameter draws from the
same three severity ranges as calibration. Validation image nuisance is
broader: exposure ×0.55–1.45, RGB noise 0–8 intensity levels, and blur kernels
1, 3, or 5 pixels. Calibration uses exposure ×0.70–1.30, noise 0–4, and blur
1 or 3.

| Threshold policy | Frozen threshold | Nominal false alarms | Subtle drift detected | Moderate drift detected | Severe drift detected |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nominal-only 95th percentile | 0.573 px | 7/95 (7.4%) | 100/100 | 100/100 | 100/100 |
| Supervised, max detection at ≤5% calibration false alarms | 0.841 px | 0/95 | 100/100 | 100/100 | 100/100 |
| Earlier fixed exploratory threshold | 3.000 px | 0/95 | 60/100 | 93/100 | 100/100 |

The nominal-only quantile met its 5% false-alarm target on calibration
(4/100), but exceeded it on validation. The supervised rule used labelled
camera shifts in the calibration split and selected the threshold with the
highest drift detection subject to at most 5% nominal calibration false
alarms. It selected 0.841 px, with 0/100 calibration false alarms and 300/300
calibration shifts detected. On held-out seeds, it flagged no nominal image
with visible landmarks and detected all 300 shifted-camera images. All five
dim probes were unavailable rather than reported as healthy.

This is a promising controlled result, not a real-camera guarantee. The split
holds out random seeds and image nuisance draws, while reusing the same scene,
landmark layout, camera-shift types, and severity ranges. The 300 camera-shift
frames per split are repeated renders grouped under five seeds, not 300
independent physical cameras. The supervised rule also depends on representative
labelled camera-shift examples. No reaching controller runs here, and these
health labels do not certify physical reaching or fresh visual confirmation.

Reproduce the split report and validation demo with:

```bash
uv run visual-servo-camera-calibration-validation --seeds 5 \
  --samples-per-seed 20 --calibration-base-seed 400 \
  --validation-base-seed 500 --max-false-alarm-rate 0.05 \
  --output artifacts/camera-calibration-health-validation.json
uv run python scripts/generate_camera_calibration_validation_demo.py
```

The demo replays one held-out nominal image under the nominal-only and frozen
supervised thresholds, then shows representative subtle drift, severe drift,
and dim-image abstention cases. Its pixel residuals and per-seed summaries are
in the ignored local JSON report.

### Frozen threshold transfer to camera poses and lens distortion

The 0.8409987 px threshold selected on East-view seeds 400–404 is now frozen
across a separate seed block, 600–604. Five synthetic camera poses cover the
East reference, a lower West view, a higher North view, a diagonal view, and a
low grazing East view. Each pose has 95 visible nominal images, five dim
availability probes, and 100 trials at each of three camera-shift severities.
The same five views also receive 25 repeats at each of eight OpenCV radial
`k1` coefficients from −0.04 to +0.04. This makes 3,000 evaluation images.
All use the same MuJoCo scene and fixed four-landmark layout; they are not five
independent physical cameras.

The frozen threshold transfers well to most tested views for camera shifts, but
its nominal false-alarm and availability behavior is pose-dependent:

| Held-out pose | Nominal frozen alarms | Subtle / moderate / severe shift alarms (available trials) | Landmark unavailability |
| --- | ---: | ---: | ---: |
| East reference | 0/95 | 100/100 / 100/100 / 100/100 | 0/300 shifted trials |
| West lower | 2/95 | 100/100 / 99/99 / 82/82 | 19/300 shifted trials unavailable |
| North higher | 0/95 | 99/100 / 100/100 / 100/100 | 0/300 shifted trials unavailable |
| Southwest diagonal | 0/95 | 100/100 / 100/100 / 99/99 | 1/300 shifted trials unavailable |
| East grazing | unavailable 95/95 | unavailable 300/300 | expected landmark centers fall outside the safe image margin |

The low grazing image originally produced a candidate RMS of 12.30 px, which
looked like drift even though the expected landmark centers were outside the
image. A 12 px image-margin guard now reports that condition as **unavailable**
and retains the raw candidate RMS for inspection. The RMS is not used for a
threshold decision when expected landmarks are clipped. The West-lower pose
still has 2/95 nominal alarms and loses landmarks in 19/300 shifted trials, so
view transfer is not uniformly reliable.

Radial distortion detection also changes with landmark placement in the image.
For example, positive `k1=+0.010` was flagged in 25/25 East images, 0/25
higher-North images, and 1/25 diagonal images. At `k1=+0.020`, all three were
flagged in 25/25. On the East view, `k1=−0.005` produced 0/25 alarms while
`k1=+0.005` produced 25/25. These coefficients are synthetic image-warp
settings, not measured lens specifications. The alarm identifies disagreement
with the pinhole model; it does not identify whether distortion, calibration,
marker geometry, or visibility caused it.

The transfer diagnostic checks whether all expected landmark centers lie
inside a 12 px image margin before using their residual. A missing/clipped
configuration is an unavailable observation, not a camera-health pass or a
drift alarm. This remains a simulator diagnostic: no reaching controller,
physical success, fresh terminal visual confirmation, recorded camera, or
hardware lens is tested.

Reproduce the frozen-threshold transfer report and video with:

```bash
uv run visual-servo-camera-calibration-transfer \
  --threshold-report artifacts/camera-calibration-health-validation.json \
  --seeds 5 --samples-per-seed 20 --distortion-repeats-per-seed 5 \
  --validation-base-seed 600 \
  --output artifacts/camera-calibration-transfer-validation.json
uv run python scripts/generate_camera_calibration_transfer_demo.py
```
