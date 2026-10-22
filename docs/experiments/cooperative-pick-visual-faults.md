# Stress testing fresh post-pick vision

[Project overview](../../README.md) · [Fresh visual confirmation](cooperative-pick-visual-confirmation.md) · [Video gallery](../video-gallery.md)

## Question and protocol

The post-pick verifier uses two fresh side-camera images, so this experiment
holds its V3 thresholds fixed and changes only those RGB inputs. The initial
overhead image, camera-guided motion plan, MuJoCo scene, and physical scoring
stay the same. It asks whether dim or bright exposure, partial fiducial
occlusion, blur, and camera movement change the image decision.

The frozen rule requires at least 50 px upward displacement in each fresh
frame, at most 12 px inter-frame drift, and at most 4 degrees fiducial rotation
from the initial frame. It receives the initial side RGB and the two fresh side
RGB images only. Simulator height, tilt, contact, and collision results are
evaluation labels recorded afterward.

The run uses base seed 39600, three seeds, and 20 target poses per seed. Each
of the same 60 poses is replayed once under each of six conditions, for 360
simulations. All use the 5 mm baseline approach, 16 mm finger pads, and
clearance release. Seed 39500 was used only for a six-run smoke check and is
explicitly excluded from the reported validation. No threshold was adjusted
after that check.

| Input condition | Image-only change |
| --- | --- |
| Clean | No post-processing. |
| Dim exposure x0.25 | Multiply side-camera RGB channel values by 0.25. |
| Bright exposure x1.5 | Multiply RGB values by 1.5 and clip to 255. |
| Fiducial occlusion 50% | Cover the central half of the detected magenta marker bounding box. |
| Horizontal blur 9 px | Apply a 9-pixel horizontal image filter. |
| Camera shift up 25 px | Translate both fresh side-camera images upward by 25 pixels; keep the initial image unchanged. |

The image transformations run after rendering. A per-pose assertion checks
that capture-time truth, payload hold, collision safety, strict physical
success, and failure reason remain identical across that pose's six replays.
The image-fault experiment does not compare different controllers or approach
heights.

## Held-out V3 results

The fixed thresholds were selected before seeds 39400 and 39600. Base seed
39600 is a new validation block after the smoke-check seed was excluded.

| Side-camera condition | TP | False confirmation | TN | False non-confirmation | Abstain | Positive recall | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 58 | 0 | 2 | 0 | 0 | 100% | 100% |
| Dim exposure x0.25 | 0 | 0 | 0 | 0 | 60 | 0% | 0% |
| Bright exposure x1.5 | 58 | 0 | 2 | 0 | 0 | 100% | 100% |
| Fiducial occlusion 50% | 26 | 0 | 2 | 32 | 0 | 45% | 100% |
| Horizontal blur 9 px | 57 | 0 | 2 | 1 | 0 | 98% | 100% |
| Camera shift up 25 px | 58 | 0 | 2 | 0 | 0 | 100% | 100% |

Dim exposure made the marker unavailable, and the verifier abstained on all
60 trials. Half-marker occlusion left the marker detectable but produced 32
false non-confirmations among 58 capture-time positives. This is a concrete
limitation of the current largest-component detector: partial occlusion split
the marker image, and the measured rise, drift, or angle gate rejected 32 real
positives. The 9 px blur condition caused one false non-confirmation. Bright
exposure and this 25 px image-plane shift did not change any V3 decisions on
the sampled poses.

Only two trials were negative at capture time. Specificity was 2/2 in the clean,
bright, occlusion, blur, and camera-shift conditions, with a Wilson 95% interval
of 34%-100%. These outcomes do not establish false-confirmation performance
under new camera conditions. For dim exposure, all decisions abstained, so
negative specificity is 0/2 with a wide 0%-66% Wilson interval.

The physical outcomes stayed identical across all six image conditions for
every target pose: 57/60 payload holds and 57/60 strict full-cycle successes.
These scores are repeated evaluation labels, not 360 independent physical
trials. A fresh two-frame visual result still does not certify continuous
holding or safe release.

## Results and demos

The [scorecard](../../results/cooperative-pick-visual-fault-validation/cooperative-pick-visual-fault-scorecard.svg),
[summary JSON](../../results/cooperative-pick-visual-fault-validation/cooperative-pick-visual-fault-summary.json),
and [360-row trial CSV](../../results/cooperative-pick-visual-fault-validation/cooperative-pick-visual-fault-trials.csv)
preserve per-condition image decisions and separate physical labels.

| Condition | Replay |
| --- | --- |
| Clean reference | [MP4](../../assets/demos/cooperative_pick_visual_fault_clean.mp4) · [GIF](../../assets/demos/cooperative_pick_visual_fault_clean.gif) |
| Dim exposure x0.25 | [MP4](../../assets/demos/cooperative_pick_visual_fault_dim_exposure_x0_25.mp4) · [GIF](../../assets/demos/cooperative_pick_visual_fault_dim_exposure_x0_25.gif) |
| Bright exposure x1.5 | [MP4](../../assets/demos/cooperative_pick_visual_fault_bright_exposure_x1_5.mp4) · [GIF](../../assets/demos/cooperative_pick_visual_fault_bright_exposure_x1_5.gif) |
| Fiducial occlusion 50% | [MP4](../../assets/demos/cooperative_pick_visual_fault_marker_occlusion_50pct.mp4) · [GIF](../../assets/demos/cooperative_pick_visual_fault_marker_occlusion_50pct.gif) |
| Horizontal blur 9 px | [MP4](../../assets/demos/cooperative_pick_visual_fault_horizontal_blur_9px.mp4) · [GIF](../../assets/demos/cooperative_pick_visual_fault_horizontal_blur_9px.gif) |
| Camera shift up 25 px | [MP4](../../assets/demos/cooperative_pick_visual_fault_camera_shift_up_25px.mp4) · [GIF](../../assets/demos/cooperative_pick_visual_fault_camera_shift_up_25px.gif) |

The six clips replay the first pose in the validation block, already included
in the CSV. They are visual examples and add no trials. The [metadata](../../assets/demos/cooperative_pick_visual_fault_demos.json)
records each transformation and keeps the physical audit separate.

## Reproduction

```bash
uv run --locked visual-servo-cooperative-pick-visual-fault-validation \
  --base-seed 39600 --seeds 3 --samples-per-seed 20
uv run --locked --extra demo python scripts/generate_cooperative_pick_visual_fault_demos.py
```

## Limits and next experiment

- The exposure, blur, occlusion, and camera movement are authored pixel
  transforms of ideal MuJoCo renders, not measurements from a real camera.
- The marker occlusion is placed relative to the detected marker bounding box;
  it does not model scene geometry or an independent foreground object.
- The 25 px shift is a uniform image-plane translation. It does not model
  parallax, lens distortion, rolling shutter, or camera rotation.
- The held-out split has only two negative labels, so specificity evidence is
  weak even where no false confirmation occurred.
- The artificial magenta fiducial remains required; natural-object tracking
  is not tested.

The next useful algorithm experiment is an opt-in gap-tolerant fiducial
component measurement with an explicit image-quality/abstention rule. Develop
it on new seeds, then freeze it before a separate occlusion-heavy validation.
The current V3 thresholds and default behavior remain unchanged until that
comparison shows a repeatable gain without added false confirmations.
