# Independent benchmark review protocol

## What this is for

The project has substantial simulator evidence, but its assumptions and conditions were largely
chosen by the author. This protocol gives a reviewer a reproducible starting point and a list of
specific assumptions to challenge. It is a plan for independent review; no outside reviewer has
completed it yet.

The four post-stop studies below test whether a later camera observation is adequate evidence of
completion. They do not replace or redefine the reaching controller. Physical reaching at the
controller stop and fresh visual confirmation at camera-frame delivery are separate outcomes.
The experiments labeled `delayed_target_current_joint_tip` use current joint state to project the
arm tip, while `fully_delayed_rgb` delays both target and tip vision. Results from those modes
must stay separately labeled.

## Reproduce from a disposable checkout

Use a fresh clone of the reviewed commit. The benchmark commands rewrite their output files, so
keep the reviewer checkout disposable. Python 3.12 and the committed lockfile were used for the
reproduction recorded below.

```bash
git clone https://github.com/Lavalx100/visual-servo-mujoco.git
cd visual-servo-mujoco
git checkout YOUR_REVIEWED_COMMIT
uv sync --locked --extra dev --extra demo
uv run --locked pytest -q
uv run --locked python -m visual_servo_mujoco.evidence_audit
```

Then rerun each study and its paired demo:

```bash
uv run visual-servo-timestamped-verification-benchmark \
  --output artifacts/timestamped-verification.json \
  --csv results/timestamped-verification-trials.csv \
  --summary-csv results/timestamped-verification-summary.csv \
  --figure assets/demos/timestamped_verification_benchmark.svg
uv run python scripts/generate_timestamped_verification_demo.py

uv run visual-servo-completion-speed-benchmark \
  --output artifacts/timestamped-completion-speed.json \
  --csv results/timestamped-completion-speed-trials.csv \
  --summary-csv results/timestamped-completion-speed-summary.csv \
  --figure assets/demos/timestamped_completion_speed_benchmark.svg
uv run --extra demo python scripts/generate_timestamped_speed_demo.py

uv run visual-servo-motion-envelope-benchmark
uv run --extra demo python scripts/generate_motion_envelope_demo.py

uv run visual-servo-conditional-uncertainty-benchmark
uv run --extra demo python scripts/generate_conditional_uncertainty_demo.py

uv run --locked --extra demo visual-servo-perception-appearance-benchmark \
  --seeds 5 --samples-per-seed 10 --base-seed 9100

git diff --exit-code -- artifacts results assets/demos
```

The last two benchmark commands use their documented default paths. Compare the regenerated CSV,
summary, calibration, diagnostic, SVG, MP4, and GIF files with the reviewed commit. Exact media
bytes are useful for same-platform replay; visual inspection is still needed to judge whether a
clip explains the result faithfully. Run the evidence auditor after regeneration to check that
published trial counts and rates reconcile with their summaries.

## Reproduction completed locally

The initial clean-clone replay used the merged `main` snapshot `bdffa624cb1b28ca64525f82bd67a2a31a1db603`,
Python 3.12.15, and the locked dependencies. It ran on the same Mac as development. This verifies
same-platform reproducibility, not independent human review or cross-platform behavior.

| Study | Trial rows / grouped outcomes | Reproduction result |
| --- | ---: | --- |
| [Timestamped verification](../assets/demos/timestamped_post_stop_verification.mp4) | 1,000 / 9,000 | Trial CSV, summary CSV, SVG, MP4, and GIF matched byte for byte. |
| [Completion-speed sweep](../assets/demos/timestamped_completion_speed_comparison.mp4) | 975 / 8,775 | Trial CSV, summary CSV, SVG, MP4, and GIF matched byte for byte. |
| [Motion envelope](../assets/demos/motion_envelope_comparison.mp4) | 1,560 / 18,720 | Calibration, validation, summary, diagnostics, SVG, MP4, and GIF matched byte for byte; demo selected the same held-out high-latency case. |
| [Conditional uncertainty](../assets/demos/conditional_uncertainty_comparison.mp4) | 1,560 / 9,360 | Calibration, summary, diagnostics, SVG, MP4, and GIF matched byte for byte. The validation trial CSV had 335 visually confirmed predictor rows with missing `decision_frame_index`; all other fields matched. |
| [Rendered appearance stress](../assets/demos/perception_appearance_comparison.mp4) | 3,200 / 64 | Same-platform seeded rerun matched the trial and summary CSVs, JSON, SVG, MP4, and GIF byte for byte. No cross-platform or outside reproduction has been performed. |

The 335 missing frame indices have been restored from the deterministic clean-clone replay. The
conditional summary counts and rates did not change. The evidence auditor checks that every
confirmed row for the three q99 policies has a decision frame index, that the four post-stop trial
tables reconcile with their aggregate counts and confirmation rates, and that the appearance
dataset's condition summaries agree with its paired trial rows and protocol.
Per-frame rejection totals are reconciled only where the published trial table contains that
per-trial field; for example, the conditional table does not include a `stale_rejections` column.

The conditional uncertainty result remains negative. Across 2,340 held-out outcomes per policy,
the prior global q99 confirmed 129, the new global q99 confirmed 109, and the conditional q99
confirmed 97; each had zero false confirmations at delivery. The more specialized model therefore
reduced coverage without improving that measured false-confirmation count. All predictor policies
remain opt-in.

## Questions for an outside reviewer

First reproduce the committed artifacts before changing the protocol. Then challenge the
assumptions below and record each new condition before running it, so the held-out set is not
chosen after seeing results.

- **Camera transport:** How do the policies behave with variable frame intervals, burst loss,
  timestamp jitter, or reordering patterns that were not selected from the current profiles?
- **Target motion:** Do the conclusions survive speed, acceleration, phase, and path shapes outside
  the current stationary, sinusoidal, and piecewise-linear grid?
- **Visual measurement:** What happens with blur, lighting and exposure changes, partial marker
  occlusion, or detector errors unlike the injected pixel noise?
- **Geometry and mechanics:** How sensitive are the conclusions to camera calibration error,
  actuator lag, joint-state error, and a changed physical success radius?
- **Information available to the policy:** Are fully delayed RGB results compared only with fully
  delayed RGB? Are current-joint-state tip projections disclosed and reported as a distinct
  proprioceptive mode?
- **Task outcome:** Are controller-stop physical success, fresh visual confirmation, false
  confirmation at delivery, missed confirmation, and confirmation coverage reported separately?
- **Calibration protocol:** Are calibration seeds disjoint from evaluation seeds, and do quantile
  margins retain their stated coverage on an untouched test set?

Candidate new conditions above are proposals, not measured evidence. Keep the measured-target
baseline and current-age guard available, preserve failed outcomes, and do not enable an
experimental predictor by default unless a preregistered held-out comparison supports it.

## Current automation boundary

`.github/workflows/ci.yml` checks the lockfile, runs the evidence audit, and runs the unit/integration
suite on Ubuntu with MuJoCo's OSMesa software renderer. The appearance evidence audit reconciles
all 3,200 trial rows without rerendering the scenes. CI does not rerun the full benchmarks or
regenerate videos. The workflow has not run on GitHub yet; it will become an observed CI result only
after the reviewed changes are published and GitHub executes it.
