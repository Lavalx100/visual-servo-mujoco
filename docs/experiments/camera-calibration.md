# Camera calibration experiments

[Project overview](../../README.md)

## Dense planar-target radius diagnostic

The transfer failures motivated a 49-corner checkerboard whose features are
known in the table plane. The target-only calibration scene uses the same table
and five camera poses; a separate paired render in the original arm scene keeps
the frozen four-marker score comparable by view, condition, and appearance
seed. No controller is run. The dense profile remains descriptive and the
0.840999 px threshold stays attached only to the original four markers.

The evaluation has 325 rows: five views, 13 conditions, and five appearance
seeds (800–804). It covers nominal images, ±5° rendered FOV, a +2 cm camera X
shift, +3° yaw, and eight radial coefficients from −0.04 to +0.04. Brightness,
noise, and blur use the same ranges as the preceding transfer test. Target-size
selection probes used seeds 700–704; those seeds are excluded from this report.

The checkerboard profile was available on all five nominal samples in East,
West, and higher-North views, four of five diagonal samples, and three of five
grazing samples. The four old landmarks were unavailable in all five grazing
samples. Dense target coverage still varies by view: its maximum feature radius
ranges from 182 px on the diagonal and 186 px on North to 287 px at grazing.
The legacy four-marker radii reach 210, 223, and 366 px in those views.

The radius profile makes the radial-distortion blind spot measurable. In East,
`k1=+0.010` raised the dense-profile median RMS from 0.28 px nominally to 0.38
px; median signed radial residual was 0.86 px in the 240–320 px band and −0.05
px in the 0–80 px band. The frozen four-marker rule alarmed on all five East
samples. With the same coefficient in higher-North, its four-marker rule
alarmed on 0/5 samples; the dense profile median was 0.27 px versus 0.16 px
nominally. The diagonal dense median was 1.38 px among four available samples,
while the four-marker rule alarmed on 0/5. These are view and radius effects in
synthetic images; the dense profile has no acceptance threshold.

The larger 0.17 m-square design probe extended feature radii to 354 px and
matched the old edge coverage more closely, then lost detector availability:
2/5 West nominal, 4/5 diagonal, and 0/5 grazing samples. Its full report is
retained at `artifacts/camera-calibration-planar-profile-wide-target-probe.json`.
The selected 0.15 m-square layout trades some edge coverage for higher
appearance-robust detection. A future physical target may need multiple board
placements to cover the outer image field without shrinking the squares.

![Planar calibration target and radius-resolved reprojection profile](../../assets/demos/camera_calibration_planar.png)

![Selected and wider checkerboard layouts under the same view and appearance settings](../../assets/demos/camera_calibration_planar_target_design.png)

Reproduce the report and demo with:

```bash
uv run visual-servo-camera-calibration-planar \
  --seeds 5 --base-seed 800 \
  --output artifacts/camera-calibration-planar-profile.json
uv run python scripts/generate_camera_calibration_planar_demo.py
```

The dense-target radius profile above remains synthetic evidence. The external
recorded-camera check below exercises checkerboard calibration on real pixels,
but it does not test this project's camera, robot reaching, or fresh visual
confirmation. The next environment-specific milestone is still to evaluate a
camera used by this project. That requires checkerboard images at multiple
poses and image radii, with nominal and shifted captures held apart for
validation. Keep the verifier and quality gate opt-in until that evidence
exists.

## Public recorded-camera calibration check

To avoid requiring Jad to own a camera or print a target, this offline check
uses OpenCV's published left-camera checkerboard sequence. The
[OpenCV calibration tutorial](https://github.com/opencv/opencv/blob/9b6287ec500ef7b327ac74c4f6c9cd84a2cbc9cd/doc/tutorials/calib3d/camera_calibration/camera_calibration.markdown)
documents that its sample images were captured with an AXIS IP camera and
demonstrates calibrating from them. The image files are fetched from the
[pinned OpenCV sample directory](https://github.com/opencv/opencv/tree/9b6287ec500ef7b327ac74c4f6c9cd84a2cbc9cd/samples/data)
with SHA-256 checks; the project stores only the numerical report and residual
plot, not the source photographs.

The split is fixed: eight images (left01–left08) fit the camera intrinsics;
four later views (left09, left12–left14) are held out. left11 has a different
9×7 inner-corner pattern, so it is excluded from the 9×6 evaluation. Both
models estimate intrinsics only on the eight training images. For each
held-out view, solvePnP estimates that board's pose with the intrinsics frozen,
then the benchmark measures corner reprojection residuals by image radius.

| Model | Training corner RMS | Held-out pooled corner RMS | Held-out image RMS median / P95 |
| --- | ---: | ---: | ---: |
| Zero distortion | 1.726 px | 1.445 px | 1.257 / 1.879 px |
| Radial k1,k2 | 0.261 px | 0.286 px | 0.282 / 0.341 px |

Download the [machine-readable JSON report](../../results/public-camera-calibration-validation.json)
or the [per-image and per-radius CSV](../../results/public-camera-calibration-validation.csv).

The radial model had lower held-out RMS on all four images in this sample.
This is an encouraging pipeline check, not independent ground-truth
calibration: the sequence contains one camera, only four held-out views, and
does not provide the camera's true intrinsics. The held-out corners do not
extend beyond 240 px from the estimated principal point, so this dataset does
not test the outermost field-of-view failure seen in simulation. The result
does not imply better robot reaching, fresh visual confirmation, or that
k1,k2 should be enabled in the controller.

![Held-out OpenCV sample camera residuals by image and corner radius](../../assets/demos/public_camera_calibration_holdout.svg)

Reproduce the report and plot without capturing any images:

```bash
OPENCV_OPENCL_RUNTIME=disabled uv run python scripts/download_opencv_calibration_samples.py \
  --output-dir artifacts/opencv-calibration-samples
OPENCV_OPENCL_RUNTIME=disabled uv run visual-servo-camera-calibration-public \
  --image-dir artifacts/opencv-calibration-samples \
  --output artifacts/public-camera-calibration-validation.json \
  --figure artifacts/public-camera-calibration-holdout.svg \
  --csv results/public-camera-calibration-validation.csv
```

The first command downloads 13 small sample images to the ignored
artifacts/ directory and verifies each checksum. The evaluator uses only its
fixed 8-image training and 4-image held-out split; left11.jpg is retained
locally to document the pattern mismatch but is not scored. The calibration
report and source link make clear that this is external camera imagery, not
physical robot validation.
