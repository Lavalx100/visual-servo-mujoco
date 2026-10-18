"""Fetch a checksum-pinned OpenCV sample sequence for offline calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from visual_servo_mujoco.public_camera_calibration import (
    DEFAULT_PATTERN_SIZE,
    DEFAULT_TRAIN_IMAGES,
    DEFAULT_VALIDATION_IMAGES,
    OPENCV_SAMPLE_REVISION,
    OPENCV_SAMPLE_SOURCE,
)

SAMPLE_SHA256 = {
    "left01.jpg": "9621899098adffa1440c8264606bb7be535ccfd9ad126ae8e0bd9b0c5a5b8676",
    "left02.jpg": "052e700bbcc865c112958ec2ea105cb1bce4deca50ac8d6efae13adca9df7a2c",
    "left03.jpg": "c85b9426c3d8fcf161e1e727356105091009c3fd0b1b1a65bc2da91fea2724d2",
    "left04.jpg": "b490599dccfbfe5eb56d33363415b02227a9548ab8c9b8e7b05d3580e97266d9",
    "left05.jpg": "40bab8934fa76f82ec614a7fd0e6ebb4d9f74391d5e8b0c6f2f0ed6080c6c4de",
    "left06.jpg": "1cb164db61b3bd52f58a7e9340bb27c5e0278895bc80b6f292fba5cf9bccc4a1",
    "left07.jpg": "d2211fe9f971646b6ff88ed93a4f0d3f443741eb2894095cba15caade7c8c6e4",
    "left08.jpg": "f7a0737894812a28243af5b409c5689f4fadfe7610958b8ca5fc79aec2f2cd1b",
    "left09.jpg": "753939bc62d093c001d9118e989bef2b887f1b2f9ac03603df915a8b85aa81d3",
    "left11.jpg": "e736ace3d515c94c547009c3c10d818c81506ed7059cb31123749b9cc3f94ee8",
    "left12.jpg": "04bdd78b3b507fe0551aa7062a3ffd94846a147ee2bea7fb38082c720bdbe231",
    "left13.jpg": "2637fb6dd5e557f4fadb06b353a453eb0b9223f4671f6f572f815302bc4a70b3",
    "left14.jpg": "d4c032971dd7b859f9d6aed5090c9888bd8039b82e0604bee5bfe05444fce834",
}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def download_samples(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, expected_hash in SAMPLE_SHA256.items():
        path = output_dir / name
        if path.exists():
            actual_hash = _sha256(path.read_bytes())
            if actual_hash != expected_hash:
                raise ValueError(
                    f"existing sample failed SHA-256 verification: {path}"
                )
            print(f"verified {name} (already present)")
            continue

        url = (
            "https://raw.githubusercontent.com/opencv/opencv/"
            f"{OPENCV_SAMPLE_REVISION}/samples/data/{name}"
        )
        request = urllib.request.Request(
            url, headers={"User-Agent": "visual-servo-mujoco-calibration-benchmark"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()
        actual_hash = _sha256(content)
        if actual_hash != expected_hash:
            raise ValueError(f"downloaded sample failed SHA-256 verification: {name}")
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_bytes(content)
        temporary_path.replace(path)
        print(f"downloaded and verified {name}")

    manifest = {
        "dataset_name": "OpenCV left-camera calibration sample images",
        "repository": "opencv/opencv",
        "revision": OPENCV_SAMPLE_REVISION,
        "source": OPENCV_SAMPLE_SOURCE,
        "source_data_is_not_copied_into_this_project": True,
        "pattern_inner_corners_columns_rows": list(DEFAULT_PATTERN_SIZE),
        "training_images": list(DEFAULT_TRAIN_IMAGES),
        "validation_images": list(DEFAULT_VALIDATION_IMAGES),
        "excluded_images": {
            "left11.jpg": (
                "The 9x6 target pattern is not present; the sample contains a "
                "9x7 inner-corner pattern instead."
            )
        },
        "sha256": SAMPLE_SHA256,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download checksum-pinned OpenCV checkerboard sample images."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/opencv-calibration-samples"),
    )
    args = parser.parse_args()
    manifest = download_samples(args.output_dir)
    print(
        f"Wrote manifest for {len(manifest['sha256'])} images to "
        f"{args.output_dir / 'dataset_manifest.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
