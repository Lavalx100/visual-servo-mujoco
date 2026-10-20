import csv
import json

import cv2
import numpy as np
import pytest

from visual_servo_mujoco.evidence_audit import (
    EvidenceAuditError,
    audit_candidate_validation_dataset,
)
from visual_servo_mujoco.perception_candidate import (
    detect_red_target_circle,
    expected_target_radius_px,
)
from visual_servo_mujoco.perception_candidate_validation import (
    DEVELOPMENT_OCCLUSION_FRACTIONS,
    DISTRACTOR_DISTANCE_RANGE_M,
    DISTRACTOR_RADIUS_FACTOR_RANGE,
    HELD_OUT_DISTRACTOR_DISTANCE_RANGES_M,
    HELD_OUT_DISTRACTOR_RADIUS_FACTOR_RANGE,
    HELD_OUT_LIGHT_INTENSITY_RANGES,
    HELD_OUT_OCCLUSION_FRACTIONS,
    LIGHT_INTENSITY_RANGE,
    run_perception_candidate_validation,
)
from visual_servo_mujoco.run import detect_red_target


EXPECTED_RADIUS_PX = expected_target_radius_px(480)


def _image() -> np.ndarray:
    return np.full((480, 640, 3), (165, 165, 165), dtype=np.uint8)


def test_circle_prior_recovers_a_dim_target_missed_by_default_detector():
    image = _image()
    center = (260, 210)
    cv2.circle(image, center, round(EXPECTED_RADIUS_PX), (42, 3, 2), -1)

    with pytest.raises(RuntimeError, match="no red target"):
        detect_red_target(image)

    detected = detect_red_target_circle(image, expected_radius_px=EXPECTED_RADIUS_PX)
    assert detected is not None
    assert np.hypot(detected.center_x_px - center[0], detected.center_y_px - center[1]) < 1.0


def test_circle_prior_uses_projected_size_to_ignore_a_larger_red_distractor():
    image = _image()
    target = (260, 210)
    cv2.circle(image, target, round(EXPECTED_RADIUS_PX), (255, 0, 0), -1)
    cv2.circle(image, (420, 190), round(EXPECTED_RADIUS_PX * 1.45), (255, 0, 0), -1)

    detected = detect_red_target_circle(image, expected_radius_px=EXPECTED_RADIUS_PX)
    assert detected is not None
    assert np.hypot(detected.center_x_px - target[0], detected.center_y_px - target[1]) < 2.0


def test_circle_prior_abstains_when_two_red_objects_have_target_like_size():
    image = _image()
    radius = round(EXPECTED_RADIUS_PX)
    cv2.circle(image, (240, 210), radius, (255, 0, 0), -1)
    cv2.circle(image, (320, 210), radius, (255, 0, 0), -1)

    assert detect_red_target_circle(image, expected_radius_px=EXPECTED_RADIUS_PX) is None


def test_final_validation_factor_ranges_are_disjoint_from_development():
    assert set(DEVELOPMENT_OCCLUSION_FRACTIONS).isdisjoint(HELD_OUT_OCCLUSION_FRACTIONS)
    assert HELD_OUT_DISTRACTOR_RADIUS_FACTOR_RANGE[1] < DISTRACTOR_RADIUS_FACTOR_RANGE[0]
    assert all(
        high < DISTRACTOR_DISTANCE_RANGE_M[0]
        or low > DISTRACTOR_DISTANCE_RANGE_M[1]
        for low, high in HELD_OUT_DISTRACTOR_DISTANCE_RANGES_M
    )
    assert all(
        high < LIGHT_INTENSITY_RANGE[0] or low > LIGHT_INTENSITY_RANGE[1]
        for low, high in HELD_OUT_LIGHT_INTENSITY_RANGES
    )


@pytest.mark.parametrize("image_height", [0, -1])
def test_projected_radius_rejects_non_positive_image_height(image_height):
    with pytest.raises(ValueError, match="image_height must be positive"):
        expected_target_radius_px(image_height)


def test_candidate_validation_writes_and_audits_paired_evidence(tmp_path):
    report = run_perception_candidate_validation(
        seeds=1,
        samples_per_seed=1,
        base_seed=9200,
        split_name="development",
        output_dir=tmp_path / "results",
        media_dir=tmp_path / "assets" / "demos",
        save_media=False,
    )
    assert report["trial_count"] == 9
    assert len(report["condition_summaries"]) == 9
    assert audit_candidate_validation_dataset(tmp_path) == (9, 9)

    summary_path = tmp_path / "results" / "perception-candidate-validation-summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        fields = next(csv.DictReader(handle)).keys()
    rows = list(csv.DictReader(summary_path.open(newline="", encoding="utf-8")))
    rows[0]["candidate_detections"] = "0"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(EvidenceAuditError, match="candidate_detections"):
        audit_candidate_validation_dataset(tmp_path)

    saved_report = json.loads(
        (tmp_path / "results" / "perception-candidate-validation-summary.json").read_text()
    )
    assert saved_report["candidate_status"] == "experimental; not the runtime default"
