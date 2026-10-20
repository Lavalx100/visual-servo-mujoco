import json
import csv

import pytest

from visual_servo_mujoco.evidence_audit import (
    EvidenceAuditError,
    audit_appearance_dataset,
)
from visual_servo_mujoco.perception_appearance_benchmark import (
    ALL_CONDITIONS,
    AppearanceCondition,
    build_appearance_scene_xml,
    run_perception_appearance_benchmark,
)


def test_appearance_grid_is_full_factorial_and_condition_names_are_stable():
    assert len(ALL_CONDITIONS) == 64
    assert len({condition.name for condition in ALL_CONDITIONS}) == 64
    assert AppearanceCondition("dim", "mixed", 0.5).name == (
        "dim__mixed__occlusion-0p50"
    )


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        (AppearanceCondition("dim", "none", 0.0), "diffuse=\"0.16 0.16 0.16\""),
        (AppearanceCondition("side", "none", 0.0), "pos=\"0.65 -0.35 2.4\""),
        (AppearanceCondition("nominal", "red_distractor", 0.0), "red_distractor"),
        (AppearanceCondition("nominal", "none", 0.5), "target_occluder"),
    ],
)
def test_scene_xml_applies_rendered_appearance_factors(condition, expected):
    assert expected in build_appearance_scene_xml(condition)


def test_one_rendered_condition_writes_localization_evidence(tmp_path):
    condition = AppearanceCondition("nominal", "none", 0.0)
    report = run_perception_appearance_benchmark(
        seeds=1,
        samples_per_seed=1,
        base_seed=9100,
        conditions=(condition,),
        output_dir=tmp_path / "results",
        media_dir=tmp_path / "media",
        save_media=False,
    )

    summary = report["condition_summaries"][0]
    assert summary["trials"] == 1
    assert summary["detections"] == 1
    assert summary["within_target_radius_localizations"] == 1
    assert (tmp_path / "results/perception-appearance-trials.csv").exists()
    assert (tmp_path / "results/perception-appearance-summary.csv").exists()
    saved_report = json.loads(
        (tmp_path / "results/perception-appearance-summary.json").read_text()
    )
    assert saved_report["scope"].startswith("target-center localization only")
    assert (tmp_path / "media/perception-appearance-scorecard.svg").exists()
    assert audit_appearance_dataset(tmp_path) == (1, 1)

    summary_path = tmp_path / "results/perception-appearance-summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    rows[0]["detections"] = "0"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(EvidenceAuditError, match="detections"):
        audit_appearance_dataset(tmp_path)


def test_occlusion_width_outside_studied_range_is_rejected():
    with pytest.raises(ValueError, match="between 0 and 0.75"):
        build_appearance_scene_xml(AppearanceCondition("nominal", "none", 1.0))
