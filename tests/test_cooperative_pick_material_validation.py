import json

from visual_servo_mujoco.cooperative_pick_material_validation import run_validation
from visual_servo_mujoco.cooperative_pick_demo import DEFAULT_PAYLOAD_BODY_MASS_KG


def test_material_validation_writes_paired_physics_evidence(tmp_path):
    output = tmp_path / "material-results"
    demo_metadata = tmp_path / "demos" / "material-demos.json"

    summary = run_validation(
        output_dir=output,
        demo_metadata_path=demo_metadata,
        base_seed=49900,
        samples_per_seed=1,
        payload_masses_kg=(DEFAULT_PAYLOAD_BODY_MASS_KG,),
        sliding_friction_levels=(2.2,),
        render_demos=False,
    )

    assert summary["total_physics_trials"] == 2
    assert summary["unique_target_poses"] == 1
    assert summary["unique_pose_material_pairs"] == 1
    condition = summary["conditions"][f"mass-{DEFAULT_PAYLOAD_BODY_MASS_KG:g}_friction-2.2"]
    assert condition["trials"] == 1
    assert set(condition["controllers"]) == {"camera_estimate", "oracle_xy"}
    assert condition["controllers"]["camera_estimate"]["trials"] == 1
    assert condition["controllers"]["camera_estimate"]["fresh_visual_confirmation"] == (
        "not measured in material sweep"
    )
    assert (output / "cooperative-pick-material-trials.csv").is_file()
    assert (output / "cooperative-pick-material-scorecard.svg").is_file()
    saved = json.loads(
        (output / "cooperative-pick-material-summary.json").read_text()
    )
    assert saved["total_physics_trials"] == 2
