import numpy as np
import pytest
import cv2

from visual_servo_mujoco.camera_guided_cooperative_pick import (
    DEMO_TARGET_XY,
    run_trial,
)
from visual_servo_mujoco.cooperative_pick_visual_verification import (
    detect_verification_marker,
    verify_raised_payload_from_orthogonal_images,
    verify_raised_payload_from_images,
)
from visual_servo_mujoco.cooperative_pick_visual_validation import (
    _condition_summary,
    _paired_physical_comparison,
)
from visual_servo_mujoco.cooperative_pick_visual_fault_validation import (
    _assert_physics_unchanged,
    _summarize as _summarize_fault_condition,
)
from visual_servo_mujoco.cooperative_pick_occlusion_validation import (
    _select_development_candidate,
)
from visual_servo_mujoco.visual_image_faults import (
    apply_visual_image_fault,
    validate_visual_image_fault,
)


def _frame_with_marker(center_x: int, center_y: int) -> np.ndarray:
    image = np.zeros((180, 240, 3), dtype=np.uint8)
    image[center_y - 20 : center_y + 20, center_x - 5 : center_x + 5] = (255, 0, 255)
    return image


def test_detector_returns_largest_magenta_component_in_rgb_image():
    image = _frame_with_marker(80, 90)
    image[20:25, 20:25] = (255, 0, 255)

    marker = detect_verification_marker(image)

    assert marker is not None
    assert marker["center_px"] == pytest.approx([79.5, 89.5])
    assert marker["area_px"] == 400
    assert abs(abs(marker["major_axis_angle_deg"]) - 90.0) < 0.01


def test_gap_tolerant_detector_rejoins_a_partially_occluded_marker():
    image = _frame_with_marker(80, 90)
    image[85:95, 75:85] = (96, 96, 96)

    strict = detect_verification_marker(image)
    tolerant = detect_verification_marker(image, vertical_gap_tolerance_px=10)
    too_small = detect_verification_marker(image, vertical_gap_tolerance_px=9)

    assert strict["center_px"][1] == pytest.approx(77.0)
    assert too_small["component_count"] == 1
    assert tolerant["center_px"] == pytest.approx([79.5, 89.5])
    assert tolerant["area_px"] == 300
    assert tolerant["component_count"] == 2
    assert abs(abs(tolerant["major_axis_angle_deg"]) - 90.0) < 0.01


@pytest.mark.parametrize("tolerance", [-1, 101, 1.5, True])
def test_detector_rejects_invalid_vertical_gap_tolerance(tolerance):
    with pytest.raises(ValueError, match="integer from 0 to 100"):
        detect_verification_marker(_frame_with_marker(80, 90), vertical_gap_tolerance_px=tolerance)


def test_image_only_verifier_confirms_a_raised_stable_marker():
    result = verify_raised_payload_from_images(
        _frame_with_marker(80, 130),
        [(8.4, _frame_with_marker(81, 80)), (8.8, _frame_with_marker(82, 81))],
    )

    assert result["decision"] == "confirmed"
    assert result["reason"] == "raised_and_stable_in_two_fresh_frames"
    assert result["upward_displacement_px"] == [50.0, 49.0]
    assert result["interframe_drift_px"] == pytest.approx(1.414)
    assert "simulator" not in result["runtime_inputs"].lower()


@pytest.mark.parametrize(
    ("frames", "expected_reason"),
    [
        (
            [(8.4, _frame_with_marker(80, 110)), (8.8, _frame_with_marker(80, 109))],
            "insufficient_upward_displacement",
        ),
        (
            [(8.4, _frame_with_marker(80, 70)), (8.8, _frame_with_marker(95, 70))],
            "marker_not_stable_between_frames",
        ),
    ],
)
def test_verifier_rejects_insufficient_lift_or_unstable_marker(frames, expected_reason):
    result = verify_raised_payload_from_images(_frame_with_marker(80, 130), frames)

    assert result["decision"] == "not_confirmed"
    assert result["reason"] == expected_reason


def test_verifier_abstains_when_marker_is_missing_from_fresh_frame():
    result = verify_raised_payload_from_images(
        _frame_with_marker(80, 130),
        [(8.4, _frame_with_marker(80, 70)), (8.8, np.zeros((180, 240, 3), np.uint8))],
    )

    assert result["decision"] == "abstain"
    assert result["reason"] == "marker_not_visible"


def test_optional_orientation_gate_rejects_translation_from_a_tilted_payload():
    center = (120, 92)
    rotation = cv2.getRotationMatrix2D(center, 22.0, 1.0)
    tilted = cv2.warpAffine(
        _frame_with_marker(*center), rotation, (240, 180), flags=cv2.INTER_NEAREST
    )
    first = np.roll(tilted, -55, axis=0)
    second = np.roll(tilted, -56, axis=0)

    result = verify_raised_payload_from_images(
        _frame_with_marker(*center),
        [(8.4, first), (8.8, second)],
        max_marker_rotation_from_start_deg=12.0,
    )

    assert result["upward_displacement_px"][0] >= 35
    assert result["interframe_drift_px"] <= 12
    assert result["decision"] == "not_confirmed"
    assert result["reason"] == "marker_orientation_changed_too_much"
    assert result["marker_rotation_from_start_deg"][0] > 12


def test_orthogonal_views_reject_rotation_hidden_from_the_original_camera():
    initial = _frame_with_marker(120, 130)
    fresh_side = _frame_with_marker(120, 75)
    initial_orthogonal = _frame_with_marker(120, 130)
    rotation = cv2.getRotationMatrix2D((120, 120), 18.0, 1.0)
    rotation[1, 2] -= 55
    tilted_orthogonal = cv2.warpAffine(
        initial_orthogonal,
        rotation,
        (240, 180),
        flags=cv2.INTER_NEAREST,
    )

    result = verify_raised_payload_from_orthogonal_images(
        initial,
        initial_orthogonal,
        [
            (8.4, fresh_side, tilted_orthogonal),
            (8.8, fresh_side, tilted_orthogonal),
        ],
        min_upward_displacement_px=50.0,
        max_marker_rotation_from_start_deg=4.0,
    )

    assert result["decision"] == "not_confirmed"
    assert result["reason"] == "orthogonal_view_orientation_changed_too_much"
    assert result["view_records"]["side"]["decision"] == "confirmed"
    assert result["view_records"]["orthogonal"]["decision"] == "not_confirmed"
    assert result["simulator_truth_is_input"] is False


def test_orthogonal_verifier_abstains_if_one_view_lacks_a_visible_marker():
    initial = _frame_with_marker(120, 130)
    fresh = _frame_with_marker(120, 75)
    missing = np.zeros_like(initial)

    result = verify_raised_payload_from_orthogonal_images(
        initial,
        missing,
        [(8.4, fresh, fresh), (8.8, fresh, fresh)],
        min_upward_displacement_px=50.0,
        max_marker_rotation_from_start_deg=4.0,
    )

    assert result["decision"] == "abstain"
    assert result["reason"] == "orthogonal_view_insufficient_evidence"


def test_verifier_validates_capture_order_and_rgb_shape():
    with pytest.raises(ValueError, match="increasing"):
        verify_raised_payload_from_images(
            _frame_with_marker(80, 130),
            [(8.8, _frame_with_marker(80, 70)), (8.4, _frame_with_marker(80, 70))],
        )
    with pytest.raises(ValueError, match="HxWx3 uint8 RGB"):
        detect_verification_marker(np.zeros((20, 20), dtype=np.uint8))
    with pytest.raises(ValueError, match="thresholds"):
        verify_raised_payload_from_images(
            _frame_with_marker(80, 130),
            [(8.4, _frame_with_marker(80, 70)), (8.8, _frame_with_marker(80, 70))],
            min_upward_displacement_px=float("nan"),
        )


def test_visual_fault_transforms_copy_frames_and_translate_only_fresh_images():
    initial = _frame_with_marker(80, 100)
    fresh = _frame_with_marker(80, 80)
    fault = {"kind": "camera_translation", "dx_px": 0, "dy_px": -25}

    initial_result = apply_visual_image_fault(
        initial, fault, is_initial_frame=True
    )
    fresh_result = apply_visual_image_fault(fresh, fault, is_initial_frame=False)

    assert np.array_equal(initial_result, initial)
    assert initial_result is not initial
    assert detect_verification_marker(fresh_result)["center_px"][1] == pytest.approx(54.5)
    assert detect_verification_marker(fresh)["center_px"][1] == pytest.approx(79.5)
    assert np.array_equal(fresh, _frame_with_marker(80, 80))


def test_visual_faults_validate_parameters_and_occlusion_changes_only_pixels():
    image = _frame_with_marker(80, 90)
    original_area = detect_verification_marker(image)["area_px"]
    faulted = apply_visual_image_fault(
        image,
        {"kind": "marker_occlusion", "occluded_fraction": 0.5},
        is_initial_frame=True,
    )

    assert detect_verification_marker(faulted)["area_px"] < original_area
    assert detect_verification_marker(faulted)["area_px"] > 0
    assert detect_verification_marker(image)["area_px"] == original_area
    with pytest.raises(ValueError, match="exposure scale"):
        validate_visual_image_fault({"kind": "exposure_scale", "scale": 0})
    with pytest.raises(ValueError, match="odd integer"):
        validate_visual_image_fault(
            {"kind": "horizontal_motion_blur", "kernel_size": 4}
        )


@pytest.mark.parametrize(
    "fault",
    [
        {"kind": "marker_occlusion", "occluded_fraction": 0.3},
        {
            "kind": "marker_occlusion",
            "occluded_fraction": 0.3,
            "shape": "vertical_band",
        },
        {
            "kind": "marker_occlusion",
            "occluded_fraction": 0.25,
            "shape": "diagonal_band",
            "angle_deg": 40,
        },
        {
            "kind": "marker_occlusion",
            "occluded_fraction": 0.5,
            "shape": "corner_patch",
            "offset_x_fraction": -0.2,
            "offset_y_fraction": -0.2,
        },
    ],
)
def test_marker_occlusion_supports_frozen_shape_and_position_transforms(fault):
    image = _frame_with_marker(80, 90)
    original = image.copy()

    faulted = apply_visual_image_fault(image, fault, is_initial_frame=True)

    assert not np.array_equal(faulted, image)
    assert np.array_equal(image, original)
    marker = detect_verification_marker(faulted)
    assert marker is not None
    assert 0 < marker["area_px"] < detect_verification_marker(image)["area_px"]


def test_marker_occlusion_rejects_invalid_shape_geometry():
    with pytest.raises(ValueError, match="supported band or corner patch"):
        validate_visual_image_fault(
            {"kind": "marker_occlusion", "occluded_fraction": 0.5, "shape": "circle"}
        )
    with pytest.raises(ValueError, match="offsets"):
        validate_visual_image_fault(
            {
                "kind": "marker_occlusion",
                "occluded_fraction": 0.5,
                "offset_x_fraction": 0.6,
            }
        )


def test_camera_guided_pick_keeps_visual_and_physics_decisions_separate():
    metrics = run_trial(
        DEMO_TARGET_XY,
        controller="camera_estimate",
        release_profile="clearance",
        lowest_approach_tool_height_offset_m=0.020,
        visual_verification=True,
        visual_verification_min_upward_displacement_px=50.0,
        visual_verification_max_marker_rotation_from_start_deg=4.0,
        visual_verification_gap_tolerance_candidates_px=(12,),
    )

    visual = metrics["visual_verification"]
    assert metrics["fresh_visual_confirmation"] == visual["decision"] == "confirmed"
    assert visual["capture_times_s"] == [8.4, 8.8]
    assert visual["runtime_inputs"] == "initial and two fresh side-camera RGB frames only"
    assert visual["evaluation_only_physics"]["available_to_visual_decision"] is False
    assert visual["evaluation_only_physics"]["raised_and_stable_at_both_frames"] is True
    assert metrics["physical_stable_lift_success"] is True
    assert metrics["visual_verification_translation_only_baseline"]["decision"] == "confirmed"
    assert metrics["visual_verification_orientation_only_baseline"]["decision"] == "confirmed"
    assert metrics["visual_verification_rise_orientation_baseline"]["decision"] == "confirmed"
    gap_candidate = metrics["visual_verification_gap_tolerant_candidates"]["12"]
    assert gap_candidate["decision"] == visual["decision"]
    assert gap_candidate["evaluation_only_physics"] == visual["evaluation_only_physics"]


def test_camera_image_fault_is_recorded_as_an_observation_only_change():
    fault = {
        "kind": "exposure_scale",
        "scale": 0.25,
        "label": "dim exposure x0.25",
    }
    metrics = run_trial(
        DEMO_TARGET_XY,
        controller="camera_estimate",
        release_profile="clearance",
        lowest_approach_tool_height_offset_m=0.020,
        visual_verification=True,
        visual_verification_min_upward_displacement_px=50.0,
        visual_verification_max_marker_rotation_from_start_deg=4.0,
        visual_verification_image_fault=fault,
    )

    assert metrics["visual_verification"]["input_image_fault"] == fault
    assert metrics["visual_verification"]["evaluation_only_physics"][
        "available_to_visual_decision"
    ] is False
    assert metrics["physical_stable_lift_success"] is True


def test_posthoc_image_fault_candidates_share_one_physical_run_and_capture():
    metrics = run_trial(
        DEMO_TARGET_XY,
        controller="camera_estimate",
        visual_verification=True,
        visual_verification_min_upward_displacement_px=50.0,
        visual_verification_max_marker_rotation_from_start_deg=4.0,
        visual_verification_gap_tolerance_candidates_px=(28,),
        visual_verification_image_fault_candidates={
            "clean": None,
            "vertical_band": {
                "kind": "marker_occlusion",
                "shape": "vertical_band",
                "occluded_fraction": 0.3,
                "label": "center vertical band",
            },
        },
    )

    candidates = metrics["visual_verification_fault_candidate_records"]
    assert set(candidates) == {"clean", "vertical_band"}
    assert candidates["clean"]["v3"]["decision"] == metrics["visual_verification"]["decision"]
    assert (
        candidates["clean"]["gap_tolerant_candidates"]["28"]["decision"]
        == metrics["visual_verification_gap_tolerant_candidates"]["28"]["decision"]
    )
    for condition in candidates.values():
        assert condition["evaluation_only_physics"] == metrics["visual_verification"][
            "evaluation_only_physics"
        ]
        assert condition["gap_tolerant_candidates"]["28"][
            "evaluation_only_physics"
        ] == metrics["visual_verification"]["evaluation_only_physics"]


def test_validation_summary_counts_false_confirmation_and_abstention_separately():
    rows = [
        {"condition": "baseline_5mm", "truth_raised_stable_at_two_frames": True, "decision": "confirmed", "translation_only_decision": "confirmed", "orientation_only_decision": "confirmed", "rise_orientation_decision": "confirmed", "payload_stable_lift_success": True, "physical_stable_lift_success": True, "full_sequence_collision_safe": True},
        {"condition": "baseline_5mm", "truth_raised_stable_at_two_frames": True, "decision": "abstain", "translation_only_decision": "abstain", "orientation_only_decision": "abstain", "rise_orientation_decision": "abstain", "payload_stable_lift_success": False, "physical_stable_lift_success": False, "full_sequence_collision_safe": False},
        {"condition": "baseline_5mm", "truth_raised_stable_at_two_frames": False, "decision": "not_confirmed", "translation_only_decision": "confirmed", "orientation_only_decision": "confirmed", "rise_orientation_decision": "confirmed", "payload_stable_lift_success": False, "physical_stable_lift_success": False, "full_sequence_collision_safe": True},
        {"condition": "baseline_5mm", "truth_raised_stable_at_two_frames": False, "decision": "not_confirmed", "translation_only_decision": "not_confirmed", "orientation_only_decision": "not_confirmed", "rise_orientation_decision": "not_confirmed", "payload_stable_lift_success": False, "physical_stable_lift_success": False, "full_sequence_collision_safe": True},
    ]

    summary = _condition_summary(rows, "baseline_5mm")

    assert summary["image_classifier"]["true_positive"] == 1
    assert summary["image_classifier"]["false_confirmation"] == 0
    assert summary["image_classifier"]["true_negative"] == 2
    assert summary["image_classifier"]["abstained"] == 1
    assert summary["image_classifier"]["coverage"] == 0.75
    assert summary["image_classifier"]["positive_recall_including_abstentions"] == 0.5
    assert summary["translation_only_baseline"]["false_confirmation"] == 1
    assert summary["orientation_only_baseline"]["false_confirmation"] == 1
    assert summary["rise_orientation_baseline"]["false_confirmation"] == 1
    assert summary["physical_outcomes"]["payload_stable_hold_successes"] == 1
    assert summary["physical_outcomes"]["strict_full_sequence_successes"] == 1


def test_matched_physical_comparison_counts_both_directions():
    rows = [
        {"pair_id": "a", "condition": "baseline_5mm", "payload_stable_lift_success": False, "physical_stable_lift_success": False},
        {"pair_id": "a", "condition": "experimental_20mm", "payload_stable_lift_success": True, "physical_stable_lift_success": True},
        {"pair_id": "b", "condition": "baseline_5mm", "payload_stable_lift_success": True, "physical_stable_lift_success": True},
        {"pair_id": "b", "condition": "experimental_20mm", "payload_stable_lift_success": True, "physical_stable_lift_success": False},
    ]

    result = _paired_physical_comparison(rows)

    assert result["matched_target_pose_pairs"] == 2
    assert result["outcomes"]["payload_stable_lift_success"]["experimental_20mm_only_successes"] == 1
    assert result["outcomes"]["payload_stable_lift_success"]["baseline_5mm_only_successes"] == 0
    assert result["outcomes"]["strict_full_sequence_success"]["experimental_20mm_only_successes"] == 1
    assert result["outcomes"]["strict_full_sequence_success"]["baseline_5mm_only_successes"] == 1


def test_fault_validation_counts_image_errors_separately_from_physics():
    rows = [
        {"fault_condition": "clean", "truth_raised_stable_at_two_frames": True, "decision": "confirmed", "payload_stable_lift_success": True, "physical_stable_lift_success": True},
        {"fault_condition": "clean", "truth_raised_stable_at_two_frames": False, "decision": "not_confirmed", "payload_stable_lift_success": False, "physical_stable_lift_success": False},
        {"fault_condition": "dim", "truth_raised_stable_at_two_frames": True, "decision": "abstain", "payload_stable_lift_success": True, "physical_stable_lift_success": True},
        {"fault_condition": "dim", "truth_raised_stable_at_two_frames": False, "decision": "confirmed", "payload_stable_lift_success": False, "physical_stable_lift_success": False},
    ]

    summary = _summarize_fault_condition(rows, "dim")

    assert summary["true_positive"] == 0
    assert summary["false_confirmation"] == 1
    assert summary["abstained"] == 1
    assert summary["coverage"] == 0.5
    assert summary["positive_recall_including_abstentions"] == 0.0
    assert summary["negative_specificity_including_abstentions"] == 0.0


def test_fault_validation_rejects_any_change_to_per_pose_physics_outcome():
    rows = [
        {"pair_id": "39600-000", "fault_condition": "clean", "truth_raised_stable_at_two_frames": True, "payload_stable_lift_success": True, "full_sequence_collision_safe": True, "physical_stable_lift_success": True, "failure_reason": "none"},
        {"pair_id": "39600-000", "fault_condition": "camera_shift", "truth_raised_stable_at_two_frames": True, "payload_stable_lift_success": True, "full_sequence_collision_safe": True, "physical_stable_lift_success": True, "failure_reason": "none"},
    ]

    _assert_physics_unchanged(rows)
    rows[1]["physical_stable_lift_success"] = False
    with pytest.raises(RuntimeError, match="changed a physical outcome"):
        _assert_physics_unchanged(rows)


def test_gap_candidate_selection_requires_clean_parity_and_zero_false_confirmations():
    rows = [
        {"condition": "clean", "truth_raised_stable_at_two_frames": True, "v3_decision": "confirmed", "gap_12_px_decision": "confirmed", "gap_20_px_decision": "confirmed", "gap_28_px_decision": "confirmed"},
        {"condition": "marker_occlusion_50pct", "truth_raised_stable_at_two_frames": True, "v3_decision": "not_confirmed", "gap_12_px_decision": "confirmed", "gap_20_px_decision": "confirmed", "gap_28_px_decision": "confirmed"},
        {"condition": "marker_occlusion_50pct", "truth_raised_stable_at_two_frames": False, "v3_decision": "not_confirmed", "gap_12_px_decision": "not_confirmed", "gap_20_px_decision": "confirmed", "gap_28_px_decision": "not_confirmed"},
    ]

    selection = _select_development_candidate(rows, (12, 20, 28))

    assert selection["selected_gap_tolerance_px"] == 12
    assert selection["candidate_assessments"]["20"]["eligible"] is False
