import numpy as np
import pytest

from visual_servo_mujoco.camera_calibration_health import (
    CalibrationScene,
    CALIBRATION_CAMERA_VIEWS,
    REPROJECTION_ALARM_THRESHOLD_PX,
    match_landmark_detections,
    project_world_point_to_pixel,
    run_camera_calibration_health_benchmark,
)
from visual_servo_mujoco.camera_calibration_validation import (
    run_camera_calibration_validation,
)
from visual_servo_mujoco.camera_calibration_transfer import (
    run_camera_calibration_transfer_validation,
)
from visual_servo_mujoco.camera_calibration_planar import (
    PLANAR_TARGET_FEATURE_COUNT,
    PLANAR_TARGET_VISIBILITY_MARGIN_PX,
    PlanarCalibrationScene,
    _validate_protocol as validate_planar_protocol,
    analyze_planar_frame,
    associate_checkerboard_corners,
    planar_checkerboard_world_points,
    run_camera_calibration_planar_benchmark,
)


def test_nominal_camera_projects_point_to_principal_point():
    pixel = project_world_point_to_pixel(
        (0.0, 0.0, 0.0),
        np.asarray((0.0, 0.0, 2.0)),
        np.eye(3),
        40.0,
    )
    assert pixel == pytest.approx((319.5, 239.5))


def test_projection_rejects_point_behind_camera():
    with pytest.raises(ValueError, match="in front"):
        project_world_point_to_pixel(
            (0.0, 0.0, 1.0),
            np.asarray((0.0, 0.0, 0.0)),
            np.eye(3),
            40.0,
        )


def test_landmark_matching_finds_lowest_cost_assignment_and_requires_four():
    expected = {
        "a": (10.0, 10.0),
        "b": (20.0, 10.0),
        "c": (10.0, 20.0),
        "d": (20.0, 20.0),
    }
    observed = [(20.2, 20.0), (10.1, 10.0), (20.0, 10.0), (10.0, 20.0)]
    matched = match_landmark_detections(expected, observed)
    assert matched is not None
    assert matched["a"]["observed_pixel_xy"] == pytest.approx([10.1, 10.0])
    assert matched["d"]["reprojection_residual_px"] == pytest.approx(0.2)
    assert match_landmark_detections(expected, observed[:3]) is None


def test_camera_health_benchmark_separates_nominal_from_renderer_drift():
    report = run_camera_calibration_health_benchmark()
    results = {row["condition"]: row for row in report["results"]}
    assert results["nominal"]["diagnostic_status"] == "healthy"
    assert results["nominal"]["reprojection_rms_px"] < 1.0
    assert results["nominal"]["reprojection_rms_px"] < REPROJECTION_ALARM_THRESHOLD_PX
    for condition in (
        "render_fovy_plus_5deg",
        "render_fovy_minus_5deg",
        "render_camera_world_x_plus_2cm",
        "render_camera_yaw_plus_3deg",
        "render_combined_fovy_x_yaw_shift",
    ):
        assert results[condition]["renderer_camera_perturbed"] is True
        assert results[condition]["diagnostic_status"] == "drift_detected"
        assert results[condition]["reprojection_rms_px"] > REPROJECTION_ALARM_THRESHOLD_PX
    assert results["bright_exposure_x1_5"]["diagnostic_status"] == "healthy"
    assert results["dim_exposure_x0_25"]["diagnostic_status"] == "unavailable"


def test_frozen_camera_health_validation_keeps_seed_splits_disjoint():
    report = run_camera_calibration_validation(
        seeds=2,
        samples_per_seed=4,
        calibration_base_seed=41,
        validation_base_seed=73,
    )
    assert report["protocol"]["calibration_seed_range"] == [41, 42]
    assert report["protocol"]["validation_seed_range"] == [73, 74]
    assert report["threshold_selection"]["calibration_nominal_available_samples"] == 8
    assert report["threshold_selection"]["calibration_false_alarm_rate"] <= 0.05
    assert report["validation_nominal"]["unavailable_samples"] == 2
    assert report["validation_dim_probe"]["available_samples"] == 0
    nominal_only = report["threshold_comparisons"]["nominal_only_p95"]
    supervised = report["threshold_comparisons"]["supervised_fpr_constrained"]
    assert (
        supervised["validation_nominal"]["false_alarm_samples"]
        <= nominal_only["validation_nominal"]["false_alarm_samples"]
    )
    for severity in ("subtle", "moderate", "severe"):
        group = report["validation_drift_by_severity"][severity]
        assert group["samples"] == 8
        assert group["drift_detected_samples"] + group["missed_drift_samples"] == 8
    assert all(
        row["frozen_threshold_rms_px"]
        == report["threshold_selection"]["threshold_rms_px"]
        for row in report["trial_results"]
    )


def test_frozen_camera_health_validation_rejects_overlapping_seeds():
    with pytest.raises(ValueError, match="must not overlap"):
        run_camera_calibration_validation(
            seeds=2,
            samples_per_seed=2,
            calibration_base_seed=10,
            validation_base_seed=11,
        )


def test_calibration_scene_records_alternate_camera_pose():
    assert "east_grazing" in CALIBRATION_CAMERA_VIEWS
    with CalibrationScene("north_higher") as scene:
        _, row = scene.render(condition_name="higher_view_nominal")
    assert row["camera_view"] == "north_higher"
    assert row["renderer_camera_perturbed"] is False
    with CalibrationScene("east_grazing") as scene:
        _, clipped = scene.render(condition_name="grazing_view_nominal")
    assert clipped["diagnostic_status"] == "unavailable"
    assert clipped["diagnostic_unavailable_reason"] == (
        "expected_landmark_outside_safe_image_margin"
    )
    assert clipped["reprojection_rms_px"] is None
    assert clipped["candidate_reprojection_rms_px"] > 3.0


def test_transfer_validation_applies_one_frozen_threshold_to_every_trial():
    threshold = 0.8409987385743665
    report = run_camera_calibration_transfer_validation(
        threshold_px=threshold,
        threshold_source="test fixture",
        seeds=1,
        samples_per_seed=2,
        distortion_repeats_per_seed=1,
        validation_base_seed=900,
    )
    assert report["protocol"]["threshold_px"] == threshold
    assert report["protocol"]["threshold_policy"].startswith("frozen")
    assert set(report["views"]) == set(CALIBRATION_CAMERA_VIEWS)
    assert len(report["trial_results"]) == len(CALIBRATION_CAMERA_VIEWS) * (2 + 6 + 8)
    assert all(
        row["frozen_threshold_rms_px"] == threshold
        for row in report["trial_results"]
    )
    assert report["views"]["east"]["radial_distortion_by_k1"]["+0.0400"][
        "frozen_alarm_samples"
    ] == 1


def test_transfer_validation_rejects_nonpositive_threshold():
    with pytest.raises(ValueError, match="finite and positive"):
        run_camera_calibration_transfer_validation(threshold_px=0.0, seeds=1)


def test_transfer_validation_rejects_threshold_development_seeds():
    with pytest.raises(ValueError, match="overlap threshold development"):
        run_camera_calibration_transfer_validation(
            threshold_px=0.8409987385743665,
            seeds=1,
            validation_base_seed=500,
        )


def test_planar_target_has_dense_known_coplanar_features():
    points = planar_checkerboard_world_points()
    assert points.shape == (PLANAR_TARGET_FEATURE_COUNT, 3)
    assert PLANAR_TARGET_FEATURE_COUNT == 49
    assert np.unique(points[:, 2]).tolist() == pytest.approx([0.1418])
    assert np.min(points[:, 0]) == pytest.approx(-0.45)
    assert np.max(points[:, 0]) == pytest.approx(0.45)
    assert np.min(points[:, 1]) == pytest.approx(-0.45)
    assert np.max(points[:, 1]) == pytest.approx(0.45)


def test_planar_target_size_probe_changes_edge_coverage_and_validates_size():
    wide_points = planar_checkerboard_world_points(0.17)
    assert np.min(wide_points[:, 0]) == pytest.approx(-0.51)
    assert np.max(wide_points[:, 0]) == pytest.approx(0.51)
    with PlanarCalibrationScene("east", square_size_m=0.17) as wide_scene:
        assert wide_scene.square_size_m == pytest.approx(0.17)
    with pytest.raises(ValueError, match="finite and positive"):
        planar_checkerboard_world_points(0.0)


def test_checkerboard_association_handles_reversed_grid_order():
    points = np.column_stack(
        (np.arange(PLANAR_TARGET_FEATURE_COUNT), np.arange(PLANAR_TARGET_FEATURE_COUNT) ** 2)
    ).astype(float)
    association = associate_checkerboard_corners(points[::-1], points)
    assert association is not None
    assert association["rms_px"] == pytest.approx(0.0)
    np.testing.assert_allclose(association["matched_corners_px"], points)


def test_planar_calibration_camera_matches_robot_scene_pose_and_detects_grid():
    with PlanarCalibrationScene("north_higher") as planar_scene, CalibrationScene(
        "north_higher"
    ) as legacy_scene:
        np.testing.assert_allclose(
            planar_scene.nominal_position, legacy_scene.nominal_position, atol=1e-8
        )
        np.testing.assert_allclose(
            planar_scene.nominal_rotation, legacy_scene.nominal_rotation, atol=1e-8
        )
        expected = np.asarray(
            [
                project_world_point_to_pixel(
                    point,
                    planar_scene.nominal_position,
                    planar_scene.nominal_rotation,
                    planar_scene.nominal_fovy,
                )
                for point in planar_checkerboard_world_points()
            ]
        )
        image = planar_scene.render(condition_name="nominal")
    diagnostic = analyze_planar_frame(image, expected)
    assert diagnostic["status"] == "available_profile_only"
    assert diagnostic["detected_feature_count"] == PLANAR_TARGET_FEATURE_COUNT
    assert diagnostic["profile_reprojection_rms_px"] < 1.0


def test_planar_profile_separates_radial_error_by_feature_radius():
    with PlanarCalibrationScene("west_lower") as scene:
        expected = np.asarray(
            [
                project_world_point_to_pixel(
                    point,
                    scene.nominal_position,
                    scene.nominal_rotation,
                    scene.nominal_fovy,
                )
                for point in planar_checkerboard_world_points()
            ]
        )
        positive_image = scene.render(
            condition_name="positive_radial_warp", radial_distortion_k1=0.04
        )
        negative_image = scene.render(
            condition_name="negative_radial_warp", radial_distortion_k1=-0.04
        )
    positive = analyze_planar_frame(positive_image, expected)
    negative = analyze_planar_frame(negative_image, expected)
    assert positive["status"] == "available_profile_only"
    assert negative["status"] == "available_profile_only"

    def median_radial_residual(diagnostic, radius_bin):
        values = [
            point["radial_residual_px"]
            for point in diagnostic["radius_bins"]
            if point["radius_bin"] == radius_bin
        ]
        return float(np.median(values))

    positive_outer = median_radial_residual(positive, "240_320px")
    positive_inner = median_radial_residual(positive, "0_80px")
    negative_outer = median_radial_residual(negative, "240_320px")
    assert positive_outer > 2.0
    assert positive_outer > positive_inner
    assert negative_outer < -2.0


def test_planar_missing_or_clipped_grid_is_never_reported_as_zero_error():
    with PlanarCalibrationScene("east") as scene:
        expected = np.asarray(
            [
                project_world_point_to_pixel(
                    point,
                    scene.nominal_position,
                    scene.nominal_rotation,
                    scene.nominal_fovy,
                )
                for point in planar_checkerboard_world_points()
            ]
        )
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    not_detected = analyze_planar_frame(blank, expected)
    assert not_detected["status"] == "unavailable_checkerboard_not_detected"
    assert not_detected["profile_reprojection_rms_px"] is None
    assert not_detected["safely_visible_expected_feature_count"] == PLANAR_TARGET_FEATURE_COUNT

    clipped = analyze_planar_frame(blank, expected + (1000.0, 0.0))
    assert clipped["status"] == "unavailable_expected_corner_outside_safe_margin"
    assert clipped["unavailable_reason"] == "expected_corner_outside_safe_image_margin"
    assert clipped["safely_visible_expected_feature_count"] == 0


def test_planar_protocol_rejects_target_design_probe_seeds():
    with pytest.raises(ValueError, match="overlap prior calibration splits"):
        validate_planar_protocol(seeds=1, base_seed=700)


def test_planar_benchmark_pairs_legacy_threshold_without_transferring_it():
    report = run_camera_calibration_planar_benchmark(seeds=1, base_seed=900)
    assert len(report["trial_results"]) == len(CALIBRATION_CAMERA_VIEWS) * 13
    assert report["protocol"]["legacy_four_marker_reference"]["threshold_px"] == (
        0.8409987385743665
    )
    assert report["protocol"]["target"]["inner_corner_count"] == 49
    assert report["protocol"]["target"]["design_probe_seed_range_excluded"] == [
        700,
        704,
    ]
    assert report["views"]["east"]["feature_radius_coverage"][
        "dense_target_max_radius_px"
    ] > 200.0
    assert report["trial_results"][0]["legacy_four_marker"][
        "frozen_threshold_px"
    ] == 0.8409987385743665
    assert "frozen_threshold_px" not in report["trial_results"][0][
        "planar_diagnostic"
    ]
    assert PLANAR_TARGET_VISIBILITY_MARGIN_PX == 12.0
    assert "feature_radius_coverage" in report["views"]["east"]
