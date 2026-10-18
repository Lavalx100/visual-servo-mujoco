import json

import cv2
import numpy as np
import pytest

from visual_servo_mujoco.public_camera_calibration import (
    export_validation_csv,
    render_validation_figure,
    run_public_camera_calibration_validation,
)


def _write_synthetic_calibration_sequence(image_dir):
    columns, rows = 9, 6
    square_px = 48
    board = np.zeros((7 * square_px, 10 * square_px, 3), dtype=np.uint8)
    for row in range(7):
        for column in range(10):
            value = 245 if (row + column) % 2 == 0 else 10
            board[
                row * square_px : (row + 1) * square_px,
                column * square_px : (column + 1) * square_px,
            ] = value

    camera_matrix = np.asarray(
        ((700.0, 0.0, 320.0), (0.0, 690.0, 240.0), (0.0, 0.0, 1.0))
    )
    distortion = np.asarray((-0.12, 0.025, 0.0, 0.0, 0.0))
    grid_x, grid_y = np.meshgrid(
        np.arange(640, dtype=np.float32), np.arange(480, dtype=np.float32)
    )
    distorted_pixels = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 1, 2)
    ideal_pixels = cv2.undistortPoints(
        distorted_pixels, camera_matrix, distortion, P=camera_matrix
    ).reshape(480, 640, 2)
    outer_board_corners = np.asarray(
        ((-1, -1, 0), (columns, -1, 0), (columns, rows, 0), (-1, rows, 0)),
        dtype=np.float32,
    )
    source_corners = np.asarray(
        (
            (0, 0),
            (10 * square_px - 1, 0),
            (10 * square_px - 1, 7 * square_px - 1),
            (0, 7 * square_px - 1),
        ),
        dtype=np.float32,
    )

    for index in range(12):
        rotation = np.asarray(
            (
                ((index % 4) - 1.5) * 0.14,
                ((index % 3) - 1) * 0.17,
                ((index % 5) - 2) * 0.09,
            )
        )
        translation = np.asarray(
            (
                -4.0 + ((index % 4) - 1.5) * 0.45,
                -2.5 + ((index % 3) - 1) * 0.35,
                18.0 + (index % 3) * 1.1,
            )
        )
        projected_outer, _ = cv2.projectPoints(
            outer_board_corners,
            rotation,
            translation,
            camera_matrix,
            np.zeros(5),
        )
        homography = cv2.getPerspectiveTransform(
            source_corners, projected_outer.reshape(-1, 2).astype(np.float32)
        )
        ideal_frame = cv2.warpPerspective(
            board,
            homography,
            (640, 480),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(120, 120, 120),
        )
        distorted_frame = cv2.remap(
            ideal_frame,
            ideal_pixels[:, :, 0],
            ideal_pixels[:, :, 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(120, 120, 120),
        )
        assert cv2.imwrite(str(image_dir / f"view{index:02d}.png"), distorted_frame)


def test_public_calibration_heldout_validation_uses_frozen_intrinsics(tmp_path):
    _write_synthetic_calibration_sequence(tmp_path)
    training = [f"view{index:02d}.png" for index in range(8)]
    validation = [f"view{index:02d}.png" for index in range(8, 12)]
    report_path = tmp_path / "report.json"
    figure_path = tmp_path / "report.svg"
    csv_path = tmp_path / "report.csv"

    report = run_public_camera_calibration_validation(
        tmp_path,
        train_images=training,
        validation_images=validation,
        output_path=report_path,
    )
    render_validation_figure(report, figure_path)
    export_validation_csv(report, csv_path)

    assert report["dataset"]["training_images"] == training
    assert report["dataset"]["validation_images"] == validation
    assert report["models"]["zero_distortion"]["validation"]["available_images"] == 4
    assert report["models"]["radial_k1_k2"]["validation"]["available_images"] == 4
    assert (
        report["models"]["radial_k1_k2"]["validation"]["pooled_corner_rms_px"]
        < report["models"]["zero_distortion"]["validation"]["pooled_corner_rms_px"]
    )
    assert json.loads(report_path.read_text())["schema_version"] == 1
    assert "<svg" in figure_path.read_text()
    assert "no corners beyond" in figure_path.read_text()
    csv_lines = csv_path.read_text().splitlines()
    assert len(csv_lines) > 2 * (len(validation) + 1)
    assert "radius_bin" in csv_lines[0]


@pytest.mark.parametrize(
    "training,validation",
    [
        (["same.png", "other.png", "third.png"], ["same.png"]),
        (["../outside.png", "other.png", "third.png"], ["heldout.png"]),
    ],
)
def test_public_calibration_rejects_overlapping_or_traversal_names(
    tmp_path, training, validation
):
    with pytest.raises(ValueError):
        run_public_camera_calibration_validation(
            tmp_path,
            train_images=training,
            validation_images=validation,
        )
