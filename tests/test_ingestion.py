from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agri_auditor.ingestion import LogLoader


def data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "provided_data"


@pytest.fixture()
def loader() -> LogLoader:
    return LogLoader(data_dir())


def test_load_manifest_adds_velocity_column(loader: LogLoader) -> None:
    df = loader.load_manifest()
    assert len(df) == 1085
    assert "velocity_mps" in df.columns
    assert len(df) == len(pd.read_csv(data_dir() / "manifest.csv"))


def test_velocity_has_finite_values_and_expected_nans(loader: LogLoader) -> None:
    df = loader.load_manifest()
    assert np.isnan(df.loc[0, "velocity_mps"])
    finite_count = int(np.isfinite(df["velocity_mps"]).sum())
    assert finite_count > 900

    pose_nan_mask = df["pose_front_center_stereo_left_x"].isna()
    assert np.isnan(df.loc[pose_nan_mask, "velocity_mps"]).all()


def test_calibration_auto_detection_plural_filename(loader: LogLoader) -> None:
    calibs = loader.load_calibrations()
    assert "front_center_stereo_left" in calibs
    assert calibs["front_center_stereo_left"]["width"] == 0
    assert calibs["front_center_stereo_left"]["height"] == 0


def test_get_camera_model_honors_image_shape_override(loader: LogLoader) -> None:
    model = loader.get_camera_model("front_center_stereo_left", image_shape=(1200, 1920))
    assert model.height == 1200
    assert model.width == 1920


def test_get_camera_model_infers_dimensions_when_missing(loader: LogLoader) -> None:
    model = loader.get_camera_model("front_center_stereo_left")
    assert model.height == 1200
    assert model.width == 1920


def test_get_camera_model_returns_expected_k_matrix(loader: LogLoader) -> None:
    model = loader.get_camera_model("front_center_stereo_left", image_shape=(1200, 1920))
    expected = np.array(
        [
            [model.fx, 0.0, model.cx],
            [0.0, model.fy, model.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    assert model.K.shape == (3, 3)
    assert np.allclose(model.K, expected)


def test_unknown_camera_raises_clear_error(loader: LogLoader) -> None:
    with pytest.raises(ValueError, match="Unknown camera"):
        loader.get_camera_model("missing_camera_name")

