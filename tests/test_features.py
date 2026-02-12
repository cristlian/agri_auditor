from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from agri_auditor import FeatureEngine, LogLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return PROJECT_ROOT.parent / "provided_data"


@pytest.fixture(scope="module")
def loader() -> LogLoader:
    return LogLoader(data_dir())


@pytest.fixture(scope="module")
def feature_engine(loader: LogLoader) -> FeatureEngine:
    return FeatureEngine(loader=loader)


@pytest.fixture(scope="module")
def features_df(feature_engine: FeatureEngine) -> pd.DataFrame:
    return feature_engine.build_features()


def test_build_features_adds_expected_columns(features_df: pd.DataFrame) -> None:
    required_cols = {
        "roughness_camera_rms",
        "roughness_syslogic_rms",
        "roughness",
        "min_clearance_m",
    }
    assert required_cols.issubset(features_df.columns)


def test_build_features_adds_phase25_columns(features_df: pd.DataFrame) -> None:
    """Phase 2.5 columns: orientation, sensor health, canopy, GPS."""
    phase25_cols = {
        "yaw", "pitch", "roll", "yaw_rate",
        "imu_correlation", "pose_confidence",
        "canopy_density_proxy",
        "gps_lat", "gps_lon",
    }
    assert phase25_cols.issubset(features_df.columns)


def test_roughness_high_pass_is_non_negative_and_informative(
    features_df: pd.DataFrame,
) -> None:
    roughness = pd.to_numeric(features_df["roughness"], errors="coerce").dropna()
    assert len(roughness) > 0
    assert (roughness >= 0).all()
    assert float(roughness.quantile(0.99)) > float(roughness.quantile(0.50))


def test_min_clearance_nan_when_depth_missing(features_df: pd.DataFrame) -> None:
    no_depth_mask = ~features_df["has_depth"].astype(bool)
    assert features_df.loc[no_depth_mask, "min_clearance_m"].isna().all()


def test_min_clearance_finite_when_depth_present_and_in_range(
    loader: LogLoader, features_df: pd.DataFrame
) -> None:
    has_depth_mask = features_df["has_depth"].astype(bool)
    finite_depth_rows = pd.to_numeric(
        features_df.loc[has_depth_mask, "min_clearance_m"], errors="coerce"
    ).dropna()
    expected_count = _count_valid_depth_frames(loader, features_df)
    assert len(finite_depth_rows) == expected_count
    assert (finite_depth_rows > 0.0).all()
    assert (finite_depth_rows < 20.0).all()


def test_single_frame_clearance_matches_manual_percentile(
    loader: LogLoader, features_df: pd.DataFrame
) -> None:
    first_depth_frame = int(
        features_df.loc[features_df["has_depth"].astype(bool), "frame_idx"].iloc[0]
    )
    expected = _manual_min_clearance(loader, first_depth_frame, 0.30, 5.0)
    actual = float(
        pd.to_numeric(
            features_df.loc[
                features_df["frame_idx"] == first_depth_frame, "min_clearance_m"
            ],
            errors="coerce",
        ).iloc[0]
    )
    assert np.isfinite(expected)
    assert np.isclose(actual, expected, atol=1e-9)


def test_step2_script_exports_csv() -> None:
    script_path = PROJECT_ROOT / "scripts" / "build_features.py"

    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        src_path
        if not existing_pythonpath
        else f"{src_path}{os.pathsep}{existing_pythonpath}"
    )

    output_path = data_dir() / f"features_test_export_{uuid.uuid4().hex}.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--data-dir",
            str(data_dir()),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"build_features.py failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert output_path.exists()

    exported = pd.read_csv(output_path)
    for col in (
        "roughness_camera_rms",
        "roughness_syslogic_rms",
        "roughness",
        "min_clearance_m",
    ):
        assert col in exported.columns


# ------------------------------------------------------------------
# Phase 2.5 feature tests
# ------------------------------------------------------------------


def test_orientation_yaw_pitch_roll_finite(features_df: pd.DataFrame) -> None:
    """Quaternion → Euler: yaw/pitch/roll should be finite where quats exist."""
    for col in ("yaw", "pitch", "roll"):
        finite = features_df[col].dropna()
        assert len(finite) > 1000, f"{col} has too few finite values: {len(finite)}"
        assert (finite.abs() <= 180).all(), f"{col} has values outside ±180°"


def test_yaw_rate_reasonable_range(features_df: pd.DataFrame) -> None:
    """Yaw rate should be finite and within a physically reasonable range (deg/s)."""
    yaw_rate = features_df["yaw_rate"].dropna()
    assert len(yaw_rate) > 1000
    # Tractor shouldn't spin faster than ~50 deg/s in normal operation
    assert (yaw_rate.abs() < 500).all(), "Yaw rate exceeds 500 deg/s — likely unwrap bug"


def test_imu_correlation_range(features_df: pd.DataFrame) -> None:
    """Rolling IMU cross-correlation must be in [-1, 1]."""
    corr = features_df["imu_correlation"].dropna()
    assert len(corr) > 1000, f"Only {len(corr)} non-null imu_correlation values"
    assert (corr >= -1.01).all() and (corr <= 1.01).all(), "Correlation outside [-1, 1]"


def test_pose_confidence_passthrough(features_df: pd.DataFrame) -> None:
    """Pose confidence is passed through from the raw manifest column."""
    conf = features_df["pose_confidence"].dropna()
    assert len(conf) > 1000
    assert (conf > 0).all(), "Pose confidence should be positive"


def test_canopy_density_proxy(features_df: pd.DataFrame) -> None:
    """Canopy density proxy is non-NaN where depth exists, NaN where not."""
    has_depth = features_df["has_depth"].astype(bool)
    # Where depth absent → NaN
    assert features_df.loc[~has_depth, "canopy_density_proxy"].isna().all()
    # Where depth present → at least some finite values (some frames may have
    # zero-valued upper crops, which would also be NaN)
    canopy = features_df.loc[has_depth, "canopy_density_proxy"].dropna()
    assert len(canopy) > 0
    assert (canopy > 0).all(), "Canopy density proxy should be > 0 where valid"


def test_gps_lat_lon_numeric(features_df: pd.DataFrame) -> None:
    """GPS lat / lon are numeric after cleaning."""
    for col in ("gps_lat", "gps_lon"):
        vals = features_df[col].dropna()
        assert len(vals) > 1000, f"{col} has too few non-null values"
        assert vals.dtype == np.float64 or np.issubdtype(vals.dtype, np.floating)


def _count_valid_depth_frames(loader: LogLoader, df: pd.DataFrame) -> int:
    count = 0
    for frame_idx in df.loc[df["has_depth"].astype(bool), "frame_idx"]:
        if np.isfinite(_manual_min_clearance(loader, int(frame_idx), 0.30, 5.0)):
            count += 1
    return count


def _manual_min_clearance(
    loader: LogLoader,
    frame_idx: int,
    crop_ratio: float,
    percentile: float,
) -> float:
    path = loader.frames_dir / "depth" / f"{frame_idx:04d}.png"
    if not path.exists():
        return float("nan")

    with Image.open(path) as image:
        depth = np.asarray(image)

    height, width = depth.shape[:2]
    crop_h = max(1, int(round(height * crop_ratio)))
    crop_w = max(1, int(round(width * crop_ratio)))
    y0 = max((height - crop_h) // 2, 0)
    x0 = max((width - crop_w) // 2, 0)
    y1 = min(y0 + crop_h, height)
    x1 = min(x0 + crop_w, width)
    roi = depth[y0:y1, x0:x1]

    valid = roi[roi > 0]
    if valid.size == 0:
        return float("nan")
    return float(np.percentile(valid, percentile) / 1000.0)
