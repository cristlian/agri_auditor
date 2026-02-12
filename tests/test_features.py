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
