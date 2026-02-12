from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .ingestion import LogLoader


DEFAULT_WINDOW_SEC = 1.0
DEFAULT_DEPTH_CROP_RATIO = 0.30
DEFAULT_CLEARANCE_PERCENTILE = 5.0
DEFAULT_WINDOW_SAMPLES_FALLBACK = 30
DEFAULT_DEPTH_SUBDIR = "depth"


class FeatureEngine:
    def __init__(
        self,
        loader: LogLoader,
        window_sec: float = DEFAULT_WINDOW_SEC,
        depth_crop_ratio: float = DEFAULT_DEPTH_CROP_RATIO,
        clearance_percentile: float = DEFAULT_CLEARANCE_PERCENTILE,
    ) -> None:
        if window_sec <= 0:
            raise ValueError("window_sec must be > 0.")
        if not 0 < depth_crop_ratio <= 1:
            raise ValueError("depth_crop_ratio must satisfy 0 < ratio <= 1.")
        if not 0 <= clearance_percentile <= 100:
            raise ValueError("clearance_percentile must satisfy 0 <= percentile <= 100.")

        self.loader = loader
        self.window_sec = float(window_sec)
        self.depth_crop_ratio = float(depth_crop_ratio)
        self.clearance_percentile = float(clearance_percentile)

    def build_features(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        base_df = self.loader.load_manifest() if df is None else df
        features_df = base_df.copy()
        self._validate_required_columns(features_df)

        window_samples = self._infer_window_samples(features_df["timestamp_sec"])
        features_df["roughness_camera_rms"] = self._compute_high_pass_rms(
            features_df["imu_camera_accel_z"], window_samples
        )
        features_df["roughness_syslogic_rms"] = self._compute_high_pass_rms(
            features_df["imu_syslogic_accel_z"], window_samples
        )
        features_df["roughness"] = features_df[
            ["roughness_camera_rms", "roughness_syslogic_rms"]
        ].mean(axis=1, skipna=True)

        min_clearance = [
            self._compute_frame_min_clearance(frame_idx, has_depth)
            for frame_idx, has_depth in zip(
                features_df["frame_idx"], features_df["has_depth"]
            )
        ]
        features_df["min_clearance_m"] = pd.Series(
            min_clearance, index=features_df.index, dtype="float64"
        )

        return features_df

    @staticmethod
    def _validate_required_columns(df: pd.DataFrame) -> None:
        required = {
            "frame_idx",
            "timestamp_sec",
            "has_depth",
            "imu_camera_accel_z",
            "imu_syslogic_accel_z",
        }
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"DataFrame missing required columns for features: {missing}")

    def _infer_window_samples(self, timestamp_sec: pd.Series) -> int:
        ts = pd.to_numeric(timestamp_sec, errors="coerce")
        dt = ts.diff()
        positive_dt = dt[(dt > 0) & np.isfinite(dt)]
        if positive_dt.empty:
            return DEFAULT_WINDOW_SAMPLES_FALLBACK

        median_dt = float(positive_dt.median())
        if not np.isfinite(median_dt) or median_dt <= 0:
            return DEFAULT_WINDOW_SAMPLES_FALLBACK

        samples = int(round(self.window_sec / median_dt))
        return max(1, samples)

    @staticmethod
    def _compute_high_pass_rms(series: pd.Series, window_samples: int) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        baseline = values.rolling(window=window_samples, min_periods=1).mean()
        high_pass = values - baseline
        rms = np.sqrt((high_pass**2).rolling(window=window_samples, min_periods=1).mean())
        return rms.astype("float64")

    def _compute_frame_min_clearance(self, frame_idx: Any, has_depth: Any) -> float:
        if not self._as_bool(has_depth):
            return float("nan")

        frame_number = pd.to_numeric(frame_idx, errors="coerce")
        if pd.isna(frame_number):
            return float("nan")

        depth_path = self._depth_path_for_frame(int(frame_number))
        if not depth_path.exists():
            return float("nan")

        try:
            with Image.open(depth_path) as image:
                depth_image = np.asarray(image)
        except OSError:
            return float("nan")

        if depth_image.ndim < 2:
            return float("nan")

        roi = self._center_crop(depth_image)
        valid_depth_mm = roi[roi > 0]
        if valid_depth_mm.size == 0:
            return float("nan")

        robust_min_mm = float(np.percentile(valid_depth_mm, self.clearance_percentile))
        return robust_min_mm / 1000.0

    def _depth_path_for_frame(self, frame_idx: int) -> Path:
        return self.loader.frames_dir / DEFAULT_DEPTH_SUBDIR / f"{frame_idx:04d}.png"

    def _center_crop(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        crop_h = max(1, int(round(height * self.depth_crop_ratio)))
        crop_w = max(1, int(round(width * self.depth_crop_ratio)))
        y0 = max((height - crop_h) // 2, 0)
        x0 = max((width - crop_w) // 2, 0)
        y1 = min(y0 + crop_h, height)
        x1 = min(x0 + crop_w, width)
        return image[y0:y1, x0:x1]

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "t", "yes", "y"}:
                return True
            if normalized in {"0", "false", "f", "no", "n", ""}:
                return False
        return bool(value)
