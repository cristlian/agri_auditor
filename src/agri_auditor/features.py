from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation

from .ingestion import LogLoader


DEFAULT_WINDOW_SEC = 1.0
DEFAULT_DEPTH_CROP_RATIO = 0.30
DEFAULT_CANOPY_CROP_RATIO = 0.30
DEFAULT_CLEARANCE_PERCENTILE = 5.0
DEFAULT_WINDOW_SAMPLES_FALLBACK = 30
DEFAULT_DEPTH_SUBDIR = "depth"
DEFAULT_VELOCITY_GATE_MPS = 0.1

_QUAT_COLS = (
    "pose_front_center_stereo_left_qx",
    "pose_front_center_stereo_left_qy",
    "pose_front_center_stereo_left_qz",
    "pose_front_center_stereo_left_qw",
)
_POSE_CONFIDENCE_COL = "pose_front_center_stereo_left_confidence"


class FeatureEngine:
    def __init__(
        self,
        loader: LogLoader,
        window_sec: float = DEFAULT_WINDOW_SEC,
        depth_crop_ratio: float = DEFAULT_DEPTH_CROP_RATIO,
        clearance_percentile: float = DEFAULT_CLEARANCE_PERCENTILE,
        canopy_crop_ratio: float = DEFAULT_CANOPY_CROP_RATIO,
        velocity_gate_mps: float = DEFAULT_VELOCITY_GATE_MPS,
    ) -> None:
        if window_sec <= 0:
            raise ValueError("window_sec must be > 0.")
        if not 0 < depth_crop_ratio <= 1:
            raise ValueError("depth_crop_ratio must satisfy 0 < ratio <= 1.")
        if not 0 <= clearance_percentile <= 100:
            raise ValueError("clearance_percentile must satisfy 0 <= percentile <= 100.")
        if not 0 < canopy_crop_ratio <= 1:
            raise ValueError("canopy_crop_ratio must satisfy 0 < ratio <= 1.")

        self.loader = loader
        self.window_sec = float(window_sec)
        self.depth_crop_ratio = float(depth_crop_ratio)
        self.clearance_percentile = float(clearance_percentile)
        self.canopy_crop_ratio = float(canopy_crop_ratio)
        self.velocity_gate_mps = float(velocity_gate_mps)

    def build_features(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        base_df = self.loader.load_manifest() if df is None else df
        features_df = base_df.copy()
        self._validate_required_columns(features_df)

        window_samples = self._infer_window_samples(features_df["timestamp_sec"])

        # --- Step 2 original: roughness ---
        features_df["roughness_camera_rms"] = self._compute_high_pass_rms(
            features_df["imu_camera_accel_z"], window_samples
        )
        features_df["roughness_syslogic_rms"] = self._compute_high_pass_rms(
            features_df["imu_syslogic_accel_z"], window_samples
        )
        features_df["roughness"] = features_df[
            ["roughness_camera_rms", "roughness_syslogic_rms"]
        ].mean(axis=1, skipna=True)

        # --- Step 2 original: min clearance ---
        clearance_and_canopy = [
            self._compute_frame_depth_features(frame_idx, has_depth)
            for frame_idx, has_depth in zip(
                features_df["frame_idx"], features_df["has_depth"]
            )
        ]
        features_df["min_clearance_m"] = pd.Series(
            [c[0] for c in clearance_and_canopy],
            index=features_df.index,
            dtype="float64",
        )

        # --- Phase 2.5: canopy density proxy ---
        features_df["canopy_density_proxy"] = pd.Series(
            [c[1] for c in clearance_and_canopy],
            index=features_df.index,
            dtype="float64",
        )

        # --- Phase 2.5: spatial (GPS) ---
        features_df = self._clean_gps(features_df)

        # --- Phase 2.5: orientation from quaternions ---
        features_df = self._compute_orientation(features_df)

        # --- Phase 2.5: sensor health ---
        features_df = self._compute_sensor_health(features_df, window_samples)

        return features_df

    # ------------------------------------------------------------------
    # GPS cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_gps(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure gps_lat and gps_lon are numeric and present."""
        for col in ("gps_lat", "gps_lon"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = np.nan
        if "gps_alt" in df.columns:
            df["gps_alt"] = pd.to_numeric(df["gps_alt"], errors="coerce")
        return df

    # ------------------------------------------------------------------
    # Quaternion → Euler + yaw rate
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_orientation(df: pd.DataFrame) -> pd.DataFrame:
        """Convert quaternion pose to yaw/pitch/roll (degrees) and yaw_rate (deg/s)."""
        has_quats = all(col in df.columns for col in _QUAT_COLS)
        if not has_quats:
            for col in ("yaw", "pitch", "roll", "yaw_rate"):
                df[col] = np.nan
            return df

        qx = pd.to_numeric(df[_QUAT_COLS[0]], errors="coerce")
        qy = pd.to_numeric(df[_QUAT_COLS[1]], errors="coerce")
        qz = pd.to_numeric(df[_QUAT_COLS[2]], errors="coerce")
        qw = pd.to_numeric(df[_QUAT_COLS[3]], errors="coerce")

        valid_mask = qx.notna() & qy.notna() & qz.notna() & qw.notna()

        yaw = pd.Series(np.nan, index=df.index, dtype="float64")
        pitch = pd.Series(np.nan, index=df.index, dtype="float64")
        roll = pd.Series(np.nan, index=df.index, dtype="float64")

        if valid_mask.any():
            quats = np.column_stack([
                qx[valid_mask].values,
                qy[valid_mask].values,
                qz[valid_mask].values,
                qw[valid_mask].values,
            ])
            # Normalize quaternions to avoid numerical issues
            norms = np.linalg.norm(quats, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            quats = quats / norms

            rot = Rotation.from_quat(quats)  # scipy expects [x, y, z, w]
            # Use 'ZYX' intrinsic (yaw-pitch-roll, standard vehicle convention)
            euler = rot.as_euler("ZYX", degrees=True)

            yaw.iloc[valid_mask.values] = euler[:, 0]
            pitch.iloc[valid_mask.values] = euler[:, 1]
            roll.iloc[valid_mask.values] = euler[:, 2]

        df["yaw"] = yaw
        df["pitch"] = pitch
        df["roll"] = roll

        # Yaw rate: handle wraparound at ±180°
        ts = pd.to_numeric(df["timestamp_sec"], errors="coerce")
        dt = ts.diff()
        dyaw = yaw.diff()
        # Unwrap: if delta > 180, subtract 360; if < -180, add 360
        dyaw = dyaw.where(dyaw.abs() <= 180, dyaw - np.sign(dyaw) * 360)
        yaw_rate = dyaw / dt
        yaw_rate[dt.isna() | (dt <= 0)] = np.nan
        df["yaw_rate"] = yaw_rate.astype("float64")

        return df

    # ------------------------------------------------------------------
    # Sensor health: IMU correlation + pose confidence
    # ------------------------------------------------------------------

    def _compute_sensor_health(
        self, df: pd.DataFrame, window_samples: int,
    ) -> pd.DataFrame:
        """Rolling IMU cross-correlation and pose confidence passthrough."""
        cam_z = pd.to_numeric(df["imu_camera_accel_z"], errors="coerce")
        sys_z = pd.to_numeric(df["imu_syslogic_accel_z"], errors="coerce")

        # Velocity gate: mask frames where vehicle is truly stationary
        # to avoid noise-dominated correlation.  Use a lenient threshold
        # because pose-derived velocity at 30Hz is noisy and often near-zero
        # even when the vehicle is crawling.
        velocity = pd.to_numeric(df.get("velocity_mps"), errors="coerce")
        stationary_mask = velocity.notna() & (velocity <= self.velocity_gate_mps)

        cam_gated = cam_z.copy()
        sys_gated = sys_z.copy()
        cam_gated[stationary_mask] = np.nan
        sys_gated[stationary_mask] = np.nan

        imu_corr = cam_gated.rolling(
            window=window_samples, min_periods=max(window_samples // 4, 2),
        ).corr(sys_gated)

        df["imu_correlation"] = imu_corr.astype("float64")

        # Pose confidence passthrough
        if _POSE_CONFIDENCE_COL in df.columns:
            df["pose_confidence"] = pd.to_numeric(
                df[_POSE_CONFIDENCE_COL], errors="coerce"
            ).astype("float64")
        else:
            df["pose_confidence"] = np.nan

        return df

    # ------------------------------------------------------------------
    # Depth features: clearance + canopy density
    # ------------------------------------------------------------------

    def _compute_frame_depth_features(
        self, frame_idx: Any, has_depth: Any,
    ) -> tuple[float, float]:
        """Return (min_clearance_m, canopy_density_proxy) for a single frame."""
        if not self._as_bool(has_depth):
            return (float("nan"), float("nan"))

        frame_number = pd.to_numeric(frame_idx, errors="coerce")
        if pd.isna(frame_number):
            return (float("nan"), float("nan"))

        depth_path = self._depth_path_for_frame(int(frame_number))
        if not depth_path.exists():
            return (float("nan"), float("nan"))

        try:
            with Image.open(depth_path) as image:
                depth_image = np.asarray(image)
        except OSError:
            return (float("nan"), float("nan"))

        if depth_image.ndim < 2:
            return (float("nan"), float("nan"))

        # Clearance: center crop, 5th percentile
        center_roi = self._center_crop(depth_image)
        valid_center = center_roi[center_roi > 0]
        if valid_center.size == 0:
            min_clearance_m = float("nan")
        else:
            min_clearance_m = float(
                np.percentile(valid_center, self.clearance_percentile)
            ) / 1000.0

        # Canopy density: upper crop, mean depth
        upper_roi = self._upper_crop(depth_image)
        valid_upper = upper_roi[upper_roi > 0]
        if valid_upper.size == 0:
            canopy_density_proxy = float("nan")
        else:
            # Lower mean depth = denser canopy (things are closer overhead)
            canopy_density_proxy = float(np.mean(valid_upper)) / 1000.0

        return (min_clearance_m, canopy_density_proxy)

    def _upper_crop(self, image: np.ndarray) -> np.ndarray:
        """Crop the upper portion of the image (canopy region)."""
        height, width = image.shape[:2]
        crop_h = max(1, int(round(height * self.canopy_crop_ratio)))
        return image[0:crop_h, :]

    # ------------------------------------------------------------------
    # Kept from Step 2 (unchanged)
    # ------------------------------------------------------------------

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
        """Legacy method kept for backward compatibility."""
        return self._compute_frame_depth_features(frame_idx, has_depth)[0]

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
