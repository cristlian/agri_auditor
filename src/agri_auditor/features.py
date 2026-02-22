from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
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
DEFAULT_DEPTH_WORKERS = 4
DEFAULT_DEPTH_CACHE_FILENAME = "depth_feature_cache.parquet"

_QUAT_COLS = (
    "pose_front_center_stereo_left_qx",
    "pose_front_center_stereo_left_qy",
    "pose_front_center_stereo_left_qz",
    "pose_front_center_stereo_left_qw",
)
_POSE_CONFIDENCE_COL = "pose_front_center_stereo_left_confidence"


@dataclass(frozen=True)
class _DepthCacheEntry:
    cache_key: str
    frame_idx: int
    mtime_ns: int
    size_bytes: int
    params_hash: str


def _center_crop_array(image: np.ndarray, crop_ratio: float) -> np.ndarray:
    height, width = image.shape[:2]
    crop_h = max(1, int(round(height * crop_ratio)))
    crop_w = max(1, int(round(width * crop_ratio)))
    y0 = max((height - crop_h) // 2, 0)
    x0 = max((width - crop_w) // 2, 0)
    y1 = min(y0 + crop_h, height)
    x1 = min(x0 + crop_w, width)
    return image[y0:y1, x0:x1]


def _upper_crop_array(image: np.ndarray, crop_ratio: float) -> np.ndarray:
    height, _width = image.shape[:2]
    crop_h = max(1, int(round(height * crop_ratio)))
    return image[0:crop_h, :]


def _compute_depth_features_from_path(
    *,
    depth_path: str | Path,
    depth_crop_ratio: float,
    canopy_crop_ratio: float,
    clearance_percentile: float,
) -> tuple[float, float]:
    path = Path(depth_path)
    try:
        with Image.open(path) as image:
            depth_image = np.asarray(image)
    except OSError:
        return (float("nan"), float("nan"))

    if depth_image.ndim < 2:
        return (float("nan"), float("nan"))

    center_roi = _center_crop_array(depth_image, depth_crop_ratio)
    valid_center = center_roi[center_roi > 0]
    if valid_center.size == 0:
        min_clearance_m = float("nan")
    else:
        min_clearance_m = float(np.percentile(valid_center, clearance_percentile)) / 1000.0

    upper_roi = _upper_crop_array(depth_image, canopy_crop_ratio)
    valid_upper = upper_roi[upper_roi > 0]
    if valid_upper.size == 0:
        canopy_density_proxy = float("nan")
    else:
        canopy_density_proxy = float(np.mean(valid_upper)) / 1000.0

    return (min_clearance_m, canopy_density_proxy)


def _depth_feature_worker(task: tuple[str, float, float, float]) -> tuple[float, float]:
    path, depth_crop_ratio, canopy_crop_ratio, clearance_percentile = task
    return _compute_depth_features_from_path(
        depth_path=path,
        depth_crop_ratio=depth_crop_ratio,
        canopy_crop_ratio=canopy_crop_ratio,
        clearance_percentile=clearance_percentile,
    )


class FeatureEngine:
    def __init__(
        self,
        loader: LogLoader,
        window_sec: float = DEFAULT_WINDOW_SEC,
        depth_crop_ratio: float = DEFAULT_DEPTH_CROP_RATIO,
        clearance_percentile: float = DEFAULT_CLEARANCE_PERCENTILE,
        canopy_crop_ratio: float = DEFAULT_CANOPY_CROP_RATIO,
        velocity_gate_mps: float = DEFAULT_VELOCITY_GATE_MPS,
        depth_workers: int = DEFAULT_DEPTH_WORKERS,
        depth_cache_dir: str | Path | None = None,
    ) -> None:
        if window_sec <= 0:
            raise ValueError("window_sec must be > 0.")
        if not 0 < depth_crop_ratio <= 1:
            raise ValueError("depth_crop_ratio must satisfy 0 < ratio <= 1.")
        if not 0 <= clearance_percentile <= 100:
            raise ValueError("clearance_percentile must satisfy 0 <= percentile <= 100.")
        if not 0 < canopy_crop_ratio <= 1:
            raise ValueError("canopy_crop_ratio must satisfy 0 < ratio <= 1.")
        if depth_workers <= 0:
            raise ValueError("depth_workers must be > 0.")

        self.loader = loader
        self.window_sec = float(window_sec)
        self.depth_crop_ratio = float(depth_crop_ratio)
        self.clearance_percentile = float(clearance_percentile)
        self.canopy_crop_ratio = float(canopy_crop_ratio)
        self.velocity_gate_mps = float(velocity_gate_mps)
        self.depth_workers = int(depth_workers)
        self.depth_cache_dir = (
            Path(depth_cache_dir).expanduser()
            if depth_cache_dir is not None and str(depth_cache_dir).strip()
            else None
        )
        self._depth_feature_memory_cache: dict[str, tuple[float, float]] = {}
        self._depth_cache_records: dict[str, dict[str, object]] = {}
        self._depth_cache_loaded = False
        self._depth_cache_dirty = False
        self._depth_params_hash = self._compute_depth_params_hash()

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
        clearance_and_canopy = self._compute_depth_features_batch(
            features_df["frame_idx"], features_df["has_depth"]
        )
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
        self, frame_idx: Any, has_depth: Any, *, flush_cache: bool = True,
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

        cache_entry = self._depth_cache_entry(int(frame_number), depth_path)
        cache_key = cache_entry.cache_key if cache_entry is not None else None
        cached = self._read_cached_depth_features(cache_key)
        if cached is not None:
            return cached

        result = _compute_depth_features_from_path(
            depth_path=depth_path,
            depth_crop_ratio=self.depth_crop_ratio,
            canopy_crop_ratio=self.canopy_crop_ratio,
            clearance_percentile=self.clearance_percentile,
        )
        self._write_cached_depth_features(cache_entry, result)
        if flush_cache:
            self._flush_depth_cache()
        return result

    def _compute_depth_features_batch(
        self,
        frame_indices: pd.Series,
        has_depth_series: pd.Series,
    ) -> list[tuple[float, float]]:
        work_items = list(zip(frame_indices.tolist(), has_depth_series.tolist()))
        if self.depth_workers <= 1 or len(work_items) <= 1:
            sequential_results = [
                self._compute_frame_depth_features(
                    frame_idx,
                    has_depth,
                    flush_cache=False,
                )
                for frame_idx, has_depth in work_items
            ]
            self._flush_depth_cache()
            return sequential_results
        results: list[tuple[float, float]] = [
            (float("nan"), float("nan")) for _ in work_items
        ]
        pending_tasks: list[tuple[str, float, float, float]] = []
        pending_meta: list[tuple[int, _DepthCacheEntry | None]] = []

        for row_idx, (frame_idx, has_depth) in enumerate(work_items):
            if not self._as_bool(has_depth):
                continue

            frame_number = pd.to_numeric(frame_idx, errors="coerce")
            if pd.isna(frame_number):
                continue

            depth_path = self._depth_path_for_frame(int(frame_number))
            if not depth_path.exists():
                continue

            cache_entry = self._depth_cache_entry(int(frame_number), depth_path)
            cache_key = cache_entry.cache_key if cache_entry is not None else None
            cached = self._read_cached_depth_features(cache_key)
            if cached is not None:
                results[row_idx] = cached
                continue

            pending_tasks.append(
                (
                    str(depth_path),
                    self.depth_crop_ratio,
                    self.canopy_crop_ratio,
                    self.clearance_percentile,
                )
            )
            pending_meta.append((row_idx, cache_entry))

        if pending_tasks:
            try:
                with ProcessPoolExecutor(max_workers=self.depth_workers) as executor:
                    computed = list(executor.map(_depth_feature_worker, pending_tasks))
            except (OSError, PermissionError):
                computed = [_depth_feature_worker(task) for task in pending_tasks]
            for (row_idx, cache_entry), value in zip(pending_meta, computed, strict=False):
                results[row_idx] = value
                self._write_cached_depth_features(cache_entry, value)
            self._flush_depth_cache()

        return results

    def _compute_depth_params_hash(self) -> str:
        parts = (
            f"{self.depth_crop_ratio:.6f}",
            f"{self.canopy_crop_ratio:.6f}",
            f"{self.clearance_percentile:.6f}",
        )
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    def _depth_cache_entry(self, frame_idx: int, depth_path: Path) -> _DepthCacheEntry | None:
        if self.depth_cache_dir is None:
            return None
        try:
            stat = depth_path.stat()
        except OSError:
            return None
        parts = (
            str(frame_idx),
            str(stat.st_mtime_ns),
            str(stat.st_size),
            self._depth_params_hash,
        )
        return _DepthCacheEntry(
            cache_key=hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest(),
            frame_idx=frame_idx,
            mtime_ns=int(stat.st_mtime_ns),
            size_bytes=int(stat.st_size),
            params_hash=self._depth_params_hash,
        )

    def _depth_cache_path(self) -> Path:
        if self.depth_cache_dir is None:
            raise ValueError("depth_cache_dir is not configured.")
        return self.depth_cache_dir / DEFAULT_DEPTH_CACHE_FILENAME

    def _ensure_depth_cache_loaded(self) -> None:
        if self.depth_cache_dir is None or self._depth_cache_loaded:
            return
        self._depth_cache_loaded = True
        cache_path = self._depth_cache_path()
        if not cache_path.exists():
            return

        try:
            cache_df = pd.read_parquet(cache_path)
        except (OSError, ValueError, ImportError):
            return

        required_columns = {
            "cache_key",
            "frame_idx",
            "min_clearance_m",
            "canopy_density_proxy",
            "mtime_ns",
            "size_bytes",
            "params_hash",
        }
        if not required_columns.issubset(cache_df.columns):
            return

        for row in cache_df.to_dict(orient="records"):
            cache_key = str(row.get("cache_key", "")).strip()
            if not cache_key:
                continue

            clearance = self._as_optional_float(row.get("min_clearance_m"))
            canopy = self._as_optional_float(row.get("canopy_density_proxy"))
            result = (
                clearance if clearance is not None else float("nan"),
                canopy if canopy is not None else float("nan"),
            )
            self._depth_feature_memory_cache[cache_key] = result
            self._depth_cache_records[cache_key] = {
                "cache_key": cache_key,
                "frame_idx": int(row.get("frame_idx", -1)),
                "min_clearance_m": clearance,
                "canopy_density_proxy": canopy,
                "mtime_ns": int(row.get("mtime_ns", 0)),
                "size_bytes": int(row.get("size_bytes", 0)),
                "params_hash": str(row.get("params_hash", "")),
            }

    def _read_cached_depth_features(self, cache_key: str | None) -> tuple[float, float] | None:
        if self.depth_cache_dir is None or cache_key is None:
            return None
        self._ensure_depth_cache_loaded()
        if cache_key in self._depth_feature_memory_cache:
            return self._depth_feature_memory_cache[cache_key]
        return None

    def _write_cached_depth_features(
        self,
        cache_entry: _DepthCacheEntry | None,
        result: tuple[float, float],
    ) -> None:
        if self.depth_cache_dir is None or cache_entry is None:
            return
        self._ensure_depth_cache_loaded()
        clearance = result[0] if np.isfinite(result[0]) else None
        canopy = result[1] if np.isfinite(result[1]) else None
        self._depth_feature_memory_cache[cache_entry.cache_key] = result
        self._depth_cache_records[cache_entry.cache_key] = {
            "cache_key": cache_entry.cache_key,
            "frame_idx": cache_entry.frame_idx,
            "min_clearance_m": clearance,
            "canopy_density_proxy": canopy,
            "mtime_ns": cache_entry.mtime_ns,
            "size_bytes": cache_entry.size_bytes,
            "params_hash": cache_entry.params_hash,
        }
        self._depth_cache_dirty = True

    def _flush_depth_cache(self) -> None:
        if self.depth_cache_dir is None or not self._depth_cache_dirty:
            return

        cache_df = pd.DataFrame.from_records(
            list(self._depth_cache_records.values()),
            columns=[
                "cache_key",
                "frame_idx",
                "min_clearance_m",
                "canopy_density_proxy",
                "mtime_ns",
                "size_bytes",
                "params_hash",
            ],
        )
        try:
            self.depth_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self._depth_cache_path()
            tmp_path = cache_path.with_suffix(".tmp.parquet")
            cache_df.to_parquet(tmp_path, index=False)
            try:
                tmp_path.replace(cache_path)
            except OSError:
                cache_df.to_parquet(cache_path, index=False)
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            self._depth_cache_dirty = False
        except (OSError, ValueError, ImportError):
            return

    @staticmethod
    def _as_optional_float(value: object) -> float | None:
        if value is None:
            return None
        if not isinstance(value, (int, float, str, bytes, bytearray)):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(parsed):
            return None
        return parsed

    def _upper_crop(self, image: np.ndarray) -> np.ndarray:
        """Crop the upper portion of the image (canopy region)."""
        return _upper_crop_array(image, self.canopy_crop_ratio)

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
        return _center_crop_array(image, self.depth_crop_ratio)

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
