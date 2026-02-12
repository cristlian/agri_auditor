from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_MANIFEST_NAME = "manifest.csv"
DEFAULT_CALIBRATION_CANDIDATES = ("calibrations.json", "calibration.json")
DEFAULT_POSE_PREFIX = "pose_front_center_stereo_left"


@dataclass(frozen=True)
class CameraModel:
    camera_name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    K: np.ndarray


class LogLoader:
    def __init__(
        self,
        data_dir: Path | str,
        manifest_name: str = DEFAULT_MANIFEST_NAME,
        calibration_name: str | None = None,
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.manifest_path = self.data_dir / manifest_name
        self.frames_dir = self.data_dir / "frames"
        self.calibration_name = calibration_name
        self._manifest_df: pd.DataFrame | None = None
        self._calibrations: dict[str, dict[str, float | int]] | None = None
        self._camera_shape_cache: dict[str, tuple[int, int]] = {}

    def load_manifest(self) -> pd.DataFrame:
        if self._manifest_df is not None:
            return self._manifest_df.copy()

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        df = pd.read_csv(self.manifest_path)
        required_cols = {
            "frame_idx",
            "timestamp_sec",
            f"{DEFAULT_POSE_PREFIX}_x",
            f"{DEFAULT_POSE_PREFIX}_y",
            f"{DEFAULT_POSE_PREFIX}_z",
        }
        missing = sorted(required_cols - set(df.columns))
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")

        numeric_cols = [
            "timestamp_sec",
            f"{DEFAULT_POSE_PREFIX}_x",
            f"{DEFAULT_POSE_PREFIX}_y",
            f"{DEFAULT_POSE_PREFIX}_z",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["velocity_mps"] = self.compute_velocity(
            df,
            x_col=f"{DEFAULT_POSE_PREFIX}_x",
            y_col=f"{DEFAULT_POSE_PREFIX}_y",
            z_col=f"{DEFAULT_POSE_PREFIX}_z",
            ts_col="timestamp_sec",
        )
        self._manifest_df = df
        return df.copy()

    def compute_velocity(
        self,
        df: pd.DataFrame,
        x_col: str = f"{DEFAULT_POSE_PREFIX}_x",
        y_col: str = f"{DEFAULT_POSE_PREFIX}_y",
        z_col: str = f"{DEFAULT_POSE_PREFIX}_z",
        ts_col: str = "timestamp_sec",
    ) -> pd.Series:
        for col in (x_col, y_col, z_col, ts_col):
            if col not in df.columns:
                raise ValueError(f"Cannot compute velocity. Missing column: {col}")

        x = pd.to_numeric(df[x_col], errors="coerce")
        y = pd.to_numeric(df[y_col], errors="coerce")
        z = pd.to_numeric(df[z_col], errors="coerce")
        t = pd.to_numeric(df[ts_col], errors="coerce")

        dx = x.diff()
        dy = y.diff()
        dz = z.diff()
        dt = t.diff()

        dist = np.sqrt((dx**2) + (dy**2) + (dz**2))
        velocity = dist / dt
        invalid = dt.isna() | (dt <= 0.0) | dist.isna()
        velocity[invalid] = np.nan
        return velocity.rename("velocity_mps")

    def load_calibrations(self) -> dict[str, dict[str, float | int]]:
        if self._calibrations is not None:
            return self._calibrations

        calibration_path = self._resolve_calibration_path()
        raw = pd.read_json(calibration_path, typ="series")
        calibrations: dict[str, dict[str, float | int]] = {}
        for camera_name, params in raw.items():
            if not isinstance(params, dict):
                raise ValueError(f"Invalid calibration payload for camera '{camera_name}'.")

            required = ("fx", "fy", "cx", "cy", "width", "height")
            missing = [key for key in required if key not in params]
            if missing:
                raise ValueError(
                    f"Camera '{camera_name}' missing calibration fields: {missing}"
                )

            calibrations[camera_name] = {
                "fx": float(params["fx"]),
                "fy": float(params["fy"]),
                "cx": float(params["cx"]),
                "cy": float(params["cy"]),
                "width": int(params["width"]),
                "height": int(params["height"]),
            }

        self._calibrations = calibrations
        return calibrations

    def get_camera_model(
        self,
        camera_name: str,
        image_shape: tuple[int, int] | tuple[int, int, int] | None = None,
    ) -> CameraModel:
        calibrations = self.load_calibrations()
        if camera_name not in calibrations:
            available = ", ".join(sorted(calibrations))
            raise ValueError(f"Unknown camera '{camera_name}'. Available: {available}")

        params = calibrations[camera_name]
        width = int(params["width"])
        height = int(params["height"])

        if image_shape is not None:
            height, width = self._parse_image_shape(image_shape)
        elif width <= 0 or height <= 0:
            height, width = self._infer_camera_shape(camera_name)

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Could not resolve non-zero dimensions for camera '{camera_name}'."
            )

        k_matrix = np.array(
            [
                [float(params["fx"]), 0.0, float(params["cx"])],
                [0.0, float(params["fy"]), float(params["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        return CameraModel(
            camera_name=camera_name,
            width=width,
            height=height,
            fx=float(params["fx"]),
            fy=float(params["fy"]),
            cx=float(params["cx"]),
            cy=float(params["cy"]),
            K=k_matrix,
        )

    def get_image(self, frame_idx: int, camera_name: str) -> np.ndarray:
        camera_dir = self.frames_dir / camera_name
        if not camera_dir.exists():
            raise ValueError(f"Unknown camera folder: {camera_name}")

        frame_stem = f"{int(frame_idx):04d}"
        for suffix in (".jpg", ".png", ".jpeg"):
            path = camera_dir / f"{frame_stem}{suffix}"
            if path.exists():
                with Image.open(path) as image:
                    return np.asarray(image)

        raise FileNotFoundError(
            f"Frame {frame_stem} not found for camera '{camera_name}' in {camera_dir}"
        )

    def _resolve_calibration_path(self) -> Path:
        if self.calibration_name:
            path = self.data_dir / self.calibration_name
            if not path.exists():
                raise FileNotFoundError(f"Calibration file not found: {path}")
            return path

        for candidate in DEFAULT_CALIBRATION_CANDIDATES:
            path = self.data_dir / candidate
            if path.exists():
                return path

        raise FileNotFoundError(
            "Calibration file not found. Looked for "
            f"{', '.join(DEFAULT_CALIBRATION_CANDIDATES)} in {self.data_dir}"
        )

    def _infer_camera_shape(self, camera_name: str) -> tuple[int, int]:
        if camera_name in self._camera_shape_cache:
            return self._camera_shape_cache[camera_name]

        camera_dir = self.frames_dir / camera_name
        if not camera_dir.exists():
            raise ValueError(f"Camera directory does not exist: {camera_dir}")

        image_path = self._find_first_image(camera_dir)
        if image_path is None:
            raise FileNotFoundError(f"No image files found in: {camera_dir}")

        with Image.open(image_path) as image:
            width, height = image.size

        self._camera_shape_cache[camera_name] = (height, width)
        return height, width

    @staticmethod
    def _parse_image_shape(
        image_shape: tuple[int, int] | tuple[int, int, int],
    ) -> tuple[int, int]:
        if len(image_shape) == 2:
            height, width = image_shape
        elif len(image_shape) == 3:
            height, width = image_shape[:2]
        else:
            raise ValueError(
                "image_shape must be (height, width) or (height, width, channels)."
            )
        return int(height), int(width)

    @staticmethod
    def _find_first_image(camera_dir: Path) -> Path | None:
        for suffix in ("*.jpg", "*.png", "*.jpeg"):
            matches = sorted(camera_dir.glob(suffix))
            if matches:
                return matches[0]
        return None
