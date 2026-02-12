from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .features import FeatureEngine
from .ingestion import LogLoader


DEFAULT_ROUGHNESS_WEIGHT = 0.35
DEFAULT_PROXIMITY_WEIGHT = 0.15
DEFAULT_YAW_RATE_WEIGHT = 0.20
DEFAULT_IMU_FAULT_WEIGHT = 0.15
DEFAULT_LOCALIZATION_FAULT_WEIGHT = 0.15
DEFAULT_CLEARANCE_SAFE_M = 10.0
DEFAULT_PEAK_DISTANCE_FRAMES = 150
DEFAULT_TOP_K = 5
DEFAULT_PRIMARY_CAMERA = "front_center_stereo_left"
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
DEFAULT_TIMEOUT_SEC = 20.0
DEFAULT_TEMPERATURE = 1.0  # Gemini 3 is optimized for temperature=1.0; lower values may cause looping
DEFAULT_THINKING_LEVEL = "low"  # Minimizes latency for simple caption tasks
DEFAULT_MAX_WORDS = 20
UNAVAILABLE_CAPTION = "AI Analysis Unavailable"

# All 6 surround-view cameras in standard order
ALL_CAMERAS = (
    "front_left",
    "front_center_stereo_left",
    "front_right",
    "rear_left",
    "rear_center_stereo_left",
    "rear_right",
)

SYSTEM_PROMPT = (
    "Autonomous tractor front camera. "
    "Terrain type, crop/canopy state, navigation hazards. "
    "Clinical, max 20 words."
)


class GeminiConfigError(RuntimeError):
    """Raised when Gemini configuration is missing or invalid."""


@dataclass(frozen=True)
class EventCandidate:
    frame_idx: int
    timestamp_sec: float
    timestamp_iso_utc: str
    severity_score: float
    roughness: float | None
    min_clearance_m: float | None
    yaw_rate: float | None
    imu_correlation: float | None
    pose_confidence: float | None
    roughness_norm: float
    proximity_norm: float
    yaw_rate_norm: float
    imu_fault_norm: float
    localization_fault_norm: float
    event_type: str
    gps_lat: float | None
    gps_lon: float | None


@dataclass(frozen=True)
class Event:
    event_rank: int
    frame_idx: int
    timestamp_sec: float
    timestamp_iso_utc: str
    severity_score: float
    roughness: float | None
    min_clearance_m: float | None
    yaw_rate: float | None
    imu_correlation: float | None
    pose_confidence: float | None
    roughness_norm: float
    proximity_norm: float
    yaw_rate_norm: float
    imu_fault_norm: float
    localization_fault_norm: float
    event_type: str
    gps_lat: float | None
    gps_lon: float | None
    primary_camera: str
    camera_paths: dict[str, str]  # camera_name -> resolved path
    gemini_caption: str
    gemini_model: str | None
    gemini_source: str
    gemini_latency_ms: float | None


@dataclass(frozen=True)
class GeminiAnalysisResult:
    caption: str
    model: str | None
    source: str
    latency_ms: float | None
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    total_tokens: int | None = None


class EventDetector:
    def __init__(
        self,
        roughness_weight: float = DEFAULT_ROUGHNESS_WEIGHT,
        proximity_weight: float = DEFAULT_PROXIMITY_WEIGHT,
        yaw_rate_weight: float = DEFAULT_YAW_RATE_WEIGHT,
        imu_fault_weight: float = DEFAULT_IMU_FAULT_WEIGHT,
        localization_fault_weight: float = DEFAULT_LOCALIZATION_FAULT_WEIGHT,
        clearance_safe_m: float = DEFAULT_CLEARANCE_SAFE_M,
    ) -> None:
        weights = [roughness_weight, proximity_weight, yaw_rate_weight,
                    imu_fault_weight, localization_fault_weight]
        if any(w < 0 for w in weights):
            raise ValueError("All scoring weights must be >= 0.")
        if sum(weights) <= 0:
            raise ValueError("At least one scoring weight must be > 0.")
        if clearance_safe_m <= 0:
            raise ValueError("clearance_safe_m must be > 0.")

        self.roughness_weight = float(roughness_weight)
        self.proximity_weight = float(proximity_weight)
        self.yaw_rate_weight = float(yaw_rate_weight)
        self.imu_fault_weight = float(imu_fault_weight)
        self.localization_fault_weight = float(localization_fault_weight)
        self.clearance_safe_m = float(clearance_safe_m)
        self.last_peak_count: int = 0
        self.last_peak_indices: np.ndarray = np.array([], dtype=np.int64)

    def find_events(
        self,
        df: pd.DataFrame,
        top_k: int = DEFAULT_TOP_K,
        distance_frames: int = DEFAULT_PEAK_DISTANCE_FRAMES,
    ) -> list[EventCandidate]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0.")
        if distance_frames <= 0:
            raise ValueError("distance_frames must be > 0.")

        scored_df = self.score_dataframe(df)
        severity = pd.to_numeric(scored_df["severity_score"], errors="coerce").fillna(0.0)
        peak_indices, _ = find_peaks(severity.to_numpy(dtype=np.float64), distance=distance_frames)
        self.last_peak_count = int(len(peak_indices))
        self.last_peak_indices = peak_indices.astype(np.int64, copy=False)

        if self.last_peak_count == 0:
            return []

        peak_rows = scored_df.iloc[peak_indices].copy()
        peak_rows = peak_rows.sort_values("severity_score", ascending=False).head(top_k)

        candidates: list[EventCandidate] = []
        for row in peak_rows.itertuples(index=False):
            frame_idx = self._coerce_int(getattr(row, "frame_idx", None), default=0)
            timestamp_sec = self._coerce_float(getattr(row, "timestamp_sec", None), default=0.0)
            roughness_norm = self._coerce_float(getattr(row, "roughness_norm", None), default=0.0)
            proximity_norm = self._coerce_float(
                getattr(row, "proximity_norm", None), default=0.0
            )
            yaw_rate_norm = self._coerce_float(getattr(row, "yaw_rate_norm", None), default=0.0)
            imu_fault_norm = self._coerce_float(getattr(row, "imu_fault_norm", None), default=0.0)
            loc_fault_norm = self._coerce_float(getattr(row, "localization_fault_norm", None), default=0.0)
            candidates.append(
                EventCandidate(
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    timestamp_iso_utc=_timestamp_to_iso(timestamp_sec),
                    severity_score=self._coerce_float(
                        getattr(row, "severity_score", None), default=0.0
                    ),
                    roughness=self._coerce_optional_float(getattr(row, "roughness", None)),
                    min_clearance_m=self._coerce_optional_float(
                        getattr(row, "min_clearance_m", None)
                    ),
                    yaw_rate=self._coerce_optional_float(getattr(row, "yaw_rate", None)),
                    imu_correlation=self._coerce_optional_float(getattr(row, "imu_correlation", None)),
                    pose_confidence=self._coerce_optional_float(getattr(row, "pose_confidence", None)),
                    roughness_norm=roughness_norm,
                    proximity_norm=proximity_norm,
                    yaw_rate_norm=yaw_rate_norm,
                    imu_fault_norm=imu_fault_norm,
                    localization_fault_norm=loc_fault_norm,
                    event_type=self._classify_event(
                        roughness_norm, proximity_norm, yaw_rate_norm,
                        imu_fault_norm, loc_fault_norm,
                    ),
                    gps_lat=self._coerce_optional_float(getattr(row, "gps_lat", None)),
                    gps_lon=self._coerce_optional_float(getattr(row, "gps_lon", None)),
                )
            )
        return candidates

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_required_columns(df)

        scored_df = df.copy()
        proximity_norm = self._proximity_normalize(scored_df["min_clearance_m"])

        # Safe-value imputation before normalization:
        # roughness -> 0, yaw_rate -> 0, imu_correlation -> 1.0, pose_confidence -> 100.
        roughness_raw = pd.to_numeric(scored_df["roughness"], errors="coerce").fillna(0.0)

        yaw_rate_raw = (
            pd.to_numeric(scored_df["yaw_rate"], errors="coerce")
            if "yaw_rate" in scored_df.columns
            else pd.Series(np.nan, index=scored_df.index, dtype="float64")
        )
        yaw_rate_abs_raw = yaw_rate_raw.abs().fillna(0.0)

        imu_corr_raw = (
            pd.to_numeric(scored_df["imu_correlation"], errors="coerce")
            if "imu_correlation" in scored_df.columns
            else pd.Series(np.nan, index=scored_df.index, dtype="float64")
        )
        imu_corr_raw = imu_corr_raw.fillna(1.0)

        pose_conf_raw = (
            pd.to_numeric(scored_df["pose_confidence"], errors="coerce")
            if "pose_confidence" in scored_df.columns
            else pd.Series(np.nan, index=scored_df.index, dtype="float64")
        )
        pose_conf_raw = pose_conf_raw.fillna(100.0)

        roughness_norm = self._minmax_normalize(roughness_raw)
        yaw_rate_norm = self._minmax_normalize(yaw_rate_abs_raw)
        # Inverse semantics: lower IMU correlation means higher sensor-fault severity.
        imu_fault_norm = self._invert_normalize(imu_corr_raw)
        # Inverse semantics: lower pose confidence means higher localization-fault severity.
        localization_fault_norm = self._invert_normalize(pose_conf_raw)

        severity_score = (
            self.roughness_weight * roughness_norm
            + self.yaw_rate_weight * yaw_rate_norm
            + self.imu_fault_weight * imu_fault_norm
            + self.localization_fault_weight * localization_fault_norm
        )
        severity_score = (
            severity_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

        scored_df["roughness_norm"] = roughness_norm.astype("float64")
        scored_df["proximity_norm"] = proximity_norm.astype("float64")
        scored_df["yaw_rate_norm"] = yaw_rate_norm.astype("float64")
        scored_df["imu_fault_norm"] = imu_fault_norm.astype("float64")
        scored_df["localization_fault_norm"] = localization_fault_norm.astype("float64")
        scored_df["severity_score"] = severity_score.astype("float64")
        return scored_df

    @staticmethod
    def _validate_required_columns(df: pd.DataFrame) -> None:
        required = {"frame_idx", "timestamp_sec", "roughness", "min_clearance_m"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(
                f"DataFrame missing required columns for event detection: {missing}"
            )

    def _proximity_normalize(self, min_clearance_m: pd.Series) -> pd.Series:
        clearance = pd.to_numeric(min_clearance_m, errors="coerce").astype("float64")
        filled = clearance.fillna(self.clearance_safe_m)

        finite = filled[np.isfinite(filled)]
        if finite.empty:
            return pd.Series(np.zeros(len(filled), dtype=np.float64), index=filled.index)

        min_clearance = float(finite.min())
        denominator = self.clearance_safe_m - min_clearance
        if not np.isfinite(denominator) or denominator <= 0.0:
            return pd.Series(np.zeros(len(filled), dtype=np.float64), index=filled.index)

        proximity_norm = (self.clearance_safe_m - filled) / denominator
        proximity_norm = proximity_norm.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return proximity_norm.clip(lower=0.0, upper=1.0)

    @staticmethod
    def _minmax_normalize(values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce").astype("float64")
        finite_values = numeric[np.isfinite(numeric)]
        if finite_values.empty:
            return pd.Series(np.zeros(len(numeric), dtype=np.float64), index=numeric.index)

        min_value = float(finite_values.min())
        max_value = float(finite_values.max())
        filled = numeric.fillna(min_value)

        denominator = max_value - min_value
        if not np.isfinite(denominator) or denominator <= 0.0:
            return pd.Series(np.zeros(len(filled), dtype=np.float64), index=filled.index)

        normalized = (filled - min_value) / denominator
        normalized = normalized.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return normalized.clip(lower=0.0, upper=1.0)

    @staticmethod
    def _invert_normalize(values: pd.Series) -> pd.Series:
        """Min-max normalize then invert: low original value 鈫?high output."""
        numeric = pd.to_numeric(values, errors="coerce").astype("float64")
        finite_values = numeric[np.isfinite(numeric)]
        if finite_values.empty:
            return pd.Series(np.zeros(len(numeric), dtype=np.float64), index=numeric.index)

        min_value = float(finite_values.min())
        max_value = float(finite_values.max())
        filled = numeric.fillna(max_value)  # NaN 鈫?high value 鈫?low fault

        denominator = max_value - min_value
        if not np.isfinite(denominator) or denominator <= 0.0:
            return pd.Series(np.zeros(len(filled), dtype=np.float64), index=filled.index)

        # Invert: low confidence 鈫?high fault score
        normalized = (max_value - filled) / denominator
        normalized = normalized.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return normalized.clip(lower=0.0, upper=1.0)

    @staticmethod
    def _classify_event(
        roughness_norm: float,
        proximity_norm: float,
        yaw_rate_norm: float = 0.0,
        imu_fault_norm: float = 0.0,
        localization_fault_norm: float = 0.0,
    ) -> str:
        """Classify by the dominant signal contributor."""
        scores = {
            "roughness": float(roughness_norm),
            "proximity": float(proximity_norm),
            "steering": float(yaw_rate_norm),
            "sensor_fault": float(imu_fault_norm),
            "localization_fault": float(localization_fault_norm),
        }
        top = max(scores, key=scores.get)  # type: ignore[arg-type]
        vals = sorted(scores.values(), reverse=True)
        if len(vals) >= 2 and abs(vals[0] - vals[1]) <= 0.1:
            return "mixed"
        return top

    @staticmethod
    def _coerce_optional_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(parsed):
            return None
        return parsed

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        optional_value = EventDetector._coerce_optional_float(value)
        return default if optional_value is None else optional_value

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed


class GeminiAnalyst:
    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str | None = None,
        prompt: str = SYSTEM_PROMPT,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        temperature: float = DEFAULT_TEMPERATURE,
        max_words: int = DEFAULT_MAX_WORDS,
        thinking_level: str = DEFAULT_THINKING_LEVEL,
    ) -> None:
        # Auto-load .env file for API key if present
        load_dotenv(override=False)
        resolved_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        if not resolved_key:
            raise GeminiConfigError("GEMINI_API_KEY is required for GeminiAnalyst.")
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be > 0.")
        if max_words <= 0:
            raise ValueError("max_words must be > 0.")

        self.api_key = resolved_key
        self.model = model
        self.prompt = prompt
        self.timeout_sec = float(timeout_sec)
        self.temperature = float(temperature)
        self.max_words = int(max_words)
        self.thinking_level = str(thinking_level)

    def analyze_image(self, image_path: str | Path, model: str | None = None) -> GeminiAnalysisResult:
        path = Path(image_path)
        selected_model = model or self.model
        if not path.exists():
            return GeminiAnalysisResult(
                caption=UNAVAILABLE_CAPTION,
                model=selected_model,
                source="unavailable",
                latency_ms=None,
                error=f"Image not found: {path}",
            )

        try:
            image_bytes = path.read_bytes()
        except OSError as exc:
            return GeminiAnalysisResult(
                caption=UNAVAILABLE_CAPTION,
                model=selected_model,
                source="unavailable",
                latency_ms=None,
                error=f"Failed to read image: {exc}",
            )

        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        try:
            result = self._analyze_with_sdk(
                image_bytes=image_bytes,
                mime_type=mime_type,
                model=selected_model,
            )
            return self._normalize_result(result)
        except Exception as sdk_error:  # pragma: no cover - exercised via mock fallback test
            sdk_message = str(sdk_error)

        try:
            result = self._analyze_with_rest(
                image_bytes=image_bytes,
                mime_type=mime_type,
                model=selected_model,
            )
            return self._normalize_result(result)
        except Exception as rest_error:
            return GeminiAnalysisResult(
                caption=UNAVAILABLE_CAPTION,
                model=selected_model,
                source="unavailable",
                latency_ms=None,
                error=f"sdk_error={sdk_message}; rest_error={rest_error}",
            )

    def _normalize_result(self, result: GeminiAnalysisResult) -> GeminiAnalysisResult:
        normalized_caption = _truncate_words(result.caption, self.max_words)
        if not normalized_caption:
            normalized_caption = UNAVAILABLE_CAPTION
        return GeminiAnalysisResult(
            caption=normalized_caption,
            model=result.model,
            source=result.source,
            latency_ms=result.latency_ms,
            error=result.error,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            thinking_tokens=result.thinking_tokens,
            total_tokens=result.total_tokens,
        )

    def _analyze_with_sdk(
        self,
        image_bytes: bytes,
        mime_type: str,
        model: str,
    ) -> GeminiAnalysisResult:
        start = time.perf_counter()
        try:
            from google import genai  # type: ignore
        except Exception as exc:
            raise RuntimeError("google-genai SDK is unavailable.") from exc

        client = genai.Client(api_key=self.api_key)
        encoded_image = base64.b64encode(image_bytes).decode("ascii")
        contents = [
            {
                "role": "user",
                "parts": [
                    {"text": self.prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": encoded_image,
                        }
                    },
                ],
            }
        ]
        config: dict[str, Any] = {"temperature": self.temperature}
        if self._model_supports_thinking_level(model):
            config["thinking_config"] = {"thinking_level": self.thinking_level}
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        caption = self._extract_sdk_caption(response)
        if not caption:
            raise RuntimeError("SDK response did not contain caption text.")

        input_tokens, output_tokens, thinking_tokens, total_tokens = self._extract_usage_metadata(response)
        return GeminiAnalysisResult(
            caption=caption,
            model=model,
            source="sdk",
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            total_tokens=total_tokens,
        )

    def _analyze_with_rest(
        self,
        image_bytes: bytes,
        mime_type: str,
        model: str,
    ) -> GeminiAnalysisResult:
        start = time.perf_counter()
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:"
            f"generateContent?key={self.api_key}"
        )
        encoded_image = base64.b64encode(image_bytes).decode("ascii")
        request_body: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": self.prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": encoded_image,
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
            },
        }
        if self._model_supports_thinking_level(model):
            request_body["generationConfig"]["thinkingConfig"] = {
                "thinkingLevel": self.thinking_level.upper()
            }
        payload = json.dumps(request_body).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"REST HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"REST connection error: {exc.reason}") from exc

        latency_ms = (time.perf_counter() - start) * 1000.0
        caption = self._extract_rest_caption(response_data)
        if not caption:
            raise RuntimeError("REST response did not contain caption text.")

        input_tokens, output_tokens, thinking_tokens, total_tokens = self._extract_usage_metadata(
            response_data
        )
        return GeminiAnalysisResult(
            caption=caption,
            model=model,
            source="rest",
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            total_tokens=total_tokens,
        )

    @staticmethod
    def _model_supports_thinking_level(model: str) -> bool:
        """Return True if the model supports the Gemini 3 ``thinking_level`` parameter."""
        return "gemini-3" in model.lower()

    @staticmethod
    def _extract_sdk_caption(response: Any) -> str | None:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        candidates = getattr(response, "candidates", None)
        if candidates is not None:
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None) if content is not None else None
                if parts is None:
                    continue
                for part in parts:
                    part_text = getattr(part, "text", None)
                    if isinstance(part_text, str) and part_text.strip():
                        return part_text.strip()

        if isinstance(response, dict):
            return GeminiAnalyst._extract_rest_caption(response)
        return None

    @staticmethod
    def _extract_rest_caption(response_payload: dict[str, Any]) -> str | None:
        candidates = response_payload.get("candidates", [])
        if not isinstance(candidates, list):
            return None

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content", {})
            if not isinstance(content, dict):
                continue
            parts = content.get("parts", [])
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
        return None

    @staticmethod
    def _extract_usage_metadata(
        payload: Any,
    ) -> tuple[int | None, int | None, int | None, int | None]:
        """Return (input_tokens, output_tokens, thinking_tokens, total_tokens)."""
        usage_obj: Any = None
        if isinstance(payload, dict):
            usage_obj = payload.get("usageMetadata") or payload.get("usage_metadata")
            input_tokens = _extract_int_from_mapping(
                usage_obj,
                ["promptTokenCount", "prompt_token_count", "inputTokenCount"],
            )
            output_tokens = _extract_int_from_mapping(
                usage_obj,
                [
                    "candidatesTokenCount",
                    "candidates_token_count",
                    "outputTokenCount",
                ],
            )
            thinking_tokens = _extract_int_from_mapping(
                usage_obj,
                ["thoughtsTokenCount", "thoughts_token_count", "thinkingTokenCount"],
            )
            total_tokens = _extract_int_from_mapping(
                usage_obj, ["totalTokenCount", "total_token_count"]
            )
        else:
            usage_obj = getattr(payload, "usage_metadata", None) or getattr(
                payload, "usageMetadata", None
            )
            input_tokens = _extract_int_from_object(
                usage_obj,
                ["prompt_token_count", "promptTokenCount", "input_token_count"],
            )
            output_tokens = _extract_int_from_object(
                usage_obj,
                [
                    "candidates_token_count",
                    "candidatesTokenCount",
                    "output_token_count",
                ],
            )
            thinking_tokens = _extract_int_from_object(
                usage_obj,
                ["thoughts_token_count", "thoughtsTokenCount", "thinking_token_count"],
            )
            total_tokens = _extract_int_from_object(
                usage_obj, ["total_token_count", "totalTokenCount"]
            )

        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        return input_tokens, output_tokens, thinking_tokens, total_tokens


class IntelligenceOrchestrator:
    def __init__(
        self,
        loader: LogLoader,
        detector: EventDetector | None = None,
        analyst: GeminiAnalyst | None = None,
        primary_camera: str = DEFAULT_PRIMARY_CAMERA,
        cameras: tuple[str, ...] = ALL_CAMERAS,
    ) -> None:
        self.loader = loader
        self.detector = detector or EventDetector()
        self.analyst = analyst
        self.primary_camera = primary_camera
        self.cameras = cameras

    def build_events(
        self,
        features_df: pd.DataFrame | None = None,
        top_k: int = DEFAULT_TOP_K,
        distance_frames: int = DEFAULT_PEAK_DISTANCE_FRAMES,
        model: str | None = None,
    ) -> list[Event]:
        feature_table = (
            FeatureEngine(loader=self.loader).build_features()
            if features_df is None
            else features_df.copy()
        )
        candidates = self.detector.find_events(
            feature_table,
            top_k=top_k,
            distance_frames=distance_frames,
        )

        events: list[Event] = []
        for rank, candidate in enumerate(candidates, start=1):
            camera_paths = self._resolve_all_camera_paths(candidate.frame_idx)
            primary_path = camera_paths.get(
                self.primary_camera,
                next(iter(camera_paths.values()), ""),
            )
            analysis = self._analyze_event_frame(
                image_path=Path(primary_path) if primary_path else Path("."),
                model=model,
            )
            events.append(
                Event(
                    event_rank=rank,
                    frame_idx=candidate.frame_idx,
                    timestamp_sec=candidate.timestamp_sec,
                    timestamp_iso_utc=candidate.timestamp_iso_utc,
                    severity_score=candidate.severity_score,
                    roughness=candidate.roughness,
                    min_clearance_m=candidate.min_clearance_m,
                    yaw_rate=candidate.yaw_rate,
                    imu_correlation=candidate.imu_correlation,
                    pose_confidence=candidate.pose_confidence,
                    roughness_norm=candidate.roughness_norm,
                    proximity_norm=candidate.proximity_norm,
                    yaw_rate_norm=candidate.yaw_rate_norm,
                    imu_fault_norm=candidate.imu_fault_norm,
                    localization_fault_norm=candidate.localization_fault_norm,
                    event_type=candidate.event_type,
                    gps_lat=candidate.gps_lat,
                    gps_lon=candidate.gps_lon,
                    primary_camera=self.primary_camera,
                    camera_paths=camera_paths,
                    gemini_caption=analysis.caption,
                    gemini_model=analysis.model,
                    gemini_source=analysis.source,
                    gemini_latency_ms=analysis.latency_ms,
                )
            )
        return events

    @staticmethod
    def serialize_events(events: list[Event]) -> list[dict[str, Any]]:
        return [asdict(event) for event in events]

    def _analyze_event_frame(
        self,
        image_path: Path,
        model: str | None,
    ) -> GeminiAnalysisResult:
        if self.analyst is None:
            return GeminiAnalysisResult(
                caption=UNAVAILABLE_CAPTION,
                model=model,
                source="unavailable",
                latency_ms=None,
                error="Gemini analyst disabled.",
            )
        try:
            return self.analyst.analyze_image(image_path=image_path, model=model)
        except Exception as exc:
            return GeminiAnalysisResult(
                caption=UNAVAILABLE_CAPTION,
                model=model,
                source="unavailable",
                latency_ms=None,
                error=f"Unexpected analyst error: {exc}",
            )

    def _resolve_all_camera_paths(self, frame_idx: int) -> dict[str, str]:
        """Resolve image paths for all surround-view cameras at this frame."""
        paths: dict[str, str] = {}
        for camera in self.cameras:
            path = self._resolve_event_image_path(frame_idx, camera)
            paths[camera] = str(path)
        return paths

    def _resolve_event_image_path(self, frame_idx: int, camera_name: str | None = None) -> Path:
        cam = camera_name or self.primary_camera
        camera_dir = self.loader.frames_dir / cam
        stem = f"{int(frame_idx):04d}"
        for suffix in (".jpg", ".png", ".jpeg"):
            candidate = camera_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        return camera_dir / f"{stem}.jpg"


def _extract_int_from_mapping(mapping: Any, keys: list[str]) -> int | None:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = mapping.get(key)
        maybe_int = _safe_int(value)
        if maybe_int is not None:
            return maybe_int
    return None


def _extract_int_from_object(obj: Any, attrs: list[str]) -> int | None:
    if obj is None:
        return None
    for attr in attrs:
        value = getattr(obj, attr, None)
        maybe_int = _safe_int(value)
        if maybe_int is not None:
            return maybe_int
    return None


def _safe_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _timestamp_to_iso(timestamp_sec: float) -> str:
    if not np.isfinite(timestamp_sec):
        return ""
    return datetime.fromtimestamp(float(timestamp_sec), tz=timezone.utc).isoformat()


def _truncate_words(text: str, max_words: int) -> str:
    words = str(text).strip().split()
    if not words:
        return ""
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


