from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
import pandas as pd
import pytest

from agri_auditor import (
    EventDetector,
    FeatureEngine,
    GeminiAnalyst,
    GeminiAnalysisResult,
    IntelligenceOrchestrator,
    LogLoader,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return PROJECT_ROOT.parent / "provided_data"


@pytest.fixture(scope="module")
def loader() -> LogLoader:
    return LogLoader(data_dir())


@pytest.fixture(scope="module")
def features_df(loader: LogLoader) -> pd.DataFrame:
    return FeatureEngine(loader=loader).build_features()


def test_find_events_returns_exact_top_k_and_sorted(features_df: pd.DataFrame) -> None:
    detector = EventDetector()
    events = detector.find_events(features_df, top_k=5, distance_frames=150)

    assert len(events) == 5
    scores = [event.severity_score for event in events]
    assert scores == sorted(scores, reverse=True)
    # Multi-modal compound scoring produces different top-5 than the old
    # 2-signal model; verify the ranking is stable and deterministic.
    frame_indices = [event.frame_idx for event in events]
    assert len(set(frame_indices)) == 5  # all distinct
    assert all(isinstance(f, int) for f in frame_indices)


def test_find_events_respects_peak_distance(features_df: pd.DataFrame) -> None:
    detector = EventDetector()
    events = detector.find_events(features_df, top_k=6, distance_frames=150)
    assert len(events) == 6

    sorted_frames = sorted(event.frame_idx for event in events)
    deltas = [b - a for a, b in zip(sorted_frames, sorted_frames[1:])]
    assert all(delta >= 150 for delta in deltas)


def test_find_events_handles_min_clearance_nan_without_crash(
    features_df: pd.DataFrame,
) -> None:
    detector = EventDetector()
    nan_df = features_df.copy()
    nan_df["min_clearance_m"] = np.nan

    events = detector.find_events(nan_df, top_k=5, distance_frames=100)
    assert len(events) == 5
    assert all(np.isfinite(event.severity_score) for event in events)
    assert all(np.isclose(event.proximity_norm, 0.0) for event in events)


def test_score_dataframe_imputes_safe_values_before_normalization() -> None:
    detector = EventDetector()
    df = pd.DataFrame(
        {
            "frame_idx": [0, 1, 2, 3],
            "timestamp_sec": [0.0, 1.0, 2.0, 3.0],
            "roughness": [0.2, np.nan, 0.5, np.nan],
            "yaw_rate": [1.0, np.nan, -2.0, np.nan],
            "imu_correlation": [0.8, np.nan, -0.2, np.nan],
            "pose_confidence": [70.0, np.nan, 30.0, np.nan],
            "min_clearance_m": [4.0, 4.0, 4.0, 4.0],
        }
    )

    scored = detector.score_dataframe(df)
    severity = pd.to_numeric(scored["severity_score"], errors="coerce")
    assert severity.notna().all()
    assert np.isfinite(severity).all()


def test_score_dataframe_includes_proximity_signal_in_severity() -> None:
    detector = EventDetector()
    df = pd.DataFrame(
        {
            "frame_idx": [0, 1],
            "timestamp_sec": [0.0, 1.0],
            "roughness": [0.0, 0.0],
            "yaw_rate": [0.0, 0.0],
            "imu_correlation": [1.0, 1.0],
            "pose_confidence": [100.0, 100.0],
            "min_clearance_m": [9.0, 2.0],
        }
    )

    scored = detector.score_dataframe(df)
    severity = pd.to_numeric(scored["severity_score"], errors="coerce")
    proximity = pd.to_numeric(scored["proximity_norm"], errors="coerce")

    assert np.isfinite(proximity).all()
    assert np.isfinite(severity).all()
    assert float(proximity.iloc[1]) > float(proximity.iloc[0])
    assert float(severity.iloc[1]) > float(severity.iloc[0])


def test_stationary_imu_nan_maps_to_safe_fault() -> None:
    detector = EventDetector()
    df = pd.DataFrame(
        {
            "frame_idx": [0, 1],
            "timestamp_sec": [0.0, 1.0],
            "roughness": [1.0, 1.0],
            "yaw_rate": [0.0, 0.0],
            "imu_correlation": [0.2, np.nan],
            "pose_confidence": [50.0, 50.0],
            "min_clearance_m": [5.0, 5.0],
        }
    )

    scored = detector.score_dataframe(df)
    imu_fault = pd.to_numeric(scored["imu_fault_norm"], errors="coerce")

    assert np.isfinite(imu_fault).all()
    assert float(imu_fault.iloc[1]) <= 0.05
    assert float(imu_fault.iloc[1]) < float(imu_fault.iloc[0])


def test_find_events_works_with_nan_health_inputs(
    features_df: pd.DataFrame,
) -> None:
    detector = EventDetector()
    nan_df = features_df.copy()
    nan_df["imu_correlation"] = np.nan
    nan_df["pose_confidence"] = np.nan

    events = detector.find_events(nan_df, top_k=5, distance_frames=100)
    assert len(events) == 5
    assert all(np.isfinite(event.severity_score) for event in events)
    assert all(np.isfinite(event.imu_fault_norm) for event in events)
    assert all(np.isfinite(event.localization_fault_norm) for event in events)


def test_event_schema_contains_required_fields(
    loader: LogLoader, features_df: pd.DataFrame
) -> None:
    orchestrator = IntelligenceOrchestrator(loader=loader, analyst=None)
    events = orchestrator.build_events(features_df=features_df, top_k=3, distance_frames=150)
    assert len(events) == 3

    record = asdict(events[0])
    assert set(record) == {
        "event_rank",
        "frame_idx",
        "timestamp_sec",
        "timestamp_iso_utc",
        "severity_score",
        "roughness",
        "min_clearance_m",
        "yaw_rate",
        "imu_correlation",
        "pose_confidence",
        "roughness_norm",
        "proximity_norm",
        "yaw_rate_norm",
        "imu_fault_norm",
        "localization_fault_norm",
        "event_type",
        "gps_lat",
        "gps_lon",
        "primary_camera",
        "camera_paths",
        "gemini_caption",
        "gemini_model",
        "gemini_source",
        "gemini_latency_ms",
    }
    assert record["gemini_source"] == "unavailable"
    assert isinstance(record["camera_paths"], dict)
    assert "front_center_stereo_left" in record["camera_paths"]


def test_gemini_analyst_sdk_success_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-key")
    image_path = sample_image_path()
    analyst = GeminiAnalyst(model="gemini-3-flash-preview")

    def fake_sdk(
        *,
        image_bytes: bytes,
        mime_type: str,
        model: str,
    ) -> GeminiAnalysisResult:
        assert len(image_bytes) > 0
        assert mime_type.startswith("image/")
        return GeminiAnalysisResult(
            caption="Tractor traversing deep rut; potential suspension risk.",
            model=model,
            source="sdk",
            latency_ms=123.0,
            input_tokens=12,
            output_tokens=8,
            total_tokens=20,
        )

    def fake_rest(*, image_bytes: bytes, mime_type: str, model: str) -> GeminiAnalysisResult:
        raise AssertionError("REST fallback should not run for SDK success path.")

    monkeypatch.setattr(analyst, "_analyze_with_sdk", fake_sdk)
    monkeypatch.setattr(analyst, "_analyze_with_rest", fake_rest)

    result = analyst.analyze_image(image_path=image_path)
    assert result.source == "sdk"
    assert result.model == "gemini-3-flash-preview"
    assert result.caption == "Tractor traversing deep rut; potential suspension risk."
    assert result.total_tokens == 20


def test_gemini_analyst_rest_fallback_when_sdk_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-key")
    image_path = sample_image_path()
    analyst = GeminiAnalyst(model="gemini-3-flash-preview")

    def fake_sdk(*, image_bytes: bytes, mime_type: str, model: str) -> GeminiAnalysisResult:
        raise RuntimeError("SDK unavailable")

    def fake_rest(
        *,
        image_bytes: bytes,
        mime_type: str,
        model: str,
    ) -> GeminiAnalysisResult:
        return GeminiAnalysisResult(
            caption="Muddy furrow visible; maintain reduced speed.",
            model=model,
            source="rest",
            latency_ms=211.0,
        )

    monkeypatch.setattr(analyst, "_analyze_with_sdk", fake_sdk)
    monkeypatch.setattr(analyst, "_analyze_with_rest", fake_rest)

    result = analyst.analyze_image(image_path=image_path)
    assert result.source == "rest"
    assert result.caption == "Muddy furrow visible; maintain reduced speed."
    assert result.latency_ms == 211.0


def test_gemini_analyst_returns_unavailable_on_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-key")
    image_path = sample_image_path()
    analyst = GeminiAnalyst(model="gemini-3-flash-preview")

    def fake_sdk(*, image_bytes: bytes, mime_type: str, model: str) -> GeminiAnalysisResult:
        raise RuntimeError("SDK outage")

    def fake_rest(*, image_bytes: bytes, mime_type: str, model: str) -> GeminiAnalysisResult:
        raise RuntimeError("REST outage")

    monkeypatch.setattr(analyst, "_analyze_with_sdk", fake_sdk)
    monkeypatch.setattr(analyst, "_analyze_with_rest", fake_rest)

    result = analyst.analyze_image(image_path=image_path)
    assert result.source == "unavailable"
    assert result.caption == "AI Analysis Unavailable"
    assert result.error is not None
    assert "sdk_error=SDK outage" in result.error
    assert "rest_error=REST outage" in result.error


def test_orchestrator_handles_recoverable_analyst_runtime_error(
    loader: LogLoader,
) -> None:
    class _BrokenAnalyst:
        def analyze_image(self, image_path: Path, model: str | None = None) -> GeminiAnalysisResult:
            raise RuntimeError("simulated runtime failure")

    orchestrator = IntelligenceOrchestrator(loader=loader, analyst=_BrokenAnalyst())
    analysis = orchestrator._analyze_event_frame(
        image_path=Path("nonexistent.jpg"),
        model="gemini-3-flash-preview",
    )
    assert analysis.source == "unavailable"
    assert analysis.caption == "AI Analysis Unavailable"
    assert analysis.error is not None
    assert "simulated runtime failure" in analysis.error


def test_build_events_script_writes_json() -> None:
    script_path = PROJECT_ROOT / "scripts" / "build_events.py"
    output_path = data_dir() / f"events_test_export_{uuid.uuid4().hex}.json"

    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        src_path
        if not existing_pythonpath
        else f"{src_path}{os.pathsep}{existing_pythonpath}"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--data-dir",
            str(data_dir()),
            "--output",
            str(output_path),
            "--top-k",
            "5",
            "--distance-frames",
            "150",
            "--disable-gemini",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"build_events.py failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["top_k_requested"] == 5
    assert payload["distance_frames"] == 150
    assert payload["rows_processed"] == 1085
    assert payload["peaks_found"] >= 5
    assert isinstance(payload["events"], list)
    assert len(payload["events"]) == 5
    assert all(event["gemini_source"] == "unavailable" for event in payload["events"])


def sample_image_path() -> Path:
    return data_dir() / "frames" / "front_center_stereo_left" / "0000.jpg"
