from __future__ import annotations

import os
from pathlib import Path

import pytest

from agri_auditor import FeatureEngine, GeminiAnalyst, IntelligenceOrchestrator, LogLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return PROJECT_ROOT.parent / "provided_data"


@pytest.fixture(scope="module")
def loader() -> LogLoader:
    return LogLoader(data_dir())


@pytest.fixture(scope="module")
def require_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        pytest.fail("GEMINI_API_KEY not set; gemini_live tests require live credentials.")
    return key


def sample_image_path() -> Path:
    return data_dir() / "frames" / "front_center_stereo_left" / "0000.jpg"


@pytest.mark.gemini_live
def test_gemini_live_sdk_returns_valid_caption(require_api_key: str) -> None:
    image_path = sample_image_path()
    analyst = GeminiAnalyst(model="gemini-3-flash-preview", api_key=require_api_key)
    result = analyst.analyze_image(image_path=image_path)

    assert result.source == "sdk", f"Expected SDK source, got {result.source} (error: {result.error})"
    assert result.caption != "AI Analysis Unavailable"
    assert len(result.caption.split()) <= 25, f"Caption too long: {result.caption}"
    assert result.latency_ms is not None and result.latency_ms > 0
    assert result.input_tokens is not None and result.input_tokens > 0
    assert result.total_tokens is not None and result.total_tokens > 0
    assert result.model == "gemini-3-flash-preview"


@pytest.mark.gemini_live
def test_gemini_live_thinking_tokens_captured(require_api_key: str) -> None:
    image_path = sample_image_path()
    analyst = GeminiAnalyst(model="gemini-3-flash-preview", api_key=require_api_key)
    result = analyst.analyze_image(image_path=image_path)

    assert result.source == "sdk"
    if result.thinking_tokens is not None:
        assert result.thinking_tokens >= 0
    if (
        result.input_tokens is not None
        and result.output_tokens is not None
        and result.total_tokens is not None
    ):
        assert result.total_tokens >= result.input_tokens + result.output_tokens


@pytest.mark.gemini_live
def test_gemini_live_build_events_end_to_end(
    loader: LogLoader,
    require_api_key: str,
) -> None:
    features_df = FeatureEngine(loader=loader).build_features()
    analyst = GeminiAnalyst(model="gemini-3-flash-preview", api_key=require_api_key)
    orchestrator = IntelligenceOrchestrator(
        loader=loader,
        analyst=analyst,
        primary_camera="front_center_stereo_left",
    )
    events = orchestrator.build_events(features_df=features_df, top_k=2, distance_frames=150)

    assert len(events) == 2
    for event in events:
        assert event.gemini_source == "sdk", f"Event {event.event_rank} source={event.gemini_source}"
        assert event.gemini_caption != "AI Analysis Unavailable"
        assert event.gemini_model == "gemini-3-flash-preview"
        assert event.gemini_latency_ms is not None and event.gemini_latency_ms > 0
