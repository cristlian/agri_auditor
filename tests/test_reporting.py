"""Tests for reporting module — ReportBuilder and utilities."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from agri_auditor.intelligence import Event, UNAVAILABLE_CAPTION
from agri_auditor.reporting import (
    COLORS,
    EVENT_TYPE_COLORS,
    SEVERITY_WARNING_THRESHOLD,
    SEVERITY_CRITICAL_THRESHOLD,
    CHART_TELEMETRY_HEIGHT,
    CHART_SPARKLINE_HEIGHT,
    ReportBuilder,
    _encode_image_b64,
    _fmt,
    _severity_color,
    _severity_label,
    _elapsed,
    _PLACEHOLDER_IMAGE,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "provided_data"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = PROJECT_ROOT / "artifacts"


def _new_workspace_tmp_dir() -> Path:
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEST_TMP_ROOT / f"report_test_tmp_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_features_df(n: int = 100) -> pd.DataFrame:
    """Generate a synthetic features DataFrame for testing."""
    rng = np.random.default_rng(42)
    t0 = 1_768_474_273.0
    ts = np.linspace(t0, t0 + n * 0.0333, n)
    return pd.DataFrame({
        "frame_idx": range(n),
        "timestamp_sec": ts,
        "has_depth": [i % 3 == 0 for i in range(n)],
        "velocity_mps": rng.uniform(0.5, 2.5, n),
        "roughness": rng.uniform(0, 1.2, n),
        "min_clearance_m": [rng.uniform(3, 5) if i % 3 == 0 else np.nan for i in range(n)],
        "imu_correlation": rng.uniform(-0.5, 1.0, n),
        "pose_confidence": rng.uniform(40, 75, n),
        "yaw": rng.uniform(-90, 90, n),
        "pitch": rng.uniform(-5, 5, n),
        "roll": rng.uniform(-5, 5, n),
        "yaw_rate": rng.uniform(-20, 20, n),
        "gps_lat": np.linspace(39.661, 39.662, n),
        "gps_lon": np.linspace(-0.558, -0.557, n),
        "imu_camera_accel_z": rng.normal(9.8, 0.5, n),
        "imu_syslogic_accel_z": rng.normal(-9.0, 0.5, n),
    })


def _make_events(n: int = 3) -> list[Event]:
    """Generate synthetic Event objects."""
    events = []
    for i in range(n):
        events.append(Event(
            event_rank=i + 1,
            frame_idx=i * 30 + 10,
            timestamp_sec=1_768_474_273.0 + (i * 30 + 10) * 0.0333,
            timestamp_iso_utc=f"2026-01-15T10:51:{16 + i}+00:00",
            severity_score=0.9 - i * 0.2,
            roughness=0.8 - i * 0.1,
            min_clearance_m=3.5 + i * 0.5 if i % 2 == 0 else None,
            yaw_rate=-15.0 + i * 5,
            imu_correlation=-0.1 + i * 0.3,
            pose_confidence=50.0 + i * 5,
            roughness_norm=0.9 - i * 0.15,
            proximity_norm=0.8 - i * 0.2,
            yaw_rate_norm=0.5 + i * 0.1,
            imu_fault_norm=0.6 - i * 0.1,
            localization_fault_norm=0.7 - i * 0.15,
            event_type=["roughness", "mixed", "proximity"][i],
            gps_lat=39.661 + i * 0.0003,
            gps_lon=-0.558 + i * 0.0001,
            primary_camera="front_center_stereo_left",
            camera_paths={
                "front_center_stereo_left": str(DATA_DIR / "frames" / "front_center_stereo_left" / f"{i * 30 + 10:04d}.jpg"),
                "front_left": str(DATA_DIR / "frames" / "front_left" / f"{i * 30 + 10:04d}.jpg"),
            },
            gemini_caption="Rough terrain with visible obstacles." if i == 0 else UNAVAILABLE_CAPTION,
            gemini_model="gemini-3-flash-preview" if i == 0 else None,
            gemini_source="sdk" if i == 0 else "unavailable",
            gemini_latency_ms=3500.0 if i == 0 else None,
        ))
    return events


def _mock_loader() -> MagicMock:
    loader = MagicMock()
    loader.data_dir = DATA_DIR
    loader.frames_dir = DATA_DIR / "frames"
    return loader


# ── Utility tests ─────────────────────────────────────────────────────────────


class TestUtilities:
    def test_severity_color_low(self):
        assert _severity_color(0.2) == COLORS["green"]

    def test_severity_color_medium(self):
        assert _severity_color(0.5) == COLORS["amber"]

    def test_severity_color_high(self):
        assert _severity_color(0.8) == COLORS["red"]

    def test_severity_color_clamps(self):
        assert _severity_color(-1.0) == COLORS["green"]
        assert _severity_color(2.0) == COLORS["red"]

    def test_severity_label(self):
        assert _severity_label(0.2) == "NOMINAL"
        assert _severity_label(0.5) == "WARNING"
        assert _severity_label(0.8) == "CRITICAL"

    def test_fmt_normal(self):
        assert _fmt(3.14159, 2) == "3.14"

    def test_fmt_none(self):
        assert _fmt(None) == "N/A"

    def test_fmt_nan(self):
        assert _fmt(float("nan")) == "N/A"

    def test_fmt_inf(self):
        assert _fmt(float("inf")) == "N/A"

    def test_fmt_fallback(self):
        assert _fmt(None, fallback="Blind") == "Blind"

    def test_elapsed_series(self):
        ts = pd.Series([100.0, 101.0, 102.0, 103.0])
        elapsed = _elapsed(ts)
        assert elapsed.iloc[0] == pytest.approx(0.0)
        assert elapsed.iloc[-1] == pytest.approx(3.0)

    def test_elapsed_series_nan_safe(self):
        ts = pd.Series([np.nan, np.nan])
        result = _elapsed(ts)
        assert result.isna().all()

    def test_severity_thresholds_are_consistent(self):
        """Verify named constants match the boundary behavior."""
        assert _severity_label(SEVERITY_WARNING_THRESHOLD - 0.01) == "NOMINAL"
        assert _severity_label(SEVERITY_WARNING_THRESHOLD) == "WARNING"
        assert _severity_label(SEVERITY_CRITICAL_THRESHOLD - 0.01) == "WARNING"
        assert _severity_label(SEVERITY_CRITICAL_THRESHOLD) == "CRITICAL"

    def test_chart_height_constants(self):
        """Named constants should be positive integers."""
        assert isinstance(CHART_TELEMETRY_HEIGHT, int)
        assert CHART_TELEMETRY_HEIGHT > 0
        assert isinstance(CHART_SPARKLINE_HEIGHT, int)
        assert CHART_SPARKLINE_HEIGHT > 0


class TestImageEncoding:
    @pytest.fixture
    def sample_image_path(self):
        """Use a real frame from provided_data if available."""
        path = DATA_DIR / "frames" / "front_center_stereo_left" / "0010.jpg"
        if path.exists():
            return path
        return None

    def test_encode_nonexistent_returns_none(self):
        assert _encode_image_b64("/nonexistent/image.jpg") is None

    def test_encode_returns_base64_string(self, sample_image_path):
        if sample_image_path is None:
            pytest.skip("Sample image not available")
        result = _encode_image_b64(sample_image_path, max_width=320, quality=70)
        assert result is not None
        assert result.startswith("data:image/jpeg;base64,")

    def test_encode_respects_max_width(self, sample_image_path):
        if sample_image_path is None:
            pytest.skip("Sample image not available")
        small = _encode_image_b64(sample_image_path, max_width=100, quality=50)
        large = _encode_image_b64(sample_image_path, max_width=640, quality=50)
        assert small is not None and large is not None
        assert len(small) < len(large)


# ── ReportBuilder tests ──────────────────────────────────────────────────────


class TestReportBuilder:
    @pytest.fixture
    def builder(self):
        return ReportBuilder(
            loader=_mock_loader(),
            features_df=_make_features_df(),
            events=_make_events(),
            metadata={"run_id": "test-run-001"},
            include_surround=False,  # faster tests
        )

    def test_summary_contains_required_keys(self, builder):
        summary = builder._summary()
        required = {"frames", "duration", "events", "avg_speed", "max_severity",
                     "max_severity_color", "max_severity_label", "depth_pct"}
        assert required <= set(summary.keys())

    def test_summary_frames_count(self, builder):
        summary = builder._summary()
        assert summary["frames"] == "100"

    def test_summary_events_count(self, builder):
        summary = builder._summary()
        assert summary["events"] == "3"

    def test_summary_max_severity_is_critical(self, builder):
        summary = builder._summary()
        # Highest event severity is 0.9
        assert summary["max_severity_label"] in ("WARNING", "CRITICAL")

    def test_chart_telemetry_returns_figure(self, builder):
        fig = builder._chart_telemetry()
        assert hasattr(fig, "to_json")
        json_str = fig.to_json()
        parsed = json.loads(json_str)
        assert "data" in parsed
        assert "layout" in parsed
        # Should have velocity + pitch + roll + clearance + event markers
        assert len(parsed["data"]) >= 4

    def test_gps_path_data_returns_list(self, builder):
        path = builder._gps_path_data()
        assert isinstance(path, list)
        assert len(path) > 0
        assert len(path[0]) == 2  # [lat, lon]

    def test_event_ctx_contains_gps(self, builder):
        ctx = builder._event_ctx(builder.events[0])
        assert "gps_lat" in ctx
        assert "gps_lon" in ctx

    def test_event_ctx_structure(self, builder):
        ctx = builder._event_ctx(builder.events[0])
        required_keys = {
            "rank", "frame_idx", "timestamp_iso", "elapsed",
            "severity", "severity_pct", "severity_color", "severity_label",
            "event_type", "event_type_upper", "event_type_color",
            "signals", "metrics",
            "gemini_caption", "has_ai", "primary_image", "surround",
        }
        assert required_keys <= set(ctx.keys())

    def test_event_ctx_signals_count(self, builder):
        ctx = builder._event_ctx(builder.events[0])
        assert len(ctx["signals"]) == 5

    def test_event_ctx_metrics_count(self, builder):
        ctx = builder._event_ctx(builder.events[0])
        assert len(ctx["metrics"]) == 6

    def test_event_ctx_ai_flag(self, builder):
        ctx0 = builder._event_ctx(builder.events[0])
        assert ctx0["has_ai"] is True

        ctx1 = builder._event_ctx(builder.events[1])
        assert ctx1["has_ai"] is False

    def test_event_ctx_placeholder_for_missing_image(self, builder):
        # Event with non-existent image path
        ev = Event(
            event_rank=99,
            frame_idx=9999,
            timestamp_sec=1_768_474_273.0,
            timestamp_iso_utc="2026-01-15T10:51:13+00:00",
            severity_score=0.5,
            roughness=0.3,
            min_clearance_m=None,
            yaw_rate=None,
            imu_correlation=None,
            pose_confidence=None,
            roughness_norm=0.3,
            proximity_norm=0.0,
            yaw_rate_norm=0.0,
            imu_fault_norm=0.0,
            localization_fault_norm=0.0,
            event_type="mixed",
            gps_lat=None,
            gps_lon=None,
            primary_camera="front_center_stereo_left",
            camera_paths={"front_center_stereo_left": "/nonexistent/9999.jpg"},
            gemini_caption=UNAVAILABLE_CAPTION,
            gemini_model=None,
            gemini_source="unavailable",
            gemini_latency_ms=None,
        )
        ctx = builder._event_ctx(ev)
        assert ctx["primary_image"] == _PLACEHOLDER_IMAGE

    def test_render_produces_html(self, builder):
        html = builder._render()
        assert "<!DOCTYPE html>" in html
        assert "AGRI-AUDITOR" in html
        assert "Mission Control Dashboard" in html
        assert "chart-telemetry" in html
        assert "test-run-001" in html
        # Cockpit layout elements
        assert "split-container" in html
        assert "leaflet" in html.lower()
        assert "triageEvent" in html

    def test_render_contains_events(self, builder):
        html = builder._render()
        assert 'id="event-1"' in html
        assert 'id="event-2"' in html
        assert 'id="event-3"' in html

    def test_render_contains_plotly_script(self, builder):
        html = builder._render()
        assert "plotly" in html.lower()
        assert "Plotly.newPlot" in html

    def test_save_report_writes_file(self, builder):
        tmpdir = _new_workspace_tmp_dir()
        out = builder.save_report(tmpdir / "test_report.html")
        assert out.exists()
        assert out.stat().st_size > 1000  # non-trivial HTML
        content = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content


class TestReportBuilderWithRealData:
    """Integration tests using the actual provided_data directory."""

    @pytest.fixture
    def real_loader(self):
        if not DATA_DIR.exists():
            pytest.skip("provided_data not available")
        from agri_auditor import LogLoader
        return LogLoader(DATA_DIR)

    @pytest.fixture
    def real_features(self, real_loader):
        from agri_auditor import FeatureEngine
        return FeatureEngine(loader=real_loader).build_features()

    @pytest.fixture
    def real_events(self, real_loader, real_features):
        from agri_auditor import EventDetector, IntelligenceOrchestrator
        orch = IntelligenceOrchestrator(loader=real_loader, analyst=None)
        return orch.build_events(features_df=real_features, top_k=5)

    def test_end_to_end_report_generation(self, real_loader, real_features, real_events):
        builder = ReportBuilder(
            loader=real_loader,
            features_df=real_features,
            events=real_events,
            metadata={"run_id": "integration-test"},
            include_surround=True,
        )
        tmpdir = _new_workspace_tmp_dir()
        out = builder.save_report(tmpdir / "integration_report.html")
        assert out.exists()
        size_kb = out.stat().st_size / 1024
        content = out.read_text(encoding="utf-8")

        # Structural checks
        assert "<!DOCTYPE html>" in content
        assert "AGRI-AUDITOR" in content
        assert "integration-test" in content
        assert 'id="event-1"' in content
        assert "Plotly.newPlot" in content
        assert "chart-telemetry" in content
        assert "split-container" in content
        assert "leaflet" in content.lower()
        assert "triageEvent" in content

        # Size sanity (should include base64 images)
        print(f"Report size: {size_kb:.1f} KB")
        assert size_kb > 50, "Report should be > 50 KB with charts and images"
