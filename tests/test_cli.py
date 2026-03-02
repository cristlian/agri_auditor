from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

import pandas as pd

from agri_auditor.cli import _backfill_event_camera_paths
from agri_auditor.ingestion import LogLoader
from agri_auditor.intelligence import Event, UNAVAILABLE_CAPTION


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return PROJECT_ROOT.parent / "provided_data"


def _pythonpath_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
    )
    return env


def _run_module(
    args: list[str],
    *,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = _pythonpath_env()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "agri_auditor", *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_python_module_help() -> None:
    result = _run_module(["--help"])
    assert result.returncode == 0, result.stderr
    assert "features" in result.stdout
    assert "events" in result.stdout
    assert "report" in result.stdout
    assert "process" in result.stdout
    assert "benchmark-gemini" in result.stdout


def test_python_module_invalid_command_returns_non_zero() -> None:
    result = _run_module(["not-a-command"])
    assert result.returncode != 0
    assert "invalid choice" in result.stderr.lower()


def test_process_disable_gemini_completes_and_writes_outputs() -> None:
    with tempfile.TemporaryDirectory(prefix="cli_process_") as tmp:
        tmp_dir = Path(tmp)
        features_output = tmp_dir / "features.csv"
        events_output = tmp_dir / "events.json"
        report_output = tmp_dir / "audit_report.html"

        result = _run_module(
            [
                "process",
                "--data-dir",
                str(data_dir()),
                "--output-features",
                str(features_output),
                "--output-events",
                str(events_output),
                "--output-report",
                str(report_output),
                "--top-k",
                "2",
                "--distance-frames",
                "150",
                "--disable-gemini",
                "--no-surround",
                "--report-mode",
                "split",
                "--report-telemetry-downsample",
                "2",
                "--report-feature-columns",
                "timestamp_sec,_elapsed,gps_lat,gps_lon,velocity_mps",
                "--gemini-workers",
                "2",
                "--gemini-retries",
                "1",
                "--gemini-backoff-ms",
                "0",
                "--depth-workers",
                "2",
                "--peak-prominence",
                "0.01",
                "--peak-width",
                "1",
                "--peak-min-distance",
                "150",
            ]
        )
        assert result.returncode == 0, (
            f"process command failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        assert features_output.exists()
        assert events_output.exists()
        assert report_output.exists()

        features = pd.read_csv(features_output)
        assert "roughness" in features.columns
        assert "min_clearance_m" in features.columns

        payload = json.loads(events_output.read_text(encoding="utf-8"))
        assert len(payload["events"]) == 2
        assert all(event["gemini_source"] == "unavailable" for event in payload["events"])
        assert "dataset_hash" in payload
        assert "code_version" in payload
        assert "config_fingerprint" in payload
        assert "latency_summary" in payload

        assets_dir = report_output.with_name(f"{report_output.stem}_assets")
        assert assets_dir.exists()
        assert (assets_dir / "events.json").exists()
        assert (assets_dir / "telemetry.json").exists()


def test_process_requires_gemini_key_when_not_disabled() -> None:
    result = _run_module(
        [
            "process",
            "--data-dir",
            str(data_dir()),
            "--top-k",
            "1",
            "--distance-frames",
            "150",
            "--no-surround",
        ],
        env_overrides={"GEMINI_API_KEY": ""},
    )
    assert result.returncode != 0
    assert "GEMINI_API_KEY" in (result.stderr + result.stdout)


def test_legacy_script_wrappers_forward_to_cli_help() -> None:
    script_paths = [
        PROJECT_ROOT / "scripts" / "build_features.py",
        PROJECT_ROOT / "scripts" / "build_events.py",
        PROJECT_ROOT / "scripts" / "build_report.py",
        PROJECT_ROOT / "scripts" / "benchmark_gemini.py",
    ]
    for script_path in script_paths:
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=PROJECT_ROOT,
            env=_pythonpath_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"{script_path.name} --help failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def test_prepare_test_data_script_creates_required_files() -> None:
    script_path = PROJECT_ROOT / "scripts" / "prepare_test_data.py"
    with tempfile.TemporaryDirectory(prefix="provided_data_test_") as tmp:
        tmp_dir = Path(tmp)
        result = subprocess.run(
            [sys.executable, str(script_path), "--output-dir", str(tmp_dir)],
            cwd=PROJECT_ROOT,
            env=_pythonpath_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"prepare_test_data.py failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert (tmp_dir / "manifest.csv").exists()
        assert (tmp_dir / "calibrations.json").exists()
        assert (tmp_dir / "frames" / "front_center_stereo_left" / "0000.jpg").exists()
        assert (tmp_dir / "frames" / "depth" / "0000.png").exists()


def test_backfill_event_camera_paths_recovers_surround_views() -> None:
    loader = LogLoader(data_dir())
    single_path_event = Event(
        event_rank=1,
        frame_idx=10,
        timestamp_sec=1_768_474_273.0,
        timestamp_iso_utc="2026-01-15T10:51:13+00:00",
        severity_score=0.5,
        roughness=0.3,
        min_clearance_m=3.2,
        yaw_rate=1.0,
        imu_correlation=0.1,
        pose_confidence=55.0,
        roughness_norm=0.3,
        proximity_norm=0.4,
        yaw_rate_norm=0.2,
        imu_fault_norm=0.1,
        localization_fault_norm=0.2,
        event_type="mixed",
        gps_lat=39.661,
        gps_lon=-0.558,
        primary_camera="front_center_stereo_left",
        camera_paths={
            "front_center_stereo_left": str(
                data_dir() / "frames" / "front_center_stereo_left" / "0010.jpg"
            )
        },
        gemini_caption=UNAVAILABLE_CAPTION,
        gemini_model=None,
        gemini_source="unavailable",
        gemini_latency_ms=None,
    )

    repaired_events, repaired_count = _backfill_event_camera_paths(
        [single_path_event], loader
    )
    assert repaired_count == 1
    assert len(repaired_events) == 1
    assert len(repaired_events[0].camera_paths) >= 2
