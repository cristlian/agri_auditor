from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import pandas as pd


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


def _run_module(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "agri_auditor", *args],
        cwd=PROJECT_ROOT,
        env=_pythonpath_env(),
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
    suffix = uuid.uuid4().hex
    features_output = data_dir() / f"features_cli_test_{suffix}.csv"
    events_output = data_dir() / f"events_cli_test_{suffix}.json"
    report_output = data_dir() / f"audit_cli_test_{suffix}.html"

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
    tmp_dir = PROJECT_ROOT / "artifacts" / f"provided_data_test_{uuid.uuid4().hex}"
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
