from __future__ import annotations

import argparse
import json
from pathlib import Path

import jsonschema
import pytest

from agri_auditor.cli import _build_run_metadata
from agri_auditor.config import load_runtime_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _data_dir() -> Path:
    return PROJECT_ROOT.parent / "provided_data"


def _schema() -> dict[str, object]:
    schema_path = PROJECT_ROOT / "schemas" / "run_metadata.schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _args_namespace() -> argparse.Namespace:
    return argparse.Namespace(
        model=None,
        gemini_workers=None,
        gemini_retries=None,
        gemini_backoff_ms=None,
        depth_workers=None,
        score_normalization=None,
        score_robust_quantile_low=None,
        score_robust_quantile_high=None,
        peak_prominence=None,
        peak_width=None,
        peak_min_distance=None,
        distance_frames=150,
        report_mode=None,
        report_telemetry_downsample=None,
    )


def test_run_metadata_payload_matches_schema() -> None:
    runtime = load_runtime_config(env={})
    metadata = _build_run_metadata(
        data_dir=_data_dir(),
        runtime=runtime,
        args=_args_namespace(),
        events=[],
    )
    jsonschema.validate(instance=metadata, schema=_schema())


def test_run_metadata_schema_rejects_missing_required_key() -> None:
    payload = {
        "dataset_hash": "abc",
        "code_version": "def",
        "latency_summary": {
            "count": 0,
            "avg_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
        },
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=payload, schema=_schema())


def test_run_metadata_schema_rejects_invalid_latency_count() -> None:
    payload = {
        "dataset_hash": "abc",
        "code_version": "def",
        "config_fingerprint": "123",
        "latency_summary": {
            "count": -1,
            "avg_ms": 1.0,
            "p50_ms": 1.0,
            "p95_ms": 2.0,
            "p99_ms": 3.0,
        },
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=payload, schema=_schema())
