from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "check_perf_budget.py"


def _load_perf_budget_module():
    spec = importlib.util.spec_from_file_location("check_perf_budget", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load check_perf_budget module spec.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_perf_budget_script_passes_with_fast_warm_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_perf_budget_module()
    calls = 0

    def _fake_measure_feature_build(*, data_dir, cache_dir, depth_workers):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            return 4.0, 1085
        return 2.0, 1085

    monkeypatch.setattr(module, "_measure_feature_build", _fake_measure_feature_build)
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    data_dir.mkdir(parents=True, exist_ok=True)
    exit_code = module.main(
        [
            "--data-dir",
            str(data_dir),
            "--cache-dir",
            str(cache_dir),
            "--max-cold-sec",
            "10",
            "--max-warm-to-cold-ratio",
            "0.75",
        ]
    )
    assert exit_code == 0


def test_perf_budget_script_fails_on_budget_violation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_perf_budget_module()

    def _fake_measure_feature_build(*, data_dir, cache_dir, depth_workers):  # type: ignore[no-untyped-def]
        return 2.0, 1085

    monkeypatch.setattr(module, "_measure_feature_build", _fake_measure_feature_build)
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    data_dir.mkdir(parents=True, exist_ok=True)
    exit_code = module.main(
        [
            "--data-dir",
            str(data_dir),
            "--cache-dir",
            str(cache_dir),
            "--max-cold-sec",
            "1.0",
            "--max-warm-to-cold-ratio",
            "0.5",
        ]
    )
    assert exit_code == 1
