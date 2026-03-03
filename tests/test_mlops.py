from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

from agri_auditor.mlops import log_run_lineage


class _NoopRun:
    def __enter__(self) -> "_NoopRun":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> bool:
        return False


def test_log_run_lineage_logs_mlflow_exception_and_returns_false(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    class FakeMlflowException(Exception):
        pass

    def _set_tag(key: str, value: str) -> None:
        raise FakeMlflowException("invalid lineage payload")

    fake_mlflow = types.SimpleNamespace(
        exceptions=types.SimpleNamespace(MlflowException=FakeMlflowException),
        set_tracking_uri=lambda uri: None,
        start_run=lambda run_name: _NoopRun(),
        set_tag=_set_tag,
        log_param=lambda key, value: None,
        log_artifact=lambda path: None,
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    with caplog.at_level(logging.WARNING):
        logged = log_run_lineage(
            run_metadata={"dataset": {"hash": "abc"}},
            run_id="run-123",
            command_name="process",
            enabled=True,
            tracking_uri=str(tmp_path / "mlruns"),
        )

    assert logged is False
    assert "Failed to log lineage to MLFlow" in caplog.text
    assert "invalid lineage payload" in caplog.text


def test_log_run_lineage_logs_mkdir_oserror_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    class FakeMlflowException(Exception):
        pass

    def _raise_mkdir(self: Path, parents: bool = False, exist_ok: bool = False) -> None:
        raise OSError("permission denied")

    fake_mlflow = types.SimpleNamespace(
        exceptions=types.SimpleNamespace(MlflowException=FakeMlflowException),
        set_tracking_uri=lambda uri: None,
        start_run=lambda run_name: _NoopRun(),
        set_tag=lambda key, value: None,
        log_param=lambda key, value: None,
        log_artifact=lambda path: None,
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setattr("agri_auditor.mlops.Path.mkdir", _raise_mkdir)

    with caplog.at_level(logging.WARNING):
        logged = log_run_lineage(
            run_metadata={"dataset_hash": "abc"},
            run_id="run-456",
            command_name="process",
            enabled=True,
            tracking_uri=str(tmp_path / "mlruns"),
        )

    assert logged is True
    assert "Failed to create MLFlow directory" in caplog.text
    assert "permission denied" in caplog.text


def test_log_run_lineage_does_not_swallow_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeMlflowException(Exception):
        pass

    def _set_tag(key: str, value: str) -> None:
        raise ValueError("unexpected bug")

    fake_mlflow = types.SimpleNamespace(
        exceptions=types.SimpleNamespace(MlflowException=FakeMlflowException),
        set_tracking_uri=lambda uri: None,
        start_run=lambda run_name: _NoopRun(),
        set_tag=_set_tag,
        log_param=lambda key, value: None,
        log_artifact=lambda path: None,
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    with pytest.raises(ValueError, match="unexpected bug"):
        log_run_lineage(
            run_metadata={"dataset_hash": "abc"},
            run_id="run-789",
            command_name="process",
            enabled=True,
            tracking_uri=str(tmp_path / "mlruns"),
        )
