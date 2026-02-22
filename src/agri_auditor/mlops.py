from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MLFLOW_TRACKING_DIR = PROJECT_ROOT / "artifacts" / "mlruns"


def mlflow_enabled_from_env(env: Mapping[str, str] | None = None) -> bool:
    source = env if env is not None else os.environ
    raw = source.get("AGRI_AUDITOR_MLFLOW_ENABLED", "")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _flatten_metadata(
    payload: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten_metadata(value, prefix=full_key))
        else:
            flat[full_key] = value
    return flat


def log_run_lineage(
    *,
    run_metadata: Mapping[str, Any],
    run_id: str,
    command_name: str,
    artifacts: Sequence[Path] | None = None,
    tracking_uri: str | None = None,
    enabled: bool | None = None,
) -> bool:
    should_log = mlflow_enabled_from_env() if enabled is None else bool(enabled)
    if not should_log:
        return False

    try:
        import mlflow
    except ModuleNotFoundError:
        return False

    resolved_tracking_uri = (
        tracking_uri
        if tracking_uri is not None and str(tracking_uri).strip()
        else os.getenv("AGRI_AUDITOR_MLFLOW_TRACKING_URI", "").strip()
    )
    if not resolved_tracking_uri:
        resolved_tracking_uri = str(DEFAULT_MLFLOW_TRACKING_DIR.resolve())

    try:
        Path(resolved_tracking_uri).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    try:
        mlflow.set_tracking_uri(resolved_tracking_uri)
        with mlflow.start_run(run_name=f"agri-auditor-{command_name}-{run_id}"):
            mlflow.set_tag("run_id", run_id)
            mlflow.set_tag("command", command_name)
            for key, value in sorted(_flatten_metadata(run_metadata).items()):
                if value is None:
                    continue
                mlflow.log_param(key, str(value))
            for artifact in artifacts or ():
                path = Path(artifact)
                if path.exists() and path.is_file():
                    mlflow.log_artifact(str(path))
        return True
    except Exception:
        return False
