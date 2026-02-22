from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import RuntimeConfig, load_runtime_config, normalize_log_format, normalize_log_level
from .features import FeatureEngine
from .ingestion import LogLoader
from .intelligence import (
    Event,
    EventDetector,
    GeminiAnalyst,
    GeminiConfigError,
    IntelligenceOrchestrator,
    UNAVAILABLE_CAPTION,
)
from .logging_config import configure_logging, get_logger, log_event
from .mlops import log_run_lineage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT.parent / "provided_data"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_FEATURES_OUTPUT = DEFAULT_ARTIFACTS_DIR / "features.csv"
DEFAULT_EVENTS_OUTPUT = DEFAULT_ARTIFACTS_DIR / "events.json"
DEFAULT_REPORT_OUTPUT = DEFAULT_ARTIFACTS_DIR / "audit_report.html"
DEFAULT_BENCHMARK_OUTPUT = DEFAULT_ARTIFACTS_DIR / "gemini_benchmark.json"

DEFAULT_TOP_K = 5
DEFAULT_DISTANCE_FRAMES = 150
DEFAULT_CAMERA = "front_center_stereo_left"
DEFAULT_RUN_ID = "audit-run"


def _get_report_builder() -> type[Any]:
    try:
        from .reporting import ReportBuilder
    except ImportError as exc:
        raise RuntimeError(
            "Report dependencies are missing. Install with: pip install agri-auditor[report]"
        ) from exc
    return ReportBuilder


def _resolve_optional_arg(args: argparse.Namespace, name: str, fallback: Any) -> Any:
    value = getattr(args, name, None)
    return fallback if value is None else value


def _parse_csv_columns(raw: str) -> tuple[str, ...]:
    parts = [item.strip() for item in str(raw).split(",")]
    return tuple(item for item in parts if item)


def _make_feature_engine(
    *,
    loader: LogLoader,
    runtime: RuntimeConfig,
    args: argparse.Namespace,
) -> FeatureEngine:
    depth_workers = int(
        _resolve_optional_arg(args, "depth_workers", runtime.depth_workers)
    )
    depth_cache_dir = _resolve_optional_arg(args, "depth_cache_dir", runtime.depth_cache_dir)
    return FeatureEngine(
        loader=loader,
        depth_workers=depth_workers,
        depth_cache_dir=depth_cache_dir,
    )


def _make_event_detector(runtime: RuntimeConfig, args: argparse.Namespace) -> EventDetector:
    normalization = str(
        _resolve_optional_arg(args, "score_normalization", runtime.score_normalization)
    )
    q_low = float(
        _resolve_optional_arg(
            args, "score_robust_quantile_low", runtime.score_robust_quantile_low
        )
    )
    q_high = float(
        _resolve_optional_arg(
            args, "score_robust_quantile_high", runtime.score_robust_quantile_high
        )
    )
    peak_prominence = float(
        _resolve_optional_arg(args, "peak_prominence", runtime.peak_prominence)
    )
    peak_width = int(_resolve_optional_arg(args, "peak_width", runtime.peak_width))
    return EventDetector(
        normalization_mode=normalization,
        robust_quantile_low=q_low,
        robust_quantile_high=q_high,
        peak_prominence=peak_prominence,
        peak_width=peak_width,
    )


def _dataset_hash(data_dir: Path) -> str:
    hasher = hashlib.sha256()
    for name in ("manifest.csv", "calibrations.json"):
        path = data_dir / name
        hasher.update(name.encode("utf-8"))
        if path.exists():
            try:
                hasher.update(path.read_bytes())
            except OSError:
                hasher.update(b"<unreadable>")
    return hasher.hexdigest()


def _code_version() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        revision = proc.stdout.strip()
        return revision or "unknown"
    except OSError:
        return "unknown"


def _config_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _latency_summary(events: list[Event]) -> dict[str, float | int | None]:
    latencies = [
        float(event.gemini_latency_ms)
        for event in events
        if event.gemini_latency_ms is not None and np.isfinite(event.gemini_latency_ms)
    ]
    if not latencies:
        return {
            "count": 0,
            "avg_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
        }
    arr = np.array(latencies, dtype=np.float64)
    return {
        "count": int(arr.size),
        "avg_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def _build_run_metadata(
    *,
    data_dir: Path,
    runtime: RuntimeConfig,
    args: argparse.Namespace,
    events: list[Event],
) -> dict[str, Any]:
    config_payload = {
        "log_level": runtime.log_level,
        "log_format": runtime.log_format,
        "gemini_model": _resolve_optional_arg(args, "model", runtime.gemini_model),
        "gemini_workers": _resolve_optional_arg(args, "gemini_workers", runtime.gemini_workers),
        "gemini_retries": _resolve_optional_arg(args, "gemini_retries", runtime.gemini_retries),
        "gemini_backoff_ms": _resolve_optional_arg(args, "gemini_backoff_ms", runtime.gemini_backoff_ms),
        "gemini_jitter_ratio": runtime.gemini_jitter_ratio,
        "depth_workers": _resolve_optional_arg(args, "depth_workers", runtime.depth_workers),
        "score_normalization": _resolve_optional_arg(args, "score_normalization", runtime.score_normalization),
        "score_q_low": _resolve_optional_arg(args, "score_robust_quantile_low", runtime.score_robust_quantile_low),
        "score_q_high": _resolve_optional_arg(args, "score_robust_quantile_high", runtime.score_robust_quantile_high),
        "peak_prominence": _resolve_optional_arg(args, "peak_prominence", runtime.peak_prominence),
        "peak_width": _resolve_optional_arg(args, "peak_width", runtime.peak_width),
        "peak_min_distance": _resolve_optional_arg(
            args, "peak_min_distance", getattr(args, "distance_frames", runtime.peak_min_distance)
        ),
        "report_mode": _resolve_optional_arg(args, "report_mode", runtime.report_mode),
        "report_telemetry_downsample": _resolve_optional_arg(
            args,
            "report_telemetry_downsample",
            runtime.report_telemetry_downsample,
        ),
    }
    return {
        "dataset_hash": _dataset_hash(data_dir),
        "code_version": _code_version(),
        "config_fingerprint": _config_fingerprint(config_payload),
        "latency_summary": _latency_summary(events),
    }


def _load_events_from_json(json_path: Path) -> list[Event]:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    events_raw = raw.get("events", raw if isinstance(raw, list) else [])

    events: list[Event] = []
    for item in events_raw:
        if not isinstance(item, dict):
            continue
        camera_paths = item.get("camera_paths", {})
        if not camera_paths and "image_path" in item:
            camera_name_raw = item.get(
                "camera_name", item.get("primary_camera", DEFAULT_CAMERA)
            )
            camera_name = (
                str(camera_name_raw).strip() if camera_name_raw is not None else ""
            )
            if not camera_name:
                camera_name = DEFAULT_CAMERA
            camera_paths = {camera_name: item["image_path"]}

        primary_camera_raw = item.get(
            "primary_camera", item.get("camera_name", DEFAULT_CAMERA)
        )
        primary_camera = (
            str(primary_camera_raw).strip() if primary_camera_raw is not None else ""
        )
        if not primary_camera:
            primary_camera = DEFAULT_CAMERA

        events.append(
            Event(
                event_rank=item.get("event_rank", 0),
                frame_idx=item.get("frame_idx", 0),
                timestamp_sec=item.get("timestamp_sec", 0.0),
                timestamp_iso_utc=item.get("timestamp_iso_utc", ""),
                severity_score=item.get("severity_score", 0.0),
                roughness=item.get("roughness"),
                min_clearance_m=item.get("min_clearance_m"),
                yaw_rate=item.get("yaw_rate"),
                imu_correlation=item.get("imu_correlation"),
                pose_confidence=item.get("pose_confidence"),
                roughness_norm=item.get("roughness_norm", 0.0),
                proximity_norm=item.get("proximity_norm", 0.0),
                yaw_rate_norm=item.get("yaw_rate_norm", 0.0),
                imu_fault_norm=item.get("imu_fault_norm", 0.0),
                localization_fault_norm=item.get("localization_fault_norm", 0.0),
                event_type=item.get("event_type", "mixed"),
                gps_lat=item.get("gps_lat"),
                gps_lon=item.get("gps_lon"),
                primary_camera=primary_camera,
                camera_paths=camera_paths if isinstance(camera_paths, dict) else {},
                gemini_caption=item.get("gemini_caption", UNAVAILABLE_CAPTION),
                gemini_model=item.get("gemini_model"),
                gemini_source=item.get("gemini_source", "unavailable"),
                gemini_latency_ms=item.get("gemini_latency_ms"),
            )
        )
    return events


def _resolve_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    runtime = load_runtime_config()
    if getattr(args, "log_level", None):
        runtime = replace(runtime, log_level=normalize_log_level(args.log_level))
    if getattr(args, "log_format", None):
        runtime = replace(runtime, log_format=normalize_log_format(args.log_format))
    return runtime


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _resolve_gemini_model(cli_model: str | None, runtime: RuntimeConfig) -> str:
    if cli_model is not None and cli_model.strip():
        return cli_model.strip()
    return runtime.gemini_model


def _init_analyst(
    *,
    disable_gemini: bool,
    model: str,
    runtime: RuntimeConfig,
    args: argparse.Namespace,
    logger: Any,
) -> tuple[GeminiAnalyst | None, str | None]:
    if disable_gemini:
        return None, "disabled_by_flag"
    try:
        retries = int(_resolve_optional_arg(args, "gemini_retries", runtime.gemini_retries))
        backoff_ms = int(
            _resolve_optional_arg(args, "gemini_backoff_ms", runtime.gemini_backoff_ms)
        )
        cache_dir = _resolve_optional_arg(args, "gemini_cache_dir", runtime.gemini_cache_dir)
        return (
            GeminiAnalyst(
                model=model,
                timeout_sec=runtime.gemini_timeout_sec,
                retries=retries,
                backoff_ms=backoff_ms,
                jitter_ratio=runtime.gemini_jitter_ratio,
                cache_dir=cache_dir,
            ),
            None,
        )
    except GeminiConfigError as exc:
        reason = f"config_error: {exc}"
        log_event(logger, "warning", "gemini_disabled", reason=reason)
        return None, reason


def _cmd_features(args: argparse.Namespace, runtime: RuntimeConfig, logger: Any) -> int:
    started = time.perf_counter()
    data_dir = _resolve_path(args.data_dir)
    output_path = _resolve_path(args.output)

    log_event(logger, "info", "features_start", run_id=args.run_id, data_dir=str(data_dir))
    loader = LogLoader(data_dir=data_dir)
    features_df = _make_feature_engine(
        loader=loader,
        runtime=runtime,
        args=args,
    ).build_features()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    elapsed = time.perf_counter() - started
    log_event(
        logger,
        "info",
        "features_complete",
        run_id=args.run_id,
        rows=int(len(features_df)),
        output_path=str(output_path),
        elapsed_sec=round(elapsed, 3),
    )
    return 0


def _cmd_events(args: argparse.Namespace, runtime: RuntimeConfig, logger: Any) -> int:
    started = time.perf_counter()
    data_dir = _resolve_path(args.data_dir)
    output_path = _resolve_path(args.output)
    model = _resolve_gemini_model(args.model, runtime)

    log_event(logger, "info", "events_start", run_id=args.run_id, data_dir=str(data_dir))
    loader = LogLoader(data_dir=data_dir)
    features_df = _make_feature_engine(
        loader=loader,
        runtime=runtime,
        args=args,
    ).build_features()
    detector = _make_event_detector(runtime, args)

    analyst, gemini_disabled_reason = _init_analyst(
        disable_gemini=args.disable_gemini,
        model=model,
        runtime=runtime,
        args=args,
        logger=logger,
    )

    orchestrator = IntelligenceOrchestrator(
        loader=loader,
        detector=detector,
        analyst=analyst,
        primary_camera=args.camera,
        gemini_workers=int(
            _resolve_optional_arg(args, "gemini_workers", runtime.gemini_workers)
        ),
    )
    events = orchestrator.build_events(
        features_df=features_df,
        top_k=args.top_k,
        distance_frames=args.distance_frames,
        model=model if analyst is not None else None,
        peak_prominence=float(
            _resolve_optional_arg(args, "peak_prominence", runtime.peak_prominence)
        ),
        peak_width=int(_resolve_optional_arg(args, "peak_width", runtime.peak_width)),
        min_distance_frames=int(
            _resolve_optional_arg(args, "peak_min_distance", args.distance_frames)
        ),
    )
    run_metadata = _build_run_metadata(
        data_dir=data_dir,
        runtime=runtime,
        args=args,
        events=events,
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "camera_name": args.camera,
        "top_k_requested": int(args.top_k),
        "distance_frames": int(args.distance_frames),
        "rows_processed": int(len(features_df)),
        "peaks_found": int(detector.last_peak_count),
        "gemini_model": model if analyst is not None else None,
        "gemini_disabled_reason": gemini_disabled_reason,
        "dataset_hash": run_metadata["dataset_hash"],
        "code_version": run_metadata["code_version"],
        "config_fingerprint": run_metadata["config_fingerprint"],
        "latency_summary": run_metadata["latency_summary"],
        "events": orchestrator.serialize_events(events),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    success_count = sum(1 for event in events if event.gemini_source in {"sdk", "rest", "cache"})
    elapsed = time.perf_counter() - started
    log_event(
        logger,
        "info",
        "events_complete",
        run_id=args.run_id,
        rows=int(len(features_df)),
        peaks_found=int(detector.last_peak_count),
        events_written=int(len(events)),
        caption_success=int(success_count),
        caption_failure=int(len(events) - success_count),
        output_path=str(output_path),
        elapsed_sec=round(elapsed, 3),
    )
    return 0


def _cmd_report(args: argparse.Namespace, runtime: RuntimeConfig, logger: Any) -> int:
    started = time.perf_counter()
    data_dir = _resolve_path(args.data_dir)
    output_path = _resolve_path(args.output)
    model = _resolve_gemini_model(args.model, runtime)

    log_event(logger, "info", "report_start", run_id=args.run_id, data_dir=str(data_dir))
    loader = LogLoader(data_dir)
    features_df = _make_feature_engine(
        loader=loader,
        runtime=runtime,
        args=args,
    ).build_features()
    detector = _make_event_detector(runtime, args)

    if args.events_json is not None:
        events_json_path = _resolve_path(args.events_json)
        events = _load_events_from_json(events_json_path)
        log_event(
            logger,
            "info",
            "report_events_loaded",
            run_id=args.run_id,
            events_json=str(events_json_path),
            events_loaded=int(len(events)),
        )
    else:
        analyst, _ = _init_analyst(
            disable_gemini=args.disable_gemini,
            model=model,
            runtime=runtime,
            args=args,
            logger=logger,
        )
        orchestrator = IntelligenceOrchestrator(
            loader=loader,
            detector=detector,
            analyst=analyst,
            primary_camera=args.camera,
            gemini_workers=int(
                _resolve_optional_arg(args, "gemini_workers", runtime.gemini_workers)
            ),
        )
        events = orchestrator.build_events(
            features_df=features_df,
            top_k=args.top_k,
            distance_frames=args.distance_frames,
            model=model if analyst is not None else None,
            peak_prominence=float(
                _resolve_optional_arg(args, "peak_prominence", runtime.peak_prominence)
            ),
            peak_width=int(_resolve_optional_arg(args, "peak_width", runtime.peak_width)),
            min_distance_frames=int(
                _resolve_optional_arg(args, "peak_min_distance", args.distance_frames)
            ),
        )
        log_event(
            logger,
            "info",
            "report_events_detected",
            run_id=args.run_id,
            events_detected=int(len(events)),
        )

    if "severity_score" not in features_df.columns:
        features_df = detector.score_dataframe(features_df)

    run_metadata = _build_run_metadata(
        data_dir=data_dir,
        runtime=runtime,
        args=args,
        events=events,
    )
    ReportBuilder = _get_report_builder()
    builder = ReportBuilder(
        loader=loader,
        features_df=features_df,
        events=events,
        metadata={
            "run_id": args.run_id,
            "dataset_hash": run_metadata["dataset_hash"],
            "code_version": run_metadata["code_version"],
            "config_fingerprint": run_metadata["config_fingerprint"],
            "latency_summary": run_metadata["latency_summary"],
        },
        include_surround=not args.no_surround,
        report_mode=str(_resolve_optional_arg(args, "report_mode", runtime.report_mode)),
        telemetry_downsample=int(
            _resolve_optional_arg(
                args,
                "report_telemetry_downsample",
                runtime.report_telemetry_downsample,
            )
        ),
        feature_columns=tuple(
            _resolve_optional_arg(
                args, "report_feature_columns", runtime.report_feature_columns
            )
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = builder.save_report(output_path)
    elapsed = time.perf_counter() - started
    log_event(
        logger,
        "info",
        "report_complete",
        run_id=args.run_id,
        output_path=str(out),
        report_size_kb=round(out.stat().st_size / 1024.0, 1),
        elapsed_sec=round(elapsed, 3),
    )
    return 0


def _cmd_process(args: argparse.Namespace, runtime: RuntimeConfig, logger: Any) -> int:
    started = time.perf_counter()
    data_dir = _resolve_path(args.data_dir)
    output_features = _resolve_path(args.output_features)
    output_events = _resolve_path(args.output_events)
    output_report = _resolve_path(args.output_report)
    model = _resolve_gemini_model(args.model, runtime)

    log_event(
        logger,
        "info",
        "process_start",
        run_id=args.run_id,
        data_dir=str(data_dir),
        output_features=str(output_features),
        output_events=str(output_events),
        output_report=str(output_report),
    )
    loader = LogLoader(data_dir=data_dir)
    features_df = _make_feature_engine(
        loader=loader,
        runtime=runtime,
        args=args,
    ).build_features()
    output_features.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_features, index=False)

    detector = _make_event_detector(runtime, args)
    analyst, gemini_disabled_reason = _init_analyst(
        disable_gemini=args.disable_gemini,
        model=model,
        runtime=runtime,
        args=args,
        logger=logger,
    )
    orchestrator = IntelligenceOrchestrator(
        loader=loader,
        detector=detector,
        analyst=analyst,
        primary_camera=args.camera,
        gemini_workers=int(
            _resolve_optional_arg(args, "gemini_workers", runtime.gemini_workers)
        ),
    )
    events = orchestrator.build_events(
        features_df=features_df,
        top_k=args.top_k,
        distance_frames=args.distance_frames,
        model=model if analyst is not None else None,
        peak_prominence=float(
            _resolve_optional_arg(args, "peak_prominence", runtime.peak_prominence)
        ),
        peak_width=int(_resolve_optional_arg(args, "peak_width", runtime.peak_width)),
        min_distance_frames=int(
            _resolve_optional_arg(args, "peak_min_distance", args.distance_frames)
        ),
    )
    run_metadata = _build_run_metadata(
        data_dir=data_dir,
        runtime=runtime,
        args=args,
        events=events,
    )

    events_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "camera_name": args.camera,
        "top_k_requested": int(args.top_k),
        "distance_frames": int(args.distance_frames),
        "rows_processed": int(len(features_df)),
        "peaks_found": int(detector.last_peak_count),
        "gemini_model": model if analyst is not None else None,
        "gemini_disabled_reason": gemini_disabled_reason,
        "dataset_hash": run_metadata["dataset_hash"],
        "code_version": run_metadata["code_version"],
        "config_fingerprint": run_metadata["config_fingerprint"],
        "latency_summary": run_metadata["latency_summary"],
        "events": orchestrator.serialize_events(events),
    }
    output_events.parent.mkdir(parents=True, exist_ok=True)
    output_events.write_text(json.dumps(events_payload, indent=2), encoding="utf-8")

    if "severity_score" not in features_df.columns:
        features_df = detector.score_dataframe(features_df)

    ReportBuilder = _get_report_builder()
    builder = ReportBuilder(
        loader=loader,
        features_df=features_df,
        events=events,
        metadata={
            "run_id": args.run_id,
            "dataset_hash": run_metadata["dataset_hash"],
            "code_version": run_metadata["code_version"],
            "config_fingerprint": run_metadata["config_fingerprint"],
            "latency_summary": run_metadata["latency_summary"],
        },
        include_surround=not args.no_surround,
        report_mode=str(_resolve_optional_arg(args, "report_mode", runtime.report_mode)),
        telemetry_downsample=int(
            _resolve_optional_arg(
                args,
                "report_telemetry_downsample",
                runtime.report_telemetry_downsample,
            )
        ),
        feature_columns=tuple(
            _resolve_optional_arg(
                args, "report_feature_columns", runtime.report_feature_columns
            )
        ),
    )
    output_report.parent.mkdir(parents=True, exist_ok=True)
    out = builder.save_report(output_report)
    mlflow_logged = log_run_lineage(
        run_metadata=run_metadata,
        run_id=args.run_id,
        command_name="process",
        artifacts=[output_features, output_events, out],
    )

    elapsed = time.perf_counter() - started
    log_event(
        logger,
        "info",
        "process_complete",
        run_id=args.run_id,
        rows=int(len(features_df)),
        peaks_found=int(detector.last_peak_count),
        events_written=int(len(events)),
        output_features=str(output_features),
        output_events=str(output_events),
        output_report=str(out),
        mlflow_logged=mlflow_logged,
        elapsed_sec=round(elapsed, 3),
    )
    return 0


def _cmd_benchmark_gemini(args: argparse.Namespace, runtime: RuntimeConfig, logger: Any) -> int:
    started = time.perf_counter()
    data_dir = _resolve_path(args.data_dir)
    output_path = _resolve_path(args.output)
    frame_indices = args.frame_indices if args.frame_indices is not None else []
    model_names = list(args.models)
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0.")
    if not model_names:
        raise ValueError("At least one model must be provided via --models.")
    if not os.getenv("GEMINI_API_KEY", "").strip():
        raise RuntimeError(
            "Gemini benchmark requires GEMINI_API_KEY. "
            "Set the environment variable and rerun."
        )

    log_event(logger, "info", "benchmark_start", run_id=args.run_id, data_dir=str(data_dir))
    loader = LogLoader(data_dir=data_dir)
    resolved_frames = _resolve_frame_indices(loader=loader, frame_indices=frame_indices, events_json=args.events_json)
    if not resolved_frames:
        raise RuntimeError("No frame indices available for benchmark.")

    analyst = GeminiAnalyst(
        model=model_names[0],
        timeout_sec=runtime.gemini_timeout_sec,
        retries=int(_resolve_optional_arg(args, "gemini_retries", runtime.gemini_retries)),
        backoff_ms=int(
            _resolve_optional_arg(args, "gemini_backoff_ms", runtime.gemini_backoff_ms)
        ),
        jitter_ratio=runtime.gemini_jitter_ratio,
        cache_dir=_resolve_optional_arg(args, "gemini_cache_dir", runtime.gemini_cache_dir),
    )

    runs: list[dict[str, object]] = []
    model_durations_sec: dict[str, float] = {}
    for model in model_names:
        model_start = time.perf_counter()
        for frame_idx in resolved_frames:
            image_path = _resolve_image_path(loader, args.camera, frame_idx)
            for repeat_idx in range(args.repeats):
                result = analyst.analyze_image(image_path=image_path, model=model)
                caption_text = result.caption.strip()
                runs.append(
                    {
                        "model": model,
                        "frame_idx": int(frame_idx),
                        "repeat": int(repeat_idx + 1),
                        "image_path": str(image_path),
                        "source": result.source,
                        "caption": caption_text,
                        "caption_word_count": len(caption_text.split()),
                        "latency_ms": result.latency_ms,
                        "error": result.error,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "thinking_tokens": result.thinking_tokens,
                        "total_tokens": result.total_tokens,
                        "success": result.source in {"sdk", "rest", "cache"},
                    }
                )
        model_durations_sec[model] = max(time.perf_counter() - model_start, 1e-9)

    summary = {
        model: _summarize_model_runs(
            [row for row in runs if row["model"] == model],
            elapsed_sec=model_durations_sec.get(model, 1e-9),
        )
        for model in model_names
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "camera_name": args.camera,
        "frame_indices": resolved_frames,
        "models": model_names,
        "repeats": int(args.repeats),
        "runs": runs,
        "summary": summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - started
    log_event(
        logger,
        "info",
        "benchmark_complete",
        run_id=args.run_id,
        frames=int(len(resolved_frames)),
        models=int(len(model_names)),
        calls=int(len(runs)),
        output_path=str(output_path),
        elapsed_sec=round(elapsed, 3),
    )
    return 0


def _resolve_frame_indices(
    *,
    loader: LogLoader,
    frame_indices: list[int],
    events_json: Path | None,
) -> list[int]:
    if frame_indices:
        return _dedupe_sorted(frame_indices)

    if events_json is not None:
        events_path = _resolve_path(events_json)
        if not events_path.exists():
            raise FileNotFoundError(f"--events-json not found: {events_path}")
        payload = json.loads(events_path.read_text(encoding="utf-8"))
        events = payload.get("events", [])
        if not isinstance(events, list):
            raise ValueError("Invalid events JSON format: expected key 'events' list.")
        extracted: list[int] = []
        for event in events:
            if isinstance(event, dict) and "frame_idx" in event:
                try:
                    extracted.append(int(event["frame_idx"]))
                except (TypeError, ValueError):
                    continue
        return _dedupe_sorted(extracted)

    features_df = FeatureEngine(loader=loader).build_features()
    detector = EventDetector()
    candidates = detector.find_events(features_df, top_k=15, distance_frames=60)
    if candidates:
        return _dedupe_sorted([candidate.frame_idx for candidate in candidates])
    frame_values = features_df["frame_idx"].head(10).tolist()
    return _dedupe_sorted([int(value) for value in frame_values])


def _resolve_image_path(loader: LogLoader, camera_name: str, frame_idx: int) -> Path:
    camera_dir = loader.frames_dir / camera_name
    stem = f"{int(frame_idx):04d}"
    for suffix in (".jpg", ".png", ".jpeg"):
        candidate = camera_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return camera_dir / f"{stem}.jpg"


def _dedupe_sorted(values: list[int]) -> list[int]:
    return sorted({int(value) for value in values})


def _summarize_model_runs(
    model_runs: list[dict[str, object]], elapsed_sec: float
) -> dict[str, object]:
    call_count = len(model_runs)
    success_count = sum(1 for row in model_runs if bool(row.get("success")))
    error_count = call_count - success_count
    error_rate = (error_count / call_count) if call_count else 0.0

    latencies: list[float] = []
    for row in model_runs:
        raw_latency = row.get("latency_ms")
        if raw_latency is None:
            continue
        if not isinstance(raw_latency, (int, float, str, bytes, bytearray)):
            continue
        try:
            parsed_latency = float(raw_latency)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(parsed_latency):
            continue
        latencies.append(parsed_latency)
    p50 = _percentile(latencies, 50.0)
    p95 = _percentile(latencies, 95.0)
    p99 = _percentile(latencies, 99.0)
    avg_latency_ms = statistics.fmean(latencies) if latencies else None

    input_tokens = _sum_optional_ints([row.get("input_tokens") for row in model_runs])
    output_tokens = _sum_optional_ints([row.get("output_tokens") for row in model_runs])
    thinking_tokens = _sum_optional_ints([row.get("thinking_tokens") for row in model_runs])
    total_tokens = _sum_optional_ints([row.get("total_tokens") for row in model_runs])
    throughput_calls_per_min = (call_count / elapsed_sec) * 60.0 if elapsed_sec > 0 else None

    return {
        "calls": call_count,
        "success_count": success_count,
        "error_count": error_count,
        "error_rate": error_rate,
        "latency_ms_avg": avg_latency_ms,
        "latency_ms_p50": p50,
        "latency_ms_p95": p95,
        "latency_ms_p99": p99,
        "throughput_calls_per_min": throughput_calls_per_min,
        "input_tokens_sum": input_tokens,
        "output_tokens_sum": output_tokens,
        "thinking_tokens_sum": thinking_tokens,
        "total_tokens_sum": total_tokens,
    }


def _sum_optional_ints(values: list[object]) -> int | None:
    converted: list[int] = []
    for value in values:
        if value is None:
            continue
        if not isinstance(value, (int, float, str, bytes, bytearray)):
            continue
        try:
            converted.append(int(value))
        except (TypeError, ValueError):
            continue
    if not converted:
        return None
    return int(sum(converted))


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--run-id",
        type=str,
        default=DEFAULT_RUN_ID,
        help=f"Run identifier for logs/report metadata. Default: {DEFAULT_RUN_ID}",
    )
    common.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    common.add_argument(
        "--log-format",
        type=str,
        default=None,
        help="Override log format (auto, json, console).",
    )
    common.add_argument(
        "--depth-workers",
        type=int,
        default=None,
        help="Depth extraction workers override.",
    )
    common.add_argument(
        "--depth-cache-dir",
        type=Path,
        default=None,
        help="Persistent depth feature cache directory.",
    )
    common.add_argument(
        "--gemini-workers",
        type=int,
        default=None,
        help="Gemini analysis worker count override.",
    )
    common.add_argument(
        "--gemini-retries",
        type=int,
        default=None,
        help="Gemini retry count override.",
    )
    common.add_argument(
        "--gemini-backoff-ms",
        type=int,
        default=None,
        help="Gemini retry backoff in milliseconds override.",
    )
    common.add_argument(
        "--gemini-cache-dir",
        type=Path,
        default=None,
        help="Deterministic Gemini cache directory.",
    )
    common.add_argument(
        "--score-normalization",
        choices=["minmax", "robust"],
        default=None,
        help="Event scoring normalization mode override.",
    )
    common.add_argument(
        "--score-robust-quantile-low",
        type=float,
        default=None,
        help="Lower quantile for robust scaling.",
    )
    common.add_argument(
        "--score-robust-quantile-high",
        type=float,
        default=None,
        help="Upper quantile for robust scaling.",
    )
    common.add_argument(
        "--peak-prominence",
        type=float,
        default=None,
        help="Peak prominence override for event detection.",
    )
    common.add_argument(
        "--peak-width",
        type=int,
        default=None,
        help="Peak width override for event detection.",
    )
    common.add_argument(
        "--peak-min-distance",
        type=int,
        default=None,
        help="Minimum frame distance between peaks override.",
    )
    common.add_argument(
        "--report-mode",
        choices=["single", "split"],
        default=None,
        help="Report output mode override.",
    )
    common.add_argument(
        "--report-telemetry-downsample",
        type=int,
        default=None,
        help="Downsample factor for report telemetry/features payloads.",
    )
    common.add_argument(
        "--report-feature-columns",
        type=_parse_csv_columns,
        default=None,
        help="Comma-separated feature columns to embed in report payload.",
    )

    parser = argparse.ArgumentParser(
        prog="agri-auditor",
        description="Agri Auditor production CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_features = subparsers.add_parser(
        "features", parents=[common], description="Build Step 2 feature table."
    )
    parser_features.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Path to dataset directory. Default: {DEFAULT_DATA_DIR}",
    )
    parser_features.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FEATURES_OUTPUT,
        help=f"Output CSV path. Default: {DEFAULT_FEATURES_OUTPUT}",
    )
    parser_features.set_defaults(handler=_cmd_features)

    parser_events = subparsers.add_parser(
        "events", parents=[common], description="Build Step 3 events."
    )
    parser_events.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Path to dataset directory. Default: {DEFAULT_DATA_DIR}",
    )
    parser_events.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EVENTS_OUTPUT,
        help=f"Output JSON path. Default: {DEFAULT_EVENTS_OUTPUT}",
    )
    parser_events.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top events to export. Default: {DEFAULT_TOP_K}",
    )
    parser_events.add_argument(
        "--distance-frames",
        type=int,
        default=DEFAULT_DISTANCE_FRAMES,
        help=f"Minimum spacing in frames between peaks. Default: {DEFAULT_DISTANCE_FRAMES}",
    )
    parser_events.add_argument(
        "--model",
        type=str,
        default=None,
        help="Gemini model name override.",
    )
    parser_events.add_argument(
        "--camera",
        type=str,
        default=DEFAULT_CAMERA,
        help=f"Camera name used for event thumbnails. Default: {DEFAULT_CAMERA}",
    )
    parser_events.add_argument(
        "--disable-gemini",
        action="store_true",
        help="Disable Gemini calls and emit unavailable captions.",
    )
    parser_events.set_defaults(handler=_cmd_events)

    parser_report = subparsers.add_parser(
        "report", parents=[common], description="Generate Mission Control HTML dashboard."
    )
    parser_report.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Path to dataset directory. Default: {DEFAULT_DATA_DIR}",
    )
    parser_report.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_OUTPUT,
        help=f"Output HTML path. Default: {DEFAULT_REPORT_OUTPUT}",
    )
    parser_report.add_argument(
        "--events-json",
        type=Path,
        default=None,
        help="Pre-built events.json path. If absent, events are detected online.",
    )
    parser_report.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top events to detect. Default: {DEFAULT_TOP_K}",
    )
    parser_report.add_argument(
        "--distance-frames",
        type=int,
        default=DEFAULT_DISTANCE_FRAMES,
        help=f"Min frames between event peaks. Default: {DEFAULT_DISTANCE_FRAMES}",
    )
    parser_report.add_argument(
        "--disable-gemini",
        action="store_true",
        help="Disable Gemini caption generation.",
    )
    parser_report.add_argument(
        "--no-surround",
        action="store_true",
        help="Exclude surround-view thumbnails from event cards.",
    )
    parser_report.add_argument(
        "--model",
        type=str,
        default=None,
        help="Gemini model name override.",
    )
    parser_report.add_argument(
        "--camera",
        type=str,
        default=DEFAULT_CAMERA,
        help=f"Primary camera for event analysis. Default: {DEFAULT_CAMERA}",
    )
    parser_report.set_defaults(handler=_cmd_report)

    parser_process = subparsers.add_parser(
        "process", parents=[common], description="Run full pipeline: features + events + report."
    )
    parser_process.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Path to dataset directory. Default: {DEFAULT_DATA_DIR}",
    )
    parser_process.add_argument(
        "--output-features",
        type=Path,
        default=DEFAULT_FEATURES_OUTPUT,
        help=f"Output CSV path for features. Default: {DEFAULT_FEATURES_OUTPUT}",
    )
    parser_process.add_argument(
        "--output-events",
        type=Path,
        default=DEFAULT_EVENTS_OUTPUT,
        help=f"Output JSON path for events. Default: {DEFAULT_EVENTS_OUTPUT}",
    )
    parser_process.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_REPORT_OUTPUT,
        help=f"Output HTML path for report. Default: {DEFAULT_REPORT_OUTPUT}",
    )
    parser_process.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top events to detect. Default: {DEFAULT_TOP_K}",
    )
    parser_process.add_argument(
        "--distance-frames",
        type=int,
        default=DEFAULT_DISTANCE_FRAMES,
        help=f"Min frames between event peaks. Default: {DEFAULT_DISTANCE_FRAMES}",
    )
    parser_process.add_argument(
        "--disable-gemini",
        action="store_true",
        help="Disable Gemini caption generation.",
    )
    parser_process.add_argument(
        "--no-surround",
        action="store_true",
        help="Exclude surround-view thumbnails from event cards.",
    )
    parser_process.add_argument(
        "--model",
        type=str,
        default=None,
        help="Gemini model name override.",
    )
    parser_process.add_argument(
        "--camera",
        type=str,
        default=DEFAULT_CAMERA,
        help=f"Primary camera for event analysis. Default: {DEFAULT_CAMERA}",
    )
    parser_process.set_defaults(handler=_cmd_process)

    parser_bench = subparsers.add_parser(
        "benchmark-gemini",
        parents=[common],
        description="Benchmark Gemini models on selected frame indices.",
    )
    parser_bench.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Path to dataset directory. Default: {DEFAULT_DATA_DIR}",
    )
    parser_bench.add_argument(
        "--frame-indices",
        nargs="*",
        type=int,
        default=None,
        help="Explicit frame indices to benchmark (space-separated).",
    )
    parser_bench.add_argument(
        "--events-json",
        type=Path,
        default=None,
        help="Optional events.json path to source frame indices.",
    )
    parser_bench.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["gemini-3-flash-preview", "gemini-2.5-flash"],
        help="Model names to benchmark.",
    )
    parser_bench.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated calls per frame per model.",
    )
    parser_bench.add_argument(
        "--camera",
        type=str,
        default=DEFAULT_CAMERA,
        help=f"Camera used to resolve frame image files. Default: {DEFAULT_CAMERA}",
    )
    parser_bench.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_BENCHMARK_OUTPUT,
        help=f"Output JSON path. Default: {DEFAULT_BENCHMARK_OUTPUT}",
    )
    parser_bench.set_defaults(handler=_cmd_benchmark_gemini)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    runtime = _resolve_runtime_config(args)
    effective_log_format = configure_logging(runtime.log_level, runtime.log_format)
    logger = get_logger("agri_auditor.cli")
    log_event(
        logger,
        "info",
        "command_start",
        run_id=args.run_id,
        command=args.command,
        log_level=runtime.log_level,
        log_format=effective_log_format,
    )

    try:
        return int(args.handler(args, runtime, logger))
    except KeyboardInterrupt:
        log_event(
            logger,
            "warning",
            "command_interrupted",
            run_id=args.run_id,
            command=args.command,
        )
        return 130
    except (FileNotFoundError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        log_event(
            logger,
            "error",
            "command_failed",
            run_id=args.run_id,
            command=args.command,
            error=str(exc),
        )
        return 1
