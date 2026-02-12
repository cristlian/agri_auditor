from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from agri_auditor import (
        EventDetector,
        FeatureEngine,
        GeminiAnalyst,
        GeminiConfigError,
        LogLoader,
    )
except ModuleNotFoundError as exc:
    if exc.name == "agri_auditor":
        raise SystemExit(
            "Unable to import 'agri_auditor'. Install the project first "
            "(for example: `python -m pip install -e .` or `poetry install`) "
            "and rerun this script."
        ) from exc
    raise


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT.parent / "provided_data"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "gemini_benchmark.json"
DEFAULT_MODELS = ("gemini-3-flash-preview", "gemini-2.5-flash")
DEFAULT_CAMERA_NAME = "front_center_stereo_left"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Gemini models on selected frame indices."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Path to dataset directory. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--frame-indices",
        nargs="*",
        type=int,
        default=None,
        help="Explicit frame indices to benchmark (space-separated).",
    )
    parser.add_argument(
        "--events-json",
        type=Path,
        default=None,
        help="Optional events.json path to source frame indices.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=list(DEFAULT_MODELS),
        help=f"Model names to benchmark. Default: {list(DEFAULT_MODELS)}",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated calls per frame per model.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=DEFAULT_CAMERA_NAME,
        help="Camera used to resolve frame image files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if args.repeats <= 0:
        raise SystemExit("--repeats must be > 0.")
    if not args.models:
        raise SystemExit("At least one model must be provided via --models.")

    loader = LogLoader(data_dir=data_dir)
    frame_indices = _resolve_frame_indices(loader=loader, args=args)
    if not frame_indices:
        raise SystemExit("No frame indices available for benchmark.")

    try:
        analyst = GeminiAnalyst(model=args.models[0])
    except GeminiConfigError as exc:
        raise SystemExit(f"Gemini benchmark requires GEMINI_API_KEY: {exc}") from exc

    runs: list[dict[str, object]] = []
    model_durations_sec: dict[str, float] = {}

    for model in args.models:
        model_start = time.perf_counter()
        for frame_idx in frame_indices:
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
                        "success": result.source in {"sdk", "rest"},
                    }
                )
        model_durations_sec[model] = max(time.perf_counter() - model_start, 1e-9)

    summary = {
        model: _summarize_model_runs(
            model_runs=[row for row in runs if row["model"] == model],
            elapsed_sec=model_durations_sec.get(model, 1e-9),
        )
        for model in args.models
    }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "camera_name": args.camera,
        "frame_indices": frame_indices,
        "models": args.models,
        "repeats": int(args.repeats),
        "runs": runs,
        "summary": summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Gemini benchmark complete.")
    print(f"frames={len(frame_indices)}")
    print(f"models={len(args.models)}")
    print(f"repeats={args.repeats}")
    print(f"calls={len(runs)}")
    print(f"output={output_path}")


def _resolve_frame_indices(loader: LogLoader, args: argparse.Namespace) -> list[int]:
    if args.frame_indices:
        return _dedupe_sorted(args.frame_indices)

    if args.events_json is not None:
        events_path = args.events_json.expanduser().resolve()
        if not events_path.exists():
            raise SystemExit(f"--events-json not found: {events_path}")
        payload = json.loads(events_path.read_text(encoding="utf-8"))
        events = payload.get("events", [])
        if not isinstance(events, list):
            raise SystemExit("Invalid events JSON format: expected key 'events' list.")
        extracted = []
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
    model_runs: list[dict[str, object]],
    elapsed_sec: float,
) -> dict[str, object]:
    call_count = len(model_runs)
    success_count = sum(1 for row in model_runs if bool(row.get("success")))
    error_count = call_count - success_count
    error_rate = (error_count / call_count) if call_count else 0.0

    latencies = [
        float(row["latency_ms"])
        for row in model_runs
        if row.get("latency_ms") is not None
    ]
    p50 = _percentile(latencies, 50.0)
    p95 = _percentile(latencies, 95.0)
    p99 = _percentile(latencies, 99.0)
    avg_latency_ms = statistics.fmean(latencies) if latencies else None

    input_tokens = _sum_optional_ints(row.get("input_tokens") for row in model_runs)
    output_tokens = _sum_optional_ints(row.get("output_tokens") for row in model_runs)
    thinking_tokens = _sum_optional_ints(row.get("thinking_tokens") for row in model_runs)
    total_tokens = _sum_optional_ints(row.get("total_tokens") for row in model_runs)
    throughput_frames_per_min = (call_count / elapsed_sec) * 60.0 if elapsed_sec > 0 else None

    return {
        "calls": call_count,
        "success_count": success_count,
        "error_count": error_count,
        "error_rate": error_rate,
        "latency_ms_avg": avg_latency_ms,
        "latency_ms_p50": p50,
        "latency_ms_p95": p95,
        "latency_ms_p99": p99,
        "throughput_calls_per_min": throughput_frames_per_min,
        "input_tokens_sum": input_tokens,
        "output_tokens_sum": output_tokens,
        "thinking_tokens_sum": thinking_tokens,
        "total_tokens_sum": total_tokens,
    }


def _sum_optional_ints(values: list[object] | object) -> int | None:
    items = list(values) if not isinstance(values, list) else values
    converted: list[int] = []
    for value in items:
        if value is None:
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


if __name__ == "__main__":
    main()
