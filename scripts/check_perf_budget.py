from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

DEFAULT_DATA_DIR = PROJECT_ROOT.parent / "provided_data"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_CACHE_DIR = DEFAULT_ARTIFACT_DIR / "perf_depth_cache"
DEFAULT_MAX_COLD_SEC = 20.0
DEFAULT_MAX_WARM_TO_COLD_RATIO = 0.95
DEFAULT_DEPTH_WORKERS = 1


def _measure_feature_build(
    *,
    data_dir: Path,
    cache_dir: Path,
    depth_workers: int,
) -> tuple[float, int]:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from agri_auditor import FeatureEngine, LogLoader

    loader = LogLoader(data_dir=data_dir)
    start = time.perf_counter()
    rows = len(
        FeatureEngine(
            loader=loader,
            depth_workers=depth_workers,
            depth_cache_dir=cache_dir,
        ).build_features()
    )
    elapsed_sec = time.perf_counter() - start
    return elapsed_sec, rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic feature-extraction perf budget checks with cold/warm cache."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Dataset directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Depth cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--depth-workers",
        type=int,
        default=DEFAULT_DEPTH_WORKERS,
        help=f"Depth workers for perf run (default: {DEFAULT_DEPTH_WORKERS})",
    )
    parser.add_argument(
        "--max-cold-sec",
        type=float,
        default=DEFAULT_MAX_COLD_SEC,
        help=f"Max allowed cold build time in seconds (default: {DEFAULT_MAX_COLD_SEC})",
    )
    parser.add_argument(
        "--max-warm-to-cold-ratio",
        type=float,
        default=DEFAULT_MAX_WARM_TO_COLD_RATIO,
        help=(
            "Max allowed warm/cold ratio. Lower values require stronger warm-cache speedup "
            f"(default: {DEFAULT_MAX_WARM_TO_COLD_RATIO})."
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path for writing the perf summary JSON payload.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    data_dir = Path(args.data_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    depth_workers = int(args.depth_workers)
    max_cold_sec = float(args.max_cold_sec)
    max_warm_to_cold_ratio = float(args.max_warm_to_cold_ratio)

    if depth_workers <= 0:
        raise ValueError("--depth-workers must be > 0.")
    if max_cold_sec <= 0:
        raise ValueError("--max-cold-sec must be > 0.")
    if max_warm_to_cold_ratio <= 0:
        raise ValueError("--max-warm-to-cold-ratio must be > 0.")
    if not data_dir.exists():
        raise FileNotFoundError(f"--data-dir not found: {data_dir}")

    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
        except OSError:
            cache_dir = cache_dir.parent / f"{cache_dir.name}_{time.time_ns()}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cold_sec, row_count = _measure_feature_build(
        data_dir=data_dir,
        cache_dir=cache_dir,
        depth_workers=depth_workers,
    )
    warm_sec, warm_rows = _measure_feature_build(
        data_dir=data_dir,
        cache_dir=cache_dir,
        depth_workers=depth_workers,
    )

    warm_to_cold = (warm_sec / cold_sec) if cold_sec > 0 else 0.0
    summary = {
        "data_dir": str(data_dir),
        "cache_dir": str(cache_dir),
        "depth_workers": depth_workers,
        "rows_cold": row_count,
        "rows_warm": warm_rows,
        "cold_sec": round(cold_sec, 4),
        "warm_sec": round(warm_sec, 4),
        "warm_to_cold_ratio": round(warm_to_cold, 4),
        "max_cold_sec": max_cold_sec,
        "max_warm_to_cold_ratio": max_warm_to_cold_ratio,
    }

    failures: list[str] = []
    if row_count != warm_rows:
        failures.append(
            f"Row-count mismatch between cold and warm runs ({row_count} != {warm_rows})."
        )
    if cold_sec > max_cold_sec:
        failures.append(
            f"Cold run exceeded budget ({cold_sec:.3f}s > {max_cold_sec:.3f}s)."
        )
    if warm_to_cold > max_warm_to_cold_ratio:
        failures.append(
            "Warm run did not meet cache speedup budget "
            f"({warm_to_cold:.3f} > {max_warm_to_cold_ratio:.3f})."
        )

    if args.summary_json is not None:
        summary_path = Path(args.summary_json).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if failures:
        for failure in failures:
            print(f"PERF_BUDGET_FAIL: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
