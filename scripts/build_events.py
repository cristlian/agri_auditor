from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from agri_auditor import (
        EventDetector,
        FeatureEngine,
        GeminiAnalyst,
        GeminiConfigError,
        IntelligenceOrchestrator,
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
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "events.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Step 3 events (peak detection + optional Gemini captions)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Path to dataset directory. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top events to export.",
    )
    parser.add_argument(
        "--distance-frames",
        type=int,
        default=150,
        help="Minimum spacing in frames between peaks.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-flash-preview",
        help="Gemini model name for caption generation.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="front_center_stereo_left",
        help="Camera name used for event thumbnails.",
    )
    parser.add_argument(
        "--disable-gemini",
        action="store_true",
        help="Disable Gemini calls and emit placeholder captions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    loader = LogLoader(data_dir=data_dir)
    features_df = FeatureEngine(loader=loader).build_features()

    detector = EventDetector()
    analyst: GeminiAnalyst | None = None
    gemini_disabled_reason: str | None = None

    if args.disable_gemini:
        gemini_disabled_reason = "disabled_by_flag"
    else:
        try:
            analyst = GeminiAnalyst(model=args.model)
        except GeminiConfigError as exc:
            gemini_disabled_reason = f"config_error: {exc}"

    orchestrator = IntelligenceOrchestrator(
        loader=loader,
        detector=detector,
        analyst=analyst,
        primary_camera=args.camera,
    )
    events = orchestrator.build_events(
        features_df=features_df,
        top_k=args.top_k,
        distance_frames=args.distance_frames,
        model=args.model if analyst is not None else None,
    )

    success_count = sum(1 for event in events if event.gemini_source in {"sdk", "rest"})
    failure_count = len(events) - success_count

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "camera_name": args.camera,
        "top_k_requested": int(args.top_k),
        "distance_frames": int(args.distance_frames),
        "rows_processed": int(len(features_df)),
        "peaks_found": int(detector.last_peak_count),
        "gemini_model": args.model if analyst is not None else None,
        "gemini_disabled_reason": gemini_disabled_reason,
        "events": orchestrator.serialize_events(events),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Events export complete.")
    print(f"rows_processed={len(features_df)}")
    print(f"peaks_found={detector.last_peak_count}")
    print(f"top_k_written={len(events)}")
    print(f"caption_success={success_count}")
    print(f"caption_failure={failure_count}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
