"""Build the Mission Control HTML dashboard from raw data or pre-built artifacts.

Usage (full pipeline):
    python scripts/build_report.py --data-dir ../provided_data --output audit_report.html

Usage (from pre-built events JSON):
    python scripts/build_report.py --data-dir ../provided_data \
        --events-json artifacts/events.json --output audit_report.html

Options:
    --top-k             Number of events to detect (default: 5)
    --distance-frames   Min frames between peaks (default: 150)
    --disable-gemini    Skip AI caption generation
    --no-surround       Exclude surround-view camera thumbnails
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from agri_auditor import LogLoader, FeatureEngine, IntelligenceOrchestrator, EventDetector
    from agri_auditor.intelligence import Event, GeminiAnalyst, GeminiConfigError, UNAVAILABLE_CAPTION
    from agri_auditor.reporting import ReportBuilder
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from agri_auditor import LogLoader, FeatureEngine, IntelligenceOrchestrator, EventDetector
    from agri_auditor.intelligence import Event, GeminiAnalyst, GeminiConfigError, UNAVAILABLE_CAPTION
    from agri_auditor.reporting import ReportBuilder


def _load_events_from_json(
    json_path: Path, loader: LogLoader,
) -> list[Event]:
    """Reconstruct Event dataclass instances from a serialized events.json."""
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    events_raw = raw.get("events", raw if isinstance(raw, list) else [])

    events: list[Event] = []
    for item in events_raw:
        # Handle both old schema (image_path) and new schema (camera_paths)
        camera_paths = item.get("camera_paths", {})
        if not camera_paths and "image_path" in item:
            cam = item.get("camera_name", item.get("primary_camera", "front_center_stereo_left"))
            camera_paths = {cam: item["image_path"]}

        events.append(Event(
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
            primary_camera=item.get("primary_camera", item.get("camera_name", "front_center_stereo_left")),
            camera_paths=camera_paths,
            gemini_caption=item.get("gemini_caption", UNAVAILABLE_CAPTION),
            gemini_model=item.get("gemini_model"),
            gemini_source=item.get("gemini_source", "unavailable"),
            gemini_latency_ms=item.get("gemini_latency_ms"),
        ))
    return events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Mission Control HTML dashboard."
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to the raw data directory (contains manifest.csv, frames/, etc.).",
    )
    parser.add_argument(
        "--output", type=str, default="audit_report.html",
        help="Output HTML file path (default: audit_report.html).",
    )
    parser.add_argument(
        "--events-json", type=str, default=None,
        help="Pre-built events.json to load instead of running the pipeline.",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top events to detect (default: 5).",
    )
    parser.add_argument(
        "--distance-frames", type=int, default=150,
        help="Min frames between event peaks (default: 150).",
    )
    parser.add_argument(
        "--disable-gemini", action="store_true",
        help="Skip AI caption generation.",
    )
    parser.add_argument(
        "--no-surround", action="store_true",
        help="Exclude surround-view camera thumbnails from event cards.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Gemini model name override.",
    )
    parser.add_argument(
        "--run-id", type=str, default="audit-run",
        help="Run identifier for the report header.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_path = Path(args.output).resolve()

    print(f"[report] Loading data from: {data_dir}")
    t0 = time.perf_counter()

    loader = LogLoader(data_dir)

    # Build features
    print("[report] Building features ...")
    engine = FeatureEngine(loader=loader)
    features_df = engine.build_features()
    print(f"[report]   features: {len(features_df)} rows")

    # Get events
    if args.events_json:
        print(f"[report] Loading events from: {args.events_json}")
        events = _load_events_from_json(Path(args.events_json), loader)
        print(f"[report]   events loaded: {len(events)}")
    else:
        print("[report] Detecting events ...")
        analyst = None
        if not args.disable_gemini:
            try:
                analyst = GeminiAnalyst(model=args.model or "gemini-3-flash-preview")
                print(f"[report]   Gemini model: {analyst.model}")
            except GeminiConfigError as exc:
                print(f"[report]   Gemini disabled: {exc}")
                analyst = None

        orchestrator = IntelligenceOrchestrator(
            loader=loader, analyst=analyst,
        )
        events = orchestrator.build_events(
            features_df=features_df,
            top_k=args.top_k,
            distance_frames=args.distance_frames,
            model=args.model,
        )
        print(f"[report]   events detected: {len(events)}")

    # Score the features dataframe for severity chart
    if "severity_score" not in features_df.columns:
        detector = EventDetector()
        features_df = detector.score_dataframe(features_df)

    # Build report
    print("[report] Generating HTML dashboard ...")
    builder = ReportBuilder(
        loader=loader,
        features_df=features_df,
        events=events,
        metadata={"run_id": args.run_id},
        include_surround=not args.no_surround,
    )
    out = builder.save_report(output_path)

    elapsed = time.perf_counter() - t0
    size_kb = out.stat().st_size / 1024
    print(f"[report] Dashboard saved: {out}")
    print(f"[report]   size: {size_kb:.1f} KB")
    print(f"[report]   elapsed: {elapsed:.1f}s")
    print("[report] Done.")


if __name__ == "__main__":
    main()
