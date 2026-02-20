# Agri Auditor

Autonomous incident auditing CLI for tractor mission logs.

The pipeline ingests raw log data, computes safety and kinematic features, detects critical events, optionally adds Gemini visual captions, and outputs a single interactive `audit_report.html`.

## Roadmap Alignment (`intelligence_roadmap_1`)

- Phase 1 Data Rig: Completed (`LogLoader` in `src/agri_auditor/ingestion.py`)
- Phase 2 Physics/Features: Completed (`FeatureEngine` in `src/agri_auditor/features.py`)
- Phase 3 Intelligence Layer: Completed (`EventDetector`, `GeminiAnalyst`, orchestrator in `src/agri_auditor/intelligence.py`)
- Phase 4 Dashboard: Completed (`ReportBuilder` in `src/agri_auditor/reporting.py`)
- Phase 5 Productionization: Partial (env-based config present; Dockerfile not yet added)
- Phase 6 Documentation/Polish: In progress (this README and tests)

## Requirements

- Python 3.13+
- Optional for AI captions: `GEMINI_API_KEY`

## Setup

```powershell
python -m pip install -e .[dev]
```

## Test

```powershell
python -m pytest -q
```

## End-to-End Usage

1. Build features:

```powershell
python scripts/build_features.py --data-dir ../provided_data --output artifacts/features.csv
```

2. Build detected events:

```powershell
python scripts/build_events.py --data-dir ../provided_data --output artifacts/events.json --top-k 5 --distance-frames 150 --disable-gemini
```

3. Generate dashboard report:

```powershell
python scripts/build_report.py --data-dir ../provided_data --output artifacts/audit_report.html --disable-gemini
```

4. Optional Gemini captions:

```powershell
$env:GEMINI_API_KEY="your-key"
python scripts/build_events.py --data-dir ../provided_data --output artifacts/events.json
python scripts/build_report.py --data-dir ../provided_data --events-json artifacts/events.json --output artifacts/audit_report.html
```

## Main Components

- `src/agri_auditor/ingestion.py`: manifest + calibration loading, velocity derivation, image access
- `src/agri_auditor/features.py`: roughness, clearance, canopy proxy, orientation, sensor health, GPS cleaning
- `src/agri_auditor/intelligence.py`: event scoring + peak detection + Gemini integration
- `src/agri_auditor/reporting.py`: interactive Mission Control HTML renderer
- `scripts/build_features.py`: feature export CLI
- `scripts/build_events.py`: event export CLI
- `scripts/build_report.py`: dashboard generation CLI
