# agri-auditor (Step 1)

This project implements Step 1 of the roadmap:
- project scaffold with Poetry
- robust `LogLoader` for manifest, calibration, image loading
- NaN-safe velocity derivation from pose deltas
- calibration filename auto-detection and camera dimension inference

## Setup

```powershell
poetry install
```

## Validate

```powershell
poetry run pytest -q
poetry run python scripts/test_load.py
```

## Notes

- Loader auto-detects `calibrations.json` or `calibration.json`.
- `velocity_mps` is computed from pose columns and `timestamp_sec`.
- If calibration width/height is `0`, camera size is inferred from `frames/<camera_name>/`.

