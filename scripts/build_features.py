from __future__ import annotations

import argparse
from pathlib import Path

try:
    from agri_auditor import FeatureEngine, LogLoader
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
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "features.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Step 2 feature table (roughness + min clearance)."
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
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    loader = LogLoader(data_dir=data_dir)
    engine = FeatureEngine(loader=loader)
    features_df = engine.build_features()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    print("Feature export complete.")
    print(f"rows={len(features_df)}")
    print(f"roughness_non_null={int(features_df['roughness'].notna().sum())}")
    print(
        f"min_clearance_m_non_null={int(features_df['min_clearance_m'].notna().sum())}"
    )
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
