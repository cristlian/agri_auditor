from __future__ import annotations

from pathlib import Path

try:
    from agri_auditor.ingestion import LogLoader
except ModuleNotFoundError as exc:
    if exc.name == "agri_auditor":
        raise SystemExit(
            "Unable to import 'agri_auditor'. Install the project first "
            "(for example: `python -m pip install -e .` or `poetry install`) "
            "and rerun this script."
        ) from exc
    raise


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data_dir = PROJECT_ROOT.parent / "provided_data"

    loader = LogLoader(data_dir=data_dir)
    manifest_df = loader.load_manifest()

    print("Manifest preview with velocity_mps:")
    print(manifest_df[["frame_idx", "timestamp_sec", "velocity_mps"]].head(10))

    model = loader.get_camera_model("front_center_stereo_left")
    print("\nCamera Model:")
    print(
        {
            "camera_name": model.camera_name,
            "width": model.width,
            "height": model.height,
            "fx": model.fx,
            "fy": model.fy,
            "cx": model.cx,
            "cy": model.cy,
        }
    )
    print("K matrix:")
    print(model.K)


if __name__ == "__main__":
    main()
