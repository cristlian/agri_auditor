from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agri_auditor.ingestion import LogLoader  # noqa: E402


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
