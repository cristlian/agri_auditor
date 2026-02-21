from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROW_COUNT = 1085


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare synthetic provided_data for CI and local testing."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1].parent / "provided_data",
        help="Target provided_data directory path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite manifest/calibration/images if they already exist.",
    )
    return parser.parse_args()


def _generate_manifest(output_dir: Path, row_count: int = ROW_COUNT) -> None:
    frame_idx = np.arange(row_count, dtype=np.int64)
    timestamp_sec = 1_700_000_000.0 + frame_idx * 0.1
    pos_x = frame_idx * 0.2
    pos_y = np.sin(frame_idx / 40.0) * 0.5
    pos_z = np.zeros_like(pos_x)
    has_depth = (frame_idx % 3) == 0

    imu_camera_accel_z = 9.8 + np.sin(frame_idx / 11.0) * 0.8
    imu_syslogic_accel_z = -9.2 + np.sin(frame_idx / 13.0) * 0.7
    gps_lat = 39.661 + frame_idx * 1e-5
    gps_lon = -0.558 + frame_idx * 1e-5
    pose_conf = 60.0 + np.sin(frame_idx / 17.0) * 8.0

    df = pd.DataFrame(
        {
            "frame_idx": frame_idx,
            "timestamp_sec": timestamp_sec,
            "pose_front_center_stereo_left_x": pos_x,
            "pose_front_center_stereo_left_y": pos_y,
            "pose_front_center_stereo_left_z": pos_z,
            "pose_front_center_stereo_left_qx": np.zeros(row_count, dtype=np.float64),
            "pose_front_center_stereo_left_qy": np.zeros(row_count, dtype=np.float64),
            "pose_front_center_stereo_left_qz": np.zeros(row_count, dtype=np.float64),
            "pose_front_center_stereo_left_qw": np.ones(row_count, dtype=np.float64),
            "pose_front_center_stereo_left_confidence": pose_conf,
            "has_depth": has_depth,
            "imu_camera_accel_z": imu_camera_accel_z,
            "imu_syslogic_accel_z": imu_syslogic_accel_z,
            "gps_lat": gps_lat,
            "gps_lon": gps_lon,
            "gps_alt": np.full(row_count, 12.0, dtype=np.float64),
        }
    )
    df.to_csv(output_dir / "manifest.csv", index=False)


def _generate_calibration(output_dir: Path) -> None:
    payload = {
        "front_center_stereo_left": {
            "fx": 1000.0,
            "fy": 1000.0,
            "cx": 960.0,
            "cy": 600.0,
            "width": 0,
            "height": 0,
        }
    }
    (output_dir / "calibrations.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _generate_images(output_dir: Path, row_count: int = ROW_COUNT) -> None:
    frames_dir = output_dir / "frames"
    front_center = frames_dir / "front_center_stereo_left"
    depth_dir = frames_dir / "depth"
    front_center.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # 1920x1200 is required by ingestion tests that infer image dimensions.
    rgb = np.zeros((1200, 1920, 3), dtype=np.uint8)
    rgb[:, :, 1] = 120
    Image.fromarray(rgb).save(front_center / "0000.jpg", quality=85)
    Image.fromarray(rgb).save(front_center / "0010.jpg", quality=85)

    for idx in range(row_count):
        if idx % 3 != 0:
            continue
        depth = np.full((16, 16), 3200 + (idx % 100), dtype=np.uint16)
        depth[0:2, 0:2] = 0
        Image.fromarray(depth).save(depth_dir / f"{idx:04d}.png")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"
    cal_path = output_dir / "calibrations.json"
    sample_image = output_dir / "frames" / "front_center_stereo_left" / "0000.jpg"

    if (
        not args.force
        and manifest_path.exists()
        and cal_path.exists()
        and sample_image.exists()
    ):
        print(f"Test data already present at: {output_dir}")
        return

    _generate_manifest(output_dir)
    _generate_calibration(output_dir)
    _generate_images(output_dir)
    print(f"Synthetic test data prepared at: {output_dir}")


if __name__ == "__main__":
    main()
