#!/usr/bin/env python3
"""Compose CAM_F0 mp4 videos from SimEngine openscene_format exports."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import time
from pathlib import Path


def ordered_cam_f0_frames(meta_pkl: Path, sensor_root: Path) -> list[Path]:
    with meta_pkl.open("rb") as f:
        frames = pickle.load(f)
    if not isinstance(frames, list):
        raise TypeError(f"Expected list in {meta_pkl}, got {type(frames)}")

    paths: list[Path] = []
    for frame in frames:
        data_path = frame["cams"]["CAM_F0"]["data_path"]
        image_path = sensor_root / data_path
        if not image_path.exists():
            raise FileNotFoundError(f"Missing rendered frame: {image_path}")
        paths.append(image_path)
    return paths


def compose_video(frame_paths: list[Path], output_mp4: Path, fps: int) -> None:
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    list_file = output_mp4.with_suffix(".txt")
    list_file.write_text(
        "\n".join(f"file '{path.resolve().as_posix()}'" for path in frame_paths) + "\n"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-vf",
        f"fps={fps}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_mp4),
    ]
    subprocess.run(cmd, check=True)
    list_file.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--we-output",
        type=Path,
        required=True,
        help="SimEngine output dir containing openscene_format/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write mp4 files",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--scenario-id",
        action="append",
        default=[],
        help="Scenario id to compose; repeat flag or omit to compose all meta pkls",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="Optional path to write a summary manifest",
    )
    args = parser.parse_args()

    openscene_root = args.we_output / "openscene_format"
    meta_dir = openscene_root / "meta_datas"
    sensor_root = openscene_root / "sensor_blobs"
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"Missing meta_datas directory: {meta_dir}")

    if args.scenario_id:
        meta_files = [meta_dir / f"{scenario_id}.pkl" for scenario_id in args.scenario_id]
    else:
        meta_files = sorted(meta_dir.glob("*.pkl"))

    manifest = {
        "fps": args.fps,
        "source_we_output": str(args.we_output.resolve()),
        "videos": [],
    }
    started = time.time()

    for meta_pkl in meta_files:
        if not meta_pkl.exists():
            raise FileNotFoundError(f"Missing meta pickle: {meta_pkl}")
        scenario_id = meta_pkl.stem
        frame_paths = ordered_cam_f0_frames(meta_pkl, sensor_root)
        output_mp4 = args.output_dir / f"{scenario_id}_CAM_F0_{args.fps}fps.mp4"
        t0 = time.time()
        compose_video(frame_paths, output_mp4, fps=args.fps)
        manifest["videos"].append(
            {
                "scenario_id": scenario_id,
                "frames": len(frame_paths),
                "video": str(output_mp4.resolve()),
                "compose_wall_s": time.time() - t0,
            }
        )
        print(f"Wrote {output_mp4} ({len(frame_paths)} frames)")

    manifest["total_compose_wall_s"] = time.time() - started
    if args.manifest_json is not None:
        args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_json.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote manifest to {args.manifest_json}")


if __name__ == "__main__":
    main()
