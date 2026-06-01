#!/usr/bin/env python3
"""Compose left/original/right comparison videos from map-constrained variant renders."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from variant_utils import VARIANT_SUFFIXES  # noqa: E402

VARIANT_ORDER = [(suffix, name) for name, suffix in VARIANT_SUFFIXES]
IDX_RE = re.compile(r"_(\d{3})-\d{3}\.jpg$")


def base_id(scene_id: str) -> str:
    parts = scene_id.rsplit("-", 1)
    return parts[0] if len(parts) == 2 and len(parts[1]) == 3 else scene_id


def short_token(scene_id: str) -> str:
    return scene_id.split("-")[-2] + "-" + scene_id.split("-")[-1]


def sorted_frames(render_root: Path, scene_id: str) -> list[Path]:
    meta_pkl = render_root / "meta_datas" / f"{scene_id}.pkl"
    sensor_root = render_root / "sensor_blobs"
    if meta_pkl.exists():
        with meta_pkl.open("rb") as f:
            frames = pickle.load(f)
        if not isinstance(frames, list):
            raise TypeError(f"Expected list in {meta_pkl}, got {type(frames)}")

        ordered_paths: list[Path] = []
        for frame in frames:
            data_path = frame["cams"]["CAM_F0"]["data_path"]
            image_path = sensor_root / data_path
            if not image_path.exists():
                raise FileNotFoundError(f"Missing rendered frame referenced by {meta_pkl}: {image_path}")
            ordered_paths.append(image_path)
        return ordered_paths

    # Fallback for ad-hoc render folders that do not have meta_datas.
    folder = render_root / "sensor_blobs" / short_token(scene_id) / "CAM_F0"
    frames: list[tuple[int, Path]] = []
    for path in folder.glob("*.jpg"):
        match = IDX_RE.search(path.name)
        if match:
            frames.append((int(match.group(1)), path))
    frames.sort(key=lambda item: item[0])
    return [path for _, path in frames]


def fit_image(img: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def draw_text(img: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.72) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def write_csv(scene: dict, path: Path, fps: int) -> None:
    sdc_id = scene["sdc_id"]
    state = scene["object_track"][sdc_id]["state"]
    pos = np.asarray(state["position"])
    heading = np.asarray(state["heading"])
    vel = np.asarray(state["velocity"])
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "t_sec", "x", "y", "z", "heading_rad", "vx", "vy", "actual_offset_m"])
        for i in range(len(pos)):
            vx, vy = (vel[i, 0], vel[i, 1]) if vel.ndim == 2 and vel.shape[1] >= 2 else (float(vel[i]), 0.0)
            writer.writerow(
                [
                    i,
                    i / fps,
                    pos[i, 0],
                    pos[i, 1],
                    pos[i, 2] if pos.shape[1] > 2 else 0.0,
                    heading[i],
                    vx,
                    vy,
                    scene["metadata"].get("actual_offset_m", 0.0),
                ]
            )


def trajectory_panel(
    scene: dict,
    frame_idx: int,
    global_bounds: tuple[float, float, float, float],
    traj_w: int,
    row_h: int,
    fps: int,
) -> np.ndarray:
    sdc_id = scene["sdc_id"]
    state = scene["object_track"][sdc_id]["state"]
    pos = np.asarray(state["position"], dtype=np.float64)
    heading = np.asarray(state["heading"], dtype=np.float64)
    offset = float(scene["metadata"].get("actual_offset_m", 0.0))
    variant = scene["metadata"].get("variant", "variant")

    panel = np.full((row_h, traj_w, 3), 245, dtype=np.uint8)
    x_min, x_max, y_min, y_max = global_bounds
    margin = 42
    sx = (traj_w - 2 * margin) / max(x_max - x_min, 1e-6)
    sy = (row_h - 2 * margin) / max(y_max - y_min, 1e-6)
    scale = min(sx, sy)

    def to_px(points: np.ndarray) -> np.ndarray:
        x = margin + (points[:, 0] - x_min) * scale
        y = row_h - margin - (points[:, 1] - y_min) * scale
        return np.stack([x, y], axis=1).round().astype(np.int32)

    pts = to_px(pos[:, :2])
    if len(pts) > 1:
        cv2.polylines(panel, [pts.reshape(-1, 1, 2)], False, (180, 180, 180), 2, cv2.LINE_AA)
        upto = pts[: min(frame_idx + 1, len(pts))]
        cv2.polylines(panel, [upto.reshape(-1, 1, 2)], False, (40, 120, 255), 3, cv2.LINE_AA)

    i = min(frame_idx, len(pos) - 1)
    cur = pts[i]
    cv2.circle(panel, tuple(cur), 6, (20, 20, 230), -1, cv2.LINE_AA)
    end = (
        int(cur[0] + 24 * np.cos(heading[i])),
        int(cur[1] - 24 * np.sin(heading[i])),
    )
    cv2.arrowedLine(panel, tuple(cur), end, (20, 20, 230), 2, cv2.LINE_AA, tipLength=0.35)

    label = f"{variant}  offset {offset:+.2f}m" if variant != "original" else "original"
    draw_text(panel, label, (18, 32), 0.7)
    draw_text(panel, f"t={frame_idx / fps:.1f}s", (18, row_h - 18), 0.62)
    cv2.rectangle(panel, (0, 0), (traj_w - 1, row_h - 1), (60, 60, 60), 1)
    return panel


def compose_columns(
    scenes: dict,
    render_root: Path,
    output_dir: Path,
    fps: int,
    panel_w: int,
    panel_h: int,
) -> list[dict]:
    by_base: dict[str, dict[str, str]] = {}
    for scene_id in scenes:
        by_base.setdefault(base_id(scene_id), {})[scene_id.rsplit("-", 1)[-1]] = scene_id

    manifest: list[dict] = []
    compose_started = time.perf_counter()
    for base, variants in sorted(by_base.items()):
        if not all(suffix in variants for suffix, _ in VARIANT_ORDER):
            continue

        frame_lists = [sorted_frames(render_root, variants[suffix]) for suffix, _ in VARIANT_ORDER]
        n = min(len(frames) for frames in frame_lists)
        if n == 0:
            continue

        item_started = time.perf_counter()
        tmp = output_dir / f"{base}_map_constrained_left_original_right_tmp.mp4"
        final = output_dir / f"{base}_map_constrained_left_original_right_{panel_w * 3}x{panel_h}_{fps}fps.mp4"
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (panel_w * 3, panel_h))

        for i in range(n):
            panels = []
            for frames, (suffix, label) in zip(frame_lists, VARIANT_ORDER):
                img = cv2.imread(str(frames[i]), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"failed to read {frames[i]}")
                panel = fit_image(img, panel_w, panel_h)
                offset = scenes[variants[suffix]]["metadata"].get("actual_offset_m", 0.0)
                text = f"{label} ({offset:+.2f}m)" if label != "original" else "original"
                draw_text(panel, text, (24, 48), 1.05)
                panels.append(panel)
            writer.write(np.concatenate(panels, axis=1))
        writer.release()

        transcode_started = time.perf_counter()
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(tmp),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(final),
            ],
            check=True,
        )
        transcode_s = time.perf_counter() - transcode_started
        tmp.unlink(missing_ok=True)

        traj_dir = output_dir / f"{base}_trajectories"
        traj_dir.mkdir(exist_ok=True)
        variant_manifest = _write_trajectory_exports(scenes, variants, traj_dir, fps, layout="columns")

        manifest.append(
            {
                "base_scene": base,
                "frames": n,
                "fps": fps,
                "layout": "left | original | right",
                "video": str(final),
                "trajectories": str(traj_dir),
                "variants": variant_manifest,
                "compose_wall_s": time.perf_counter() - item_started,
                "transcode_wall_s": transcode_s,
            }
        )

    _write_manifest(output_dir, manifest, compose_started, "manifest_map_constrained.json", "compose_timing.json")
    return manifest


def compose_rows(
    scenes: dict,
    render_root: Path,
    output_dir: Path,
    fps: int,
    traj_w: int,
    row_h: int,
    video_w: int,
    video_h: int,
) -> list[dict]:
    by_base: dict[str, dict[str, str]] = {}
    for scene_id in scenes:
        by_base.setdefault(base_id(scene_id), {})[scene_id.rsplit("-", 1)[-1]] = scene_id

    out_w = traj_w + video_w
    out_h = row_h * 3
    manifest: list[dict] = []
    compose_started = time.perf_counter()

    for base, variants in sorted(by_base.items()):
        if not all(suffix in variants for suffix, _ in VARIANT_ORDER):
            continue

        frame_lists = [sorted_frames(render_root, variants[suffix]) for suffix, _ in VARIANT_ORDER]
        n = min(len(frames) for frames in frame_lists)
        if n == 0:
            continue

        all_xy = np.concatenate(
            [
                np.asarray(scenes[variants[suffix]]["object_track"][scenes[variants[suffix]]["sdc_id"]]["state"]["position"])[:, :2]
                for suffix, _ in VARIANT_ORDER
            ],
            axis=0,
        )
        pad = 5.0
        global_bounds = (
            float(all_xy[:, 0].min() - pad),
            float(all_xy[:, 0].max() + pad),
            float(all_xy[:, 1].min() - pad),
            float(all_xy[:, 1].max() + pad),
        )

        item_started = time.perf_counter()
        tmp = output_dir / f"{base}_map_constrained_rows_tmp.mp4"
        final = output_dir / f"{base}_map_constrained_rows_{out_w}x{out_h}_{fps}fps.mp4"
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

        for i in range(n):
            rows = []
            for frames, (suffix, _) in zip(frame_lists, VARIANT_ORDER):
                scene_id = variants[suffix]
                img = cv2.imread(str(frames[i]), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"failed to read {frames[i]}")
                video = fit_image(img, video_w, video_h)
                traj = trajectory_panel(scenes[scene_id], i, global_bounds, traj_w, row_h, fps)
                rows.append(np.concatenate([traj, video], axis=1))
            writer.write(np.concatenate(rows, axis=0))
        writer.release()

        transcode_started = time.perf_counter()
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(tmp),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(final),
            ],
            check=True,
        )
        transcode_s = time.perf_counter() - transcode_started
        tmp.unlink(missing_ok=True)

        traj_dir = output_dir / f"{base}_trajectories"
        traj_dir.mkdir(exist_ok=True)
        variant_manifest = _write_trajectory_exports(
            scenes,
            variants,
            traj_dir,
            fps,
            layout="three rows; each row is trajectory panel + CAM_F0 video",
        )

        manifest.append(
            {
                "base_scene": base,
                "frames": n,
                "fps": fps,
                "layout": "three rows; each row: trajectory left, CAM_F0 video right",
                "video": str(final),
                "trajectories": str(traj_dir),
                "variants": variant_manifest,
                "compose_wall_s": time.perf_counter() - item_started,
                "transcode_wall_s": transcode_s,
            }
        )

    _write_manifest(
        output_dir,
        manifest,
        compose_started,
        "manifest_map_constrained_rows.json",
        "compose_timing_rows.json",
    )
    return manifest


def _write_trajectory_exports(
    scenes: dict,
    variants: dict[str, str],
    traj_dir: Path,
    fps: int,
    layout: str,
) -> dict:
    variant_manifest: dict = {}
    for suffix, label in VARIANT_ORDER:
        scene_id = variants[suffix]
        scene = scenes[scene_id]
        write_csv(scene, traj_dir / f"{label}_ego_trajectory_10hz.csv", fps)
        sdc_id = scene["sdc_id"]
        np.save(
            traj_dir / f"{label}_ego_position_heading_10hz.npy",
            {
                "position": np.asarray(scene["object_track"][sdc_id]["state"]["position"]),
                "heading": np.asarray(scene["object_track"][sdc_id]["state"]["heading"]),
                "velocity": np.asarray(scene["object_track"][sdc_id]["state"]["velocity"]),
                "fps": fps,
                "actual_offset_m": scene["metadata"].get("actual_offset_m", 0.0),
                "target_offset_m": scene["metadata"].get("target_offset_m", 0.0),
                "map_constraint": scene["metadata"].get("map_constraint", {}),
                "layout": layout,
            },
            allow_pickle=True,
        )
        variant_manifest[label] = {
            "scenario_id": scene_id,
            "actual_offset_m": scene["metadata"].get("actual_offset_m", 0.0),
            "target_offset_m": scene["metadata"].get("target_offset_m", 0.0),
            "map_constraint": scene["metadata"].get("map_constraint", {}),
        }
    return variant_manifest


def _write_manifest(
    output_dir: Path,
    manifest: list[dict],
    compose_started: float,
    manifest_name: str,
    timing_name: str,
) -> None:
    with (output_dir / manifest_name).open("w") as f:
        json.dump(manifest, f, indent=2)
    with (output_dir / timing_name).open("w") as f:
        json.dump(
            {
                "total_compose_wall_s": time.perf_counter() - compose_started,
                "videos": [
                    {
                        "base_scene": item["base_scene"],
                        "frames": item["frames"],
                        "compose_wall_s": item["compose_wall_s"],
                        "transcode_wall_s": item["transcode_wall_s"],
                    }
                    for item in manifest
                ],
            },
            f,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-pkl", type=Path, required=True)
    parser.add_argument(
        "--render-root",
        type=Path,
        required=True,
        help="Path to WE_output/openscene_format",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layout", choices=("columns", "rows"), default="columns")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--panel-width", type=int, default=1024)
    parser.add_argument("--panel-height", type=int, default=576)
    parser.add_argument("--traj-width", type=int, default=640)
    parser.add_argument("--row-height", type=int, default=360)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=360)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.variant_pkl.open("rb") as f:
        scenes = pickle.load(f)

    if args.layout == "columns":
        manifest = compose_columns(
            scenes,
            args.render_root,
            args.output_dir,
            args.fps,
            args.panel_width,
            args.panel_height,
        )
    else:
        manifest = compose_rows(
            scenes,
            args.render_root,
            args.output_dir,
            args.fps,
            args.traj_width,
            args.row_height,
            args.video_width,
            args.video_height,
        )

    print(f"Wrote {len(manifest)} videos to {args.output_dir}")
    for item in manifest:
        print(item["frames"], item["video"])


if __name__ == "__main__":
    main()
