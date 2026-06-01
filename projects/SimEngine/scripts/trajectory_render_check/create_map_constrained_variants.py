#!/usr/bin/env python3
"""Build left/original/right 10 Hz variants constrained by scenario HD map polygons."""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from variant_utils import (
    VariantConfig,
    apply_variant_state,
    ego_dimensions,
    load_scene_ids,
    local_lateral_variant,
    make_ego2globals,
    resample_dynamic_map_states,
    resample_infos,
    resample_track,
    select_scenes,
    variant_offsets,
)

DRIVABLE_TYPES = {"LANE_SURFACE_STREET", "LANE_SURFACE_UNSTRUCTURE"}


def build_drivable_area(scene: dict) -> Polygon | None:
    polygons = []
    for feature in scene.get("map_features", {}).values():
        if not isinstance(feature, dict) or feature.get("type") not in DRIVABLE_TYPES:
            continue
        points = np.asarray(feature.get("polygon", []), dtype=np.float64)
        if points.shape[0] < 3:
            continue
        poly = Polygon(points).buffer(0)
        if not poly.is_empty and poly.area > 1e-3:
            polygons.append(poly)
    if not polygons:
        return None
    return unary_union(polygons)


def ego_footprint(position: np.ndarray, heading: float, length: float, width: float) -> Polygon:
    xy = np.asarray(position[:2], dtype=np.float64)
    forward = np.array([math.cos(float(heading)), math.sin(float(heading))])
    left = np.array([-forward[1], forward[0]])
    half_l = float(length) / 2.0
    half_w = float(width) / 2.0
    corners = [
        xy + forward * half_l + left * half_w,
        xy + forward * half_l - left * half_w,
        xy - forward * half_l - left * half_w,
        xy - forward * half_l + left * half_w,
    ]
    return Polygon(corners)


def drivable_stats(
    area: Polygon,
    pos: np.ndarray,
    heading: np.ndarray,
    length: float,
    width: float,
    render_frames: int,
) -> dict:
    bad_center: list[int] = []
    bad_footprint: list[int] = []
    for i in range(render_frames):
        if not area.covers(Point(pos[i, :2])):
            bad_center.append(i)
        if not area.covers(ego_footprint(pos[i], heading[i], length, width)):
            bad_footprint.append(i)
    return {
        "center_ok": render_frames - len(bad_center),
        "footprint_ok": render_frames - len(bad_footprint),
        "bad_center_frames": bad_center[:20],
        "bad_footprint_frames": bad_footprint[:20],
        "is_valid": not bad_center and not bad_footprint,
    }


def find_safe_offset(
    area: Polygon,
    base_pos: np.ndarray,
    base_heading: np.ndarray,
    length: float,
    width: float,
    target_offset: float,
    cfg: VariantConfig,
    search_iters: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, dict]:
    if abs(target_offset) < 1e-9:
        pos, heading, vel = local_lateral_variant(base_pos, base_heading, 0.0, cfg)
        stats = drivable_stats(area, pos, heading, length, width, cfg.render_frames)
        return 0.0, pos, heading, vel, stats

    sign = 1.0 if target_offset > 0.0 else -1.0
    lo, hi = 0.0, abs(float(target_offset))
    best = None
    for _ in range(search_iters):
        mid = (lo + hi) / 2.0
        pos, heading, vel = local_lateral_variant(base_pos, base_heading, sign * mid, cfg)
        stats = drivable_stats(area, pos, heading, length, width, cfg.render_frames)
        if stats["is_valid"]:
            lo = mid
            best = (sign * mid, pos, heading, vel, stats)
        else:
            hi = mid

    if best is None:
        pos, heading, vel = local_lateral_variant(base_pos, base_heading, 0.0, cfg)
        stats = drivable_stats(area, pos, heading, length, width, cfg.render_frames)
        return 0.0, pos, heading, vel, stats

    final_offset, pos, heading, vel, _ = best
    stats = drivable_stats(area, pos, heading, length, width, cfg.render_frames)
    return final_offset, pos, heading, vel, stats


def build_variants(scenes: dict, cfg: VariantConfig, search_iters: int) -> tuple[dict, dict]:
    variant_scenes: dict = {}
    report: dict = {}

    for scene_id, scene in scenes.items():
        area = build_drivable_area(scene)
        if area is None:
            raise RuntimeError(f"No drivable map polygons found for {scene_id}")

        tracks = {track_id: resample_track(track, cfg) for track_id, track in scene["object_track"].items()}
        ego_state = tracks[scene["sdc_id"]]["state"]
        base_pos = np.asarray(ego_state["position"][: cfg.num_frames], dtype=np.float64)
        base_heading = np.asarray(ego_state["heading"][: cfg.num_frames], dtype=np.float64)
        length, width = ego_dimensions(scene)

        report[scene_id] = {
            "drivable_area_m2": float(area.area),
            "target_offset_m": cfg.target_offset_m,
            "variants": {},
        }

        for name, suffix, target_offset in variant_offsets(cfg.target_offset_m):
            actual_offset, pos, heading, vel, stats = find_safe_offset(
                area,
                base_pos,
                base_heading,
                length,
                width,
                target_offset,
                cfg,
                search_iters,
            )

            variant_scene = copy.deepcopy(scene)
            new_id = f"{scene_id}-{suffix}"
            variant_scene["id"] = new_id
            variant_scene["name"] = new_id
            variant_scene["token"] = new_id
            variant_scene["log_length"] = cfg.num_frames
            variant_scene["sample_rate"] = 2
            variant_scene["object_track"] = copy.deepcopy(tracks)
            apply_variant_state(variant_scene, scene, pos, heading, vel, cfg)
            variant_scene["dynamic_map_states"] = resample_dynamic_map_states(
                scene.get("dynamic_map_states", {}),
                cfg,
            )
            variant_scene["metadata"] = copy.deepcopy(scene["metadata"])
            variant_scene["metadata"]["digitaltwin_asset_id"] = scene_id
            variant_scene["metadata"]["variant"] = name
            variant_scene["metadata"]["target_offset_m"] = float(target_offset)
            variant_scene["metadata"]["actual_offset_m"] = float(actual_offset)
            variant_scene["metadata"]["map_constraint"] = {
                "drivable_types": sorted(DRIVABLE_TYPES),
                "check": "ego center and full ego footprint must be covered by drivable polygon union",
                **stats,
            }
            variant_scene["metadata"]["local_ground_constraint"] = (
                "p_new(t)=p_old(t)+smooth_offset(t)*local_left(t), z_new(t)=z_old(t)"
            )
            variant_scene["metadata"]["digitaltwin_ego2globals"] = make_ego2globals(scene, pos, heading, cfg)
            variant_scene["metadata"]["openscene_data_infos_dict"] = resample_infos(scene, suffix, cfg)
            variant_scenes[new_id] = variant_scene

            report[scene_id]["variants"][name] = {
                "scenario_id": new_id,
                "target_offset_m": float(target_offset),
                "actual_offset_m": float(actual_offset),
                **stats,
            }

    return variant_scenes, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pkl", type=Path, required=True)
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--scene-ids-file", type=Path, default=None)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--exclude-scene-id", action="append", default=[])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--target-offset-m", type=float, default=3.0)
    parser.add_argument("--num-frames", type=int, default=82)
    parser.add_argument("--render-frames", type=int, default=82)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--search-iters", type=int, default=10)
    args = parser.parse_args()

    cfg = VariantConfig(
        num_frames=args.num_frames,
        render_frames=args.render_frames,
        dt=args.dt,
        target_offset_m=args.target_offset_m,
    )

    with args.source_pkl.open("rb") as f:
        all_scenes = pickle.load(f)

    scene_ids = load_scene_ids(args.scene_ids_file) if args.scene_ids_file else None
    scenes = select_scenes(
        all_scenes,
        scene_ids=scene_ids,
        asset_root=args.asset_root,
        exclude_scene_ids=args.exclude_scene_id,
        max_scenes=args.max_scenes,
    )

    variant_scenes, report = build_variants(scenes, cfg, args.search_iters)
    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pkl.open("wb") as f:
        pickle.dump(variant_scenes, f, protocol=pickle.HIGHEST_PROTOCOL)
    with args.report_json.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {len(variant_scenes)} variant scenarios to {args.output_pkl}")
    print(f"Wrote constraint report to {args.report_json}")
    for scene_id, item in report.items():
        offsets = {name: round(variant["actual_offset_m"], 3) for name, variant in item["variants"].items()}
        valid = {name: variant["is_valid"] for name, variant in item["variants"].items()}
        print(scene_id, offsets, valid)


if __name__ == "__main__":
    main()
