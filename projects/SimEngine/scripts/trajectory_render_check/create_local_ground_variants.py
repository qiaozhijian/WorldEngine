#!/usr/bin/env python3
"""Build left/original/right 10 Hz scenario variants with fixed lateral offsets."""

from __future__ import annotations

import argparse
import copy
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from variant_utils import (
    VariantConfig,
    apply_variant_state,
    load_scene_ids,
    local_lateral_variant,
    make_ego2globals,
    resample_dynamic_map_states,
    resample_infos,
    resample_track,
    select_scenes,
    variant_offsets,
)


def build_variants(scenes: dict, cfg: VariantConfig) -> dict:
    variant_scenes: dict = {}
    for scene_id, scene in scenes.items():
        tracks = {track_id: resample_track(track, cfg) for track_id, track in scene["object_track"].items()}
        ego_state = tracks[scene["sdc_id"]]["state"]
        base_pos = np.asarray(ego_state["position"][: cfg.num_frames], dtype=np.float64)
        base_heading = np.asarray(ego_state["heading"][: cfg.num_frames], dtype=np.float64)

        for name, suffix, offset in variant_offsets(cfg.target_offset_m):
            pos, heading, vel = local_lateral_variant(base_pos, base_heading, offset, cfg)
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
            variant_scene["metadata"]["target_offset_m"] = float(offset)
            variant_scene["metadata"]["actual_offset_m"] = float(offset)
            variant_scene["metadata"]["local_ground_constraint"] = (
                "p_new(t)=p_old(t)+smooth_offset(t)*local_left(t), z_new(t)=z_old(t)"
            )
            variant_scene["metadata"]["digitaltwin_ego2globals"] = make_ego2globals(scene, pos, heading, cfg)
            variant_scene["metadata"]["openscene_data_infos_dict"] = resample_infos(scene, suffix, cfg)
            variant_scenes[new_id] = variant_scene
    return variant_scenes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pkl", type=Path, required=True)
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument("--scene-ids-file", type=Path, default=None)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--exclude-scene-id", action="append", default=[])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--target-offset-m", type=float, default=3.0)
    parser.add_argument("--num-frames", type=int, default=82)
    parser.add_argument("--render-frames", type=int, default=82)
    parser.add_argument("--dt", type=float, default=0.1)
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

    variant_scenes = build_variants(scenes, cfg)
    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pkl.open("wb") as f:
        pickle.dump(variant_scenes, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote {len(variant_scenes)} variant scenarios to {args.output_pkl}")


if __name__ == "__main__":
    main()
