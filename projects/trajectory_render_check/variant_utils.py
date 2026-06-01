"""Shared helpers for 10 Hz lateral trajectory variant scenario generation."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

VARIANT_SUFFIXES = (
    ("left", "001"),
    ("original", "000"),
    ("right", "002"),
)


@dataclass(frozen=True)
class VariantConfig:
    num_frames: int = 82
    render_frames: int = 82
    dt: float = 0.1
    target_offset_m: float = 3.0
    source_hz: float = 2.0

    @property
    def source_dt(self) -> float:
        return 1.0 / self.source_hz


def load_scene_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    if not ids:
        raise ValueError(f"No scene ids found in {path}")
    return ids


def select_scenes(
    all_scenes: dict,
    *,
    scene_ids: Iterable[str] | None = None,
    asset_root: Path | None = None,
    exclude_scene_ids: Iterable[str] | None = None,
    max_scenes: int | None = None,
) -> dict:
    excluded = set(exclude_scene_ids or ())
    selected: dict = {}

    candidates = scene_ids if scene_ids is not None else all_scenes.keys()
    for scene_id in candidates:
        if scene_id in excluded:
            continue
        if scene_id not in all_scenes:
            raise KeyError(f"Scene id not found: {scene_id}")
        scene = all_scenes[scene_id]
        if asset_root is not None and not (asset_root / scene_id).exists():
            continue
        selected[scene_id] = scene
        if max_scenes is not None and len(selected) >= max_scenes:
            break

    if not selected:
        raise ValueError("No scenes selected")
    return selected


def variant_offsets(target_offset_m: float) -> list[tuple[str, str, float]]:
    return [
        ("left", "001", target_offset_m),
        ("original", "000", 0.0),
        ("right", "002", -target_offset_m),
    ]


def interp_array(arr: np.ndarray, cfg: VariantConfig) -> np.ndarray:
    arr = np.asarray(arr)
    src_count = min(arr.shape[0], 18)
    src = arr[:src_count]
    src_t = np.arange(src_count) * cfg.source_dt
    dst_t = np.clip(np.arange(cfg.num_frames) * cfg.dt, src_t[0], src_t[-1])
    if arr.ndim == 1:
        if np.issubdtype(src.dtype, np.number):
            return np.interp(dst_t, src_t, src.astype(np.float64))
        idx = np.clip(np.rint(dst_t / cfg.source_dt).astype(int), 0, src_count - 1)
        return src[idx]
    flat = src.reshape(src_count, -1)
    out = np.stack(
        [np.interp(dst_t, src_t, flat[:, i].astype(np.float64)) for i in range(flat.shape[1])],
        axis=1,
    )
    return out.reshape((cfg.num_frames,) + src.shape[1:])


def resample_track(track: dict, cfg: VariantConfig) -> dict:
    new_track = copy.deepcopy(track)
    state = new_track.get("state", {})
    for key, value in list(state.items()):
        if isinstance(value, np.ndarray):
            state[key] = interp_array(value, cfg)
    if isinstance(new_track.get("metadata"), dict):
        new_track["metadata"]["track_length"] = cfg.num_frames
    return new_track


def resample_dynamic_map_states(dynamic_map_states: dict, cfg: VariantConfig) -> dict:
    out = copy.deepcopy(dynamic_map_states)
    for item in out.values():
        state = item.get("state", {}) if isinstance(item, dict) else {}
        for key, value in list(state.items()):
            if isinstance(value, np.ndarray):
                state[key] = interp_array(value, cfg)
            elif isinstance(value, list) and value:
                src = value[: min(len(value), 18)]
                idx = np.clip(
                    np.rint((np.arange(cfg.num_frames) * cfg.dt) / cfg.source_dt).astype(int),
                    0,
                    len(src) - 1,
                )
                state[key] = [copy.deepcopy(src[i]) for i in idx]
    return out


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def local_lateral_variant(
    base_pos: np.ndarray,
    base_heading: np.ndarray,
    target_offset: float,
    cfg: VariantConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.asarray(base_pos, dtype=np.float64).copy()
    heading = np.asarray(base_heading, dtype=np.float64).copy()
    if abs(target_offset) >= 1e-9:
        left = np.stack([-np.sin(heading), np.cos(heading)], axis=1)
        offset = target_offset * smoothstep(np.linspace(0.0, 1.0, len(pos)))
        pos[:, :2] = pos[:, :2] + left * offset[:, None]
        pos[:, 2] = base_pos[:, 2]

    vel = np.gradient(pos[:, :2], cfg.dt, axis=0)
    new_heading = np.arctan2(vel[:, 1], vel[:, 0])
    return pos, new_heading, vel


def make_ego2globals(scene: dict, pos: np.ndarray, heading: np.ndarray, cfg: VariantConfig) -> list[np.ndarray]:
    old = np.asarray(scene["metadata"]["digitaltwin_ego2globals"])
    src = old[: min(len(old), 18)]
    mats: list[np.ndarray] = []
    for i in range(cfg.num_frames):
        j = int(np.clip(round((i * cfg.dt) / cfg.source_dt), 0, len(src) - 1))
        mat = np.array(src[j], dtype=np.float64).copy()
        c, s = math.cos(float(heading[i])), math.sin(float(heading[i]))
        mat[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        mat[:3, 3] = pos[i, :3]
        mats.append(mat)
    return mats


def resample_infos(scene: dict, suffix: str, cfg: VariantConfig) -> dict:
    infos = list(scene["metadata"]["openscene_data_infos_dict"].values())
    src = infos[: min(len(infos), 18)]
    out: dict = {}
    for i in range(cfg.render_frames):
        j = int(np.clip(round((i * cfg.dt) / cfg.source_dt), 0, len(src) - 1))
        info = copy.deepcopy(src[j])
        token = f"{info.get('token', 'token')}_{suffix}_{i:03d}"
        info["token"] = token
        info["frame_idx"] = i
        info["timestamp"] = int(scene.get("base_timestamp", 0) + i * cfg.dt * 1e6)
        info["sample_prev"] = (
            None
            if i == 0
            else f"{src[int(np.clip(round(((i - 1) * cfg.dt) / cfg.source_dt), 0, len(src) - 1))].get('token', 'token')}_{suffix}_{i - 1:03d}"
        )
        info["sample_next"] = (
            None
            if i == cfg.render_frames - 1
            else f"{src[int(np.clip(round(((i + 1) * cfg.dt) / cfg.source_dt), 0, len(src) - 1))].get('token', 'token')}_{suffix}_{i + 1:03d}"
        )
        for cam_name, cam_info in info.get("cams", {}).items():
            old = Path(cam_info.get("data_path", f"{cam_name}.jpg"))
            cam_info["data_path"] = str(old.with_name(f"{old.stem}_{suffix}_{i:03d}{old.suffix or '.jpg'}"))
        out[token] = info
    return out


def ego_dimensions(scene: dict) -> tuple[float, float]:
    ego_state = scene["object_track"][scene["sdc_id"]]["state"]
    length = float(np.asarray(ego_state["length"])[0].reshape(-1)[0])
    width = float(np.asarray(ego_state["width"])[0].reshape(-1)[0])
    return length, width


def apply_variant_state(
    variant_scene: dict,
    source_scene: dict,
    pos: np.ndarray,
    heading: np.ndarray,
    vel: np.ndarray,
    cfg: VariantConfig,
) -> None:
    state = variant_scene["object_track"][variant_scene["sdc_id"]]["state"]
    state["position"] = pos
    state["heading"] = heading
    state["velocity"] = vel
    state["valid"] = np.ones((cfg.num_frames,), dtype=np.float64)
    for dim in ("length", "width", "height"):
        if dim in state:
            first = np.asarray(source_scene["object_track"][source_scene["sdc_id"]]["state"][dim])[0]
            state[dim] = np.repeat(np.asarray(first).reshape(1, -1), cfg.num_frames, axis=0)
