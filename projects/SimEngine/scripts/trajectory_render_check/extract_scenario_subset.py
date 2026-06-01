#!/usr/bin/env python3
"""Extract a scenario subset pickle from all_scenarios.pkl by scene id list."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def load_scene_ids(path: Path) -> list[str]:
    ids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    if not ids:
        raise ValueError(f"No scene ids found in {path}")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-pkl",
        type=Path,
        required=True,
        help="Path to source all_scenarios.pkl",
    )
    parser.add_argument(
        "--scene-ids-file",
        type=Path,
        required=True,
        help="Text file with one scenario id per line",
    )
    parser.add_argument(
        "--output-pkl",
        type=Path,
        required=True,
        help="Output subset pickle path",
    )
    args = parser.parse_args()

    scene_ids = load_scene_ids(args.scene_ids_file)
    with args.source_pkl.open("rb") as f:
        all_scenarios = pickle.load(f)

    missing = [scene_id for scene_id in scene_ids if scene_id not in all_scenarios]
    if missing:
        raise KeyError(f"Scene ids not found in {args.source_pkl}: {missing}")

    subset = {scene_id: all_scenarios[scene_id] for scene_id in scene_ids}
    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pkl.open("wb") as f:
        pickle.dump(subset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote {len(subset)} scenarios to {args.output_pkl}")


if __name__ == "__main__":
    main()
