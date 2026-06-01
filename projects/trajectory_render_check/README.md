# Trajectory Render Check Scripts

SimEngine-only log trajectory replay + MTGS rendering helpers.

Outputs stay under `experiments/trajectory_render_check/` (gitignored). Only these scripts and scene id lists are tracked in git.

## Prerequisites

```bash
export WORLDENGINE_ROOT=/path/to/WorldEngine
conda activate simengine
```

Required data:

- GS assets: `data/sim_engine/assets/navtest_failures/assets/`
- Scenarios: `data/sim_engine/scenarios/original/navtest_failures/all_scenarios.pkl`

## 1. Build a scenario subset

```bash
python projects/trajectory_render_check/extract_scenario_subset.py \
  --source-pkl "${WORLDENGINE_ROOT}/data/sim_engine/scenarios/original/navtest_failures/all_scenarios.pkl" \
  --scene-ids-file projects/trajectory_render_check/scene_ids/few_5.txt \
  --output-pkl "${WORLDENGINE_ROOT}/experiments/trajectory_render_check/scenario_subsets/few_5.pkl"
```

## 2. Build lateral 10 Hz variants

Each base scene becomes three scenarios: `*-001` (left), `*-000` (original), `*-002` (right).

Fixed lateral offset (local ground plane, `z` unchanged):

```bash
python projects/trajectory_render_check/create_local_ground_variants.py \
  --source-pkl "${WORLDENGINE_ROOT}/experiments/trajectory_render_check/scenario_subsets/few_5.pkl" \
  --output-pkl "${WORLDENGINE_ROOT}/experiments/trajectory_render_check/scenario_subsets/few_5_local_ground_10hz_variants.pkl" \
  --target-offset-m 3.0
```

HD map constrained (uses `map_features` polygons; binary search for max safe offset):

```bash
python projects/trajectory_render_check/create_map_constrained_variants.py \
  --source-pkl "${WORLDENGINE_ROOT}/experiments/trajectory_render_check/scenario_subsets/few_5.pkl" \
  --output-pkl "${WORLDENGINE_ROOT}/experiments/trajectory_render_check/scenario_subsets/few_5_map_constrained_10hz_variants.pkl" \
  --report-json "${WORLDENGINE_ROOT}/experiments/trajectory_render_check/scenario_subsets/few_5_map_constrained_10hz_report.json" \
  --target-offset-m 3.0
```

Pick scenes that have GS assets (example: first 10 with assets):

```bash
python projects/trajectory_render_check/create_map_constrained_variants.py \
  --source-pkl "${WORLDENGINE_ROOT}/data/sim_engine/scenarios/original/navtest_failures/all_scenarios.pkl" \
  --asset-root "${WORLDENGINE_ROOT}/data/sim_engine/assets/navtest_failures/assets" \
  --max-scenes 10 \
  --output-pkl "${WORLDENGINE_ROOT}/experiments/trajectory_render_check/scenario_subsets/ten_map_constrained_10hz_variants.pkl" \
  --report-json "${WORLDENGINE_ROOT}/experiments/trajectory_render_check/scenario_subsets/ten_map_constrained_10hz_report.json"
```

By default, `create_map_constrained_variants.py` skips base scenes whose original ego footprint is already outside the drivable polygons. This keeps `--max-scenes 10` as "10 valid base scenes" instead of "first 10 candidates".

Shared implementation lives in `variant_utils.py`.

## 3. Run MTGS rendering

Default profile (12 frames @ 0.5s):

```bash
bash projects/trajectory_render_check/run_trajectory_render.sh \
  "${WORLDENGINE_ROOT}/experiments/trajectory_render_check/scenario_subsets/few_5.pkl" \
  "${WORLDENGINE_ROOT}/experiments/trajectory_render_check/few_5/WE_output" \
  trajectory_render_few_5 \
  default
```

10 Hz profile (`num_history=1`, `num_future=81`; use a 10 Hz variant pkl from step 2):

```bash
bash projects/trajectory_render_check/run_trajectory_render.sh \
  "${WORLDENGINE_ROOT}/experiments/trajectory_render_check/scenario_subsets/few_5_map_constrained_10hz_variants.pkl" \
  "${WORLDENGINE_ROOT}/experiments/trajectory_render_check/map_constrained_10hz_variants/WE_output" \
  map_constrained_10hz_variants \
  10hz
```

Hydra overrides used by the shell script match saved runs under `experiments/**/.hydra/overrides.yaml`.

## 4. Compose videos

Single-stream CAM_F0 export (uses `meta_datas/*.pkl` frame order):

```bash
python projects/trajectory_render_check/compose_cam_f0_videos.py \
  --we-output "${WORLDENGINE_ROOT}/experiments/trajectory_render_check/map_constrained_10hz_variants/WE_output" \
  --output-dir "${WORLDENGINE_ROOT}/experiments/trajectory_render_check/few_5/deliverables/videos" \
  --fps 10 \
  --manifest-json "${WORLDENGINE_ROOT}/experiments/trajectory_render_check/few_5/deliverables/manifest_compose.json"
```

Recommended comparison layout: three rows. Each row is one variant's trajectory panel plus the matching CAM_F0 video.

```bash
python projects/trajectory_render_check/compose_map_constrained_videos.py \
  --variant-pkl "${WORLDENGINE_ROOT}/experiments/trajectory_render_check/scenario_subsets/ten_map_constrained_10hz_variants.pkl" \
  --render-root "${WORLDENGINE_ROOT}/experiments/trajectory_render_check/ten_map_constrained_10hz_variants/WE_output/openscene_format" \
  --output-dir "${WORLDENGINE_ROOT}/experiments/trajectory_render_check/few_5/deliverables/front_trajectory_10fps/trajectory_augmented_map_constrained_10_rows" \
  --layout rows
```

Legacy/debug layout: left | original | right videos in one horizontal strip, without trajectory panels.

```bash
python projects/trajectory_render_check/compose_map_constrained_videos.py \
  --variant-pkl "${WORLDENGINE_ROOT}/experiments/trajectory_render_check/scenario_subsets/few_5_map_constrained_10hz_variants.pkl" \
  --render-root "${WORLDENGINE_ROOT}/experiments/trajectory_render_check/map_constrained_10hz_variants/WE_output/openscene_format" \
  --output-dir "${WORLDENGINE_ROOT}/experiments/trajectory_render_check/few_5/deliverables/front_trajectory_10fps/trajectory_augmented_map_constrained_columns" \
  --layout columns
```

## Notes

- `run_trajectory_render.sh` replays log trajectories (`log_play_controller` + `trajectory_policy`); it does not run AlgEngine planners.
- Map constraint reads drivable polygons from each scenario's `map_features`, not from nuPlan map API at runtime.
- Default variant timing: 82 frames @ 10 Hz (`--num-frames 82 --render-frames 82 --dt 0.1`). The extra generated frame is intentional: SimEngine saves after stepping, so 82 states/metadata entries avoid the final-frame off-by-one failure when rendering with `num_history=1` + `num_future=81`.
- Video composition reads `meta_datas/*.pkl` first to preserve frame order; filename sorting is only a fallback for ad-hoc outputs without metadata.
