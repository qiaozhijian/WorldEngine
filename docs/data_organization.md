# Data Organization Guide

This guide explains how to organize data for WorldEngine. The platform uses a modular data structure where each subsystem (AlgEngine, SimEngine) has its own data requirements while sharing common formats.

## Quick Overview

```
WorldEngine/
├── data/                          # Main data directory
│   ├── raw/                       # Raw datasets (nuPlan, OpenScene)
│   ├── alg_engine/                # AlgEngine-specific data
│   └── sim_engine/                # SimEngine-specific data
├── projects/
│   ├── AlgEngine/                 # Algorithm training & evaluation
│   └── SimEngine/                 # Closed-loop photorealistic simulation
├── experiments/                   # Experiment outputs
└── data_output/                   # Default data outputs dir
```

---

## Data Download

We provide pre-processed datasets and model checkpoints via **[ModelScope](https://www.modelscope.cn/datasets/OpenDriveLab/WorldEngine)** and **[Hugging Face](https://huggingface.co/datasets/OpenDriveLab/WorldEngine)**.

### Option 1: Download from ModelScope (Recommended for Users in China)

```bash
pip install modelscope
modelscope download --dataset OpenDriveLab/WorldEngine
```

### Option 2: Download from Hugging Face

```bash
# Install Hugging Face CLI
curl -LsSf https://hf.co/cli/install.sh | bash

# Download the dataset
hf download OpenDriveLab/WorldEngine --repo-type dataset --local-dir /path/to/your/WorldEngine_repo
```

> **Note:** Stay tuned to our [News section](../README.md#-news).

---

## Directory Structure

### 1. Raw Data (`data/raw/`)

Raw datasets from nuPlan and OpenScene.

```bash
data/raw/
├── nuplan/                        # nuPlan raw dataset
│   └── dataset/
│      ├── maps/                  # HD maps (required for all modules)
│      │   ├── nuplan-maps-v1.0.json
│      │   ├── us-nv-las-vegas-strip/
│      │   ├── us-ma-boston/
│      │   ├── us-pa-pittsburgh-hazelwood/
│      │   └── sg-one-north/
│      └── nuplan-v1.1/
│          ├── sensor_blobs/      # Camera images and LiDAR
│          └── splits/            # Train/val/test splits
│   
│
└── openscene-v1.1/                # OpenScene dataset (nuPlan-based)
    ├── sensor_blobs/
    │   ├── trainval/              # Training sensor data
    │   └── test/                  # Test sensor data
    └── meta_datas/
        ├── trainval/              # Training metadata
        └── test/                  # Test metadata
```

**Setup commands:**

```bash
cd WorldEngine/data/raw

# Create symlink to nuPlan dataset
ln -s /path/to/nuplan nuplan

# Create symlink to OpenScene dataset
ln -s /path/to/openscene-v1.1 openscene-v1.1
```

---

### 2. AlgEngine Data (`data/alg_engine/`)

Data for end-to-end model training and evaluation.

```bash
data/alg_engine/
├── openscene-synthetic/           # Synthetic data from SimEngine
│   ├── sensor_blobs/
│   ├── meta_datas/
│   └── pdms_pkl/
│
├── ckpts/                         # Pre-trained model checkpoints
│
├── pdms_cache/                    # Pre-computed PDM metrics cache
│   ├── pdm_8192_gt_cache_navtrain.pkl
│   └── pdm_8192_gt_cache_navtest.pkl
│
├── merged_infos_navformer/
│   ├── nuplan_openscene_navtrain.pkl
│   └── nuplan_openscene_navtest.pkl
│
│
└── test_8192_kmeans.npy          # K-means clustering for PDM
```

---

### 3. SimEngine Data (`data/sim_engine/`)

Data for closed-loop simulation. Sizes below are from the [OpenDriveLab/WorldEngine](https://huggingface.co/datasets/OpenDriveLab/WorldEngine) Hugging Face release (dataset total **~4.89 TB**; `sim_engine/` **~4.81 TB**).

**On Hugging Face (as downloaded)** — scene assets are split tarballs; scenarios are single `.pkl` files:

```bash
data/sim_engine/                                          # ~4.81 TB
├── assets/                                               # ~4.76 TB
│   ├── navtrain/                                         # ~4.18 TB, 129× part*.tar.gz + configs.tar.gz
│   ├── navtest/                                          # ~489 GB,  15× part*.tar.gz + configs.tar.gz
│   └── navtest_failures/                                 # ~88 GB,   3× part*.tar.gz + configs.tar.gz
│       ├── assets/
│       │   ├── part001.tar.gz
│       │   ├── part002.tar.gz
│       │   └── ...
│       └── configs.tar.gz
│
└── scenarios/                                            # ~49 GB
    ├── original/                                         # ~33 GB
    │   ├── navtest_failures/all_scenarios.pkl            # ~912 MB
    │   ├── navtrain_50pct_collision/all_scenarios.pkl    # ~19 GB
    │   ├── navtrain_ep_per1/all_scenarios.pkl            # ~2.5 GB
    │   ├── navtrain_failures_per1/all_scenarios.pkl      # ~2.6 GB
    │   └── navtrain_hydramdp_failures/all_scenarios.pkl  # ~7.5 GB
    │
    └── augmented/                                        # ~16 GB, from BWM
        ├── navtrain_50pct_collision/all_scenarios.pkl    # ~3.1 GB
        ├── navtrain_50pct_ep_1pct/all_scenarios.pkl      # ~5.0 GB
        └── navtrain_50pct_offroad/all_scenarios.pkl      # ~7.9 GB
```

Extract `configs.tar.gz` and every `part*.tar.gz` under each split (see HF Usage step 2). Archives unpack with a `{split}/assets/{road_block_name}/...` prefix; after extraction, SimEngine expects the layout below.

**`assets/{navtrain,navtest,navtest_failures}/` — after extraction:**

```bash
navtrain/                                                 # navtest / navtest_failures: same tree
├── configs/                                              # from configs.tar.gz
│   ├── {road_block_name}.yaml
│   └── ...
└── assets/
    ├── {road_block_name}/
    │   ├── background/
    │   │   └── {road_block_name}.ckpt              # dict: background, skybox, rigid_object_*
    │   └── road_height_map/
    │       ├── road_height_map.npy
    │       ├── sim2.json
    │       └── road_height_map_preview.png
    ├── {road_block_name_2}/
    │   └── ...
    └── ...
```

Verified against HF file listing (174 files), `part001.tar.gz` contents, and extracted `navtest_failures` assets (290 blocks, identical 4-file layout per block). **`video_scene_dict.pkl` is not shipped on Hugging Face**; it is only read by the internal DigitalTwin scenario conversion script (`digitaltwin_nuplan_converter_navsim_filter.py`). SimEngine runtime loads `{road_block_name}.ckpt` via `MTGSAssetManager`; `road_height_map/` is present in release assets but currently unused in code.

---

### 4. Experiments Output (`experiments/`)

Generated experiment results and logs.

```bash
experiments/
└── closed_loop_exps/              # Closed-loop simulation results
    ├── exp_vadv2_50pct_ep20/
    │   ├── navtest_failures_NR/   # Non-reactive results
    │   │   ├── split_0/ ... split_7/
    │   │   ├── plan_traj/
    │   │   └── WE_output/
    │   │       └── openscene_format/
    │   │           ├── meta_datas/
    │   │           ├── pdms_pkl/
    │   │           ├── sensor_blobs/
    │   │           └── all_scenes_pdm_averages_NR.csv
    │   │
    │   └── navtest_failures_R/    # Reactive results
    │       └── (same structure as NR)
    │
    └── exp_vadv2_100pct_ep20/
        └── (similar structure)
```

This directory is automatically created during simulation runs.

---

## Environment Variables

Set these environment variables for proper data access:

```bash
# Add to ~/.bashrc or ~/.zshrc
export WORLDENGINE_ROOT="/path/to/WorldEngine"
export NUPLAN_MAPS_ROOT="${WORLDENGINE_ROOT}/data/raw/nuplan/maps"
export PYTHONPATH=$WORLDENGINE_ROOT:$PYTHONPATH
```

Apply changes:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

---