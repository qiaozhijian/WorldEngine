#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH." >&2
  exit 1
fi

if [[ -z "${WORLDENGINE_ROOT:-}" ]]; then
  echo "ERROR: WORLDENGINE_ROOT is not set." >&2
  exit 1
fi

SIMENGINE_ENV_NAME="${SIMENGINE_ENV_NAME:-simengine}"
SIMENGINE_ROOT="${WORLDENGINE_ROOT}/projects/SimEngine"

DATA_PKL="${1:?Usage: $0 <data_pkl> <output_dir> <job_name> [profile]}"
OUTPUT_DIR="${2:?Usage: $0 <data_pkl> <output_dir> <job_name> [profile]}"
JOB_NAME="${3:?Usage: $0 <data_pkl> <output_dir> <job_name> [profile]}"
PROFILE="${4:-default}"

ASSET_NAME="${ASSET_NAME:-navtest_failures}"
ASSET_FOLDER_PATH="${ASSET_FOLDER_PATH:-${WORLDENGINE_ROOT}/data/sim_engine/assets/${ASSET_NAME}/assets}"

if [[ ! -f "${DATA_PKL}" ]]; then
  echo "ERROR: data pickle not found: ${DATA_PKL}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

EXTRA_ARGS=(
  use_planner_actions=false
  ego_controller=log_play_controller
  ego_policy=trajectory_policy
  ego_navigation=trajectory_navigation
  agent_controller=log_play_controller
  agent_policy=trajectory_policy
  agent_navigation=trajectory_navigation
  with_metric_manager=false
  with_dense_reward_manager=false
  visualize_video=false
  visualize_BEV=false
  NC_DAC_video_capture=false
  distributed_mode=SINGLE_NODE
)

case "${PROFILE}" in
  default)
    ;;
  10hz)
    EXTRA_ARGS+=(
      num_history=1
      num_future=81
    )
    ;;
  *)
    echo "ERROR: unknown profile '${PROFILE}'. Use 'default' or '10hz'." >&2
    exit 1
    ;;
esac

cd "${SIMENGINE_ROOT}"
export PYTHONPATH="${SIMENGINE_ROOT}:${PYTHONPATH:-}"

echo "Running trajectory render:"
echo "  data_pkl=${DATA_PKL}"
echo "  asset_folder_path=${ASSET_FOLDER_PATH}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  job_name=${JOB_NAME}"
echo "  profile=${PROFILE}"

conda run --no-capture-output -n "${SIMENGINE_ENV_NAME}" \
  python worldengine/runner/run_simulation.py \
    data_file_path="${DATA_PKL}" \
    asset_folder_path="${ASSET_FOLDER_PATH}" \
    output_dir="${OUTPUT_DIR}" \
    job_name="${JOB_NAME}" \
    "${EXTRA_ARGS[@]}"
