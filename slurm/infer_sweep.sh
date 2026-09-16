#!/bin/bash
#SBATCH --job-name=topo_infer
#SBATCH --output=logs_topo/infer_%A_%a.out
#SBATCH --error=logs_topo/infer_%A_%a.err

#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=clara
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-9   # 10 cells; see slurm/_grid.sh for the grid

# Validation inference, one array task per sweep cell -- task N infers the model
# that training task N produced, because both resolve the cell through
# slurm/_grid.sh.
#
# Run once before the first submit:  mkdir -p logs_topo
#
#   sbatch --export=ALL,MARKER=ER slurm/infer_sweep.sh                  # all 38
#   sbatch --export=ALL,MARKER=ER --array=0-18 slurm/infer_sweep.sh     # lambda_cycle=10 only
#   sbatch --export=ALL,MARKER=ER,LIMIT=8 --array=13 slurm/infer_sweep.sh
#
# Submit from the repository root so SLURM_SUBMIT_DIR locates _grid.sh, or
# export REPO=/path/to/TopoCycleGAN.

set -eo pipefail

: "${MARKER:?set MARKER, e.g. --export=ALL,MARKER=ER}"
TASK_ID=${SLURM_ARRAY_TASK_ID:?submit with sbatch -- there is no array index}

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
source "${REPO}/slurm/_grid.sh"
grid_select "$TASK_ID"

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    set +u
    conda activate "${CONDA_ENV:-topocg}"
    set -u
fi

MARKER_LC=$(echo "$MARKER" | tr 'A-Z' 'a-z')
BASE=${BASE:-/work2/bz66izin-TopoCG/Outputs_${MARKER_LC}}
DATA_DIR=${DATA_DIR:-/work2/bz66izin-TopoCG/MIST_tiles/${MARKER}/TrainValAB}
VAL_A=${VAL_A:-${DATA_DIR}/valA}
DIRECTION=${DIRECTION:-A2B}
IMAGE_SIZE=${IMAGE_SIZE:-256}
LIMIT=${LIMIT:-0}

RUN_DIR="${BASE}/results/${RUN_NAME}"
OUT_DIR="${BASE}/preds/${RUN_NAME}"

echo "task ${TASK_ID}: ${RUN_NAME}"
echo "  input  ${VAL_A}"
echo "  run    ${RUN_DIR}"
echo "  output ${OUT_DIR}"

# Newest numbered checkpoint, else the rolling one. Collect with a glob into an
# array -- piping a glob into `ls` lists the CWD when nothing matches, because
# nullglob removes the argument entirely.
shopt -s nullglob
CKPT=""
CKPTS=( "${RUN_DIR}/checkpoints"/step_[0-9]*.pt )
if (( ${#CKPTS[@]} )); then
    CKPT=$(printf '%s\n' "${CKPTS[@]}" \
           | sed 's/.*step_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
elif [ -f "${RUN_DIR}/checkpoints/step_latest.pt" ]; then
    CKPT="${RUN_DIR}/checkpoints/step_latest.pt"
fi

# A cell that has not trained yet is not a failure -- exit clean so the array
# task does not show up as failed.
if [ -z "$CKPT" ]; then
    echo "  no checkpoint yet; nothing to infer"
    exit 0
fi

topo-infer --ckpt "$CKPT" --data "$VAL_A" --outdir "$OUT_DIR" \
           --direction "$DIRECTION" --image-size "$IMAGE_SIZE" \
           ${LIMIT:+--limit "$LIMIT"} --resume

echo "Done: ${RUN_NAME}"
