#!/bin/bash
#SBATCH --job-name=topo_infer_base
#SBATCH --output=logs_topo/infer_baseline_%A_%a.out
#SBATCH --error=logs_topo/infer_baseline_%A_%a.err

#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-5  # 2 models x 3 sizes; see ARMS below

# Validation inference for the baselines train_baseline.sh trained -- one array
# task per (model, size), all for one MARKER:
#
#   task  0 cyclegan small    1 cyclegan medium    2 cyclegan large
#   task  3 dclgan   small    4 dclgan   medium    5 dclgan   large
#
# Inference goes through the zoo's i2i-inference, not topo-infer: these are
# the zoo's own CycleGAN/DCLGAN, and topo-infer only rebuilds TopoCycleGAN. The
# width is in the checkpoint's config, so the size needs no flag here.
#
#   mkdir -p logs_topo
#   sbatch --export=ALL,MARKER=ER infer_baseline.sh               # all six
#   sbatch --export=ALL,MARKER=ER --array=0,3 infer_baseline.sh   # small only
#   sbatch --partition=paula --export=ALL,MARKER=VS  infer_baseline.sh
#   sbatch --partition=paula --export=ALL,MARKER=BCI infer_baseline.sh
#   sbatch --partition=paula --export=ALL,MARKER=BCI,SPLIT=testA infer_baseline.sh
#
# CKPT_STEP=400000 infers that checkpoint and nothing else. Use it when the
# rungs are compared: medium and large reach their budget over several slots,
# and the default (newest checkpoint) would quietly compare a large model at
# 200k against a small one at 400k.
#
# A (model, size) that was never trained exits clean, so the full array is
# safe to submit whatever subset exists.
#
# Output: ${BASE}/preds/${RUN_NAME}, or preds_${SPLIT}/ for a split other than
# valA -- the same layout infer_sweep.sh writes, under the root train_baseline.sh
# trained into (Outputs_<marker>_<model>_<size>).

set -eo pipefail

: "${MARKER:?set MARKER, e.g. --export=ALL,MARKER=ER}"
TASK_ID=${SLURM_ARRAY_TASK_ID:?submit with sbatch -- there is no array index}

ARMS=(
  "cyclegan:small" "cyclegan:medium" "cyclegan:large"
  "dclgan:small"   "dclgan:medium"   "dclgan:large"
)
if (( TASK_ID < 0 || TASK_ID >= ${#ARMS[@]} )); then
    echo "ERROR: task ${TASK_ID} is outside 0-$(( ${#ARMS[@]} - 1 ))" >&2
    exit 1
fi
MODEL=${ARMS[$TASK_ID]%%:*}
SIZE=${ARMS[$TASK_ID]#*:}

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-topocg}"
fi

# Kept in step with train_baseline.sh's BASE and RUN_NAME.
MARKER_LC=$(echo "$MARKER" | tr 'A-Z' 'a-z')
BASE=${BASE:-/work2/bz66izin-TopoCG/Outputs_${MARKER_LC}_${MODEL}_${SIZE}}
RUN_NAME=${RUN_NAME:-${MARKER}_${MODEL}_${SIZE}}
RUN_DIR="${BASE}/results/${RUN_NAME}"

# Tile root per marker: the same lookup as training and infer_sweep.sh.
TILES=${TILES:-/work2/bz66izin-TopoCG/MIST_tiles}
TILES_BCI=${TILES_BCI:-/work2/bz66izin-TopoCG/BCI_tiles}
TILES_VS=${TILES_VS:-/work2/bz66izin-TopoCG/VS_tiles}
tiles_root() {
    local var="TILES_$1"
    echo "${!var:-$TILES}"
}
TILE_ROOT="$(tiles_root "$MARKER")"
DATA_DIR=${DATA_DIR:-${TILE_ROOT}/${MARKER}/TrainValAB}

if [ -n "${VAL_A:-}" ]; then
    SPLIT=${SPLIT:-$(basename "$VAL_A")}
else
    SPLIT=${SPLIT:-valA}
    VAL_A="${DATA_DIR}/${SPLIT}"
fi
DIRECTION=${DIRECTION:-A2B}

# i2i-inference walks the whole tree and has no --subdir, so a raw per-case
# tiling would have its masks translated too. Refuse rather than do that.
if [ -n "${SUBDIR:-}" ]; then
    echo "ERROR: SUBDIR is not supported -- i2i-inference has no --subdir." >&2
    echo "  Point VAL_A at a directory holding only the tiles." >&2
    exit 1
fi

if [ "$SPLIT" = "valA" ]; then
    OUT_DIR=${OUT_DIR:-${BASE}/preds/${RUN_NAME}}
else
    OUT_DIR=${OUT_DIR:-${BASE}/preds_${SPLIT}/${RUN_NAME}}
fi

echo "task ${TASK_ID}: ${MODEL} / ${SIZE}  (${RUN_NAME})"
echo "  split  ${SPLIT}"
echo "  input  ${VAL_A}"
echo "  run    ${RUN_DIR}"
echo "  output ${OUT_DIR}"

if [ ! -d "$VAL_A" ]; then
    echo "ERROR: no such input directory: ${VAL_A}" >&2
    echo "  tile root for ${MARKER}: ${TILE_ROOT}" >&2
    echo "  set TILES_${MARKER}=/path/to/tiles, or DATA_DIR/VAL_A directly" >&2
    exit 1
fi

shopt -s nullglob
CKPT=""
if [ -n "${CKPT_STEP:-}" ]; then
    want="${RUN_DIR}/checkpoints/step_${CKPT_STEP}.pt"
    if [ ! -f "$want" ]; then
        echo "  no step_${CKPT_STEP}.pt yet; nothing to infer"
        exit 0
    fi
    CKPT="$want"
else
    # Newest numbered checkpoint, else the rolling one -- as infer_sweep.sh.
    CKPTS=( "${RUN_DIR}/checkpoints"/step_[0-9]*.pt )
    if (( ${#CKPTS[@]} )); then
        CKPT=$(printf '%s\n' "${CKPTS[@]}" \
               | sed 's/.*step_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
    elif [ -f "${RUN_DIR}/checkpoints/step_latest.pt" ]; then
        CKPT="${RUN_DIR}/checkpoints/step_latest.pt"
    fi
fi

if [ -z "$CKPT" ]; then
    echo "  no checkpoint yet; nothing to infer"
    exit 0
fi
echo "  ckpt   ${CKPT}"

i2i-inference --model "$MODEL" --ckpt "$CKPT" --data "$VAL_A" \
              --outdir "$OUT_DIR" --direction "$DIRECTION" --resume

echo "Done: ${RUN_NAME}"
