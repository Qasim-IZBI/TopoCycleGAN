#!/bin/bash
#SBATCH --job-name=topo_infer
#SBATCH --output=logs_topo/infer_%A_%a.out
#SBATCH --error=logs_topo/infer_%A_%a.err

#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-12  # 13 cells; see _grid.sh for the grid

# Validation inference, one array task per sweep cell -- task N infers the model
# that training task N produced, because both resolve the cell through
# _grid.sh.
#
# Run once before the first submit:  mkdir -p logs_topo
#
#   sbatch --export=ALL,MARKER=ER infer_sweep.sh                  # all 13 cells
#   sbatch --export=ALL,MARKER=ER --array=0 infer_sweep.sh        # the baseline only
#   sbatch --export=ALL,MARKER=ER,LIMIT=8 --array=0 infer_sweep.sh
#
# VS and BCI train on paula (see sweep_vs.sh / sweep_bci.sh), so their
# inference goes there too -- the partition is a #SBATCH directive here and a
# submit-time flag overrides it:
#
#   sbatch --partition=paula --export=ALL,MARKER=VS  infer_sweep.sh
#   sbatch --partition=paula --export=ALL,MARKER=BCI infer_sweep.sh
#
# BCI also has a real held-out test split from prepare_bci.sh. SPLIT picks it,
# and its predictions go to their own directory rather than over the
# validation ones:
#
#   sbatch --partition=paula --export=ALL,MARKER=BCI,SPLIT=testA infer_sweep.sh
#
# VAL_A takes a path directly, for a set that was never linked into the flat
# TrainValAB layout -- the VS H&E test tiles, say, which sit in the raw
# per-case tiling as <case>/images/<id>.tif. SUBDIR keeps the sibling
# <case>/masks/ out of the run; without it the generator would translate the
# masks too. Predictions keep the <case>/images/ structure, so ids repeated
# across cases do not collide:
#
#   sbatch --partition=paula --export=ALL,MARKER=VS,SUBDIR=images,\
#       VAL_A=/work2/bz66izin-UC_project/ID_HE/no_overlap/testA/tiles/testA \
#       infer_sweep.sh
#
# The audit's field selection is NOT needed here: it is baked into the
# checkpoint's config, which is what topo-infer rebuilds the model from.
#
# Submit from the directory holding these scripts (or from the repo root --
# both resolve), or export REPO=/path/to/the/scripts.

set -eo pipefail

: "${MARKER:?set MARKER, e.g. --export=ALL,MARKER=ER}"
TASK_ID=${SLURM_ARRAY_TASK_ID:?submit with sbatch -- there is no array index}

# Sibling scripts live next to this one. SLURM runs a batch script from a copy
# in its spool directory, so BASH_SOURCE cannot locate them -- the directory
# sbatch was called from can. The fallback keeps a submit from the repo root
# working too.
SLURM_DIR=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[ -f "${SLURM_DIR}/_grid.sh" ] || SLURM_DIR="${SLURM_DIR}/slurm"
source "${SLURM_DIR}/_grid.sh"
grid_select "$TASK_ID"

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-topocg}"
fi

MARKER_LC=$(echo "$MARKER" | tr 'A-Z' 'a-z')
BASE=${BASE:-/work2/bz66izin-TopoCG/Outputs_${MARKER_LC}}

# Tile root per marker: TILES_<MARKER> wins if it is set, else TILES. Kept in
# step with the same lookup in _sweep_common.sh -- training and inference have
# to agree on where a dataset's tiles live, and BCI and VS are tiled into their
# own roots by prepare_bci.sh / prepare_vs.sh.
TILES=${TILES:-/work2/bz66izin-TopoCG/MIST_tiles}
TILES_BCI=${TILES_BCI:-/work2/bz66izin-TopoCG/BCI_tiles}
TILES_VS=${TILES_VS:-/work2/bz66izin-TopoCG/VS_tiles}
tiles_root() {
    local var="TILES_$1"
    echo "${!var:-$TILES}"
}
TILE_ROOT="$(tiles_root "$MARKER")"
DATA_DIR=${DATA_DIR:-${TILE_ROOT}/${MARKER}/TrainValAB}

# Which split to infer. valA is the held-out validation input every dataset
# has; BCI additionally has testA. A VAL_A given as a full path still works and
# names the split itself, so its predictions stay separate as well.
if [ -n "${VAL_A:-}" ]; then
    SPLIT=${SPLIT:-$(basename "$VAL_A")}
else
    SPLIT=${SPLIT:-valA}
    VAL_A="${DATA_DIR}/${SPLIT}"
fi

DIRECTION=${DIRECTION:-A2B}
IMAGE_SIZE=${IMAGE_SIZE:-256}
LIMIT=${LIMIT:-0}
# Empty for the flat TrainValAB layouts, where every image under the split is a
# tile. Set it to 'images' for a raw per-case tiling that carries masks beside
# the tiles.
SUBDIR=${SUBDIR:-}

RUN_DIR="${BASE}/results/${RUN_NAME}"
# valA keeps the original preds/ layout; any other split gets its own root so a
# test run does not overwrite the validation predictions.
if [ "$SPLIT" = "valA" ]; then
    OUT_DIR=${OUT_DIR:-${BASE}/preds/${RUN_NAME}}
else
    OUT_DIR=${OUT_DIR:-${BASE}/preds_${SPLIT}/${RUN_NAME}}
fi

echo "task ${TASK_ID}: ${RUN_NAME}"
echo "  split  ${SPLIT}"
echo "  input  ${VAL_A}${SUBDIR:+  (only */${SUBDIR}/)}"
echo "  run    ${RUN_DIR}"
echo "  output ${OUT_DIR}"

# A missing input directory is a misconfigured tile root, not an empty split --
# say so here rather than letting topo-infer report zero tiles, which is what a
# marker tiled outside TILES used to look like.
if [ ! -d "$VAL_A" ]; then
    echo "ERROR: no such input directory: ${VAL_A}" >&2
    echo "  tile root for ${MARKER}: ${TILE_ROOT}" >&2
    echo "  set TILES_${MARKER}=/path/to/tiles (the root holding ${MARKER}/TrainValAB)," >&2
    echo "  or DATA_DIR/VAL_A directly" >&2
    exit 1
fi

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
           ${SUBDIR:+--subdir "$SUBDIR"} \
           ${LIMIT:+--limit "$LIMIT"} --resume

echo "Done: ${RUN_NAME}"
