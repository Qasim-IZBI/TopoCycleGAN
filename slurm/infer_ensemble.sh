#!/bin/bash
#SBATCH --job-name=topo_infer_ens
#SBATCH --output=logs_topo/infer_ens_%A_%a.out
#SBATCH --error=logs_topo/infer_ens_%A_%a.err

#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-29  # 3 models x 10 seeds, as train_ensemble.sh

# Inference for the deep-ensemble members train_ensemble.sh trained. Task N is
# the member training task N produced:
#
#    0- 9  TopoCycleGAN   (topo-infer)
#   10-19  CycleGAN small (i2i-inference)
#   20-29  DCLGAN small   (i2i-inference)
#
# ONLY FINISHED MEMBERS. A member is inferred from its newest numbered
# checkpoint once that has reached the seed-0 run's step budget; a member still
# training exits clean and says so. (Inference resumes, skipping tiles already
# written -- predictions from a half-trained member would survive a later run.)
# So submit the full array whenever you like; rerun it after stragglers finish.
#
# Seed 0 already has predictions -- train_ensemble.sh links them as
# preds/seed0 for valA, and for another split this links <seed-0 root>/
# preds_<SPLIT>/<run> when it exists. Otherwise seed 0 is inferred like the rest.
#
#   mkdir -p logs_topo
#   for M in Ki67 ER HER2 PR; do
#       sbatch --export=ALL,MARKER=$M infer_ensemble.sh
#   done
#   sbatch --partition=paula --export=ALL,MARKER=BCI infer_ensemble.sh
#   sbatch --partition=paula --export=ALL,MARKER=BCI,SPLIT=testA infer_ensemble.sh
#   sbatch --partition=paula --export=ALL,MARKER=VS,SPLIT=testA infer_ensemble.sh
#
# Output: ${ENS_ROOT}/<marker>/<model>/preds/seed<k>             (valA)
#         ${ENS_ROOT}/<marker>/<model>/preds_<SPLIT>/seed<k>     (any other split)

set -eo pipefail

: "${MARKER:?set MARKER, e.g. --export=ALL,MARKER=ER}"
TASK_ID=${SLURM_ARRAY_TASK_ID:?submit with sbatch -- there is no array index}

ROOT=${ROOT:-/work2/bz66izin-TopoCG}
ENS_ROOT=${ENS_ROOT:-${ROOT}/ensemble}
BEST_RUN=${BEST_RUN:-lt0.002_cyc1_trans1}   # as train_ensemble.sh
BEST_ARM_ROOT=${BEST_ARM_ROOT:-}
MODELS=(topo cyclegan dclgan)

mi=$(( TASK_ID / 10 )); SEED=$(( TASK_ID % 10 ))
(( mi < 3 )) || { echo "task ${TASK_ID}: tasks are 0-29"; exit 0; }
model=${MODELS[$mi]}

# Kept in step with train_ensemble.sh's seed0_paths.
MARKER_LC=$(echo "$MARKER" | tr 'A-Z' 'a-z')
case "$model" in
    topo)
        SRC_BASE=${BEST_ARM_ROOT:-${ROOT}/Outputs_${MARKER_LC}}
        SRC_NAME="${MARKER}_${BEST_RUN}" ;;
    *)
        SRC_BASE="${ROOT}/Outputs_${MARKER_LC}_${model}_small"
        SRC_NAME="${MARKER}_${model}_small" ;;
esac
SRC_RUN="${SRC_BASE}/results/${SRC_NAME}"

# Tile root per marker: the same lookup as training and the other inference scripts.
TILES=${TILES:-${ROOT}/MIST_tiles}
TILES_BCI=${TILES_BCI:-${ROOT}/BCI_tiles}
TILES_VS=${TILES_VS:-${ROOT}/VS_tiles}
var="TILES_${MARKER}"; TILE_ROOT=${!var:-$TILES}
DATA_DIR=${DATA_DIR:-${TILE_ROOT}/${MARKER}/TrainValAB}
SPLIT=${SPLIT:-valA}
VAL_A="${DATA_DIR}/${SPLIT}"

MODEL_DIR="${ENS_ROOT}/${MARKER}/${model}"
RUN_DIR="${MODEL_DIR}/results/seed${SEED}"
if [ "$SPLIT" = "valA" ]; then
    PRED_ROOT="${MODEL_DIR}/preds"; SRC_PREDS="${SRC_BASE}/preds/${SRC_NAME}"
else
    PRED_ROOT="${MODEL_DIR}/preds_${SPLIT}"; SRC_PREDS="${SRC_BASE}/preds_${SPLIT}/${SRC_NAME}"
fi
OUT_DIR="${PRED_ROOT}/seed${SEED}"

echo "ensemble ${MARKER} / ${model} / seed ${SEED}"
echo "  split  ${SPLIT}   input ${VAL_A}"
echo "  run    ${RUN_DIR}"
echo "  output ${OUT_DIR}"

if [ ! -d "$VAL_A" ]; then
    echo "ERROR: no such input directory: ${VAL_A}" >&2
    exit 1
fi

# Only where seed 0 is the linked existing run -- a schedule variant trained
# its own seed 0 (a real directory), which is inferred like the rest.
if [ "$SEED" = "0" ] && [ -d "$SRC_PREDS" ] && { [ -L "$RUN_DIR" ] || [ ! -e "$RUN_DIR" ]; }; then
    mkdir -p "$PRED_ROOT"
    if [ ! -e "$OUT_DIR" ]; then
        ln -sfn "$SRC_PREDS" "$OUT_DIR"
    fi
    echo "  seed 0: the existing predictions, ${SRC_PREDS}"
    exit 0
fi

if [ ! -d "$RUN_DIR" ]; then
    echo "  member not trained yet; nothing to infer"
    exit 0
fi

# The step budget is the seed-0 run's, as train_ensemble.sh status reads it.
target=$(grep -o '"total_steps": *[0-9]*' "${SRC_RUN}/training_meta.json" 2>/dev/null \
         | grep -o '[0-9]*$' || echo 400000)
target=${STEPS:-$target}     # a screen's shorter budget, as train_ensemble.sh
shopt -s nullglob
CKPT=""; step=0
for f in "${RUN_DIR}/checkpoints"/step_[0-9]*.pt; do
    n=${f##*step_}; n=${n%.pt}
    if (( n > step )); then step=$n; CKPT=$f; fi
done
if (( step < target )); then
    echo "  newest checkpoint step ${step} < budget ${target}: still training, not inferred"
    exit 0
fi
echo "  ckpt   ${CKPT}"

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-topocg}"
fi

mkdir -p "$PRED_ROOT"
if [ "$model" = "topo" ]; then
    topo-infer --ckpt "$CKPT" --data "$VAL_A" --outdir "$OUT_DIR" \
               --direction A2B --image-size 256 --resume
else
    i2i-inference --model "$model" --ckpt "$CKPT" --data "$VAL_A" \
                  --outdir "$OUT_DIR" --direction A2B --resume
fi

echo "Done: ${MARKER} / ${model} / seed ${SEED}"
