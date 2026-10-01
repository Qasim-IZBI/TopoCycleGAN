#!/bin/bash
#SBATCH --job-name=topo_dclgan
#SBATCH --output=logs_topo/dclgan_%j.out
#SBATCH --error=logs_topo/dclgan_%j.err

#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# No --array: DCLGAN has no topological terms, so there is no grid to sweep.
# One run per marker is the whole baseline.

# DCLGAN as an external baseline, trained through the zoo's own i2i-train.
#
# This is deliberately NOT a topo_i2i model: it is the published method, run by
# the zoo's code, on the same tiles, for the same number of steps, so a
# difference in the result is a difference in method rather than in plumbing.
#
#   mkdir -p logs_topo                      # SLURM will not create it for you
#   MARKER=ER   sbatch --export=ALL train_dclgan.sh
#   MARKER=VS   TILES_VS=/work2/bz66izin-TopoCG/VS_tiles \
#               sbatch --export=ALL --partition=paula train_dclgan.sh
#   MARKER=BCI  sbatch --export=ALL --partition=paula train_dclgan.sh
#
# SIZE -- and read this before choosing, because the default is not the
# smallest option and that is on purpose:
#
#   matched  ngf=64 n_blocks=9  28.81M total (23.28M generators)   [default]
#   small    ngf=32 n_blocks=9  11.63M total ( 6.10M generators)
#   tiny     ngf=32 n_blocks=6   9.85M total ( 4.33M generators)
#
# TopoCycleGAN at the settings every sweep here trains is 28.29M total, 22.76M
# of it generators -- the stain fields add no learnable parameters, being
# buffers rather than nn.Parameter. So DCLGAN AT ITS OWN DEFAULTS is 102% of
# the model it is being compared against, which is what makes it a baseline.
# `small` is 41% of it: a useful arm if you want to show the comparison is not
# merely a capacity contest, but on its own it would confound method with size
# and a reviewer would say so. Pick with SIZE=, or set DCLGAN_NGF/DCLGAN_BLOCKS
# directly:
#
#   MARKER=ER SIZE=small sbatch --export=ALL train_dclgan.sh
#
# IT DOES NOT RESUME. i2i-train never calls resume_if_exists(), unlike
# topo-train, so a job that hits the 48h wall restarts from step 0 and
# overwrites what the previous one wrote. This script therefore REFUSES to
# start on top of existing checkpoints; see the message it prints if that
# happens. --init_ckpt warm-starts the weights but resets the step counter and
# keeps none of the optimiser state, so it is not a resume either. If a marker
# cannot finish STEPS in one slot, lower STEPS for every marker rather than
# letting some run longer than others.
#
# Output: ${BASE}/results/${MARKER}_dclgan/{checkpoints,samples}

set -eo pipefail

MARKER=${MARKER:?set MARKER, e.g. --export=ALL,MARKER=ER}

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-topocg}"
fi

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi || true   # diagnostics must never kill a long job under `set -e`

# Tile root per marker: TILES_<MARKER> wins if it is set, else TILES. The same
# lookup the sweeps and infer_sweep.sh use, so the baseline reads exactly the
# tiles the method does.
TILES=${TILES:-/work2/bz66izin-TopoCG/MIST_tiles}
TILES_BCI=${TILES_BCI:-/work2/bz66izin-TopoCG/BCI_tiles}
TILES_VS=${TILES_VS:-/work2/bz66izin-TopoCG/VS_tiles}
tiles_root() {
    local var="TILES_$1"
    echo "${!var:-$TILES}"
}
TILE_ROOT="$(tiles_root "$MARKER")"
DATA_DIR=${DATA_DIR:-${TILE_ROOT}/${MARKER}/TrainValAB}
DATA_A=${DATA_A:-${DATA_DIR}/trainA/}
DATA_B=${DATA_B:-${DATA_DIR}/trainB/}

# Matched to the sweeps so the baseline is not handed a different budget.
STEPS=${STEPS:-400000}
BATCH_SIZE=${BATCH_SIZE:-1}
SAVE_STEPS=${SAVE_STEPS:-100000}
LOG_STEPS=${LOG_STEPS:-1000}
LR=${LR:-2e-4}
SEED=${SEED:-0}

# No persistence here, so every core past the training loop can load data.
CPUS=${SLURM_CPUS_PER_TASK:-4}
NUM_WORKERS=${NUM_WORKERS:-$(( CPUS > 1 ? CPUS - 1 : 1 ))}

SIZE=${SIZE:-matched}
case "$SIZE" in
    matched) def_ngf=64; def_blocks=9 ;;
    small)   def_ngf=32; def_blocks=9 ;;
    tiny)    def_ngf=32; def_blocks=6 ;;
    *) echo "ERROR: SIZE must be matched, small or tiny (got '${SIZE}')" >&2; exit 1 ;;
esac
DCLGAN_NGF=${DCLGAN_NGF:-$def_ngf}
DCLGAN_BLOCKS=${DCLGAN_BLOCKS:-$def_blocks}

# DCLGAN's own loss weights, left at the paper's values. They are not swept:
# this is a baseline, and tuning it here but not the published way round would
# make the comparison worse, not better.
LAMBDA_DCL=${LAMBDA_DCL:-1.0}
LAMBDA_CYCLE=${LAMBDA_CYCLE:-10.0}
LAMBDA_IDENTITY=${LAMBDA_IDENTITY:-0.0}
N_PATCHES=${N_PATCHES:-256}
PROJ_DIM=${PROJ_DIM:-256}

MARKER_LC=$(echo "$MARKER" | tr 'A-Z' 'a-z')
BASE=${BASE:-/work2/bz66izin-TopoCG/Outputs_${MARKER_LC}_dclgan}
RUN_NAME=${RUN_NAME:-${MARKER}_dclgan}
OUTPUT="${BASE}/results/${RUN_NAME}"

if [ ! -d "$DATA_A" ] || [ ! -d "$DATA_B" ]; then
    echo "ERROR: ${DATA_A} or ${DATA_B} is missing" >&2
    echo "  tile root for ${MARKER}: ${TILE_ROOT}" >&2
    echo "  set TILES_${MARKER}=/path/to/tiles, or DATA_DIR directly" >&2
    exit 1
fi

# i2i-train starts from step 0 whatever is already in save_dir, so a second
# submit would quietly overwrite a finished or half-finished run. Refuse rather
# than destroy it.
shopt -s nullglob
existing=( "${OUTPUT}/checkpoints"/*.pt )
if (( ${#existing[@]} )) && [ "${RESTART:-0}" != "1" ]; then
    echo "ERROR: ${OUTPUT}/checkpoints already holds ${#existing[@]} checkpoint(s)." >&2
    echo "i2i-train does not resume -- it would restart at step 0 and overwrite them." >&2
    echo "  to keep them:    train somewhere else with BASE=... or RUN_NAME=..." >&2
    echo "  to discard them: resubmit with RESTART=1" >&2
    exit 1
fi

mkdir -p "$OUTPUT"

echo
echo "================ DCLGAN baseline: ${MARKER} ================"
echo "  size      ${SIZE}  (ngf=${DCLGAN_NGF}, n_blocks=${DCLGAN_BLOCKS})"
echo "  dataA     ${DATA_A}"
echo "  dataB     ${DATA_B}"
echo "  output    ${OUTPUT}"
echo "  steps     ${STEPS}  batch ${BATCH_SIZE}  workers ${NUM_WORKERS}"
echo

# What the model actually weighs, in the log, so the baseline's capacity is on
# the record next to its result rather than assumed later.
i2i-train --model dclgan --count_params \
    --dclgan_ngf "$DCLGAN_NGF" --dclgan_n_blocks "$DCLGAN_BLOCKS" || true

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# Note the underscores: i2i-train spells its flags differently from topo-train.
run_cmd i2i-train \
    --model dclgan \
    --dataA "$DATA_A" \
    --dataB "$DATA_B" \
    --output "$OUTPUT" \
    --steps "$STEPS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --num_workers "$NUM_WORKERS" \
    --save_steps "$SAVE_STEPS" \
    --log_steps "$LOG_STEPS" \
    --seed "$SEED" \
    --dclgan_ngf "$DCLGAN_NGF" \
    --dclgan_n_blocks "$DCLGAN_BLOCKS" \
    --lambda_dcl "$LAMBDA_DCL" \
    --dclgan_lambda_cycle "$LAMBDA_CYCLE" \
    --dclgan_lambda_identity "$LAMBDA_IDENTITY" \
    --n_patches "$N_PATCHES" \
    --proj_dim "$PROJ_DIM" \
    ${AMP:+--amp}

echo "Done: ${RUN_NAME}"
