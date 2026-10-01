#!/bin/bash
#SBATCH --job-name=topo_baseline
#SBATCH --output=logs_topo/baseline_%j.out
#SBATCH --error=logs_topo/baseline_%j.err

#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# No --array: a baseline has no topological terms, so there is no grid. One run
# per (model, size, marker).

# CycleGAN and DCLGAN baselines at three capacities, trained through the zoo's
# own i2i-train. Nothing here reimplements either method: the script chooses
# the tile root, the size, and where the result goes.
#
#   mkdir -p logs_topo
#   MODEL=dclgan   SIZE=small  MARKER=ER  sbatch --export=ALL train_baseline.sh
#   MODEL=cyclegan SIZE=medium MARKER=VS  TILES_VS=/work2/bz66izin-TopoCG/VS_tiles \
#       sbatch --export=ALL --partition=paula train_baseline.sh
#   MODEL=cyclegan SIZE=large  MARKER=BCI sbatch --export=ALL --partition=paula \
#       train_baseline.sh
#
# SIZES, measured as the A->B generator -- Enc_A + Bn_A + Dec_B, which is
# exactly what `i2i-train --count_params` reports, so these numbers match what
# the log prints. Depth is fixed at 9 bottleneck blocks and only width moves,
# so the three rungs differ in one variable rather than two:
#
#   small   ngf=64    11.38M A->B   (cyclegan 28.29M total, dclgan 28.81M)
#   medium  ngf=136   51.32M A->B   (cyclegan 108.17M total, dclgan 108.99M)
#   large   ngf=192  102.26M A->B   (cyclegan 210.05M total, dclgan 211.10M)
#
# Both models give the same A->B count at a given width: they share the
# generator architecture and differ in their losses, which is what makes the
# comparison between them a comparison of method.
#
# TopoCycleGAN, at the settings every sweep in this repo trains, is ngf=64 --
# 11.38M A->B, 28.29M total. So `small` is the size-matched baseline and the
# other two rungs are deliberately LARGER than the method they are compared
# against. That asymmetry is the point of running them: a baseline that loses
# at four or nine times the capacity has lost on method.
#
# TWO THINGS WILL BITE AT medium AND large.
#
# 1. COST. Convolution cost goes as the square of the width, so medium is
#    ~4.5x and large ~9x the per-step cost of small. 400k steps do not fit in
#    one 48h slot at either size, and probably not at 2 slots for large.
#
# 2. THE WALL, which is survivable. Trained through topo-baseline rather than
#    i2i-train directly: the two are the same CLI and the same code, except
#    that topo-baseline calls resume_if_exists() first, which the zoo's entry
#    point omits. So a job killed at 48h is resumed by submitting THE SAME LINE
#    AGAIN -- model, both optimisers, the step counter and the elapsed-time
#    total all come back from the furthest-ahead checkpoint, and step_latest.pt
#    is written every LOG_STEPS, so at most that many steps are repeated.
#
#    A `large` arm therefore reaches 400k over several slots rather than
#    needing to fit in one. Resubmit until the log says it reached STEPS:
#
#      until grep -q "Done:" logs_topo/baseline_<jobid>.out; do ...resubmit...; done
#
#    Keep STEPS equal across every arm you intend to compare. The capacity
#    ladder only means something if the rungs got the same budget, and the
#    temptation when large is slow is to quietly give it less.
#
# Output: ${BASE}/results/${MARKER}_${MODEL}_${SIZE}/{checkpoints,samples}

set -eo pipefail

MODEL=${MODEL:?set MODEL=cyclegan or MODEL=dclgan}
MARKER=${MARKER:?set MARKER, e.g. MARKER=ER}
SIZE=${SIZE:-small}

case "$MODEL" in
    cyclegan|dclgan) ;;
    *) echo "ERROR: MODEL must be cyclegan or dclgan (got '${MODEL}')" >&2; exit 1 ;;
esac
case "$SIZE" in
    small)  def_ngf=64  ;;
    medium) def_ngf=136 ;;
    large)  def_ngf=192 ;;
    *) echo "ERROR: SIZE must be small, medium or large (got '${SIZE}')" >&2; exit 1 ;;
esac
NGF=${NGF:-$def_ngf}
BLOCKS=${BLOCKS:-9}

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-topocg}"
fi

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi || true   # diagnostics must never kill a long job under `set -e`

# Tile root per marker: TILES_<MARKER> wins if set, else TILES. The same lookup
# the sweeps use, so a baseline reads exactly the tiles the method does.
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

STEPS=${STEPS:-400000}
BATCH_SIZE=${BATCH_SIZE:-1}
SAVE_STEPS=${SAVE_STEPS:-100000}
LOG_STEPS=${LOG_STEPS:-1000}
LR=${LR:-2e-4}
SEED=${SEED:-0}

# No persistence in a baseline, so every core past the training loop can load.
CPUS=${SLURM_CPUS_PER_TASK:-4}
NUM_WORKERS=${NUM_WORKERS:-$(( CPUS > 1 ? CPUS - 1 : 1 ))}

# DCLGAN's own loss weights, at the paper's values. Not swept: tuning the
# baseline here but not the published way round would make the comparison
# worse, not fairer.
LAMBDA_DCL=${LAMBDA_DCL:-1.0}
DCL_LAMBDA_CYCLE=${DCL_LAMBDA_CYCLE:-10.0}
DCL_LAMBDA_IDENTITY=${DCL_LAMBDA_IDENTITY:-0.0}
N_PATCHES=${N_PATCHES:-256}
PROJ_DIM=${PROJ_DIM:-256}

MARKER_LC=$(echo "$MARKER" | tr 'A-Z' 'a-z')
BASE=${BASE:-/work2/bz66izin-TopoCG/Outputs_${MARKER_LC}_${MODEL}_${SIZE}}
RUN_NAME=${RUN_NAME:-${MARKER}_${MODEL}_${SIZE}}
OUTPUT="${BASE}/results/${RUN_NAME}"

if [ ! -d "$DATA_A" ] || [ ! -d "$DATA_B" ]; then
    echo "ERROR: ${DATA_A} or ${DATA_B} is missing" >&2
    echo "  tile root for ${MARKER}: ${TILE_ROOT}" >&2
    echo "  set TILES_${MARKER}=/path/to/tiles, or DATA_DIR directly" >&2
    exit 1
fi

# Existing checkpoints are resumed from, not overwritten -- topo-baseline calls
# resume_if_exists(). Report what is there so the log says which slot this is.
shopt -s nullglob
existing=( "${OUTPUT}/checkpoints"/*.pt )
if (( ${#existing[@]} )); then
    if [ "${RESTART:-0}" = "1" ]; then
        echo "RESTART=1: discarding ${#existing[@]} checkpoint(s) under ${OUTPUT}/checkpoints"
        rm -f "${OUTPUT}/checkpoints"/*.pt
    else
        echo "resuming: ${#existing[@]} checkpoint(s) already under ${OUTPUT}/checkpoints"
        echo "  (RESTART=1 would discard them and train from step 0 instead)"
    fi
fi

mkdir -p "$OUTPUT"

# Relative per-step cost against the size-matched rung, so the log carries the
# reason a job did or did not reach STEPS.
# LC_ALL=C: awk honours the locale's decimal separator, and a comma here
# would read as a thousands mark in the log.
cost=$(LC_ALL=C awk -v n="$NGF" 'BEGIN { printf "%.1f", (n/64.0)*(n/64.0) }')

echo
echo "================ ${MODEL} / ${SIZE} baseline: ${MARKER} ================"
echo "  width     ngf=${NGF}  n_blocks=${BLOCKS}"
echo "  rel cost  ~${cost}x per step vs ngf=64 (width squared)"
echo "  dataA     ${DATA_A}"
echo "  dataB     ${DATA_B}"
echo "  output    ${OUTPUT}"
echo "  steps     ${STEPS}  batch ${BATCH_SIZE}  workers ${NUM_WORKERS}"
echo

size_args=()
if [ "$MODEL" = "cyclegan" ]; then
    size_args=(--cyclegan_ngf "$NGF" --cyclegan_n_blocks "$BLOCKS")
else
    size_args=(--dclgan_ngf "$NGF" --dclgan_n_blocks "$BLOCKS"
               --lambda_dcl "$LAMBDA_DCL"
               --dclgan_lambda_cycle "$DCL_LAMBDA_CYCLE"
               --dclgan_lambda_identity "$DCL_LAMBDA_IDENTITY"
               --n_patches "$N_PATCHES" --proj_dim "$PROJ_DIM")
fi

# The capacity on the record, next to the result, rather than assumed later.
# --count_params exits before training, so either entry point does; use the
# zoo's, since nothing is being resumed here.
i2i-train --model "$MODEL" --count_params "${size_args[@]}" || true

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# topo-baseline, not i2i-train: same CLI and same code, plus the
# resume_if_exists() the zoo's entry point leaves out. Note the underscores --
# these flags are i2i-train's, which spells them differently from topo-train.
run_cmd topo-baseline \
    --model "$MODEL" \
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
    "${size_args[@]}" \
    ${AMP:+--amp}

echo "Done: ${RUN_NAME}"
