# Shared body of the per-marker sweeps -- not submitted directly.
#
# Sourced by sweep_ki67.sh / sweep_er.sh / sweep_her2.sh / sweep_pr.sh, each of
# which carries its own #SBATCH header and sets MARKER before sourcing. Keeping
# the logic here means the 3x3x3 grid, the schedule and the flags are defined
# once rather than drifting across four near-identical files.

: "${MARKER:?source this from a per-marker wrapper, do not sbatch it directly}"

set -eo pipefail

module purge
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
# conda's activate scripts reference unset variables, so -u has to come off
# across the activation and back on for the rest of the script.
set +u
conda activate "${CONDA_ENV:-topocg}"
set -u

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi || true   # diagnostics must never kill a 48h job under `set -e`

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Persistent homology is CPU-bound and single-threaded per image, and runs while
# the GPU idles. Keep BLAS to one thread so it does not fight the dataloader.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
CPUS=${SLURM_CPUS_PER_TASK:-8}
NUM_WORKERS=$(( CPUS > 2 ? CPUS - 2 : 1 ))

# -----------------------------
# Helper: echo and run a command
# -----------------------------
run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# -----------------------------
# Grid -- defined once in _grid.sh so training and inference agree on the cells
# -----------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID:?submit with sbatch -- there is no array index to sweep over}
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/_grid.sh"
grid_select "$TASK_ID"

# -----------------------------
# Knobs: override at submit time with --export=ALL,NAME=value
# -----------------------------
STEPS=${STEPS:-400000}              # 8 epochs of ~50k tiles; fits one 48h slot
BATCH_SIZE=${BATCH_SIZE:-1}
SAVE_STEPS=${SAVE_STEPS:-100000}    # permanent checkpoints at 100k/200k/300k/400k
IMAGE_SIZE=${IMAGE_SIZE:-256}
TOPO_DOWNSAMPLE=${TOPO_DOWNSAMPLE:-1}
TOPO_EVERY=${TOPO_EVERY:-2}
TOPO_START=${TOPO_START:-100000}    # let the GAN find its footing first
TOPO_WARMUP=${TOPO_WARMUP:-50000}   # then ramp the PH weight in over 50k steps,
                                    # keeping the ramp at half the start step as
                                    # before (was 5k after 10k)

echo "MARKER=${MARKER}"
echo "TASK_ID=${TASK_ID}"
echo "LAMBDA_CYCLE=${LAMBDA_CYCLE}  FIELD_A=${FIELD_A}  FIELD_B=${FIELD_B} (${FIELD_COMBINE})"
echo "LAMBDA_PH_CYC=${PH_CYC}  LAMBDA_PH_TRANS=${PH_TRANS}  LAMBDA_TOPO=${LAMBDA_TOPO}"

# -----------------------------
# Paths
# -----------------------------
# Tiles as produced by topo-crop, which mirrors trainA/trainB/valA/valB:
#   topo-crop --input  /work2/bz66izin-TopoCG/MIST/${MARKER}/TrainValAB/ \
#             --output /work2/bz66izin-TopoCG/MIST_tiles/${MARKER}/TrainValAB/ \
#             --tile_size 512 --resize_to 256
DATA_DIR=${DATA_DIR:-/work2/bz66izin-TopoCG/MIST_tiles/${MARKER}/TrainValAB}
DATA_A=${DATA_A:-${DATA_DIR}/trainA/}
DATA_B=${DATA_B:-${DATA_DIR}/trainB/}

# valA/valB sit alongside these; nothing in the training loop reads them yet.

# tr rather than ${MARKER,,} so this stays portable to bash 3.x
MARKER_LC=$(echo "$MARKER" | tr 'A-Z' 'a-z')
BASE=${BASE:-/work2/bz66izin-TopoCG/Outputs_${MARKER_LC}}
OUTPUT="${BASE}/results/${RUN_NAME}"

mkdir -p "${OUTPUT}"
echo "Output directory: ${OUTPUT}"

# -----------------------------
# Training
# Resumes automatically from OUTPUT if checkpoints are already there, and the
# PH schedule resumes with it (the step counter is a checkpointed buffer).
#
# No --amp: the stain fields go through -log10, and fp16 there ties values that
# should be distinct, which changes the critical-cell structure the persistence
# diagrams are built from. Add --amp back only if the PH terms are off.
# -----------------------------
run_cmd topo-train \
    --dataA "${DATA_A}" \
    --dataB "${DATA_B}" \
    --output "${OUTPUT}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --image-size "${IMAGE_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --save-steps "${SAVE_STEPS}" \
    --field-A "${FIELD_A}" \
    --field-B "${FIELD_B}" \
    --field-combine "${FIELD_COMBINE}" \
    --lambda-cycle "${LAMBDA_CYCLE}" \
    --lambda-topo "${LAMBDA_TOPO}" \
    --lambda-ph-cyc "${PH_CYC}" \
    --lambda-ph-trans "${PH_TRANS}" \
    --topo-downsample "${TOPO_DOWNSAMPLE}" \
    --topo-every "${TOPO_EVERY}" \
    --topo-start-step "${TOPO_START}" \
    --topo-warmup-steps "${TOPO_WARMUP}"

echo "Done: ${RUN_NAME} finished successfully."
