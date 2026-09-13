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
# Grid: an explicit cell list, so duplicate combinations are simply absent.
#
#   lambda_cycle  {10, 0}          CycleGAN L1 cycle weight
#   ph_cyc        {0, 1}           cycle-topology weight
#   ph_trans      {0, 1}           translation-topology weight
#   lambda_topo   {2e-4, 2e-2}     overall PH scale
#   field_B       hematoxylin/dab | dab/hematoxylin | dab+hematoxylin (sum)
#
# field_A is fixed at hematoxylin/eosin. The '/' forms deconvolve; a bare stain
# name projects and does NOT separate stains -- see fields.StainField.
#
# The full factorial is 48, but ph_cyc=0 AND ph_trans=0 switches off every
# topological term, making lambda_topo and both fields inert -- those 12 collapse
# to one anchor run per lambda_cycle. The 38 that remain are listed below.
#
#   tasks  0-18  lambda_cycle=10  (the critical path)
#   tasks 19-37  lambda_cycle=0   (does topology substitute for L1 cycle?)
#
# Format: lambda_cycle:ph_cyc:ph_trans:lambda_topo:field_B:field_combine
# -----------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID:?submit with sbatch -- there is no array index to sweep over}

CELLS=(
  "10:0:0:0:hematoxylin/dab:max"   # anchor: no PH, fields inert
  "10:0:1:0.0002:hematoxylin/dab:max"
  "10:0:1:0.0002:dab/hematoxylin:max"
  "10:0:1:0.0002:dab+hematoxylin:sum"
  "10:0:1:0.02:hematoxylin/dab:max"
  "10:0:1:0.02:dab/hematoxylin:max"
  "10:0:1:0.02:dab+hematoxylin:sum"
  "10:1:0:0.0002:hematoxylin/dab:max"
  "10:1:0:0.0002:dab/hematoxylin:max"
  "10:1:0:0.0002:dab+hematoxylin:sum"
  "10:1:0:0.02:hematoxylin/dab:max"
  "10:1:0:0.02:dab/hematoxylin:max"
  "10:1:0:0.02:dab+hematoxylin:sum"
  "10:1:1:0.0002:hematoxylin/dab:max"
  "10:1:1:0.0002:dab/hematoxylin:max"
  "10:1:1:0.0002:dab+hematoxylin:sum"
  "10:1:1:0.02:hematoxylin/dab:max"
  "10:1:1:0.02:dab/hematoxylin:max"
  "10:1:1:0.02:dab+hematoxylin:sum"
  "0:0:0:0:hematoxylin/dab:max"    # anchor: no PH, fields inert
  "0:0:1:0.0002:hematoxylin/dab:max"
  "0:0:1:0.0002:dab/hematoxylin:max"
  "0:0:1:0.0002:dab+hematoxylin:sum"
  "0:0:1:0.02:hematoxylin/dab:max"
  "0:0:1:0.02:dab/hematoxylin:max"
  "0:0:1:0.02:dab+hematoxylin:sum"
  "0:1:0:0.0002:hematoxylin/dab:max"
  "0:1:0:0.0002:dab/hematoxylin:max"
  "0:1:0:0.0002:dab+hematoxylin:sum"
  "0:1:0:0.02:hematoxylin/dab:max"
  "0:1:0:0.02:dab/hematoxylin:max"
  "0:1:0:0.02:dab+hematoxylin:sum"
  "0:1:1:0.0002:hematoxylin/dab:max"
  "0:1:1:0.0002:dab/hematoxylin:max"
  "0:1:1:0.0002:dab+hematoxylin:sum"
  "0:1:1:0.02:hematoxylin/dab:max"
  "0:1:1:0.02:dab/hematoxylin:max"
  "0:1:1:0.02:dab+hematoxylin:sum"
)
(( TASK_ID < ${#CELLS[@]} )) || { echo "task ${TASK_ID} is past the end of the ${#CELLS[@]}-cell grid" >&2; exit 1; }
IFS=: read -r LAMBDA_CYCLE PH_CYC PH_TRANS CELL_TOPO CELL_FIELD_B CELL_COMBINE <<< "${CELLS[$TASK_ID]}"

# -----------------------------
# Knobs: override at submit time with --export=ALL,NAME=value
# -----------------------------
FIELD_A=${FIELD_A:-hematoxylin/eosin}
FIELD_B=${FIELD_B:-$CELL_FIELD_B}
FIELD_COMBINE=${FIELD_COMBINE:-$CELL_COMBINE}
STEPS=${STEPS:-400000}              # 8 epochs of ~50k tiles; fits one 48h slot
BATCH_SIZE=${BATCH_SIZE:-1}
SAVE_STEPS=${SAVE_STEPS:-100000}    # permanent checkpoints at 100k/200k/300k/400k
IMAGE_SIZE=${IMAGE_SIZE:-256}
LAMBDA_TOPO=${LAMBDA_TOPO:-$CELL_TOPO}
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
FIELD_B_TAG=${FIELD_B//\//-}          # '/' is not safe in a directory name
RUN_NAME="${MARKER}_lc${LAMBDA_CYCLE}_lt${LAMBDA_TOPO}_cyc${PH_CYC}_trans${PH_TRANS}_${FIELD_B_TAG}"
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
