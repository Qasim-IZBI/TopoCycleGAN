#!/bin/bash
#SBATCH --job-name=topo_sweep
#SBATCH --output=logs_topo/topo_%A_%a.out
#SBATCH --error=logs_topo/topo_%A_%a.err

#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=clara
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-8   # 9 jobs = 3 lambda_ph_cyc x 3 lambda_ph_trans

# NOTE: SLURM will not create logs_topo/ for you -- `mkdir -p logs_topo` once
# before the first sbatch, or the jobs fail with no output to tell you why.
#
# Submit:   sbatch slurm/train_sweep.sh
# One cell: sbatch --array=4 slurm/train_sweep.sh
# Baseline: sbatch --array=0 --export=ALL,LAMBDA_TOPO=0 slurm/train_sweep.sh
# H&E->SR:  sbatch --export=ALL,PRESET=he-sr slurm/train_sweep.sh

set -euo pipefail

module purge
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-topocg}"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi || true   # diagnostics must never kill a 48h job under `set -e`

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Persistent homology is CPU-bound and single-threaded per image, and runs while
# the GPU idles. Keep BLAS to one thread so it does not fight the dataloader.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
NUM_WORKERS=$(( SLURM_CPUS_PER_TASK > 2 ? SLURM_CPUS_PER_TASK - 2 : 1 ))

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
# Axes
# lambda_ph_trans varies fastest, lambda_ph_cyc slowest.
#
# Job layout:
#   0: cyc=0.25 trans=0.25    5: cyc=0.5  trans=1
#   1: cyc=0.25 trans=0.5     6: cyc=1    trans=0.25
#   2: cyc=0.25 trans=1       7: cyc=1    trans=0.5
#   3: cyc=0.5  trans=0.25    8: cyc=1    trans=1
#   4: cyc=0.5  trans=0.5
# -----------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID}

LAMBDAS=(0.25 0.5 1)
PH_CYC=${LAMBDAS[$(( TASK_ID / 3 ))]}
PH_TRANS=${LAMBDAS[$(( TASK_ID % 3 ))]}

# -----------------------------
# Knobs: override at submit time with --export=ALL,NAME=value
# -----------------------------
PRESET=${PRESET:-he-ihc}            # he-ihc (H / H+DAB) or he-sr (E / DAB)
STEPS=${STEPS:-750000}
BATCH_SIZE=${BATCH_SIZE:-4}
IMAGE_SIZE=${IMAGE_SIZE:-256}
LAMBDA_TOPO=${LAMBDA_TOPO:-1.0}
TOPO_DOWNSAMPLE=${TOPO_DOWNSAMPLE:-1}
TOPO_EVERY=${TOPO_EVERY:-1}
TOPO_START=${TOPO_START:-100000}    # let the GAN find its footing first
TOPO_WARMUP=${TOPO_WARMUP:-5000}    # then ramp the PH weight in over 5k steps

echo "TASK_ID=${TASK_ID}"
echo "PRESET=${PRESET}"
echo "LAMBDA_PH_CYC=${PH_CYC}  LAMBDA_PH_TRANS=${PH_TRANS}  LAMBDA_TOPO=${LAMBDA_TOPO}"

# -----------------------------
# Paths
# -----------------------------
# Tiles as produced by topo-crop, which mirrors trainA/trainB/valA/valB:
#   topo-crop --input  /work2/bz66izin-TopoCG/MIST/Ki67/TrainValAB/ \
#             --output /work2/bz66izin-TopoCG/MIST_tiles/Ki67/TrainValAB/ \
#             --tile_size 512 --resize_to 256
DATA_DIR=${DATA_DIR:-/work2/bz66izin-TopoCG/MIST_tiles/Ki67/TrainValAB}
DATA_A=${DATA_A:-${DATA_DIR}/trainA/}
DATA_B=${DATA_B:-${DATA_DIR}/trainB/}

# valA/valB sit alongside these; nothing in the training loop reads them yet.

BASE=${BASE:-/work2/bz66izin-TopoCG/Outputs_topo}
RUN_NAME="${PRESET}_cyc${PH_CYC}_trans${PH_TRANS}"
OUTPUT="${BASE}/results/${RUN_NAME}"

mkdir -p "${OUTPUT}"
echo "Output directory: ${OUTPUT}"

# -----------------------------
# Training
# Resumes automatically from OUTPUT if checkpoints are already there, and the
# PH schedule resumes with it (the step counter is a checkpointed buffer).
# -----------------------------
run_cmd topo-train \
    --dataA "${DATA_A}" \
    --dataB "${DATA_B}" \
    --output "${OUTPUT}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --image-size "${IMAGE_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --amp \
    --preset "${PRESET}" \
    --lambda-topo "${LAMBDA_TOPO}" \
    --lambda-ph-cyc "${PH_CYC}" \
    --lambda-ph-trans "${PH_TRANS}" \
    --topo-downsample "${TOPO_DOWNSAMPLE}" \
    --topo-every "${TOPO_EVERY}" \
    --topo-start-step "${TOPO_START}" \
    --topo-warmup-steps "${TOPO_WARMUP}"

echo "Done: ${RUN_NAME} finished successfully."
