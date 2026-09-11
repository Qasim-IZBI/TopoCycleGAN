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
#SBATCH --array=0-26  # 27 jobs = 3 lambda_topo x 3 ph_cyc x 3 ph_trans

# NOTE: SLURM will not create logs_topo/ for you -- `mkdir -p logs_topo` once
# before the first sbatch, or the jobs fail with no output to tell you why.
#
# Submit:   sbatch slurm/train_sweep.sh
# One cell: sbatch --array=13 slurm/train_sweep.sh
# Skip dups: sbatch --array=0-8,10-17,19-26 slurm/train_sweep.sh
# Baseline: sbatch --array=0 slurm/train_sweep.sh  (ph_cyc=0, ph_trans=0)
# H&E->SR:  sbatch --export=ALL,PRESET=he-sr slurm/train_sweep.sh

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
# Axes: 3 x 3 x 3
#   lambda_topo  {2e-4, 2e-3, 2e-2}  overall scale  (slowest)
#   ph_cyc       {0, 0.5, 1}         cycle-topology weight
#   ph_trans     {0, 0.5, 1}         translation-topology weight (fastest)
#
# 2e-4 puts the PH gradient on a par with the cycle gradient, 2e-2 leaves it
# ~80x larger; the old 0.25-1 range sat at 1000-4000x, i.e. entirely saturated.
# A 0 on either weight switches that family off, so the grid contains the
# ablations for both redundancy questions.
#
# NOTE: cells with ph_cyc=0 AND ph_trans=0 have no topological term at all, so
# tasks 0, 9 and 18 are the same CycleGAN baseline three times over. Run one and
# skip the others with e.g.  --array=0-8,10-17,19-26
# -----------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID:?submit with sbatch -- there is no array index to sweep over}

LAMBDA_TOPOS=(0.0002 0.002 0.02)
PH_WEIGHTS=(0 0.5 1)

TOPO_ID=$(( TASK_ID / 9 ))
CYC_ID=$(( (TASK_ID / 3) % 3 ))
TRANS_ID=$(( TASK_ID % 3 ))

PH_CYC=${PH_WEIGHTS[$CYC_ID]}
PH_TRANS=${PH_WEIGHTS[$TRANS_ID]}

# -----------------------------
# Knobs: override at submit time with --export=ALL,NAME=value
# -----------------------------
PRESET=${PRESET:-he-ihc}            # he-ihc (H / H+DAB) or he-sr (E / DAB)
STEPS=${STEPS:-750000}
BATCH_SIZE=${BATCH_SIZE:-1}
IMAGE_SIZE=${IMAGE_SIZE:-256}
LAMBDA_TOPO=${LAMBDA_TOPO:-${LAMBDA_TOPOS[$TOPO_ID]}}
TOPO_DOWNSAMPLE=${TOPO_DOWNSAMPLE:-1}
TOPO_EVERY=${TOPO_EVERY:-2}
TOPO_START=${TOPO_START:-100000}    # let the GAN find its footing first
TOPO_WARMUP=${TOPO_WARMUP:-50000}   # then ramp the PH weight in over 50k steps,
                                    # keeping the ramp at half the start step as
                                    # before (was 5k after 10k)

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
RUN_NAME="${PRESET}_lt${LAMBDA_TOPO}_cyc${PH_CYC}_trans${PH_TRANS}"
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
    --preset "${PRESET}" \
    --lambda-topo "${LAMBDA_TOPO}" \
    --lambda-ph-cyc "${PH_CYC}" \
    --lambda-ph-trans "${PH_TRANS}" \
    --topo-downsample "${TOPO_DOWNSAMPLE}" \
    --topo-every "${TOPO_EVERY}" \
    --topo-start-step "${TOPO_START}" \
    --topo-warmup-steps "${TOPO_WARMUP}"

echo "Done: ${RUN_NAME} finished successfully."
