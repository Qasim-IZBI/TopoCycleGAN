#!/bin/bash
#SBATCH --job-name=topo-sweep
#SBATCH --array=0-8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# --- adjust for your cluster -------------------------------------------- #
# #SBATCH --partition=gpu
# #SBATCH --account=<your-account>
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=<you>
# ------------------------------------------------------------------------ #

# 3x3 sweep over the two persistent-homology weights.
#   task id -> (lambda_ph_cyc, lambda_ph_trans)
#   0:(.25,.25) 1:(.25,.5) 2:(.25,1)
#   3:(.5 ,.25) 4:(.5 ,.5) 5:(.5 ,1)
#   6:(1  ,.25) 7:(1  ,.5) 8:(1  ,1)
#
# Submit:   sbatch slurm/train_sweep.sh
# One cell: sbatch --array=4 slurm/train_sweep.sh
# Baseline: sbatch --array=0 --export=ALL,LAMBDA_TOPO=0 slurm/train_sweep.sh
# H&E->SR:  sbatch --export=ALL,PRESET=he-sr slurm/train_sweep.sh

set -euo pipefail

LAMBDAS=(0.25 0.5 1)
i=${SLURM_ARRAY_TASK_ID:?run this with sbatch, not bash}
PH_CYC=${LAMBDAS[$((i / 3))]}
PH_TRANS=${LAMBDAS[$((i % 3))]}

# --- paths and knobs: override at submit time with --export ------------- #
DATA_A=${DATA_A:-$PWD/tiles/HE}
DATA_B=${DATA_B:-$PWD/tiles/IHC}
RUNS=${RUNS:-$PWD/runs}
STEPS=${STEPS:-100000}
BATCH_SIZE=${BATCH_SIZE:-4}
IMAGE_SIZE=${IMAGE_SIZE:-256}
LAMBDA_TOPO=${LAMBDA_TOPO:-1.0}
PRESET=${PRESET:-he-ki67}          # he-ki67 (H / H+DAB) or he-sr (E / DAB)
TOPO_DOWNSAMPLE=${TOPO_DOWNSAMPLE:-2}
TOPO_EVERY=${TOPO_EVERY:-1}

RUN_NAME="${PRESET}_cyc${PH_CYC}_trans${PH_TRANS}"
OUTPUT="${RUNS}/${RUN_NAME}"

# --- environment: replace with whatever your cluster uses ---------------- #
# module load cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate topo-i2i
source "${VENV:-$PWD/.venv}/bin/activate"

# Persistent homology is CPU-bound, single-threaded per image, and runs while
# the GPU idles. Leave BLAS threads alone so they don't fight the dataloader.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
NUM_WORKERS=$(( SLURM_CPUS_PER_TASK > 2 ? SLURM_CPUS_PER_TASK - 2 : 1 ))

mkdir -p "$OUTPUT"
echo "[$(date '+%Y-%m-%dT%H:%M:%S')] task ${i}: ${RUN_NAME}  host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun topo-train \
  --dataA "$DATA_A" \
  --dataB "$DATA_B" \
  --output "$OUTPUT" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --image-size "$IMAGE_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --amp \
  --lambda-topo "$LAMBDA_TOPO" \
  --lambda-ph-cyc "$PH_CYC" \
  --lambda-ph-trans "$PH_TRANS" \
  --preset "$PRESET" \
  --topo-downsample "$TOPO_DOWNSAMPLE" \
  --topo-every "$TOPO_EVERY"

echo "[$(date '+%Y-%m-%dT%H:%M:%S')] task ${i}: ${RUN_NAME} finished"
