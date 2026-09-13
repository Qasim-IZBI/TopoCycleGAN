#!/bin/bash
#SBATCH --job-name=topo_ki67
#SBATCH --output=logs_topo/ki67_%A_%a.out
#SBATCH --error=logs_topo/ki67_%A_%a.err

#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=clara
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-37  # 38 cells; see slurm/_sweep_common.sh for the grid

# MIST Ki67: H&E -> Ki67 IHC.
#
# Run once before the first submit:  mkdir -p logs_topo
#
#   sbatch slurm/sweep_ki67.sh              # all 38
#   sbatch --array=0-18 slurm/sweep_ki67.sh # lambda_cycle=10 only (critical path)
#   sbatch --array=13 slurm/sweep_ki67.sh   # one cell
#
# Submit from the repository root so SLURM_SUBMIT_DIR points at it, or export
# REPO=/path/to/TopoCycleGAN.

MARKER=Ki67
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
source "${REPO}/slurm/_sweep_common.sh"
