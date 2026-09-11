#!/bin/bash
#SBATCH --job-name=topo_her2
#SBATCH --output=logs_topo/her2_%A_%a.out
#SBATCH --error=logs_topo/her2_%A_%a.err

#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=clara
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-26  # 27 jobs = 3 lambda_topo x 3 ph_cyc x 3 ph_trans

# MIST HER2: H&E -> HER2 IHC.
#
# Run once before the first submit:  mkdir -p logs_topo
#
#   sbatch slurm/sweep_her2.sh                          # all 27
#   sbatch --array=0-8,10-17,19-26 slurm/sweep_her2.sh  # skip duplicate baselines
#   sbatch --array=13 slurm/sweep_her2.sh               # one cell
#
# Submit from the repository root so SLURM_SUBMIT_DIR points at it, or export
# REPO=/path/to/TopoCycleGAN.

MARKER=HER2
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
source "${REPO}/slurm/_sweep_common.sh"
