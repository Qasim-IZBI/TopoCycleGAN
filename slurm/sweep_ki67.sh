#!/bin/bash
#SBATCH --job-name=topo_ki67
#SBATCH --output=logs_topo/ki67_%A_%a.out
#SBATCH --error=logs_topo/ki67_%A_%a.err

#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-12  # 13 cells; see _grid.sh for the grid

# MIST Ki67: H&E -> Ki67 IHC.
#
# Run once before the first submit:  mkdir -p logs_topo
#
#   sbatch sweep_ki67.sh            # all 13 cells
#   sbatch --array=0 sweep_ki67.sh  # the vanilla CycleGAN baseline only
#
# Needs stain vectors: run estimate_stains.sh first, or use
# run_pipeline.sh which chains the two with a dependency.
#
# Submit from the directory holding these scripts (or from the repo root --
# both resolve), or export REPO=/path/to/the/scripts.

MARKER=Ki67
# Sibling scripts live next to this one. SLURM runs a batch script from a copy
# in its spool directory, so BASH_SOURCE cannot locate them -- the directory
# sbatch was called from can. The fallback keeps a submit from the repo root
# working too.
SLURM_DIR=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[ -f "${SLURM_DIR}/_sweep_common.sh" ] || SLURM_DIR="${SLURM_DIR}/slurm"
source "${SLURM_DIR}/_sweep_common.sh"
