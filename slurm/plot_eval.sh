#!/bin/bash
#SBATCH --job-name=topo_plot_eval
#SBATCH --output=logs_topo/plot_eval_%j.out
#SBATCH --error=logs_topo/plot_eval_%j.err

#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
# No --gres: this only reads the CSVs evaluate.sh wrote.

# One figure per marker and metric -- FID, patch SSIM, LPIPS each on its own --
# from whatever evaluate.sh has scored so far. Each figure puts the four sweep
# arms (best/worst field x merged/split ph_cyc) and the baseline size ladder
# side by side on one shared y-axis. See topo_i2i/plot_eval.py for the layout.
#
#   mkdir -p logs_topo
#   sbatch plot_eval.sh                                   # 5 markers, valA
#   sbatch --export=ALL,MARKERS="ER PR" plot_eval.sh
#   sbatch --export=ALL,MARKERS=BCI,SPLIT=testA plot_eval.sh
#   bash plot_eval.sh                                     # on the login node: seconds
#
# Output: ${EVAL_ROOT}/<marker>/<split>/plots/{fid,patch_ssim,lpips}.{png,pdf}

set -eo pipefail

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-topocg}"
fi

ROOT=${ROOT:-/work2/bz66izin-TopoCG}
EVAL_ROOT=${EVAL_ROOT:-${ROOT}/eval}
MARKERS=${MARKERS:-"Ki67 ER HER2 PR BCI"}
SPLIT=${SPLIT:-valA}

# python -m rather than an entry point: an editable install picks up a new
# module at once, but a new console script only after a reinstall.
python -m topo_i2i.plot_eval --eval-root "$EVAL_ROOT" --markers $MARKERS --split "$SPLIT"

echo
echo "To pull them back:"
echo "  rsync -av --include='*/' --include='plots/*' --exclude='*' <cluster>:${EVAL_ROOT}/ ./eval_plots/"
