#!/bin/bash
#SBATCH --job-name=topo_decide
#SBATCH --output=logs_topo/decide_%j.out
#SBATCH --error=logs_topo/decide_%j.err

#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=clara
#SBATCH --ntasks=1

# Apply the pre-registered rule to whatever the audit array produced and write
# the recommendation. Submitted automatically by slurm/run_audit.sh with a
# dependency on the array; safe to re-run by hand at any time, since it only
# reads the JSON the array already wrote.

set -eo pipefail

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    set +u
    conda activate "${CONDA_ENV:-topocg}"
    set -u
fi

AUDIT=${AUDIT:-/work2/bz66izin-TopoCG/field_audit}
topo-audit decide --dir "$AUDIT" ${MARKERS:+--markers $MARKERS}

echo
echo "================================================================"
echo "Read ${AUDIT}/RECOMMENDATION.txt. To train with what it chose:"
echo "  AUDIT_DIR=${AUDIT} sbatch slurm/sweep_<marker>.sh"
echo "================================================================"
