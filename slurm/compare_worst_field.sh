#!/bin/bash
#SBATCH --job-name=topo_compare_worst
#SBATCH --output=logs_topo/compare_worstfield_%j.out
#SBATCH --error=logs_topo/compare_worstfield_%j.err

#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
# No --gres: the models already ran. This only reads their output.

# The field audit's negative control on one sheet: what the sweep made of each
# tile with the field the audit ranked BEST, beside the same cells trained on
# the field it ranked WORST.
#
# This is compare.sh with ARMS set per marker to the two ends of the audit:
#
#   PH_CYC_SPLIT=0 (default)
#     best   Outputs_<marker>                         all 13 cells
#     worst  Outputs_<marker>_worstfield_cycmerged    cells 1-12
#
#     The worst arm has no cell 0 -- the baseline touches no field, so the one
#     in the best arm stands for both. 15 + 12 panels: at three columns the
#     worst arm starts on a row boundary and each of its rows lines up under
#     the same lambda_topo decade in the best arm.
#
#   PH_CYC_SPLIT=1
#     best         Outputs_<marker>                       all 13 cells
#     best_split   Outputs_<marker>_cycsplit              ph_cyc=1 cells
#     worst_split  Outputs_<marker>_worstfield_cycsplit   ph_cyc=1 cells
#
# Infer first (infer_worst_field.sh for the worst arm, infer_sweep.sh for the
# best). A cell not inferred yet keeps its slot, labelled.
#
#   mkdir -p logs_topo
#   sbatch compare_worst_field.sh                              # 4 MIST markers
#   sbatch --export=ALL,PH_CYC_SPLIT=1 compare_worst_field.sh
#   sbatch --export=ALL,MARKERS=BCI compare_worst_field.sh
#   sbatch --export=ALL,MARKERS=VS  compare_worst_field.sh
#   MARKERS=ER SHEETS=2 bash compare_worst_field.sh            # locally, quick
#
# Every other compare.sh knob (SPLIT, INPUT, TRUTH, SUBDIR, SHEETS, SEED,
# SAMPLE, PANEL, COLS, TILES_<M>) passes straight through.
#
# Output: ${OUT}/<marker>/<tile>.png, under compare_worstfield_<variant>/ so
# these never overwrite the main sweep's sheets.

set -eo pipefail

MARKERS=${MARKERS:-"Ki67 ER HER2 PR"}
ROOT=${ROOT:-/work2/bz66izin-TopoCG}

PH_CYC_SPLIT=${PH_CYC_SPLIT:-0}
case "$PH_CYC_SPLIT" in
    0) variant="cycmerged" ;;
    1) variant="cycsplit" ;;
    *) echo "ERROR: PH_CYC_SPLIT must be 0 or 1, got '${PH_CYC_SPLIT}'" >&2; exit 1 ;;
esac

SLURM_DIR=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[ -f "${SLURM_DIR}/compare.sh" ] || SLURM_DIR="${SLURM_DIR}/slurm"

export OUT=${OUT:-${ROOT}/compare_worstfield_${variant}}

# One compare.sh run per marker, because ARMS names roots after the marker and
# compare.sh takes it as a single fixed list.
for MARKER in $MARKERS; do
    MARKER_LC=$(echo "$MARKER" | tr 'A-Z' 'a-z')
    best="${ROOT}/Outputs_${MARKER_LC}"
    worst="${ROOT}/Outputs_${MARKER_LC}_worstfield_${variant}"
    if [ "$PH_CYC_SPLIT" = "0" ]; then
        arms="best=${best} worst=${worst}"
        cells="$(seq 1 12 | tr '\n' ' ')"
    else
        arms="best=${best} best_split=${ROOT}/Outputs_${MARKER_LC}_cycsplit worst_split=${worst}"
        cells="2 3 5 6 8 9 11 12"
    fi

    MARKERS="$MARKER" ARMS="$arms" EXTRA_ARM_CELLS="$cells" REPO="$SLURM_DIR" \
        bash "${SLURM_DIR}/compare.sh"
done
