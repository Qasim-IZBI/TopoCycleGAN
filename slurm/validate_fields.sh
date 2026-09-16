#!/bin/bash
#SBATCH --job-name=topo_valfields
#SBATCH --output=logs_topo/valfields_%j.out
#SBATCH --error=logs_topo/valfields_%j.err

#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=clara
#SBATCH --ntasks=1
# No --gres: persistence is CPU-only, so this needs no GPU allocation.

# Score field / downsample / dims / projection combinations against MIST's
# registered pairs, for one marker or all of them. Run this BEFORE committing
# GPU time to a training sweep -- a combination that cannot separate true pairs
# from random ones here will not teach ph_trans anything.
#
#   sbatch slurm/validate_fields.sh                            # all four markers
#   sbatch --export=ALL,MARKERS=ER slurm/validate_fields.sh    # just ER
#   MARKERS=ER LIMIT=32 bash slurm/validate_fields.sh          # locally, quick
#
# Reports land in ${OUT}/fields_<marker>.txt as well as the job log.

set -eo pipefail

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    set +u
    conda activate "${CONDA_ENV:-topocg}"
    set -u
fi

MARKERS=${MARKERS:-"Ki67 ER HER2 PR"}
TILES=${TILES:-/work2/bz66izin-TopoCG/MIST_tiles}
OUT=${OUT:-/work2/bz66izin-TopoCG/field_validation}

LIMIT=${LIMIT:-128}            # 128 gives a 95% AUROC band of about +/-0.013,
                               # enough to rank field choices; 4000 would buy
                               # +/-0.003 and cost 1.6 h per marker
OFFSET=${OFFSET:-0}            # skip this many tiles: use it to re-score the
                               # leaders on a slice the screen never saw
SAMPLE=${SAMPLE:-random}       # tiles are named <slide>_r<row>c<col>, so 'head'
                               # would sample only the first few slides
SEED=${SEED:-0}                # keep FIXED between screen and re-score, or
                               # OFFSET will not carve a disjoint slice
SHUFFLES=${SHUFFLES:-5}
IMAGE_SIZE=${IMAGE_SIZE:-256}
COMBINE=${COMBINE:-sum}        # merge rule for every 'a+b' spec in the run

FIELDS_A=${FIELDS_A:-"hematoxylin/eosin eosin/hematoxylin hematoxylin+eosin"}
FIELDS_B=${FIELDS_B:-"hematoxylin/dab dab/hematoxylin hematoxylin+dab"}
DOWNSAMPLE=${DOWNSAMPLE:-"1 2 4"}
PROJECTIONS=${PROJECTIONS:-"auto birth death lifetime"}
DIMS_SETS=${DIMS_SETS:-"0 1 0,1"}

mkdir -p "$OUT"

for MARKER in $MARKERS; do
    VAL_A="${TILES}/${MARKER}/TrainValAB/valA"
    VAL_B="${TILES}/${MARKER}/TrainValAB/valB"

    if [ ! -d "$VAL_A" ] || [ ! -d "$VAL_B" ]; then
        echo "[skip] ${MARKER}: ${VAL_A} or ${VAL_B} is missing"
        continue
    fi

    report="${OUT}/fields_${MARKER}.txt"
    echo
    echo "================ ${MARKER} ================"
    echo "  A ${VAL_A}"
    echo "  B ${VAL_B}"
    echo "  -> ${report}"
    echo

    # valA/valB must be REGISTERED for this to mean anything; the tool warns
    # loudly if their filenames do not correspond.
    topo-validate-fields \
        --dataA "$VAL_A" --dataB "$VAL_B" \
        --field-A $FIELDS_A \
        --field-B $FIELDS_B \
        --downsample $DOWNSAMPLE \
        --topo-projection $PROJECTIONS \
        --dims-set $DIMS_SETS \
        --field-combine "$COMBINE" \
        --image-size "$IMAGE_SIZE" \
        --limit "$LIMIT" \
        --shuffles "$SHUFFLES" \
        --offset "$OFFSET" \
        --sample "$SAMPLE" \
        --seed "$SEED" \
        | tee "$report"
done

echo
echo "reports written under ${OUT}/"
echo "Two-stage use: screen the whole grid here, then re-score just the leading"
echo "rows on a disjoint slice, e.g."
echo "  OFFSET=${LIMIT} LIMIT=512 SEED=${SEED} FIELDS_A=<winner> FIELDS_B=<winner> \\"
echo "    DOWNSAMPLE=<n> PROJECTIONS=<p> DIMS_SETS=<d> bash slurm/validate_fields.sh"
