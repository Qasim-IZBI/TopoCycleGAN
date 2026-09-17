#!/bin/bash
#SBATCH --job-name=topo_estval
#SBATCH --output=logs_topo/estval_%j.out
#SBATCH --error=logs_topo/estval_%j.err

#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
# No --gres: Macenko and persistence are both CPU-only.

# Per marker: estimate stain vectors from the TRAIN tiles, then score field
# combinations on the VAL tiles using those vectors. Estimating on train and
# validating on val keeps the two disjoint.
#
# The field specs are positional (stain1/stain2), because once vectors come from
# data the names "hematoxylin"/"dab" no longer mean anything -- stain1 is simply
# whichever vector was ordered first. The grid mirrors the fixed-vector run, so
# the two are directly comparable:
#
#   stain1/stain2  ~ hematoxylin/eosin (A) or hematoxylin/dab (B)
#   stain2/stain1  ~ eosin/hematoxylin (A) or dab/hematoxylin (B)
#   stain1+stain2  ~ hematoxylin+eosin (A) or hematoxylin+dab (B)
#
#   sbatch estimate_and_validate.sh
#   sbatch --export='ALL,MARKERS=ER,WITHIN_SLIDE=1,SLIDE_REGEX=_[0-9]+_[0-9]+_r[0-9]+c[0-9]+$' \
#          estimate_and_validate.sh

set -eo pipefail

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-topocg}"
fi

MARKERS=${MARKERS:-"Ki67 ER HER2 PR"}
TILES=${TILES:-/work2/bz66izin-TopoCG/MIST_tiles}
OUT=${OUT:-/work2/bz66izin-TopoCG/field_validation_estimated}

EST_LIMIT=${EST_LIMIT:-2000}     # tiles pooled per domain; each contributes
PIXELS_PER_TILE=${PIXELS_PER_TILE:-20000}  # at most this many pixels, so coverage
                                # scales across slides without the memory scaling
PIN_SHARED=${PIN_SHARED:-0}     # 1 = force domain B stain1 := domain A stain1

LIMIT=${LIMIT:-2000}
OFFSET=${OFFSET:-0}
SAMPLE=${SAMPLE:-random}
SEED=${SEED:-0}
SHUFFLES=${SHUFFLES:-5}
IMAGE_SIZE=${IMAGE_SIZE:-256}
COMBINE=${COMBINE:-sum}
WITHIN_SLIDE=${WITHIN_SLIDE:-0}

FIELDS_A=${FIELDS_A:-"stain1/stain2 stain2/stain1 stain1+stain2"}
FIELDS_B=${FIELDS_B:-"stain1/stain2 stain2/stain1 stain1+stain2"}
DOWNSAMPLE=${DOWNSAMPLE:-"1 2 4"}
PROJECTIONS=${PROJECTIONS:-"auto birth death lifetime"}
DIMS_SETS=${DIMS_SETS:-"0 1 0,1"}

mkdir -p "$OUT"

for MARKER in $MARKERS; do
    TRAIN_A="${TILES}/${MARKER}/TrainValAB/trainA"
    TRAIN_B="${TILES}/${MARKER}/TrainValAB/trainB"
    VAL_A="${TILES}/${MARKER}/TrainValAB/valA"
    VAL_B="${TILES}/${MARKER}/TrainValAB/valB"

    for d in "$TRAIN_A" "$TRAIN_B" "$VAL_A" "$VAL_B"; do
        if [ ! -d "$d" ]; then
            echo "[skip] ${MARKER}: ${d} is missing"
            continue 2
        fi
    done

    stains="${OUT}/stains_${MARKER}.json"
    report="${OUT}/fields_${MARKER}.txt"

    echo
    echo "================ ${MARKER} ================"
    echo "--- estimating stain vectors from TRAIN ---"
    topo-estimate-stains \
        --dataA "$TRAIN_A" --dataB "$TRAIN_B" \
        --out "$stains" --limit "$EST_LIMIT" --seed "$SEED" \
        --pixels-per-tile "$PIXELS_PER_TILE" \
        $( [ "$PIN_SHARED" = "1" ] && echo --pin-shared )

    echo
    echo "--- scoring field combinations on VAL ---"
    topo-validate-fields \
        --dataA "$VAL_A" --dataB "$VAL_B" \
        --stains "$stains" \
        --field-A $FIELDS_A \
        --field-B $FIELDS_B \
        --downsample $DOWNSAMPLE \
        --topo-projection $PROJECTIONS \
        --dims-set $DIMS_SETS \
        --field-combine "$COMBINE" \
        --image-size "$IMAGE_SIZE" \
        --limit "$LIMIT" \
        --offset "$OFFSET" \
        --sample "$SAMPLE" \
        --seed "$SEED" \
        --shuffles "$SHUFFLES" \
        $( [ "$WITHIN_SLIDE" = "1" ] && echo --shuffle-within-slide ) \
        ${SLIDE_REGEX:+--slide-regex "$SLIDE_REGEX"} \
        | tee "$report"
done

echo
echo "stain vectors in ${OUT}/stains_<marker>.json  (report these in the paper)"
echo "field scores   in ${OUT}/fields_<marker>.txt"
