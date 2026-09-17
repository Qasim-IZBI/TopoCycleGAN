#!/bin/bash
#SBATCH --job-name=topo_stains
#SBATCH --output=logs_topo/stains_%j.out
#SBATCH --error=logs_topo/stains_%j.err

#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=clara
#SBATCH --ntasks=1
# No --gres: Macenko is CPU-only.

# Step 1 of the pipeline: estimate stain vectors per marker from the TRAINING
# tiles, once, and write them to JSON. Training reads those files; nothing
# re-estimates later, because vectors that move during training would make the
# loss non-stationary.
#
# Already-present JSONs are left alone, so this is safe to re-run and is a no-op
# if field_validation_estimated has already produced them.
#
#   sbatch slurm/estimate_stains.sh
#   sbatch --export=ALL,MARKERS=ER,FORCE=1 slurm/estimate_stains.sh

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
# BCI is tiled into its own root by prepare_bci.sh, so a run that mixes it with
# MIST markers needs a per-marker root rather than one TILES for all of them.
TILES_BCI=${TILES_BCI:-/work2/bz66izin-TopoCG/BCI_tiles}
STAINS_DIR=${STAINS_DIR:-/work2/bz66izin-TopoCG/field_validation_estimated}
EST_LIMIT=${EST_LIMIT:-2000}     # tiles pooled per domain; each contributes
PIXELS_PER_TILE=${PIXELS_PER_TILE:-20000}  # at most this many pixels, so coverage
                                # scales across slides without the memory scaling
SEED=${SEED:-0}
FORCE=${FORCE:-0}

mkdir -p "$STAINS_DIR"

for MARKER in $MARKERS; do
    out="${STAINS_DIR}/stains_${MARKER}.json"
    if [ -f "$out" ] && [ "$FORCE" != "1" ]; then
        echo "[keep] ${MARKER}: ${out} already exists (FORCE=1 to redo)"
        continue
    fi
    case "$MARKER" in
        BCI) root="$TILES_BCI" ;;
        *)   root="$TILES" ;;
    esac
    TRAIN_A="${root}/${MARKER}/TrainValAB/trainA"
    TRAIN_B="${root}/${MARKER}/TrainValAB/trainB"
    if [ ! -d "$TRAIN_A" ] || [ ! -d "$TRAIN_B" ]; then
        echo "[skip] ${MARKER}: training tiles missing"
        continue
    fi
    echo
    echo "=== ${MARKER} ==="
    topo-estimate-stains --dataA "$TRAIN_A" --dataB "$TRAIN_B" \
        --out "$out" --limit "$EST_LIMIT" --seed "$SEED" \
        --pixels-per-tile "$PIXELS_PER_TILE"
done

echo
echo "stain vectors in ${STAINS_DIR}/  -- report these in the paper"
