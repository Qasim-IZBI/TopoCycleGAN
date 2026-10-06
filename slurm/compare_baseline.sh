#!/bin/bash
#SBATCH --job-name=topo_compare_base
#SBATCH --output=logs_topo/compare_baseline_%j.out
#SBATCH --error=logs_topo/compare_baseline_%j.err

#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
# No --gres: the models already ran. This only reads their output.

# One sheet per tile for the external baselines: the H&E that went in, the
# real IHC/SR where it exists, one TopoCycleGAN cell for reference, and every
# CycleGAN and DCLGAN rung train_baseline.sh trained, side by side.
#
# At three columns that is three rows, each reading left to right in size:
#
#   input             truth              TopoCycleGAN (METHOD_CELL)
#   cyclegan small    cyclegan medium    cyclegan large
#   dclgan   small    dclgan   medium    dclgan   large
#
# so reading down a column holds the capacity fixed and changes the method.
#
# METHOD_CELL is a _grid.sh task index into the main sweep in Outputs_<marker>
# and defaults to 0, the in-repo vanilla CycleGAN at ngf=64 -- the same model
# as `cyclegan small`, trained by topo-train rather than the zoo's trainer, so
# those two panels should look alike. Set it to the cell you are reporting to
# put the method itself on the sheet.
#
# Run it after infer_baseline.sh (and infer_sweep.sh for the method panel). A
# rung not inferred yet keeps its slot, labelled.
#
#   mkdir -p logs_topo
#   sbatch compare_baseline.sh                                 # 4 MIST markers
#   sbatch --export=ALL,METHOD_CELL=6 compare_baseline.sh
#   sbatch --export=ALL,MARKERS=BCI compare_baseline.sh
#   sbatch --export=ALL,MARKERS=BCI,SPLIT=testA compare_baseline.sh
#   MARKERS=ER SHEETS=2 bash compare_baseline.sh               # locally, quick
#
# SPLIT, INPUT, TRUTH, SUBDIR, SHEETS, SEED, SAMPLE, PANEL and COLS mean what
# they do in compare.sh.
#
# Output: ${OUT}/<marker>/<tile>.png

set -eo pipefail

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-topocg}"
fi

SLURM_DIR=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[ -f "${SLURM_DIR}/_grid.sh" ] || SLURM_DIR="${SLURM_DIR}/slurm"
source "${SLURM_DIR}/_grid.sh"

MARKERS=${MARKERS:-"Ki67 ER HER2 PR"}
ROOT=${ROOT:-/work2/bz66izin-TopoCG}
TILES=${TILES:-${ROOT}/MIST_tiles}
TILES_BCI=${TILES_BCI:-${ROOT}/BCI_tiles}
TILES_VS=${TILES_VS:-${ROOT}/VS_tiles}
tiles_root() {
    local var="TILES_$1"
    echo "${!var:-$TILES}"
}

OUT=${OUT:-${ROOT}/compare_baseline}
SPLIT=${SPLIT:-valA}
SHEETS=${SHEETS:-8}
SEED=${SEED:-0}
SAMPLE=${SAMPLE:-random}
PANEL=${PANEL:-256}
COLS=${COLS:-3}
SUBDIR=${SUBDIR:-}
INPUT=${INPUT:-}
TRUTH=${TRUTH:-}
METHOD_CELL=${METHOD_CELL:-0}

MODELS="cyclegan dclgan"
SIZES="small medium large"

preds_root_of() {
    if [ "$SPLIT" = "valA" ]; then echo "$1/preds"; else echo "$1/preds_${SPLIT}"; fi
}

mkdir -p "$OUT"

for MARKER in $MARKERS; do
  (
    MARKER_LC=$(echo "$MARKER" | tr 'A-Z' 'a-z')
    root="$(tiles_root "$MARKER")"
    dir_a=${INPUT:-${root}/${MARKER}/TrainValAB/${SPLIT}}
    dir_b=${TRUTH:-${root}/${MARKER}/TrainValAB/${SPLIT%A}B}

    if [ ! -d "$dir_a" ]; then
        echo "[skip] ${MARKER}: input ${dir_a} is missing"
        exit 0
    fi

    truth_arg=()
    if [ -d "$dir_b" ]; then
        truth_arg=(--truth "$dir_b")
        truth_note="$dir_b"
    else
        truth_note="none (${dir_b} is not there)"
    fi

    # The reference panel: one cell of the main sweep.
    grid_select "$METHOD_CELL"
    if [ "$PH_CYC" = "0" ] && [ "$PH_TRANS" = "0" ]; then
        method_label="TopoCycleGAN baseline (cell 0)"
    else
        method_label="TopoCycleGAN lt ${LAMBDA_TOPO}  cyc${PH_CYC} trans${PH_TRANS}"
    fi
    method_dir="$(preds_root_of "${ROOT}/Outputs_${MARKER_LC}")/${RUN_NAME}"
    pred_args=(--pred "${method_label}=${method_dir}")
    n_ready=0 n_panels=1
    [ -d "$method_dir" ] && n_ready=1
    notes="
    ${method_label}  ${method_dir}$( [ -d "$method_dir" ] || echo "   (not inferred yet)" )"

    # Where train_baseline.sh / infer_baseline.sh put each rung.
    for model in $MODELS; do
        for size in $SIZES; do
            run="${MARKER}_${model}_${size}"
            dir="$(preds_root_of "${ROOT}/Outputs_${MARKER_LC}_${model}_${size}")/${run}"
            pred_args+=(--pred "${model} ${size}=${dir}")
            n_panels=$((n_panels + 1))
            if [ -d "$dir" ]; then
                n_ready=$((n_ready + 1))
            fi
            notes="${notes}
    ${model} ${size}  ${dir}$( [ -d "$dir" ] || echo "   (not inferred yet)" )"
        done
    done

    if [ "$n_ready" = "0" ]; then
        echo "[skip] ${MARKER}: nothing inferred yet -- run infer_baseline.sh first"
        echo "       looked in:${notes}"
        exit 0
    fi

    echo
    echo "================ ${MARKER} ================"
    echo "  input   ${dir_a}${SUBDIR:+  (only */${SUBDIR}/)}"
    echo "  truth   ${truth_note}"
    echo "  panels  ${n_ready} of ${n_panels} inferred so far:${notes}"
    echo "  sheets  ${SHEETS} tiles, ${SAMPLE} (seed ${SEED})"

    topo-compare \
        --input "$dir_a" \
        ${truth_arg[@]+"${truth_arg[@]}"} \
        "${pred_args[@]}" \
        --outdir "${OUT}/${MARKER}" \
        --tiles "$SHEETS" \
        --sample "$SAMPLE" \
        --seed "$SEED" \
        --panel-size "$PANEL" \
        --cols "$COLS" \
        ${SUBDIR:+--subdir "$SUBDIR"}
  )
done

echo
echo "sheets under ${OUT}/"
echo "To pull them back:  tar czf compare_baseline.tgz -C ${OUT} ."
