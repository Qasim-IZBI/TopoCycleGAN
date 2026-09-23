#!/bin/bash
#SBATCH --job-name=topo_compare
#SBATCH --output=logs_topo/compare_%j.out
#SBATCH --error=logs_topo/compare_%j.err

#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
# No --gres: the models already ran. This only reads their output.

# One sheet per tile: the H&E that went in, the real IHC/SR where it exists,
# and what all 13 sweep cells made of it, side by side.
#
# Run it after infer_sweep.sh. The cells come from _grid.sh, the same list
# training and inference resolve, so a cell cannot be left off the sheet by
# accident -- and one that has not been inferred yet keeps its slot, labelled,
# rather than shifting every panel after it.
#
#   mkdir -p logs_topo                      # SLURM will not create it for you
#   sbatch compare.sh                                   # 4 MIST markers
#   sbatch --export=ALL,MARKERS=BCI compare.sh
#   sbatch --export=ALL,MARKERS="Ki67 ER",SHEETS=16 compare.sh
#   MARKERS=ER SHEETS=2 bash compare.sh                 # locally, quick
#
#   # BCI's held-out test split, which infer_sweep.sh wrote to preds_testA/:
#   sbatch --export=ALL,MARKERS=BCI,SPLIT=testA compare.sh
#
#   # the VS H&E test tiles, which are not in a flat TrainValAB layout and
#   # have no registered partner -- the ground-truth panel says so:
#   sbatch --export=ALL,MARKERS=VS,SUBDIR=images,\
#       INPUT=/work2/bz66izin-UC_project/ID_HE/no_overlap/testA/tiles/testA \
#       compare.sh
#
# SPLIT names the split directory, as in infer_sweep.sh (valA, testA) rather
# than inspect.sh's bare prefix -- these sheets read the predictions, so they
# follow the inference side's spelling. The truth directory is that name with
# its trailing A swapped for B, and is used only if it is there.
#
# Output: ${OUT}/<marker>/<tile>.png

set -eo pipefail

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-topocg}"
fi

# Sibling scripts live next to this one. SLURM runs a batch script from a copy
# in its spool directory, so BASH_SOURCE cannot locate them -- the directory
# sbatch was called from can. The fallback keeps a submit from the repo root
# working too.
SLURM_DIR=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[ -f "${SLURM_DIR}/_grid.sh" ] || SLURM_DIR="${SLURM_DIR}/slurm"
source "${SLURM_DIR}/_grid.sh"

MARKERS=${MARKERS:-"Ki67 ER HER2 PR"}
# Tile root per marker: TILES_<MARKER> wins if it is set, else TILES. The same
# lookup infer_sweep.sh and inspect.sh use.
TILES=${TILES:-/work2/bz66izin-TopoCG/MIST_tiles}
TILES_BCI=${TILES_BCI:-/work2/bz66izin-TopoCG/BCI_tiles}
TILES_VS=${TILES_VS:-/work2/bz66izin-TopoCG/VS_tiles}
tiles_root() {
    local var="TILES_$1"
    echo "${!var:-$TILES}"
}

OUT=${OUT:-/work2/bz66izin-TopoCG/compare}
SPLIT=${SPLIT:-valA}
# Sheets per marker. Named SHEETS, not TILES: TILES is the tile ROOT here and
# in inspect.sh, and quietly overloading it would repoint the data.
SHEETS=${SHEETS:-8}
SEED=${SEED:-0}
SAMPLE=${SAMPLE:-random}
PANEL=${PANEL:-256}
COLS=${COLS:-3}
SUBDIR=${SUBDIR:-}
# Set INPUT to sheet a directory that was never linked into TrainValAB. The
# truth is then whatever TRUTH says, if anything.
INPUT=${INPUT:-}
TRUTH=${TRUTH:-}

mkdir -p "$OUT"

for MARKER in $MARKERS; do
  (
    MARKER_LC=$(echo "$MARKER" | tr 'A-Z' 'a-z')
    BASE=${BASE:-/work2/bz66izin-TopoCG/Outputs_${MARKER_LC}}
    root="$(tiles_root "$MARKER")"

    dir_a=${INPUT:-${root}/${MARKER}/TrainValAB/${SPLIT}}
    # valA -> valB, testA -> testB. Only used if it is actually there: the VS
    # test tiles have no registered partner, and a sheet without a truth panel
    # is still worth having.
    dir_b=${TRUTH:-${root}/${MARKER}/TrainValAB/${SPLIT%A}B}

    # Where infer_sweep.sh put this split's predictions.
    if [ "$SPLIT" = "valA" ]; then
        preds_root="${BASE}/preds"
    else
        preds_root="${BASE}/preds_${SPLIT}"
    fi

    if [ ! -d "$dir_a" ]; then
        echo "[skip] ${MARKER}: input ${dir_a} is missing"
        exit 0
    fi
    if [ ! -d "$preds_root" ]; then
        echo "[skip] ${MARKER}: no predictions under ${preds_root} -- run infer_sweep.sh first"
        exit 0
    fi

    truth_arg=()
    if [ -d "$dir_b" ]; then
        truth_arg=(--truth "$dir_b")
        truth_note="$dir_b"
    else
        truth_note="none (${dir_b} is not there)"
    fi

    # One --pred per cell, in grid order. At three columns that puts the input,
    # the truth and the baseline on the first row, then one lambda_topo decade
    # per row -- so reading down a column holds the PH terms fixed and sweeps
    # the weight, and reading across a row does the opposite.
    pred_args=()
    n_cells=0 n_ready=0
    for task in $(seq 0 12); do
        grid_select "$task"
        if [ "$PH_CYC" = "0" ] && [ "$PH_TRANS" = "0" ]; then
            label="baseline (CycleGAN)"
        else
            # No "=" in the label: --pred splits on the last one, and a
            # caption is easier to read without it anyway.
            label="lt ${LAMBDA_TOPO}  cyc${PH_CYC} trans${PH_TRANS}"
        fi
        pred_args+=(--pred "${label}=${preds_root}/${RUN_NAME}")
        n_cells=$((n_cells + 1))
        [ -d "${preds_root}/${RUN_NAME}" ] && n_ready=$((n_ready + 1))
    done

    echo
    echo "================ ${MARKER} ================"
    echo "  input   ${dir_a}${SUBDIR:+  (only */${SUBDIR}/)}"
    echo "  truth   ${truth_note}"
    echo "  preds   ${preds_root}"
    echo "  cells   ${n_ready} of ${n_cells} inferred so far"
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
echo "To pull them back:  tar czf compare.tgz -C ${OUT} ."
