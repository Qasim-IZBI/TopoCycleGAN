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
# and what every sweep cell made of it, side by side.
#
# Run it after infer_sweep.sh. The cells come from _grid.sh, the same list
# training and inference resolve, so a cell cannot be left off the sheet by
# accident -- and one that has not been inferred yet keeps its slot, labelled,
# rather than shifting every panel after it.
#
# ARMS is how the two cycle-topology sweeps land on one sheet: a space-
# separated list of <label>=<output root>, taken in order. The default is the
# merged-field sweep in Outputs_<marker> followed by the per-channel one in
# Outputs_<marker>_cycsplit, which is 23 panels -- and at three columns the
# first arm fills exactly five rows, so the second starts on a row boundary.
#
#   ARMS="merged=/work2/.../Outputs_er" compare.sh   # one arm, as before
#
# Only the FIRST arm contributes all 13 cells. Every later one contributes
# EXTRA_ARM_CELLS, which defaults to the 8 cells where ph_cyc is on -- the
# split arm's other five are bit-for-bit the merged arm's, so putting them up
# twice would just pad the sheet. Set EXTRA_ARM_CELLS="$(seq 0 12)" to show a
# later arm in full.
#
# The split arm has to be INFERRED into its own root before any of it can be
# sheeted; infer_sweep.sh defaults BASE to Outputs_<marker>, so it needs the
# root passing explicitly:
#
#   sbatch --export=ALL,MARKER=ER,BASE=/work2/bz66izin-TopoCG/Outputs_er_cycsplit \
#          --array=2,3,5,6,8,9,11,12 infer_sweep.sh
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
# Cells taken from each arm after the first. ph_cyc_split cannot change a cell
# whose ph_cyc is 0, so those five are shared with the first arm rather than
# missing from this one.
EXTRA_ARM_CELLS=${EXTRA_ARM_CELLS:-"2 3 5 6 8 9 11 12"}
# Set INPUT to sheet a directory that was never linked into TrainValAB. The
# truth is then whatever TRUTH says, if anything.
INPUT=${INPUT:-}
TRUTH=${TRUTH:-}

mkdir -p "$OUT"

for MARKER in $MARKERS; do
  (
    MARKER_LC=$(echo "$MARKER" | tr 'A-Z' 'a-z')
    BASE=${BASE:-/work2/bz66izin-TopoCG/Outputs_${MARKER_LC}}
    SPLIT_BASE=${SPLIT_BASE:-/work2/bz66izin-TopoCG/Outputs_${MARKER_LC}_cycsplit}
    # Resolved here, inside the per-marker subshell, because both roots are
    # named after the marker.
    arms=${ARMS:-"merged=${BASE} split=${SPLIT_BASE}"}
    root="$(tiles_root "$MARKER")"

    dir_a=${INPUT:-${root}/${MARKER}/TrainValAB/${SPLIT}}
    # valA -> valB, testA -> testB. Only used if it is actually there: the VS
    # test tiles have no registered partner, and a sheet without a truth panel
    # is still worth having.
    dir_b=${TRUTH:-${root}/${MARKER}/TrainValAB/${SPLIT%A}B}

    # Where infer_sweep.sh put a given arm's predictions for this split.
    preds_root_of() {
        if [ "$SPLIT" = "valA" ]; then echo "$1/preds"; else echo "$1/preds_${SPLIT}"; fi
    }

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

    # One --pred per cell, in grid order, for each arm in turn. At three
    # columns the first arm puts the input, the truth and the baseline on row
    # one and then one lambda_topo decade per row -- so reading down a column
    # holds the PH terms fixed and sweeps the weight, and reading across a row
    # does the opposite. That is 15 panels, exactly five rows, so a second arm
    # starts on a row boundary instead of wrapping into the first one's last.
    n_arms=0
    for arm in $arms; do n_arms=$((n_arms + 1)); done

    pred_args=()
    n_cells=0 n_ready=0 arm_i=0 roots_found=0 arm_note=""
    for arm in $arms; do
        arm_label="${arm%%=*}"          # first "=" splits label from root; the
        arm_base="${arm#*=}"            # root may contain further ones
        arm_i=$((arm_i + 1))
        arm_preds="$(preds_root_of "$arm_base")"
        [ -d "$arm_preds" ] && roots_found=$((roots_found + 1))
        arm_note="${arm_note}
    ${arm_label}  ${arm_preds}$( [ -d "$arm_preds" ] || echo "   (not inferred yet)" )"

        # The first arm is the reference and shows the whole grid; a later one
        # shows only the cells it can actually differ in.
        if [ "$arm_i" = "1" ]; then cells=$(seq 0 12); else cells="$EXTRA_ARM_CELLS"; fi

        for task in $cells; do
            grid_select "$task"
            if [ "$PH_CYC" = "0" ] && [ "$PH_TRANS" = "0" ]; then
                label="baseline (CycleGAN)"
            else
                # No "=" in the label: --pred splits on the last one, and a
                # caption is easier to read without it anyway.
                label="lt ${LAMBDA_TOPO}  cyc${PH_CYC} trans${PH_TRANS}"
            fi
            # Name the arm on the panel only when there is more than one, so a
            # single-arm sheet reads exactly as it did before.
            [ "$n_arms" = "1" ] || label="[${arm_label}] ${label}"
            pred_args+=(--pred "${label}=${arm_preds}/${RUN_NAME}")
            n_cells=$((n_cells + 1))
            [ -d "${arm_preds}/${RUN_NAME}" ] && n_ready=$((n_ready + 1))
        done
    done

    # Every arm missing means nothing has been inferred for this marker at all.
    # One missing arm is ordinary -- its panels say so on the sheet.
    if [ "$roots_found" = "0" ]; then
        echo "[skip] ${MARKER}: no predictions under any arm -- run infer_sweep.sh first"
        echo "       looked in:${arm_note}"
        exit 0
    fi

    echo
    echo "================ ${MARKER} ================"
    echo "  input   ${dir_a}${SUBDIR:+  (only */${SUBDIR}/)}"
    echo "  truth   ${truth_note}"
    echo "  arms    ${n_arms}:${arm_note}"
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
