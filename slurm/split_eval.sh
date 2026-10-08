#!/bin/bash
#SBATCH --job-name=topo_split_eval
#SBATCH --output=logs_topo/split_eval_%A_%a.out
#SBATCH --error=logs_topo/split_eval_%A_%a.err

#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0

# Choose the TopoCycleGAN configuration on one half of val and report it on the
# other. Three steps, in order:
#
# 1. split   (login node, seconds) -- divide each marker's valA into a selection
#    half and a test half, by slide, so no slide is in both. Written once;
#    re-splitting after a selection has been made would undo the point of it,
#    so an existing split is kept unless FORCE=1.
#
#      bash split_eval.sh split
#
# 2. select  (login node, seconds) -- rank the ph_trans configurations on the
#    sel half by mean nuclei recall over the MIST markers, pick the best, and
#    report it against CycleGAN small and DCLGAN small on the test half, with a
#    slide-bootstrap interval on the recall difference. Every number except FID
#    comes from per-tile results already on disk; it prints the FID jobs still
#    needed, with the chosen configuration's task filled in.
#
#      bash split_eval.sh select
#
# 3. fid     (GPU array) -- FID is a statistic over the whole set, so it has to
#    be recomputed on the half itself. Task N means the same model as in
#    evaluate.sh (both source _eval_arms.sh). The commands step 2 prints, e.g.:
#
#      sbatch --array=6     --export=ALL,MARKER=ER,ARM=main,HALF=test split_eval.sh fid
#      sbatch --array=13,14 --export=ALL,MARKER=ER,ARM=main,HALF=test split_eval.sh fid
#
#    then `bash split_eval.sh select` again fills the FID column.
#
# Output: <eval>/<marker>/<split>/val_test_split.csv,
#         <eval>/<marker>/<split>/<arm>/<name>/fid_<half>.csv,
#         <eval>/selection/{selection.json, sel_ranking.csv, test_report.csv,
#                           all_models_by_half.csv}

set -eo pipefail

MODE=${1:?give a mode: split, select or fid}
ROOT=${ROOT:-/work2/bz66izin-TopoCG}
EVAL_ROOT=${EVAL_ROOT:-${ROOT}/eval}
SPLIT=${SPLIT:-valA}
MARKERS=${MARKERS:-"Ki67 ER HER2 PR BCI"}
REPO_ROOT=${REPO_ROOT:-/home/sc.uni-leipzig.de/bz66izin/TopoCG_Project/TopoCycleGAN}
REPO=${REPO:-${REPO_ROOT}/slurm}
PY3=${PY3:-python3}     # split and select use the standard library only

case "$MODE" in
split)
    for MARKER in $MARKERS; do
        (
            PRED=/dev/null NAME=split      # only the tile-root lookup is wanted
            source "${REPO}/_eval_arms.sh"
            out="${EVAL_ROOT}/${MARKER}/${SPLIT}/val_test_split.csv"
            if [ -f "$out" ] && [ "${FORCE:-0}" != "1" ]; then
                echo "${MARKER}: split exists, kept (${out}); FORCE=1 replaces it"
                exit 0
            fi
            "$PY3" "${REPO_ROOT}/tools/split_val.py" \
                --tiles "$(tiles_root "$MARKER")/${MARKER}/TrainValAB/${SPLIT}" \
                --out "$out" --seed "${SEED:-0}"
        )
    done
    ;;

select)
    "$PY3" "${REPO_ROOT}/tools/select_config.py" --eval-root "$EVAL_ROOT" --split "$SPLIT" \
        ${SELECT_MARKERS:+--select-markers $SELECT_MARKERS} \
        ${REPORT_MARKERS:+--report-markers $REPORT_MARKERS}
    ;;

fid)
    : "${MARKER:?set MARKER}"
    HALF=${HALF:?set HALF=test or HALF=sel}
    source "${REPO}/_eval_arms.sh"
    EVAL_DIR="${EVAL_ROOT}/${MARKER}/${SPLIT}"
    SPLIT_CSV="${EVAL_DIR}/val_test_split.csv"
    TRUTH=${TRUTH:-$(tiles_root "$MARKER")/${MARKER}/TrainValAB/${SPLIT%A}B}
    OUT="${EVAL_DIR}/${ARM}/${NAME}"
    CSV="${OUT}/fid_${HALF}.csv"

    echo "${ARM} / ${NAME}  FID on the ${HALF} half"
    echo "  pred   ${PRED}"
    echo "  truth  ${TRUTH}"
    [ -f "$SPLIT_CSV" ] || { echo "ERROR: no split at ${SPLIT_CSV}; run 'split' first" >&2; exit 1; }
    [ -d "$PRED" ]      || { echo "  no predictions; nothing to do"; exit 0; }
    if [ -f "$CSV" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "[skip] ${CSV} exists (FORCE=1 recomputes)"; exit 0
    fi

    if command -v module >/dev/null 2>&1; then
        module purge
        module load Anaconda3/2025.06-1
        eval "$(conda shell.bash hook)"
        conda activate "${CONDA_ENV:-topocg}"
    fi

    # The half as two directories of links -- real and generated tiles whose
    # stem is in it -- so i2i-evaluate scores exactly that half.
    work=$(mktemp -d "${TMPDIR:-/tmp}/fid_${MARKER}_${HALF}_XXXX")
    trap 'rm -rf "$work"' EXIT
    mkdir -p "$work/real" "$work/fake"
    # Strip any \r first: a CSV written with \r\n endings would otherwise
    # leave "test\r" in the half column, and no tile would match.
    awk -F, -v h="$HALF" '{ sub(/\r$/, "") } NR > 1 && $3 == h { print $1 }' "$SPLIT_CSV" > "$work/stems"
    link_half() {   # <src dir> <dst dir>: link every image whose stem is listed
        local src="$1" dst="$2" f
        ls "$src" | awk 'NR == FNR { keep[$0]; next }
                         { stem = $0; sub(/\.[^.]*$/, "", stem); if (stem in keep) print }' \
                        "$work/stems" - |
        while IFS= read -r f; do ln -s "$src/$f" "$dst/$f"; done
    }
    link_half "$TRUTH" "$work/real"
    link_half "$PRED" "$work/fake"
    n_real=$(ls "$work/real" | wc -l | tr -d ' ')
    n_fake=$(ls "$work/fake" | wc -l | tr -d ' ')
    echo "  ${HALF} half: ${n_real} real, ${n_fake} generated tiles"
    [ "$n_fake" -gt 0 ] || { echo "ERROR: no generated tile is in the ${HALF} half" >&2; exit 1; }

    mkdir -p "$OUT"
    i2i-evaluate --metric fid --path_real "$work/real" --path_fake "$work/fake" \
        --device cuda --num_workers "${SLURM_CPUS_PER_TASK:-4}" --save_csv "$CSV"
    echo "Done: ${CSV}"
    ;;

*)
    echo "ERROR: unknown mode '${MODE}' -- split, select or fid" >&2
    exit 1
    ;;
esac
