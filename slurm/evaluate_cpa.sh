#!/bin/bash
#SBATCH --job-name=topo_eval_cpa
#SBATCH --output=logs_topo/eval_cpa_%A_%a.out
#SBATCH --error=logs_topo/eval_cpa_%A_%a.err

#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-14  # ARM=main; see _eval_arms.sh for every arm's tasks

# VS task metric: collagen proportionate area (CPA) per test slide, real vs
# generated Sirius Red, through the frozen nnU-Net Dataset314_SR_light -- the
# zoo paper's CPA MAE, computed tile by tile so no stitched slide, mask or
# whole-slide logits ever reach the disk. tools/cpa_eval.py says how each step
# of the old pipeline (stitch, nnU-Net, H&E mask, fill holes) is reproduced.
#
# Task N means the same model as in evaluate.sh: both source _eval_arms.sh.
# The test set is the 20 registered UC slides as prepare_vs_test.sh lays them
# out, so SPLIT is testA here.
#
# THREE STEPS.
#
# 1. The real-SR reference, once -- every model is compared with it:
#
#      mkdir -p logs_topo
#      ref=$(sbatch --parsable --array=0 evaluate_cpa.sh ref)
#
# 2. Check it reproduces the old pipeline's whole-slide masks (CPA per slide
#    and label agreement on the thumbnails; reads each old mask once, ~1-2 GB):
#
#      sbatch --dependency=afterok:$ref --array=0 evaluate_cpa.sh check
#
# 3. The models, after the reference:
#
#      sbatch --dependency=afterok:$ref evaluate_cpa.sh                     # main 0-14
#      sbatch --dependency=afterok:$ref --export=ALL,ARM=cycsplit \
#             --array=2,3,5,6,8,9,11,12 evaluate_cpa.sh
#      sbatch --dependency=afterok:$ref --export=ALL,ARM=baselines --array=0-5 evaluate_cpa.sh
#
#    then the table:   bash evaluate_cpa.sh summary
#
# A slide already done is skipped, so a task that ran out of time resumes
# where it stopped when resubmitted. FORCE=1 redoes every slide.
#
# Cost: nnU-Net walks ~4 windows per 512 px tile, x4 with mirroring (as the
# old pipeline did). NO_MIRROR=1 is 4x faster but is not the old protocol.
#
# Output: ${EVAL_ROOT}/VS/testA/cpa_ref/                  the real SR
#         ${EVAL_ROOT}/VS/testA/<arm>/<name>/cpa_wsi.csv  per slide
#                                           cpa_tiles.csv per tile
#                                           cpa.csv       MAE, bias, Pearson,
#                                                         Spearman vs cpa_ref
#                                           thumbs/       label maps, 1/16

set -eo pipefail

MARKER=${MARKER:-VS}
SPLIT=${SPLIT:-testA}
ROOT=${ROOT:-/work2/bz66izin-TopoCG}
EVAL_ROOT=${EVAL_ROOT:-${ROOT}/eval}
EVAL_DIR="${EVAL_ROOT}/${MARKER}/${SPLIT}"
REF="${EVAL_DIR}/cpa_ref"
HE_TILES=${HE_TILES:-/work2/bz66izin-UC_project/ID_HE/no_overlap/testA/tiles/testA}
SR_TILES=${SR_TILES:-/work2/bz66izin-UC_project/ID_SR/no_overlap/testB/tiles/testB}
NNUNET_MODEL=${NNUNET_MODEL:-/work2/bz66izin-VSproject/nnunet/nnUNet_results/Dataset314_SR_light/nnUNetTrainer__nnUNetPlans__2d}
OLD_MASKS=${OLD_MASKS:-/work2/bz66izin-UC_project/psr_masks/real/psr_masks_wsi_final}
REPO_ROOT=${REPO_ROOT:-/home/sc.uni-leipzig.de/bz66izin/TopoCG_Project/TopoCycleGAN}
CPA_PY="${REPO_ROOT}/tools/cpa_eval.py"

[ -f "$CPA_PY" ] || { echo "ERROR: no tools/cpa_eval.py under REPO_ROOT=${REPO_ROOT}" >&2; exit 1; }

MODE=${1:-models}

# -----------------------------
# summary: one row per scored model, read off cpa.csv (login node)
# -----------------------------
if [ "$MODE" = "summary" ]; then
    [ -d "$EVAL_DIR" ] || { echo "nothing evaluated under ${EVAL_DIR} yet" >&2; exit 1; }
    val() { awk -F, -v k="$2" '{ sub(/\r$/, "") } $1 == k { print $2 }' "$1"; }
    printf "%-22s %-32s %9s %9s %9s %9s %7s\n" arm model "CPA MAE" bias pearson spearman slides
    for f in "$EVAL_DIR"/*/*/cpa.csv; do
        [ -f "$f" ] || continue
        d=$(dirname "$f")
        LC_ALL=C printf "%-22s %-32s %9s %9s %9s %9s %7s\n" "$(basename "$(dirname "$d")")" \
            "$(basename "$d")" "$(val "$f" cpa_mae)" "$(val "$f" cpa_bias)" \
            "$(val "$f" cpa_pearson)" "$(val "$f" cpa_spearman)" "$(val "$f" n_slides)"
    done | sort -t' ' -k1,1 -k2,2
    exit 0
fi

if [ "$MODE" = "ref" ] || [ "$MODE" = "check" ]; then
    # One job however it was submitted: the header's default array would
    # otherwise start fifteen identical ones.
    if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ "$SLURM_ARRAY_TASK_ID" != "0" ]; then
        echo "${MODE}: task ${SLURM_ARRAY_TASK_ID} has nothing to do (task 0 does it)"
        exit 0
    fi
fi

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-i2istain}"      # the env with nnunetv2 (the zoo's [cpa])
fi
PYTHON=${PYTHON:-${CONDA_PREFIX:+${CONDA_PREFIX}/bin/}python}

if [ "$MODE" = "check" ]; then
    echo "real-SR reference ${REF} against the old whole-slide masks ${OLD_MASKS}"
    "$PYTHON" "$CPA_PY" --out "$REF" --check-old "$OLD_MASKS"
    exit 0
fi

"$PYTHON" -c "import nnunetv2" 2>/dev/null || {
    echo "ERROR: ${PYTHON} cannot import nnunetv2; set CONDA_ENV or PYTHON to the env that has it" >&2
    exit 1; }
[ -f "${NNUNET_MODEL}/plans.json" ] || {
    echo "ERROR: no nnU-Net model at NNUNET_MODEL=${NNUNET_MODEL}" >&2; exit 1; }
# nnU-Net warns when these are unset; it reads nothing from them here.
export nnUNet_results=${nnUNet_results:-$(dirname "$(dirname "$NNUNET_MODEL")")}
export nnUNet_raw=${nnUNet_raw:-$nnUNet_results} nnUNet_preprocessed=${nnUNet_preprocessed:-$nnUNet_results}

common=(--he-tiles "$HE_TILES" --model "$NNUNET_MODEL"
        --workers "${SLURM_CPUS_PER_TASK:-8}" --batch "${BATCH:-8}")
if [ "${NO_MIRROR:-0}" = "1" ]; then common+=(--no-mirror); fi
if [ "${FORCE:-0}" = "1" ]; then common+=(--force); fi

if [ "$MODE" = "ref" ]; then
    echo "real-SR reference: ${SR_TILES} -> ${REF}"
    "$PYTHON" "$CPA_PY" "${common[@]}" --pred "$SR_TILES" --out "$REF"
    exit 0
fi

if [ "$MODE" != "models" ]; then
    echo "ERROR: unknown mode '${MODE}' -- ref, check, models (default) or summary" >&2
    exit 1
fi

REPO=${REPO:-${REPO_ROOT}/slurm}
source "${REPO}/_eval_arms.sh"
OUT="${EVAL_DIR}/${ARM}/${NAME}"
echo "${ARM} / ${NAME}"
echo "  pred  ${PRED}"
echo "  out   ${OUT}"
if [ ! -d "$PRED" ]; then
    echo "  no predictions yet; nothing to evaluate"
    exit 0
fi
[ -d "$REF/slides" ] || echo "NOTE: no reference under ${REF} yet -- cpa.csv waits for it"
"$PYTHON" "$CPA_PY" "${common[@]}" --pred "$PRED" --out "$OUT" --ref "$REF"
