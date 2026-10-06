#!/bin/bash
#SBATCH --job-name=topo_eval
#SBATCH --output=logs_topo/eval_%A_%a.out
#SBATCH --error=logs_topo/eval_%A_%a.err

#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-14  # 13 TopoCycleGAN cells + 2 small baselines; see below

# FID, patch SSIM and LPIPS of one set of predictions against the ground truth,
# through the zoo's own i2i-evaluate -- the same metric code the zoo's paper
# reports, not a copy of it. Every model is scored the same way; only the
# prediction directory changes.
#
# ONE ARRAY PER DATASET. Task N of a MARKER scores:
#
#   0-12  the TopoCycleGAN sweep in Outputs_<marker>, one task per _grid.sh
#         cell -- the audit's best field across the lambda_topo / ph_cyc /
#         ph_trans grid. Task 0 is that sweep's vanilla CycleGAN cell.
#   13    cyclegan small   (Outputs_<marker>_cyclegan_small)
#   14    dclgan   small   (Outputs_<marker>_dclgan_small)
#
#   mkdir -p logs_topo
#   sbatch --export=ALL,MARKER=ER   evaluate.sh
#   sbatch --export=ALL,MARKER=Ki67 evaluate.sh
#   sbatch --export=ALL,MARKER=BCI  evaluate.sh
#   sbatch --export=ALL,MARKER=BCI,SPLIT=testA evaluate.sh
#   sbatch --export=ALL,MARKER=ER --array=13,14 evaluate.sh     # baselines only
#
# Then, once the array is done, one table per dataset (no sbatch, no GPU):
#
#   MARKER=ER bash evaluate.sh summary
#
# ANY OTHER PAIR OF DIRECTORIES -- the worst-field arm, a medium baseline, a
# test split nobody planned for -- goes through the same path with PRED, NAME
# and (optionally) TRUTH set, and lands in the same summary:
#
#   sbatch --array=0 --export=ALL,MARKER=ER,NAME=ER_dclgan_medium,\
#       PRED=/work2/bz66izin-TopoCG/Outputs_er_dclgan_medium/preds/ER_dclgan_medium \
#       evaluate.sh
#
# THE METRICS
#   fid         unpaired: InceptionV3 pool3 statistics of all predictions vs all
#               truth tiles. Lower is better.
#   patch_ssim  paired by filename: 16 random 64px patches per tile, seed fixed,
#               so every model is scored on the same patches. Higher is better.
#   lpips       paired by filename. Lower is better. NOTE: the zoo's LPIPS is
#               the unweighted VGG16 variant -- it has no learned linear
#               layers, so its values are not on the scale of the published
#               LPIPS (Zhang et al. 2018). Comparable across models here, not
#               against numbers from other papers.
#
# FID and LPIPS load ImageNet weights through torchvision. A compute node
# without internet cannot fetch them -- run this once on the login node so they
# are cached in ~/.cache/torch first:
#
#   python -c "from torchvision.models import inception_v3, vgg16, \
#       Inception_V3_Weights as I, VGG16_Weights as V; \
#       inception_v3(weights=I.DEFAULT); vgg16(weights=V.DEFAULT)"
#
# A metric whose CSV already exists is skipped; FORCE=1 recomputes. A model
# with no predictions yet exits clean.
#
# Paired metrics need a registered truth: VS has none, so for VS set
# METRICS=fid and TRUTH to a directory of real target tiles.
#
# Output: ${EVAL_ROOT}/<marker>/<split>/<name>/{fid,patch_ssim,lpips}.csv
#         ${EVAL_ROOT}/<marker>/<split>/summary.csv   (from `summary`)

set -eo pipefail

: "${MARKER:?set MARKER, e.g. --export=ALL,MARKER=ER}"

ROOT=${ROOT:-/work2/bz66izin-TopoCG}
EVAL_ROOT=${EVAL_ROOT:-${ROOT}/eval}
SPLIT=${SPLIT:-valA}
METRICS=${METRICS:-"fid patch_ssim lpips"}
MARKER_LC=$(echo "$MARKER" | tr 'A-Z' 'a-z')
EVAL_DIR="${EVAL_ROOT}/${MARKER}/${SPLIT}"

# -----------------------------
# summary: one row per scored model, read off the CSVs i2i-evaluate wrote
# -----------------------------
if [ "${1:-}" = "summary" ]; then
    if [ ! -d "$EVAL_DIR" ]; then
        echo "nothing evaluated under ${EVAL_DIR} yet" >&2
        exit 1
    fi
    # FID's CSV is metric,value,n_real,n_fake; the paired metrics end on a
    # MEAN row. Missing ones print as "-" so a half-finished array still reads.
    value_of() {
        local csv="$1"
        [ -f "$csv" ] || { echo "-"; return; }
        case "$csv" in
            */fid.csv) awk -F, 'NR==2 { print $2 }' "$csv" ;;
            *)         awk -F, '$1=="MEAN" { print $2 }' "$csv" ;;
        esac
    }
    out="${EVAL_DIR}/summary.csv"
    echo "name,fid,patch_ssim,lpips" > "$out"
    for d in "${EVAL_DIR}"/*/; do
        name=$(basename "$d")
        echo "${name},$(value_of "${d}fid.csv"),$(value_of "${d}patch_ssim.csv"),$(value_of "${d}lpips.csv")" >> "$out"
    done
    echo "${MARKER} / ${SPLIT}   (fid lower, patch_ssim higher, lpips lower is better)"
    column -s, -t < "$out"
    echo
    echo "written to ${out}"
    exit 0
fi

# -----------------------------
# Which predictions this task scores
# -----------------------------
preds_root_of() {
    if [ "$SPLIT" = "valA" ]; then echo "$1/preds"; else echo "$1/preds_${SPLIT}"; fi
}

if [ -n "${PRED:-}" ]; then
    NAME=${NAME:?PRED is set -- set NAME too, it names the output directory}
else
    TASK_ID=${SLURM_ARRAY_TASK_ID:?submit with sbatch, or set PRED and NAME}
    SLURM_DIR=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
    [ -f "${SLURM_DIR}/_grid.sh" ] || SLURM_DIR="${SLURM_DIR}/slurm"
    source "${SLURM_DIR}/_grid.sh"
    n_cells=${#CELLS[@]}

    if (( TASK_ID < n_cells )); then
        grid_select "$TASK_ID"
        NAME="$RUN_NAME"
        PRED="$(preds_root_of "${ROOT}/Outputs_${MARKER_LC}")/${RUN_NAME}"
    else
        BASELINES=( "cyclegan:small" "dclgan:small" )
        i=$(( TASK_ID - n_cells ))
        if (( i >= ${#BASELINES[@]} )); then
            echo "ERROR: task ${TASK_ID} is past the last baseline (task $(( n_cells + ${#BASELINES[@]} - 1 )))" >&2
            exit 1
        fi
        model=${BASELINES[$i]%%:*}
        size=${BASELINES[$i]#*:}
        NAME="${MARKER}_${model}_${size}"
        PRED="$(preds_root_of "${ROOT}/Outputs_${MARKER_LC}_${model}_${size}")/${NAME}"
    fi
fi

# Truth: the registered partner split -- valA -> valB, testA -> testB -- under
# the same per-marker tile root inference read from.
TILES=${TILES:-${ROOT}/MIST_tiles}
TILES_BCI=${TILES_BCI:-${ROOT}/BCI_tiles}
TILES_VS=${TILES_VS:-${ROOT}/VS_tiles}
tiles_root() {
    local var="TILES_$1"
    echo "${!var:-$TILES}"
}
TRUTH=${TRUTH:-$(tiles_root "$MARKER")/${MARKER}/TrainValAB/${SPLIT%A}B}

OUT="${EVAL_DIR}/${NAME}"

echo "${NAME}"
echo "  pred     ${PRED}"
echo "  truth    ${TRUTH}"
echo "  metrics  ${METRICS}"
echo "  output   ${OUT}"

if [ ! -d "$PRED" ]; then
    echo "  no predictions yet; nothing to evaluate"
    exit 0
fi
if [ ! -d "$TRUTH" ]; then
    echo "ERROR: no truth directory at ${TRUTH}" >&2
    echo "  set TRUTH=/path/to/real/target/tiles" >&2
    exit 1
fi

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-topocg}"
fi

mkdir -p "$OUT"
CPUS=${SLURM_CPUS_PER_TASK:-4}

# Tissue filtering is the zoo's, off by default; MIN_TISSUE=0.1 turns it on
# for tilings that carry masks.
tissue_args=()
[ -n "${MIN_TISSUE:-}" ] && tissue_args+=(--min_tissue_fraction "$MIN_TISSUE")
[ -n "${MASK_DIR:-}" ]   && tissue_args+=(--mask_dir "$MASK_DIR")

for metric in $METRICS; do
    csv="${OUT}/${metric}.csv"
    if [ -f "$csv" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "[skip] ${metric}: ${csv} exists (FORCE=1 recomputes)"
        continue
    fi
    echo
    echo "---- ${metric} ----"
    i2i-evaluate --metric "$metric" \
                 --path_real "$TRUTH" --path_fake "$PRED" \
                 --device cuda --num_workers "$CPUS" \
                 ${tissue_args[@]+"${tissue_args[@]}"} \
                 --save_csv "$csv"
done

echo
echo "Done: ${NAME}"
