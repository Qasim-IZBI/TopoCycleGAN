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
#SBATCH --array=0-14  # ARM=main: 13 TopoCycleGAN cells + 2 small baselines

# FID, patch SSIM and LPIPS of one set of predictions against the ground truth,
# through the zoo's own i2i-evaluate -- the same metric code the zoo's paper
# reports, not a copy of it. Every model is scored the same way; only the
# prediction directory changes.
#
# ONE ARRAY PER DATASET AND ARM. ARM picks which trained models task N means:
#
#   ARM=main (default)       --array=0-14
#     0-12  the TopoCycleGAN sweep in Outputs_<marker>, one task per _grid.sh
#           cell: the audit's best field, merged ph_cyc. Task 0 is that
#           sweep's vanilla CycleGAN cell.
#     13    cyclegan small   (Outputs_<marker>_cyclegan_small)
#     14    dclgan   small   (Outputs_<marker>_dclgan_small)
#
#   ARM=cycsplit             --array=2,3,5,6,8,9,11,12
#     the per-channel ph_cyc sweep in Outputs_<marker>_cycsplit. Only the
#     ph_cyc=1 cells were trained; the others are main's, bit for bit.
#
#   ARM=worstfield_cycmerged --array=1-12
#   ARM=worstfield_cycsplit  --array=2,3,5,6,8,9,11,12
#     the field audit's negative control, Outputs_<marker>_worstfield_<variant>.
#     No cell 0: the baseline touches no field, so main's stands for both.
#
#   ARM=baselines            --array=0-5
#     0-2 cyclegan small/medium/large, 3-5 dclgan small/medium/large
#     (Outputs_<marker>_<model>_<size>), as train_baseline.sh named them.
#
#   ARM=identity             --array=0
#     the H&E input itself scored as the prediction -- the "do nothing" row.
#
# The arms are defined in _eval_arms.sh, which evaluate_nuclei.sh sources too,
# so task N is the same model in both and their results share a directory.
#
# A task its arm did not train exits clean with a note, so --array=0-12 is
# also safe for any arm -- the lists above just avoid the empty jobs.
#
#   mkdir -p logs_topo
#   sbatch --export=ALL,MARKER=ER evaluate.sh
#   sbatch --export=ALL,MARKER=ER,ARM=cycsplit --array=2,3,5,6,8,9,11,12 evaluate.sh
#   sbatch --export=ALL,MARKER=ER,ARM=worstfield_cycmerged --array=1-12 evaluate.sh
#   sbatch --export=ALL,MARKER=ER,ARM=worstfield_cycsplit \
#          --array=2,3,5,6,8,9,11,12 evaluate.sh
#   sbatch --export=ALL,MARKER=ER,ARM=baselines --array=0-5 evaluate.sh
#   sbatch --export=ALL,MARKER=BCI,SPLIT=testA evaluate.sh
#
# Then, once the arrays are done, one table per dataset with every arm in it
# (no sbatch, no GPU):
#
#   MARKER=ER bash evaluate.sh summary
#
# ANY OTHER PAIR OF DIRECTORIES goes through the same path with PRED and NAME
# set (and TRUTH if it is not the registered partner split). It is filed under
# ARM=custom unless ARM says otherwise:
#
#   sbatch --array=0 --export=ALL,MARKER=ER,NAME=ER_mything,PRED=/path/to/preds \
#       evaluate.sh
#
# Every arm reuses the grid's run names (ER_lt0.002_cyc1_trans1 exists in main,
# cycsplit and both worstfield arms), which is why results are filed per arm:
# under one directory a second arm would find the first one's CSVs and skip.
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
# Output: ${EVAL_ROOT}/<marker>/<split>/<arm>/<name>/{fid,patch_ssim,lpips}.csv
#         ${EVAL_ROOT}/<marker>/<split>/summary.csv   (from `summary`; it also
#         carries evaluate_nuclei.sh's recall / precision / F1 where they exist)

set -eo pipefail

: "${MARKER:?set MARKER, e.g. --export=ALL,MARKER=ER}"

ROOT=${ROOT:-/work2/bz66izin-TopoCG}
EVAL_ROOT=${EVAL_ROOT:-${ROOT}/eval}
SPLIT=${SPLIT:-valA}
METRICS=${METRICS:-"fid patch_ssim lpips"}
MARKER_LC=$(echo "$MARKER" | tr 'A-Z' 'a-z')
EVAL_DIR="${EVAL_ROOT}/${MARKER}/${SPLIT}"

# -----------------------------
# summary: one row per scored model, every arm, read off i2i-evaluate's CSVs
# -----------------------------
if [ "${1:-}" = "summary" ]; then
    if [ ! -d "$EVAL_DIR" ]; then
        echo "nothing evaluated under ${EVAL_DIR} yet" >&2
        exit 1
    fi
    # Results from before the per-arm layout sit directly under the split.
    # They are all ARM=main; say how to file them rather than mislabel them.
    shopt -s nullglob
    legacy=( "${EVAL_DIR}"/*/fid.csv "${EVAL_DIR}"/*/patch_ssim.csv "${EVAL_DIR}"/*/lpips.csv )
    if (( ${#legacy[@]} )); then
        echo "NOTE: results from before the per-arm layout are under ${EVAL_DIR}/<name>/." >&2
        echo "      They are all ARM=main. File them once with:" >&2
        echo "        cd ${EVAL_DIR} && mkdir -p main && for d in ${MARKER}_*/; do mv \"\$d\" main/; done" >&2
        echo "      They are left out of this table until then." >&2
        echo >&2
    fi
    # FID's CSV is metric,value,n_real,n_fake; the paired metrics end on a
    # MEAN row. Missing ones print as "-" so a half-finished array still reads.
    # The zoo writes these with csv.DictWriter, whose lines end in \r\n -- strip
    # the \r, or it rides along into the table and each value's carriage return
    # overwrites the start of the row when printed.
    value_of() {
        local csv="$1"
        [ -f "$csv" ] || { echo "-"; return; }
        case "$csv" in
            */fid.csv) tr -d '\r' < "$csv" | awk -F, 'NR==2 { print $2 }' ;;
            *)         tr -d '\r' < "$csv" | awk -F, '$1=="MEAN" { print $2 }' ;;
        esac
    }
    # nuclei.csv (evaluate_nuclei.sh) is metric,value rows; its headline is
    # centroid recall / precision / F1 at the default matching radius.
    nuc_of() {
        local csv="$1" key="$2"
        [ -f "$csv" ] || { echo "-"; return; }
        # LC_ALL=C: awk's printf honours the locale's decimal separator, and a
        # comma there would split the CSV column.
        tr -d '\r' < "$csv" | LC_ALL=C awk -F, -v k="$key" '$1==k { printf "%.4f", $2 }'
    }
    out="${EVAL_DIR}/summary.csv"
    echo "arm,name,fid,patch_ssim,lpips,nuc_recall,nuc_precision,nuc_f1" > "$out"
    # Arms in a fixed order, so the table reads main first and the controls
    # after it; anything else (custom, a new arm) follows alphabetically.
    known="main cycsplit worstfield_cycmerged worstfield_cycsplit baselines identity"
    arms="$known"
    for a in "${EVAL_DIR}"/*/; do
        a=$(basename "$a")
        case " $known plots nuclei_ref " in *" $a "*) ;; *) [ -f "${EVAL_DIR}/${a}/fid.csv" ] || arms="$arms $a" ;; esac
    done
    for arm in $arms; do
        for d in "${EVAL_DIR}/${arm}"/*/; do
            name=$(basename "$d")
            echo "${arm},${name},$(value_of "${d}fid.csv"),$(value_of "${d}patch_ssim.csv"),$(value_of "${d}lpips.csv"),$(nuc_of "${d}nuclei.csv" recall),$(nuc_of "${d}nuclei.csv" precision),$(nuc_of "${d}nuclei.csv" f1)" >> "$out"
        done
    done
    echo "${MARKER} / ${SPLIT}   (fid, lpips lower; patch_ssim, nuc_* higher is better)"
    column -s, -t < "$out"
    echo
    echo "written to ${out}"
    exit 0
fi

# -----------------------------
# Which predictions this task scores -- shared with evaluate_nuclei.sh, so
# both file a model under the same <arm>/<name>
# -----------------------------
_slurm_dir=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[ -f "${_slurm_dir}/_eval_arms.sh" ] || _slurm_dir="${_slurm_dir}/slurm"
source "${_slurm_dir}/_eval_arms.sh"

# Truth: the registered partner split -- valA -> valB, testA -> testB -- under
# the same per-marker tile root inference read from.
TRUTH=${TRUTH:-$(tiles_root "$MARKER")/${MARKER}/TrainValAB/${SPLIT%A}B}

OUT="${EVAL_DIR}/${ARM}/${NAME}"

echo "${ARM} / ${NAME}"
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
