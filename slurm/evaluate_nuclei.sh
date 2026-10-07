#!/bin/bash
#SBATCH --job-name=topo_eval_nuclei
#SBATCH --output=logs_topo/eval_nuclei_%A_%a.out
#SBATCH --error=logs_topo/eval_nuclei_%A_%a.err

#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
#SBATCH --array=0-14  # ARM=main; see _eval_arms.sh for every arm's tasks
# No --gres: StarDist runs on TensorFlow's CPU build here, which needs no
# CUDA match with the node. A tile takes well under a second.

# Nuclei evaluation: do the real H&E's nuclei reappear in the generated IHC?
# Segments both with StarDist at the settings the pilot chose, matches nuclei
# by centroid distance (and IoU 0.5), and writes recall / precision / F1 next
# to the FID, SSIM and LPIPS of the same model -- evaluate.sh's summary and
# plot_eval.sh pick them up. See tools/nuclei_eval.py for the method and for
# everything that is saved.
#
# Task N means the same model as in evaluate.sh: both source _eval_arms.sh.
# ARM=identity (task 0) scores the H&E input itself as the prediction -- run
# it, it is the row that shows how much of a score copying alone earns.
#
# Needs the stardist environment and model copy from nuclei_pilot.sh's setup.
#
# PER MARKER, two steps. First the H&E reference, once -- every model is
# matched against it, and building it inside the array would have all fifteen
# tasks segment the same H&E at once:
#
#   mkdir -p logs_topo
#   ref=$(sbatch --parsable --array=0 --export=ALL,MARKER=ER evaluate_nuclei.sh ref)
#
# then the models, after it:
#
#   sbatch --dependency=afterok:$ref --export=ALL,MARKER=ER evaluate_nuclei.sh
#   sbatch --dependency=afterok:$ref --export=ALL,MARKER=ER,ARM=identity --array=0 evaluate_nuclei.sh
#   sbatch --dependency=afterok:$ref --export=ALL,MARKER=ER,ARM=baselines --array=0-5 evaluate_nuclei.sh
#   sbatch --dependency=afterok:$ref --export=ALL,MARKER=ER,ARM=worstfield_cycmerged --array=1-12 evaluate_nuclei.sh
#
# BCI goes the same way with MARKER=BCI. VS has no IHC, so no settings exist
# for it.
#
# A model whose nuclei.csv exists is skipped; FORCE=1 recomputes it.
# SAVE_MASKS=1 keeps every generated-IHC label mask (~10-20 MB per model).
#
# Output: ${EVAL_ROOT}/<marker>/<split>/<arm>/<name>/{nuclei.csv, nuclei_tiles.csv,
#         nuclei_centroids.npz, figures/}  and  .../<split>/nuclei_ref/

set -eo pipefail

: "${MARKER:?set MARKER, e.g. --export=ALL,MARKER=ER}"

ROOT=${ROOT:-/work2/bz66izin-TopoCG}
EVAL_ROOT=${EVAL_ROOT:-${ROOT}/eval}
SPLIT=${SPLIT:-valA}
EVAL_DIR="${EVAL_ROOT}/${MARKER}/${SPLIT}"
REF_CACHE="${EVAL_DIR}/nuclei_ref"
MODELS_DIR=${MODELS_DIR:-${ROOT}/stardist_models}
REPO_ROOT=${REPO_ROOT:-/home/sc.uni-leipzig.de/bz66izin/TopoCG_Project/TopoCycleGAN}
EVAL_PY="${REPO_ROOT}/tools/nuclei_eval.py"
FIGURES=${FIGURES:-12}

[ -f "$EVAL_PY" ] || { echo "ERROR: no tools/nuclei_eval.py under REPO_ROOT=${REPO_ROOT}" >&2; exit 1; }

MODE=${1:-models}
if [ "$MODE" = "ref" ]; then
    # One reference job per marker, however it was submitted: the header's
    # default array would otherwise start fifteen identical ones.
    if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ "$SLURM_ARRAY_TASK_ID" != "0" ]; then
        echo "ref: task ${SLURM_ARRAY_TASK_ID} has nothing to do (task 0 builds it)"
        exit 0
    fi
    # No model to resolve -- give _eval_arms.sh a placeholder so it only
    # defines the tile-root lookup.
    PRED=/dev/null NAME=ref
fi

# _eval_arms.sh sits beside this script, in the checkout's slurm/.
REPO=${REPO:-${REPO_ROOT}/slurm}
source "${REPO}/_eval_arms.sh"
HE="$(tiles_root "$MARKER")/${MARKER}/TrainValAB/${SPLIT}"

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    conda activate "${STARDIST_ENV:-stardist}"
fi
# The env's own interpreter, by path -- see nuclei_pilot.sh for why.
PYTHON=${PYTHON:-${CONDA_PREFIX:+${CONDA_PREFIX}/bin/}python}
"$PYTHON" -c "import stardist" 2>/dev/null || {
    echo "ERROR: ${PYTHON} cannot import stardist; set PYTHON=~/.conda/envs/stardist/bin/python" >&2
    exit 1; }
for m in 2D_versatile_he 2D_versatile_fluo; do
    [ -f "${MODELS_DIR}/${m}/weights_best.h5" ] || {
        echo "ERROR: no ${m} under MODELS_DIR=${MODELS_DIR}; once, on the login node:" >&2
        echo "  ${PYTHON} ${REPO_ROOT}/tools/nuclei_pilot.py --fetch-models --models-dir ${MODELS_DIR}" >&2
        exit 1; }
done

# -----------------------------
# ref: the H&E reference cache, once per marker
# -----------------------------
if [ "$MODE" = "ref" ]; then
    echo "H&E reference for ${MARKER}/${SPLIT}: ${HE} -> ${REF_CACHE}"
    "$PYTHON" "$EVAL_PY" --marker "$MARKER" --he "$HE" --ref-cache "$REF_CACHE" \
        --models-dir "$MODELS_DIR" --ref-only
    exit 0
fi

OUT="${EVAL_DIR}/${ARM}/${NAME}"
echo "${ARM} / ${NAME}"
echo "  H&E    ${HE}"
echo "  pred   ${PRED}"
echo "  output ${OUT}"

if [ ! -d "$PRED" ]; then
    echo "  no predictions yet; nothing to evaluate"
    exit 0
fi
if [ -f "${OUT}/nuclei.csv" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[skip] ${OUT}/nuclei.csv exists (FORCE=1 recomputes)"
    exit 0
fi

"$PYTHON" "$EVAL_PY" --marker "$MARKER" --he "$HE" --pred "$PRED" \
    --ref-cache "$REF_CACHE" --models-dir "$MODELS_DIR" --out "$OUT" \
    --figures "$FIGURES" ${SAVE_MASKS:+--save-masks}

echo "Done: ${ARM} / ${NAME}"
