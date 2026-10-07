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
# RERUNNING FAILURES. Whatever killed a task -- timeout, memory, a node -- it
# leaves predictions without a nuclei.csv. This lists every such model of a
# marker with the sbatch line that reruns exactly those, and SUBMIT=1 submits
# them (refused while evaluate_nuclei jobs are still queued or running):
#
#   bash evaluate_nuclei.sh missing                        # MARKER=ER etc. in env
#   for M in Ki67 ER HER2 PR BCI; do MARKER=$M bash evaluate_nuclei.sh missing; done
#   MARKER=ER SUBMIT=1 bash evaluate_nuclei.sh missing
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

# -----------------------------
# missing: which models of this marker have predictions but no nuclei.csv --
# i.e. whose task failed, timed out, ran out of memory or never ran -- and
# the sbatch line that reruns exactly those. Read off the result directories,
# not SLURM's records, so it does not matter how a task died. SUBMIT=1 submits
# the lines instead of printing them. Login node, no environment needed.
# -----------------------------
if [ "$MODE" = "missing" ]; then
    _slurm=${REPO:-${REPO_ROOT}/slurm}
    # Every arm and the tasks it has -- the same lists as the header.
    ARM_TASKS="main:0-14 identity:0 cycsplit:2,3,5,6,8,9,11,12
               worstfield_cycmerged:1-12 worstfield_cycsplit:2,3,5,6,8,9,11,12
               baselines:0-5"
    expand() {     # "0-3,7" -> "0 1 2 3 7"
        local part out=""
        for part in ${1//,/ }; do
            case "$part" in
                *-*) out="$out $(seq "${part%-*}" "${part#*-}" | tr '\n' ' ')" ;;
                *)   out="$out $part" ;;
            esac
        done
        echo $out
    }
    if [ ! -d "${REF_CACHE}" ] || [ -z "$(ls -A "${REF_CACHE}" 2>/dev/null)" ]; then
        echo "${MARKER}: no H&E reference under ${REF_CACHE} -- run the ref job first:"
        echo "  sbatch --array=0 --export=ALL,MARKER=${MARKER} evaluate_nuclei.sh ref"
        exit 0
    fi
    # A task still queued or running has no nuclei.csv yet either, and would
    # be submitted twice. Say so, and do not submit while any are in flight.
    # Reruns are named per marker, so one marker's reruns do not hold up the
    # next marker in a loop; first submissions carry no marker in their name,
    # so they hold up every marker, which errs on the safe side.
    in_flight=0
    if command -v squeue >/dev/null 2>&1; then
        in_flight=$(squeue -h -u "$USER" -n "topo_eval_nuclei,topo_eval_nuclei_${MARKER}" \
                        -t PENDING,RUNNING,REQUEUED 2>/dev/null | wc -l | tr -d " ")
    fi
    if (( in_flight > 0 )); then
        echo "NOTE: ${in_flight} evaluate_nuclei job(s) still queued or running -- the list"
        echo "      below includes theirs. Rerun this once they have finished."
        [ "${SUBMIT:-0}" = "1" ] && { echo "      Not submitting anything until then."; SUBMIT=0; }
    fi
    n_done=0 n_todo=0 n_nopred=0
    for spec in $ARM_TASKS; do
        arm=${spec%%:*}
        todo=()
        for t in $(expand "${spec#*:}"); do
            # Resolve task t of this arm exactly as a job would, in a subshell
            # (a task the arm never trained exits it clean, printing no '|').
            line=$( (unset PRED NAME; ARM=$arm SLURM_ARRAY_TASK_ID=$t REPO=$_slurm
                     source "${_slurm}/_eval_arms.sh" >/dev/null 2>&1
                     echo "${PRED}|${NAME}") )
            case "$line" in *"|"*) ;; *) continue ;; esac
            pred=${line%%|*} name=${line#*|}
            [ -n "$name" ] || continue
            if [ -f "${EVAL_DIR}/${arm}/${name}/nuclei.csv" ]; then
                n_done=$((n_done + 1))
            elif [ -d "$pred" ]; then
                todo+=("$t"); n_todo=$((n_todo + 1))
                echo "  missing  ${arm}/${name}"
            else
                n_nopred=$((n_nopred + 1))
            fi
        done
        (( ${#todo[@]} )) || continue
        tasks=$(IFS=,; echo "${todo[*]}")
        cmd=(sbatch --time="${TIME:-02:00:00}" --array="$tasks" --job-name="topo_eval_nuclei_${MARKER}"
             --export=ALL,MARKER="$MARKER",ARM="$arm" evaluate_nuclei.sh)
        if [ "${SUBMIT:-0}" = "1" ]; then
            echo "  -> $(cd "$_slurm" && "${cmd[@]}")"
        else
            echo "  rerun:   ${cmd[*]}"
        fi
    done
    echo "${MARKER}/${SPLIT}: ${n_done} done, ${n_todo} failed or not run, ${n_nopred} without predictions yet"
    exit 0
fi

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
