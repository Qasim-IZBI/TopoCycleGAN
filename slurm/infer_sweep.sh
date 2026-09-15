#!/bin/bash
#SBATCH --job-name=topo_infer
#SBATCH --output=logs_topo/infer_%j.out
#SBATCH --error=logs_topo/infer_%j.err

#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=clara
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# Run validation inference for every cell of a marker's sweep.
#
#   sbatch --export=ALL,MARKER=ER slurm/infer_sweep.sh
#   MARKER=ER LIMIT=8 bash slurm/infer_sweep.sh     # quick look, 8 tiles per cell
#
# Run directories are discovered by globbing BASE/results rather than
# recomputing cell names, so this stays correct if the grid changes and only
# covers cells that actually produced a checkpoint.

set -eo pipefail

: "${MARKER:?set MARKER, e.g. --export=ALL,MARKER=ER}"

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    set +u
    conda activate "${CONDA_ENV:-topocg}"
    set -u
fi

MARKER_LC=$(echo "$MARKER" | tr 'A-Z' 'a-z')
BASE=${BASE:-/work2/bz66izin-TopoCG/Outputs_${MARKER_LC}}
DATA_DIR=${DATA_DIR:-/work2/bz66izin-TopoCG/MIST_tiles/${MARKER}/TrainValAB}
VAL_A=${VAL_A:-${DATA_DIR}/valA}
DIRECTION=${DIRECTION:-A2B}
IMAGE_SIZE=${IMAGE_SIZE:-256}
LIMIT=${LIMIT:-0}

echo "marker    ${MARKER}"
echo "input     ${VAL_A}"
echo "runs      ${BASE}/results"
echo "output    ${BASE}/preds/<run>/"
echo

shopt -s nullglob
done_n=0; skipped_n=0
for run_dir in "${BASE}"/results/*/; do
    run=$(basename "$run_dir")

    # Newest numbered checkpoint, else the rolling one. Collect with a glob into
    # an array -- piping a glob into `ls` lists the CWD when nothing matches,
    # because nullglob removes the argument entirely.
    ckpt=""
    ckpts=( "${run_dir}checkpoints"/step_[0-9]*.pt )
    if (( ${#ckpts[@]} )); then
        ckpt=$(printf '%s\n' "${ckpts[@]}" \
               | sed 's/.*step_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
    elif [ -f "${run_dir}checkpoints/step_latest.pt" ]; then
        ckpt="${run_dir}checkpoints/step_latest.pt"
    fi

    if [ -z "$ckpt" ]; then
        echo "[skip] ${run}: no checkpoint yet"
        skipped_n=$((skipped_n+1))
        continue
    fi

    out="${BASE}/preds/${run}"
    echo "[run ] ${run}  <- $(basename "$ckpt")"
    topo-infer --ckpt "$ckpt" --data "$VAL_A" --outdir "$out" \
               --direction "$DIRECTION" --image-size "$IMAGE_SIZE" \
               ${LIMIT:+--limit "$LIMIT"} --resume
    done_n=$((done_n+1))
done

echo
echo "inferred ${done_n} runs, skipped ${skipped_n} without checkpoints"
echo "predictions under ${BASE}/preds/"
