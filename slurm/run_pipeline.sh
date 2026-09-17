#!/bin/bash
# Submit the whole pipeline: estimate stain vectors, then train the grid for
# every marker once the vectors exist.
#
# Run this on the login node -- it is a submitter, not a job.
#
#   bash run_pipeline.sh
#   MARKERS="ER Ki67" bash run_pipeline.sh
#   SKIP_ESTIMATE=1 bash run_pipeline.sh   # vectors already estimated
#
# The training arrays are submitted with --dependency=afterok on the estimation
# job, so they queue immediately but only start once the vectors are written.

set -eo pipefail

# This is a SUBMITTER, not a job: it has no #SBATCH directives and calls sbatch
# itself. Running it under sbatch would burn an allocation on cluster defaults
# (wrong partition, no account) just to issue four sbatch calls.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: run this with 'bash run_pipeline.sh', not 'sbatch'." >&2
    echo "It is a submitter script -- it submits the jobs for you." >&2
    exit 1
fi

MARKERS=${MARKERS:-"Ki67 ER HER2 PR"}
STAINS_DIR=${STAINS_DIR:-/work2/bz66izin-TopoCG/field_validation_estimated}
SKIP_ESTIMATE=${SKIP_ESTIMATE:-0}

# These run under bash, not sbatch, so BASH_SOURCE really does point at this
# file and the siblings can be found from any working directory.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p logs_topo

# If every marker already has vectors -- as it does when
# field_validation_estimated has been run -- skip straight to training.
missing=0
for m in $MARKERS; do
    [ -f "${STAINS_DIR}/stains_${m}.json" ] || missing=1
done
if [ "$missing" = "0" ] && [ "$SKIP_ESTIMATE" != "2" ]; then
    echo "stain vectors already present in ${STAINS_DIR} -- skipping estimation"
    SKIP_ESTIMATE=1
fi

dep=""
if [ "$SKIP_ESTIMATE" != "1" ]; then
    jid=$(sbatch --parsable --export="ALL,MARKERS=${MARKERS},STAINS_DIR=${STAINS_DIR}" \
                 "$HERE/estimate_stains.sh")
    echo "estimation job: ${jid}"
    dep="--dependency=afterok:${jid}"
fi

for m in $MARKERS; do
    lower=$(echo "$m" | tr 'A-Z' 'a-z')
    script="${HERE}/sweep_${lower}.sh"
    if [ ! -f "$script" ]; then
        echo "[skip] ${m}: ${script} not found"
        continue
    fi
    jid=$(sbatch --parsable $dep --export="ALL,STAINS_DIR=${STAINS_DIR}" "$script")
    echo "training array for ${m}: ${jid}  (13 cells, task 0 is the CycleGAN baseline)"
done

echo
echo "watch with: squeue -u \$USER"
echo "when training finishes, run inference with:"
for m in $MARKERS; do
    echo "  sbatch --export=ALL,MARKER=${m} infer_sweep.sh"
done
