#!/bin/bash
# Run the whole field-validation audit with one command, then tell you -- or
# whoever picks the project up next -- what to train with.
#
# Run this on the login node; it is a submitter, not a job.
#
#   bash run_audit.sh                                  # 4 MIST markers
#   MARKERS=BCI TILES_BCI=/work2/bz66izin-TopoCG/BCI_tiles bash run_audit.sh
#   MARKERS="Ki67 ER HER2 PR BCI" bash run_audit.sh    # everything
#
# It submits, in order:
#   1. stain estimation, for any marker that has no vectors yet
#   2. an array of 4 cells per marker -- {estimated, literature} vectors x
#      {strict, unstratified} control -- each screening the full field grid on
#      slice 1 and, for the strict cells, confirming its top candidates on a
#      disjoint slice 2
#   3. one decision job that applies the pre-registered rule and writes
#      ${AUDIT}/RECOMMENDATION.txt, decision.json and recommended_<marker>.env
#
# Everything runs in the background under SLURM. Come back to RECOMMENDATION.txt.
#
# The decision step depends on the array with afterany rather than afterok, so
# one failed cell still yields a report over the cells that did finish -- the
# report names what is missing.

set -eo pipefail

if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: run this with 'bash run_audit.sh', not 'sbatch'." >&2
    echo "It is a submitter script -- it submits the jobs for you." >&2
    exit 1
fi

MARKERS=${MARKERS:-"Ki67 ER HER2 PR"}
STAINS_DIR=${STAINS_DIR:-/work2/bz66izin-TopoCG/field_validation_estimated}
AUDIT=${AUDIT:-/work2/bz66izin-TopoCG/field_audit}
TILES=${TILES:-/work2/bz66izin-TopoCG/MIST_tiles}
TILES_BCI=${TILES_BCI:-/work2/bz66izin-TopoCG/BCI_tiles}
LIMIT=${LIMIT:-auto}   # per marker: half its matched pairs
SEED=${SEED:-0}

# These run under bash, not sbatch, so BASH_SOURCE really does point at this
# file and the siblings can be found from any working directory.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p logs_topo "$AUDIT"

n_markers=$(echo "$MARKERS" | wc -w | tr -d ' ')
last=$(( n_markers * 4 - 1 ))

# The estimated-vector arm cannot run without vectors, so make them first if any
# marker is missing them.
missing=""
for m in $MARKERS; do
    [ -f "${STAINS_DIR}/stains_${m}.json" ] || missing="${missing} ${m}"
done

common="MARKERS=${MARKERS},STAINS_DIR=${STAINS_DIR},AUDIT=${AUDIT}"
common="${common},TILES=${TILES},TILES_BCI=${TILES_BCI},LIMIT=${LIMIT},SEED=${SEED}"
[ -n "${SLIDE_REGEX:-}" ] && common="${common},SLIDE_REGEX=${SLIDE_REGEX}"

dep=""
if [ -n "$missing" ]; then
    echo "estimating stain vectors for:${missing}"
    jid=$(sbatch --parsable \
                 --export="ALL,MARKERS=${missing# },STAINS_DIR=${STAINS_DIR},TILES=${TILES},TILES_BCI=${TILES_BCI}" \
                 "$HERE/estimate_stains.sh")
    echo "  estimation job: ${jid}"
    dep="--dependency=afterok:${jid}"
else
    echo "stain vectors already present in ${STAINS_DIR}"
fi

array_jid=$(sbatch --parsable $dep --array=0-${last} \
                   --export="ALL,${common}" "$HERE/audit_fields.sh")
echo "audit array: ${array_jid}  ($(( last + 1 )) cells: ${n_markers} markers x 4)"

decide_jid=$(sbatch --parsable --dependency=afterany:${array_jid} \
                    --export="ALL,AUDIT=${AUDIT},MARKERS=${MARKERS}" \
                    "$HERE/audit_decide.sh")
echo "decision job: ${decide_jid}"

echo
echo "watch with: squeue -u \$USER"
echo "when it finishes:"
echo "  cat ${AUDIT}/RECOMMENDATION.txt"
echo "then train with what it chose:"
for m in $MARKERS; do
    lower=$(echo "$m" | tr 'A-Z' 'a-z')
    echo "  AUDIT_DIR=${AUDIT} sbatch sweep_${lower}.sh"
done
