#!/bin/bash
#SBATCH --job-name=topo_audit
#SBATCH --output=logs_topo/audit_%A_%a.out
#SBATCH --error=logs_topo/audit_%A_%a.err

#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
# No --gres: everything here is CPU-only.

# One cell of the field-validation audit. Submit it through run_audit.sh
# rather than directly -- that script works out the array size and chains the
# decision step onto the end.
#
# The array runs, per marker, the four cells
#
#     estimated vectors x strict control        <- screened, then confirmed
#     estimated vectors x unstratified control  <- screened only
#     fixed (literature) vectors x strict       <- screened, then confirmed
#     fixed (literature) vectors x unstratified <- screened only
#
# The strict cells are what any recommendation may be based on. The unstratified
# cells exist only so the gap between the two can be reported: that gap is how
# much of an apparent correspondence signal was really specimen recognition.
#
# Screening runs on slice 1 and confirmation on slice 2, disjoint by
# construction -- same SEED, offset by exactly one slice length -- so the
# candidates are fixed before slice 2 is ever touched.

set -eo pipefail

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-topocg}"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

TASK_ID=${SLURM_ARRAY_TASK_ID:?submit through run_audit.sh}

MARKERS=${MARKERS:-"Ki67 ER HER2 PR"}
TILES=${TILES:-/work2/bz66izin-TopoCG/MIST_tiles}
TILES_BCI=${TILES_BCI:-/work2/bz66izin-TopoCG/BCI_tiles}
STAINS_DIR=${STAINS_DIR:-/work2/bz66izin-TopoCG/field_validation_estimated}
AUDIT=${AUDIT:-/work2/bz66izin-TopoCG/field_audit}

LIMIT=${LIMIT:-auto}           # tiles per slice; the run uses 2 x LIMIT in all.
                               # 'auto' is half the matched pairs, so the two
                               # slices together use every tile. That is what you
                               # want: under the strict control a tile is dropped
                               # unless a groupmate lands in the SAME slice, so a
                               # small LIMIT breaks the groups up and costs more
                               # power than it saves time. On BCI's 1560 tiles
                               # LIMIT=512 leaves 355 usable per slice against
                               # 682 for auto -- a 1.4x wider noise band for no
                               # gain. Give a number only to subsample on purpose.
SEED=${SEED:-0}                # MUST match between screen and confirm, or the
                               # two slices stop being disjoint
SHUFFLES=${SHUFFLES:-5}
IMAGE_SIZE=${IMAGE_SIZE:-256}
COMBINE=${COMBINE:-sum}
DOWNSAMPLE=${DOWNSAMPLE:-"1 2 4"}
PROJECTIONS=${PROJECTIONS:-"auto birth death lifetime"}
DIMS_SETS=${DIMS_SETS:-"0 1 0,1"}

# The two vector arms sweep different specs, because a stain NAME means nothing
# once the vectors come from data and an estimated INDEX means nothing once they
# come from the literature.
EST_FIELDS_A=${EST_FIELDS_A:-"stain1/stain2 stain2/stain1 stain1+stain2"}
EST_FIELDS_B=${EST_FIELDS_B:-"stain1/stain2 stain2/stain1 stain1+stain2"}
FIX_FIELDS_A=${FIX_FIELDS_A:-"hematoxylin/eosin eosin/hematoxylin hematoxylin+eosin"}
FIX_FIELDS_B=${FIX_FIELDS_B:-"hematoxylin/dab dab/hematoxylin hematoxylin+dab"}

# -----------------------------
# Decode the array index: marker-major, then vectors, then control.
# -----------------------------
read -r -a MARKER_ARR <<< "$MARKERS"
# Both arms by default. Drop to "estimated" for a stain combination the
# literature table cannot express -- running a fixed arm against vectors that
# are not on the slide does not test the literature, it just adds a wrong answer
# the rule might then pick.
read -r -a VECTOR_ARR <<< "${VECTOR_ARMS:-estimated fixed}"
CONTROL_ARR=(strict unstratified)
PER_MARKER=$(( ${#VECTOR_ARR[@]} * ${#CONTROL_ARR[@]} ))

m_idx=$(( TASK_ID / PER_MARKER ))
rem=$(( TASK_ID % PER_MARKER ))
v_idx=$(( rem / ${#CONTROL_ARR[@]} ))
c_idx=$(( rem % ${#CONTROL_ARR[@]} ))

if [ "$m_idx" -ge "${#MARKER_ARR[@]}" ]; then
    echo "task ${TASK_ID} is past the end of MARKERS -- nothing to do"
    exit 0
fi
MARKER=${MARKER_ARR[$m_idx]}
VECTORS=${VECTOR_ARR[$v_idx]}
CONTROL=${CONTROL_ARR[$c_idx]}

# Tile root per marker: TILES_<MARKER> wins if it is set, else TILES. That is
# how a dataset tiled somewhere else (BCI, VS) joins in without editing this.
tiles_root() {
    local var="TILES_$1"
    echo "${!var:-$TILES}"
}
root="$(tiles_root "$MARKER")"
VAL_A="${root}/${MARKER}/TrainValAB/valA"
VAL_B="${root}/${MARKER}/TrainValAB/valB"
if [ ! -d "$VAL_A" ] || [ ! -d "$VAL_B" ]; then
    echo "ERROR: ${VAL_A} or ${VAL_B} is missing" >&2
    exit 1
fi

cell_dir="${AUDIT}/${MARKER}"
mkdir -p "$cell_dir"

# Resolve LIMIT before anything uses it. Every cell of a marker derives the same
# number from the same directory, so the four cells stay comparable and slice 2
# stays exactly disjoint from slice 1.
if [ "$LIMIT" = "auto" ] || [ "$LIMIT" = "0" ]; then
    LIMIT=$(python -c "
import sys
from topo_i2i.validate_fields import matched_pairs
pairs, _ = matched_pairs(sys.argv[1], sys.argv[2], limit=0)
print(max(1, len(pairs) // 2))
" "$VAL_A" "$VAL_B")
    echo "LIMIT=auto -> ${LIMIT} tiles per slice (half the matched pairs)"
fi

stain_arg=()
if [ "$VECTORS" = "estimated" ]; then
    FIELDS_A="$EST_FIELDS_A"
    FIELDS_B="$EST_FIELDS_B"
    stains="${STAINS_DIR}/stains_${MARKER}.json"
    if [ ! -f "$stains" ]; then
        echo "ERROR: ${stains} missing -- run estimate_stains.sh first" >&2
        exit 1
    fi
    stain_arg=(--stains "$stains")
else
    # No --stains at all: topo-validate-fields then uses fields.STAIN_VECTORS.
    FIELDS_A="$FIX_FIELDS_A"
    FIELDS_B="$FIX_FIELDS_B"
fi

within_arg=()
[ "$CONTROL" = "strict" ] && within_arg=(--shuffle-within-slide)

echo "================================================================"
echo "task ${TASK_ID}: marker=${MARKER} vectors=${VECTORS} control=${CONTROL}"
echo "  A ${VAL_A}"
echo "  B ${VAL_B}"
echo "  slice 1: offset 0 limit ${LIMIT}   slice 2: offset ${LIMIT} limit ${LIMIT}"
echo "================================================================"

screen_json="${cell_dir}/screen_${VECTORS}_${CONTROL}.json"

# -----------------------------
# Stage 1 -- screen the whole grid on slice 1.
# -----------------------------
topo-validate-fields \
    --dataA "$VAL_A" --dataB "$VAL_B" \
    ${stain_arg[@]+"${stain_arg[@]}"} \
    --field-A $FIELDS_A \
    --field-B $FIELDS_B \
    --downsample $DOWNSAMPLE \
    --topo-projection $PROJECTIONS \
    --dims-set $DIMS_SETS \
    --field-combine "$COMBINE" \
    --image-size "$IMAGE_SIZE" \
    --limit "$LIMIT" --offset 0 \
    --shuffles "$SHUFFLES" \
    --sample random --seed "$SEED" \
    ${within_arg[@]+"${within_arg[@]}"} \
    ${SLIDE_REGEX:+--slide-regex "$SLIDE_REGEX"} \
    --json "$screen_json" \
    | tee "${cell_dir}/screen_${VECTORS}_${CONTROL}.txt"

if [ "$CONTROL" != "strict" ]; then
    echo
    echo "unstratified cell: screening only -- nothing here may be recommended,"
    echo "it exists to measure how much signal the strict control removes."
    exit 0
fi

# -----------------------------
# Stage 2 -- confirm the top candidates on slice 2.
#
# topo-audit picks them; this script never chooses. The candidates are decided
# from slice 1 alone, before slice 2 is read.
# -----------------------------
echo
echo "---- confirming the top candidates on the held-out slice ----"
k=0
while read -r flags; do
    [ -n "$flags" ] || continue
    out="${cell_dir}/confirm_${VECTORS}_$(printf '%02d' "$k").json"
    echo
    echo "candidate ${k}: ${flags}"
    topo-validate-fields \
        --dataA "$VAL_A" --dataB "$VAL_B" \
        ${stain_arg[@]+"${stain_arg[@]}"} \
        $flags \
        --field-combine "$COMBINE" \
        --image-size "$IMAGE_SIZE" \
        --limit "$LIMIT" --offset "$LIMIT" \
        --shuffles "$SHUFFLES" \
        --sample random --seed "$SEED" \
        --shuffle-within-slide \
        ${SLIDE_REGEX:+--slide-regex "$SLIDE_REGEX"} \
        --json "$out"
    k=$(( k + 1 ))
done < <(topo-audit plan --screen "$screen_json")

echo
echo "cell done: ${cell_dir}"
