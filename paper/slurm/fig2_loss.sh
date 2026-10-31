#!/bin/bash
#SBATCH --job-name=topo_figure
#SBATCH --output=logs_topo/figure_%j.out
#SBATCH --error=logs_topo/figure_%j.err

#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
# No --gres: two generator passes per pair and persistence are fine on CPU.

# The loss figure's panels (paper/fig2_loss.py) for H&E / IHC pairs: the two
# translations, the four stain fields, their persistence diagrams, the
# reduced-axis diagrams and the two ph_trans pairings, from a trained model.
#
# The model defaults to the rerun's chosen cell, SWEEP_ROOT/<marker>/results/
# <MARKER>_<RUN> (lt0.01_cyc0_trans1). CKPT_DIR overrides it with any run
# directory, checkpoints/ directory or .pt; STEP picks a checkpoint other than
# the newest.
#
# Which pairs, one of:
#   HE=<path> IHC=<path>   exactly this pair
#   TILE=<file name>       that tile from ${SPLIT}A and ${SPLIT}B
#   (neither)              PAIRS candidate pairs, drawn as inspect.sh draws them
#                          (matched_pairs, SEED, SAMPLE) -- look through them and
#                          rerun with the one for the figure
#
#   cd paper/slurm && mkdir -p logs_topo
#   sbatch --export=ALL,MARKER=Ki67 fig2_loss.sh                      # 8 candidates
#   sbatch --export=ALL,MARKER=Ki67,PAIRS=20,SEED=1 fig2_loss.sh
#   sbatch --export=ALL,MARKER=Ki67,TILE=<name>.png fig2_loss.sh
#   sbatch --export=ALL,MARKER=ER,HE=/path/he.png,IHC=/path/ihc.png,FORMAT=svg fig2_loss.sh
#   sbatch --export=ALL,MARKER=Ki67,CKPT_DIR=$R/worstfield_layout/ki67/results/Ki67_lt0.01_cyc0_trans1 fig2_loss.sh
#
# Output: ${OUT}/<marker>/<tile>/ -- image_*.png, field_*.png/.npy,
# diagram_*, projection_*, pair_trans_{H,I}, overview.png, summary.json.
# The log ends with each pair's ph_trans values.

set -eo pipefail

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-topocg}"
fi

: "${MARKER:?set MARKER, e.g. --export=ALL,MARKER=Ki67}"
ROOT=${ROOT:-/work2/bz66izin-TopoCG}
REPO_ROOT=${REPO_ROOT:-/home/sc.uni-leipzig.de/bz66izin/TopoCG_Project/TopoCycleGAN}
PAPER="${REPO_ROOT}/paper"
[ -d "$PAPER" ] || { echo "ERROR: no paper/ under REPO_ROOT=${REPO_ROOT}" >&2; exit 1; }
MARKER_LC=$(echo "$MARKER" | tr 'A-Z' 'a-z')
SWEEP_ROOT=${SWEEP_ROOT:-${ROOT}/sweep_layout}
RUN=${RUN:-lt0.01_cyc0_trans1}
CKPT_DIR=${CKPT_DIR:-${SWEEP_ROOT}/${MARKER_LC}/results/${MARKER}_${RUN}}
OUT=${OUT:-${ROOT}/figure_loss}

# Tile root per marker, as everywhere else.
TILES=${TILES:-${ROOT}/MIST_tiles}
TILES_BCI=${TILES_BCI:-${ROOT}/BCI_tiles}
TILES_VS=${TILES_VS:-${ROOT}/VS_tiles}
var="TILES_${MARKER}"; TILE_ROOT=${!var:-$TILES}
SPLIT=${SPLIT:-val}            # val / test / train
DIR_A="${TILE_ROOT}/${MARKER}/TrainValAB/${SPLIT}A"
DIR_B="${TILE_ROOT}/${MARKER}/TrainValAB/${SPLIT}B"

PAIRS=${PAIRS:-8}
SEED=${SEED:-0}
SAMPLE=${SAMPLE:-random}
OFFSET=${OFFSET:-0}
IMAGE_SIZE=${IMAGE_SIZE:-256}
FORMAT=${FORMAT:-pdf}          # pdf / svg / png, for the plots
EXTRA=()
[ -n "${STEP:-}" ] && EXTRA+=(--step "$STEP")
[ -n "${CMAP:-}" ] && EXTRA+=(--cmap "$CMAP")
[ -n "${SUBLEVEL:-}" ] && EXTRA+=(--sublevel)

if [ ! -e "$CKPT_DIR" ]; then
    echo "ERROR: no such checkpoint or run directory: ${CKPT_DIR}" >&2
    exit 1
fi

# The pairs, as "<he path>\t<ihc path>" lines.
if [ -n "${HE:-}" ] || [ -n "${IHC:-}" ]; then
    : "${HE:?IHC is set -- set HE too}" "${IHC:?HE is set -- set IHC too}"
    pairlist=$(printf '%s\t%s\n' "$HE" "$IHC")
elif [ -n "${TILE:-}" ]; then
    pairlist=$(printf '%s\t%s\n' "${DIR_A}/${TILE}" "${DIR_B}/${TILE}")
else
    if [ ! -d "$DIR_A" ] || [ ! -d "$DIR_B" ]; then
        echo "ERROR: ${DIR_A} or ${DIR_B} is missing" >&2
        exit 1
    fi
    pairlist=$(python - "$DIR_A" "$DIR_B" "$PAIRS" "$OFFSET" "$SAMPLE" "$SEED" <<'PY'
import os, sys
from topo_i2i.validate_fields import matched_pairs
dir_a, dir_b, limit, offset, sample, seed = sys.argv[1:]
pairs, how = matched_pairs(dir_a, dir_b, limit=int(limit), offset=int(offset),
                           sample=sample, seed=int(seed))
print("#", how, file=sys.stderr)
for a, b in pairs:
    print("%s\t%s" % (os.path.join(dir_a, a), os.path.join(dir_b, b)))
PY
)
fi

echo "figure ${MARKER}"
echo "  model  ${CKPT_DIR}${STEP:+ (step ${STEP})}"
echo "  output ${OUT}/${MARKER}/"

while IFS=$'\t' read -r he ihc; do
    [ -n "$he" ] || continue
    for f in "$he" "$ihc"; do
        [ -f "$f" ] || { echo "ERROR: no such tile: $f" >&2; exit 1; }
    done
    tile=$(basename "${he%.*}")
    echo
    echo "--- ${MARKER} / ${tile}"
    python "${PAPER}/fig2_loss.py" --he "$he" --ihc "$ihc" --ckpt-dir "$CKPT_DIR" \
        --outdir "${OUT}/${MARKER}/${tile}" --image-size "$IMAGE_SIZE" \
        --format "$FORMAT" ${EXTRA[@]+"${EXTRA[@]}"}
done <<< "$pairlist"

echo
echo "================ ph_trans per pair ================"
python - "${OUT}/${MARKER}" <<'PY'
import json, os, sys
root = sys.argv[1]
print("%-40s %12s %12s" % ("tile", "trans_H", "trans_I"))
for tile in sorted(os.listdir(root)):
    p = os.path.join(root, tile, "summary.json")
    if os.path.exists(p):
        d = json.load(open(p))["distances"]
        print("%-40s %12.4g %12.4g" % (tile[:40], d["trans_H"]["total"], d["trans_I"]["total"]))
PY

echo
echo "To pull the panels back:  tar czf figure_${MARKER}.tgz -C ${OUT} ${MARKER}"
