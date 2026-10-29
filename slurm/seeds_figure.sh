#!/bin/bash
#SBATCH --job-name=topo_seeds_fig
#SBATCH --output=logs_topo/seeds_figure_%j.out
#SBATCH --error=logs_topo/seeds_figure_%j.err

#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# The inversion figure (topo_i2i/seeds_figure.py): one tile through the first
# N_SEEDS seeds of an ensemble, one grid row per ensemble, every output
# labelled with its layout r (inverted ones framed red).
#
# By default, two rows for MODEL: the original ensemble (no layout term, where
# a third to a half of the seeds invert) and the one trained with the layout
# term (stop 20k). The original Ki67 TopoCycleGAN members were archived, so for
# MODEL=topo on Ki67 point OLD_DIR at the archive.
#
#   OLD_DIR  ${ROOT}/ensemble/<MARKER>/<MODEL>
#   NEW_DIR  ${ROOT}/ensemble_layout1_stop20k/<MARKER>/<MODEL>
#   ENS      instead of both: "label=dir;label=dir;..." for any rows
#
# The tile: IMAGE=<path>, or TILE=<file name> from TrainValAB/${SPLIT}A (H&E,
# DOMAIN=he) or ${SPLIT}B (IHC, DOMAIN=ihc). Pick one with seeds that invert,
# e.g. from the inverted members' preds and inversion.csv.
#
#   mkdir -p logs_topo
#   sbatch --export=ALL,MARKER=Ki67,MODEL=cyclegan,TILE=<name>.png seeds_figure.sh
#   sbatch --export=ALL,MARKER=Ki67,MODEL=topo,TILE=<name>.png,OLD_DIR=$R/archive_ki67_topo_lt0.002_cyc1 seeds_figure.sh
#   sbatch --export=ALL,MARKER=Ki67,MODEL=dclgan,TILE=<name>.png,DOMAIN=ihc,N_SEEDS=6 seeds_figure.sh
#   sbatch --export=ALL,MARKER=Ki67,IMAGE=/path/he.png,ENS="CycleGAN=$R/ensemble/Ki67/cyclegan;DCLGAN=$R/ensemble/Ki67/dclgan" seeds_figure.sh
#
# Output: ${OUT}/<MARKER>_<MODEL>_<tile>/ -- grid.<FORMAT>, and per row
# input.png, seed<k>.png and summary.json (r, inverted, checkpoint, step).

set -eo pipefail

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-topocg}"
fi

: "${MARKER:?set MARKER, e.g. --export=ALL,MARKER=Ki67}"
ROOT=${ROOT:-/work2/bz66izin-TopoCG}
MODEL=${MODEL:-cyclegan}
DOMAIN=${DOMAIN:-he}
N_SEEDS=${N_SEEDS:-10}
SPLIT=${SPLIT:-val}
FORMAT=${FORMAT:-pdf}
OUT=${OUT:-${ROOT}/figure_seeds}

TILES=${TILES:-${ROOT}/MIST_tiles}
TILES_BCI=${TILES_BCI:-${ROOT}/BCI_tiles}
TILES_VS=${TILES_VS:-${ROOT}/VS_tiles}
var="TILES_${MARKER}"; TILE_ROOT=${!var:-$TILES}

if [ -z "${IMAGE:-}" ]; then
    : "${TILE:?set TILE=<file name> or IMAGE=<path>}"
    side=A; [ "$DOMAIN" = "ihc" ] && side=B
    IMAGE="${TILE_ROOT}/${MARKER}/TrainValAB/${SPLIT}${side}/${TILE}"
fi
[ -f "$IMAGE" ] || { echo "ERROR: no such tile: ${IMAGE}" >&2; exit 1; }

ARGS=()
if [ -n "${ENS:-}" ]; then
    IFS=';' read -ra rows <<< "$ENS"
    for r in "${rows[@]}"; do ARGS+=(--ens "$r"); done
else
    OLD_DIR=${OLD_DIR:-${ROOT}/ensemble/${MARKER}/${MODEL}}
    NEW_DIR=${NEW_DIR:-${ROOT}/ensemble_layout1_stop20k/${MARKER}/${MODEL}}
    for d in "$OLD_DIR" "$NEW_DIR"; do
        [ -d "$d" ] || { echo "ERROR: no such ensemble directory: $d" >&2; exit 1; }
    done
    ARGS+=(--ens "without layout term=${OLD_DIR}" --ens "with layout term=${NEW_DIR}")
fi
[ -n "${SEEDS:-}" ] && ARGS+=(--seeds $SEEDS)        # e.g. SEEDS="0 3 4 7"
[ -n "${STEP:-}" ] && ARGS+=(--step "$STEP")
[ -n "${NO_R:-}" ] && ARGS+=(--no-r)

tile=$(basename "${IMAGE%.*}")
DEST="${OUT}/${MARKER}_${MODEL}_${tile}"
echo "seeds figure ${MARKER} / ${MODEL}: ${IMAGE} (${DOMAIN})"
echo "  output ${DEST}"

python -m topo_i2i.seeds_figure --image "$IMAGE" --domain "$DOMAIN" \
    --n-seeds "$N_SEEDS" --outdir "$DEST" --format "$FORMAT" "${ARGS[@]}"

echo
echo "To pull it back:  tar czf seeds_${MARKER}_${MODEL}.tgz -C ${OUT} $(basename "$DEST")"
