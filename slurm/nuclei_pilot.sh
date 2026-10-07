#!/bin/bash
#SBATCH --job-name=topo_nuclei_pilot
#SBATCH --output=logs_topo/nuclei_pilot_%j.out
#SBATCH --error=logs_topo/nuclei_pilot_%j.err

#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
# No --gres: ~20 tiles per marker at three scales runs in minutes on CPU, and
# TensorFlow on CPU avoids matching its CUDA build to the node's driver.

# Pilot for the nuclei evaluation: segments ~20 real H&E and real IHC tiles
# per marker with StarDist's pretrained models at several scales and writes an
# overlay sheet per tile, so the IHC segmenter and the scale can be chosen by
# looking. See tools/nuclei_pilot.py for what each candidate is.
#
# ONE-TIME SETUP, on the login node. StarDist runs on TensorFlow, so it gets
# its own environment rather than going into topocg's PyTorch one:
#
#   module load Anaconda3/2025.06-1
#   conda create -n stardist python=3.10 -y
#   conda activate stardist
#   P=~/.conda/envs/stardist/bin/python
#   $P -m pip install tensorflow stardist scikit-image matplotlib tifffile
#
# The env's python BY PATH, not `pip` or `python`: on this cluster an activated
# env can still leave the Anaconda module's python 3.13 first on PATH, and its
# pip then installs into ~/.local -- where every environment, topocg included,
# would see it. Tested with TensorFlow 2.21 and StarDist 0.9.2; no pin needed.
#
# and fetch both pretrained models while there is internet (compute nodes have
# none; they are cached under ~/.keras):
#
#   $P -c "from stardist.models import StarDist2D as S; \
#       S.from_pretrained('2D_versatile_he'); S.from_pretrained('2D_versatile_fluo')"
#
# THEN:
#
#   mkdir -p logs_topo
#   sbatch nuclei_pilot.sh                                  # 4 MIST markers
#   sbatch --export=ALL,MARKERS=BCI nuclei_pilot.sh
#
#   # also segment one model's fake IHC for the same tiles, and preview the
#   # match against the H&E nuclei:
#   sbatch --export=ALL,MARKERS=ER,FAKE="ER=/work2/bz66izin-TopoCG/Outputs_er/preds/ER_lt0.002_cyc1_trans1" \
#       nuclei_pilot.sh
#
# It is quick enough to run on the login node too:  bash nuclei_pilot.sh
#
# Output: ${OUT}/<marker>/{<tile>.png, counts.csv, summary.txt}

set -eo pipefail

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    conda activate "${STARDIST_ENV:-stardist}"
fi

# The environment's own interpreter, by path. On this cluster an activated env
# can still leave the Anaconda module's python first on PATH, which has no
# StarDist; CONDA_PREFIX is set by activate either way. PYTHON overrides.
PYTHON=${PYTHON:-${CONDA_PREFIX:+${CONDA_PREFIX}/bin/}python}
if ! "$PYTHON" -c "import stardist" 2>/dev/null; then
    echo "ERROR: ${PYTHON} cannot import stardist." >&2
    echo "  Set PYTHON=~/.conda/envs/stardist/bin/python, or see the setup notes above." >&2
    exit 1
fi
echo "python: ${PYTHON} ($("$PYTHON" --version 2>&1))"

ROOT=${ROOT:-/work2/bz66izin-TopoCG}
MARKERS=${MARKERS:-"Ki67 ER HER2 PR"}
OUT=${OUT:-${ROOT}/nuclei_pilot}
N=${N:-20}
SCALES=${SCALES:-"1 1.5 2"}

# Tile root per marker -- the same lookup the rest of the pipeline uses.
TILES=${TILES:-${ROOT}/MIST_tiles}
TILES_BCI=${TILES_BCI:-${ROOT}/BCI_tiles}
tiles_root() {
    local var="TILES_$1"
    echo "${!var:-$TILES}"
}

REPO_ROOT=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[ -f "${REPO_ROOT}/tools/nuclei_pilot.py" ] || REPO_ROOT="${REPO_ROOT}/.."
PILOT="${REPO_ROOT}/tools/nuclei_pilot.py"
[ -f "$PILOT" ] || { echo "ERROR: cannot find tools/nuclei_pilot.py from ${REPO_ROOT}" >&2; exit 1; }

# One call per tile root: BCI is tiled into its own.
for M in $MARKERS; do
    fake_args=()
    case " ${FAKE:-} " in *" ${M}="*)
        for f in ${FAKE}; do [ "${f%%=*}" = "$M" ] && fake_args+=(--fake "$f"); done ;;
    esac
    "$PYTHON" "$PILOT" --tiles-root "$(tiles_root "$M")" --markers "$M" \
        --n "$N" --scales $SCALES --out "$OUT" \
        ${fake_args[@]+"${fake_args[@]}"}
done

echo
echo "Look at the sheets under ${OUT}/<marker>/ and read each summary.txt."
echo "To pull them back:  rsync -av <cluster>:${OUT}/ ./nuclei_pilot/"
