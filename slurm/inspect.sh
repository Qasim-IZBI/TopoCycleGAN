#!/bin/bash
#SBATCH --job-name=topo_inspect
#SBATCH --output=logs_topo/inspect_%j.out
#SBATCH --error=logs_topo/inspect_%j.err

#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
# No --gres: deconvolution and persistence are both CPU-only.

# Walk a handful of real validation pairs through the whole topological loss and
# write every intermediate, so you can look at what the loss is actually seeing.
# This is a diagnostic, not part of the pipeline -- nothing downstream reads its
# output.
#
# The pairs are drawn with the same matched_pairs() the field validation uses,
# with the same SEED and SAMPLE, so these tiles come from the population that
# produced the AUROC numbers rather than from wherever `ls` happens to start.
# The field settings default to the ones _sweep_common.sh trains with, so the
# pictures show the field the loss really filters.
#
#   mkdir -p logs_topo                      # SLURM will not create it for you
#   sbatch inspect.sh                                  # 4 MIST markers
#   sbatch --export=ALL,MARKERS=BCI inspect.sh         # just BCI
#   sbatch --export=ALL,MARKERS=Ki67,PAIRS=16 inspect.sh
#   MARKERS=ER PAIRS=2 bash inspect.sh                 # locally, quick
#
# Output: ${OUT}/<marker>/<tile>/ with the deconvolved channels, the merged
# image the diagram is built from, the filtered field as PNG and .npy, both
# persistence diagrams as CSV, overview.png and summary.json. A table of the
# distances lands at the end of the job log.

set -eo pipefail

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-topocg}"
fi

MARKERS=${MARKERS:-"Ki67 ER HER2 PR"}
TILES=${TILES:-/work2/bz66izin-TopoCG/MIST_tiles}
TILES_BCI=${TILES_BCI:-/work2/bz66izin-TopoCG/BCI_tiles}
STAINS_DIR=${STAINS_DIR:-/work2/bz66izin-TopoCG/field_validation_estimated}
OUT=${OUT:-/work2/bz66izin-TopoCG/inspect}

PAIRS=${PAIRS:-8}              # tiles per marker; each takes a few seconds
SPLIT=${SPLIT:-val}            # val / test / train -- which A/B pair of dirs
SEED=${SEED:-0}                # keep at the validation run's seed to inspect
SAMPLE=${SAMPLE:-random}       # tiles from the same slice it scored
OFFSET=${OFFSET:-0}

# Defaults deliberately mirror _sweep_common.sh: change them there and
# here together, or the diagnostic stops describing the training run.
FIELD_A=${FIELD_A:-stain1/stain2}
FIELD_B=${FIELD_B:-stain1+stain2}
FIELD_COMBINE=${FIELD_COMBINE:-sum}
DOWNSAMPLE=${DOWNSAMPLE:-2}
DIMS=${DIMS:-"0 1"}
PROJECTION=${PROJECTION:-birth}
IMAGE_SIZE=${IMAGE_SIZE:-256}

tiles_root() {
    case "$1" in
        BCI) echo "$TILES_BCI" ;;
        *)   echo "$TILES" ;;
    esac
}

mkdir -p "$OUT"

for MARKER in $MARKERS; do
    root="$(tiles_root "$MARKER")"
    DIR_A="${root}/${MARKER}/TrainValAB/${SPLIT}A"
    DIR_B="${root}/${MARKER}/TrainValAB/${SPLIT}B"
    if [ ! -d "$DIR_A" ] || [ ! -d "$DIR_B" ]; then
        echo "[skip] ${MARKER}: ${DIR_A} or ${DIR_B} is missing"
        continue
    fi

    stains="${STAINS_DIR}/stains_${MARKER}.json"
    stain_arg=()
    if [ -f "$stains" ]; then
        stain_arg=(--stains "$stains")
    else
        # Falling back to per-tile estimation is fine for a look, but one tile
        # per domain is a noisy Macenko -- do not read vectors off these.
        echo "[warn] ${MARKER}: ${stains} not found; estimating vectors per tile"
    fi

    echo
    echo "================ ${MARKER} ================"
    echo "  A ${DIR_A}"
    echo "  B ${DIR_B}"
    echo "  field-A ${FIELD_A}   field-B ${FIELD_B} (${FIELD_COMBINE})"
    echo "  downsample ${DOWNSAMPLE}  dims ${DIMS}  projection ${PROJECTION}"

    # Same pairing, seed and slice as topo-validate-fields, so the tiles here are
    # drawn from the population the AUROC was measured on.
    pairlist=$(python - "$DIR_A" "$DIR_B" "$PAIRS" "$OFFSET" "$SAMPLE" "$SEED" <<'PY'
import sys
from topo_i2i.validate_fields import matched_pairs
dir_a, dir_b, limit, offset, sample, seed = sys.argv[1:]
pairs, how = matched_pairs(dir_a, dir_b, limit=int(limit), offset=int(offset),
                           sample=sample, seed=int(seed))
print("#", how, file=sys.stderr)
for a, b in pairs:
    print("%s\t%s" % (a, b))
PY
)

    while IFS=$'\t' read -r a b; do
        [ -n "$a" ] || continue
        tile="${a%.*}"
        echo
        echo "--- ${MARKER} / ${tile}"
        # The +"..." guard matters: expanding an empty array is an error under
        # `set -u` on bash before 4.4, and empty is exactly the no-stains case.
        topo-inspect \
            --imageA "${DIR_A}/${a}" \
            --imageB "${DIR_B}/${b}" \
            ${stain_arg[@]+"${stain_arg[@]}"} \
            --field-A "$FIELD_A" \
            --field-B "$FIELD_B" \
            --field-combine "$FIELD_COMBINE" \
            --downsample "$DOWNSAMPLE" \
            --topo-dims $DIMS \
            --topo-projection "$PROJECTION" \
            --image-size "$IMAGE_SIZE" \
            --outdir "${OUT}/${MARKER}/${tile}"
    done <<< "$pairlist"
done

# One table of every distance computed, so the log ends with something readable
# rather than N screens of per-tile output.
echo
echo "================ summary ================"
python - "$OUT" <<'PY'
import json, os, sys
root = sys.argv[1]
rows = []
for marker in sorted(os.listdir(root)):
    mdir = os.path.join(root, marker)
    if not os.path.isdir(mdir):
        continue
    for tile in sorted(os.listdir(mdir)):
        path = os.path.join(mdir, tile, "summary.json")
        if not os.path.exists(path):
            continue
        s = json.load(open(path))
        rows.append((marker, tile, s["distance_total"], s["distance_per_dim"],
                     s["A"]["points"], s["B"]["points"]))
if not rows:
    print("no summaries found under %s" % root)
    raise SystemExit
print("%-6s %-34s %10s %10s %10s %14s %14s"
      % ("marker", "tile", "total", "H0", "H1", "A points", "B points"))
for marker, tile, total, per_dim, pa, pb in rows:
    pts = lambda d: ",".join("%s:%d" % (k, d[k]) for k in sorted(d))
    print("%-6s %-34s %10.3f %10.3f %10.3f %14s %14s"
          % (marker, tile[:34], total, per_dim.get("0", float("nan")),
             per_dim.get("1", float("nan")), pts(pa), pts(pb)))
by_marker = {}
for marker, _, total, _, _, _ in rows:
    by_marker.setdefault(marker, []).append(total)
print()
for marker, vals in sorted(by_marker.items()):
    print("%-6s n=%-3d mean %.3f   min %.3f   max %.3f"
          % (marker, len(vals), sum(vals) / len(vals), min(vals), max(vals)))
PY

echo
echo "results under ${OUT}/"
echo "To pull the figures back:  tar czf inspect.tgz -C ${OUT} ."
