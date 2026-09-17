#!/bin/bash
#SBATCH --job-name=topo_prep_bci
#SBATCH --output=logs_topo/prep_bci_%j.out
#SBATCH --error=logs_topo/prep_bci_%j.err

#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=clara
#SBATCH --ntasks=1
# No --gres: cropping is CPU-only.

# Tile the BCI dataset into the canonical layout the rest of the pipeline uses.
#
# BCI ships as  HE/{train,test}  and  IHC/{train,test}  -- laid out by domain
# rather than by split -- so the crop step renames on the way out.
#
# The held-out split is carved from TRAIN, not from test: test is reserved for
# the final numbers, and using it to choose field hyperparameters would leak.
# The holdout is by SOURCE IMAGE, so the four crops of one patch never straddle
# the split.
#
#   sbatch slurm/prepare_bci.sh
#   sbatch --export=ALL,VAL_FRAC=0.15 slurm/prepare_bci.sh

set -eo pipefail

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    set +u
    conda activate "${CONDA_ENV:-topocg}"
    set -u
fi

SRC=${SRC:-/work2/bz66izin-TopoCG/BIC/BCI_dataset}
OUT=${OUT:-/work2/bz66izin-TopoCG/BCI_tiles/BCI/TrainValAB}
TILE_SIZE=${TILE_SIZE:-512}
RESIZE_TO=${RESIZE_TO:-256}
VAL_FRAC=${VAL_FRAC:-0.10}
SEED=${SEED:-0}

echo "source ${SRC}"
echo "output ${OUT}"
echo

topo-crop --input "$SRC" --output "$OUT" \
    --subdirs HE/train:trainA IHC/train:trainB HE/test:testA IHC/test:testB \
    --tile_size "$TILE_SIZE" --resize_to "$RESIZE_TO"

echo
echo "--- carving val out of train (${VAL_FRAC} of source images, seed ${SEED}) ---"
python - "$OUT" "$VAL_FRAC" "$SEED" <<'PY'
import os, re, shutil, sys
out, frac, seed = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
import numpy as np

stem_of = lambda f: re.sub(r"_r\d+c\d+$", "", os.path.splitext(f)[0])
a_dir, b_dir = os.path.join(out, "trainA"), os.path.join(out, "trainB")

# Split on the stems present in BOTH domains, so a held-out patch takes its
# partner with it and the val pair stays registered.
stems_a = {stem_of(f) for f in os.listdir(a_dir)}
stems_b = {stem_of(f) for f in os.listdir(b_dir)}
shared = sorted(stems_a & stems_b)
if not shared:
    raise SystemExit("trainA and trainB share no source stems -- check the crop step")

rng = np.random.default_rng(seed)
n_val = max(1, int(round(len(shared) * frac)))
hold = set(np.array(shared)[rng.permutation(len(shared))[:n_val]].tolist())

moved = 0
for split_dir, val_name in ((a_dir, "valA"), (b_dir, "valB")):
    dst = os.path.join(out, val_name)
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(split_dir):
        if stem_of(f) in hold:
            shutil.move(os.path.join(split_dir, f), os.path.join(dst, f))
            moved += 1

for name in ("trainA", "trainB", "valA", "valB", "testA", "testB"):
    d = os.path.join(out, name)
    n = len(os.listdir(d)) if os.path.isdir(d) else 0
    print("  %-7s %6d tiles" % (name, n))
print("  held out %d of %d source images (%d tiles moved)" % (len(hold), len(shared), moved))
PY

echo
echo "next:  bash slurm/run_pipeline.sh  with"
echo "       MARKERS=BCI TILES=$(dirname "$(dirname "$OUT")")"
