#!/bin/bash
#SBATCH --job-name=topo_prep_vs
#SBATCH --output=logs_topo/prep_vs_%j.out
#SBATCH --error=logs_topo/prep_vs_%j.err

#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1

# Present the VS H&E / Sirius Red tiles in the flat layout the rest of the
# pipeline expects, WITHOUT copying or moving anything.
#
# The tiles arrive grouped by case, one directory per case:
#     QP_HE/tiles/trainA/<case>/images/<id>.tif
#     QP_SR/tiles/trainB/<case>/images/<id>.tif
# and every other script wants
#     <root>/VS/TrainValAB/{trainA,trainB,valA,valB}/<name>.tif
#
# So this builds that as a tree of SYMLINKS named <case>_<id>.tif. The case
# prefix is not decoration: it is what lets the within-slide control group tiles
# by case, via --slide-regex '_[0-9]+$'. Without it every tile would be its own
# group and the strict control could not run at all.
#
#   mkdir -p logs_topo
#   sbatch prepare_vs.sh
#   COPY=1 sbatch prepare_vs.sh     # real copies instead of symlinks
#
# It then checks the one thing that decides whether any of this is meaningful:
# whether a tile id means the SAME TISSUE in both domains. If the ids do not
# correspond, matched_pairs silently falls back to sorted order and every AUROC
# after that is measuring nothing. This refuses to finish quietly in that case.

set -eo pipefail

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-topocg}"
fi

SRC_A=${SRC_A:-/work2/bz66izin-VSproject/VS_Data/QP_HE/tiles/trainA}
SRC_B=${SRC_B:-/work2/bz66izin-VSproject/VS_Data/QP_SR/tiles/trainB}
OUT=${OUT:-/work2/bz66izin-TopoCG/VS_tiles/VS/TrainValAB}
VAL_FRAC=${VAL_FRAC:-0.15}     # held out BY CASE, never by tile
SEED=${SEED:-0}
COPY=${COPY:-0}

echo "A (H&E)       ${SRC_A}"
echo "B (SiriusRed) ${SRC_B}"
echo "out           ${OUT}"
echo

python - "$SRC_A" "$SRC_B" "$OUT" "$VAL_FRAC" "$SEED" "$COPY" <<'PY'
import os, shutil, sys
import numpy as np

src_a, src_b, out, val_frac, seed, copy = sys.argv[1:]
val_frac, seed, copy = float(val_frac), int(seed), copy == "1"
EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def index(root):
    """{case: {tile id: absolute path}} over <root>/<case>/images/."""
    out = {}
    for case in sorted(os.listdir(root)):
        img_dir = os.path.join(root, case, "images")
        if not os.path.isdir(img_dir):
            continue
        out[case] = {os.path.splitext(f)[0]: os.path.join(img_dir, f)
                     for f in sorted(os.listdir(img_dir))
                     if f.lower().endswith(EXTS)}
    return out


a, b = index(src_a), index(src_b)
print("cases: %d in A, %d in B, %d in both"
      % (len(a), len(b), len(set(a) & set(b))))

# The decisive check. A tile id has to name the same tissue in both domains, or
# the pairing that every AUROC rests on is fictional.
shared_cases = sorted(set(a) & set(b))
tot_a = sum(len(v) for v in a.values())
tot_b = sum(len(v) for v in b.values())
paired = {c: sorted(set(a[c]) & set(b[c])) for c in shared_cases}
n_paired = sum(len(v) for v in paired.values())
print("tiles: %d in A, %d in B, %d ids present in BOTH domains of the same case"
      % (tot_a, tot_b, n_paired))
if tot_a:
    print("       -> %.1f%% of A's tiles have a same-id partner" % (100.0 * n_paired / tot_a))

print("\nper case (first 10):")
for c in shared_cases[:10]:
    print("  %-6s A %5d   B %5d   shared %5d" % (c, len(a[c]), len(b[c]), len(paired[c])))

if n_paired == 0:
    sys.exit(
        "\nERROR: no tile id appears in both domains for any case.\n"
        "The audit pairs tiles BY NAME; with no shared ids it would fall back to\n"
        "sorted order and every AUROC would be meaningless. Either the two domains\n"
        "use independent numbering, or they are not registered tile-for-tile.\n"
        "Resolve that before running the audit -- see tiles_metadata.csv for the\n"
        "coordinates that would let a correct pairing be built.")
if n_paired < 0.5 * tot_a:
    print("\n[WARN] fewer than half of A's tiles have a same-id partner. Only the\n"
          "       paired ones are linked; check that this is expected.")

# Split by CASE, so a validation tile never comes from a slide seen in training.
rng = np.random.default_rng(seed)
cases = list(shared_cases)
rng.shuffle(cases)
n_val = max(1, int(round(val_frac * len(cases))))
val_cases, train_cases = set(cases[:n_val]), set(cases[n_val:])
print("\nheld out %d of %d cases for validation: %s"
      % (len(val_cases), len(cases), ", ".join(sorted(val_cases))))

for sub in ("trainA", "trainB", "valA", "valB"):
    d = os.path.join(out, sub)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)

counts = {}
for case in shared_cases:
    split = "val" if case in val_cases else "train"
    for tile in paired[case]:
        for dom, src in (("A", a[case][tile]), ("B", b[case][tile])):
            # <case>_<id> keeps the case recoverable from the filename, which is
            # what --slide-regex '_[0-9]+$' groups on.
            name = "%s_%s%s" % (case, tile, os.path.splitext(src)[1])
            dst = os.path.join(out, split + dom, name)
            if copy:
                shutil.copy2(src, dst)
            else:
                os.symlink(os.path.abspath(src), dst)
            counts[split + dom] = counts.get(split + dom, 0) + 1

print()
for k in ("trainA", "trainB", "valA", "valB"):
    print("  %-7s %6d" % (k, counts.get(k, 0)))
PY

echo
echo "A metadata file, in case it carries tile coordinates -- with those you"
echo "could group by SOURCE TILE like MIST and BCI do, a stricter control than"
echo "grouping by case:"
meta=$(find "$SRC_A" -name tiles_metadata.csv 2>/dev/null | head -1)
if [ -n "$meta" ]; then
    echo "  ${meta}"
    echo "  header: $(head -1 "$meta")"
    echo "  row:    $(sed -n 2p "$meta")"
fi

echo
echo "================================================================"
echo "Next, with VS's own tile root and a CASE-level control:"
echo
echo "  export MARKERS=VS"
echo "  export TILES_VS=$(dirname "$(dirname "$OUT")")"
echo "  export SLIDE_REGEX='_[0-9]+\$'"
echo "  export AUDIT=/work2/bz66izin-TopoCG/field_audit_vs"
echo "  export FIX_FIELDS_B='sirius_red/hematoxylin hematoxylin/sirius_red hematoxylin+sirius_red'"
echo "  bash run_audit.sh"
echo
echo "NOTE: the sirius_red vector in the built-in table is a GUESS with no"
echo "      source behind it, so the audit's 'fixed' arm is only as good as that"
echo "      number on this dataset. The 'estimated' arm is the one to trust here."
echo "================================================================"
