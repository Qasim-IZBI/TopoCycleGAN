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
# VS has two sets, and they play different roles:
#
#   TRAIN  QP_HE/tiles/trainA/<case>/images/<id>.tif   35 cases, UNREGISTERED
#          QP_SR/tiles/trainB/<case>/images/<id>.tif
#          CycleGAN is unpaired, so training needs no correspondence at all and
#          every tile is linked regardless of whether an id exists in B.
#
#   VAL    temp/tiles/testA/<case>/images/<id>.tif      5 cases, REGISTERED
#          temp/tiles/testB/<case>/images/<id>.tif
#          The audit lives or dies on real correspondence, so ONLY these become
#          valA/valB -- the directories topo-validate-fields reads.
#
# Everything else wants
#     <root>/VS/TrainValAB/{trainA,trainB,valA,valB}/<name>.tif
#
# So this builds that as a tree of SYMLINKS named <case>_<id>.tif. The case
# prefix is not decoration: it is what lets the within-slide control group tiles
# by case, via --slide-regex '_[0-9]+$'. Without it every tile would be its own
# group and the strict control could not run at all.
#
#   mkdir -p logs_topo
#   sbatch prepare_vs.sh
#   COPY=1 sbatch prepare_vs.sh                  # real copies, not symlinks
#   EXCLUDE_VAL_CASES=1 sbatch prepare_vs.sh     # keep the registered cases out
#                                                # of training so they stay
#                                                # usable as a paired test set
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

TRAIN_A=${TRAIN_A:-/work2/bz66izin-VSproject/VS_Data/QP_HE/tiles/trainA}
TRAIN_B=${TRAIN_B:-/work2/bz66izin-VSproject/VS_Data/QP_SR/tiles/trainB}
VAL_A=${VAL_A:-/work2/bz66izin-VSproject/VS_Data/temp/tiles/testA}
VAL_B=${VAL_B:-/work2/bz66izin-VSproject/VS_Data/temp/tiles/testB}
OUT=${OUT:-/work2/bz66izin-TopoCG/VS_tiles/VS/TrainValAB}
COPY=${COPY:-0}
# The registered cases are the only tiles that can ever give you a PAIRED
# metric (SSIM, PSNR, or a real per-tile comparison against ground truth). Set
# this to 1 to keep them out of training so they stay usable as a held-out test
# set. Only meaningful if the two sets share a case numbering -- the script
# reports whether they do.
EXCLUDE_VAL_CASES=${EXCLUDE_VAL_CASES:-0}

echo "train A (H&E, unregistered) ${TRAIN_A}"
echo "train B (SR,  unregistered) ${TRAIN_B}"
echo "val   A (H&E, REGISTERED)   ${VAL_A}"
echo "val   B (SR,  REGISTERED)   ${VAL_B}"
echo "out                         ${OUT}"
echo "exclude val cases from train: ${EXCLUDE_VAL_CASES}"
echo

python - "$TRAIN_A" "$TRAIN_B" "$VAL_A" "$VAL_B" "$OUT" "$COPY" "$EXCLUDE_VAL_CASES" <<'PY'
import os, shutil, sys

train_a, train_b, val_a, val_b, out, copy, exclude = sys.argv[1:]
copy, exclude = copy == "1", exclude == "1"
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


ta, tb, va, vb = (index(d) for d in (train_a, train_b, val_a, val_b))
n = lambda idx: sum(len(v) for v in idx.values())
print("train: %d cases / %d tiles in A, %d cases / %d tiles in B"
      % (len(ta), n(ta), len(tb), n(tb)))
print("val:   %d cases / %d tiles in A, %d cases / %d tiles in B"
      % (len(va), n(va), len(vb), n(vb)))

# The decisive check, and it only applies to the validation set: a tile id has
# to name the same tissue in both domains, or the pairing every AUROC rests on
# is fictional. Training is unpaired and needs none of this.
shared_cases = sorted(set(va) & set(vb))
paired = {c: sorted(set(va[c]) & set(vb[c])) for c in shared_cases}
n_paired = sum(len(v) for v in paired.values())
print("\nREGISTERED PAIRS (what the audit will use)")
print("  %d cases in both domains, %d tile ids shared" % (len(shared_cases), n_paired))
for c in shared_cases:
    print("    case %-6s A %5d   B %5d   shared %5d"
          % (c, len(va[c]), len(vb[c]), len(paired[c])))

if n_paired == 0:
    sys.exit(
        "\nERROR: no tile id appears in both domains for any validation case.\n"
        "The audit pairs tiles BY NAME; with no shared ids it would fall back to\n"
        "sorted order and every AUROC would be meaningless. Check whether the two\n"
        "domains share a tile numbering, or build the pairing from the coordinates\n"
        "in tiles_metadata.csv.")

# With n tiles per slice the pre-registered bar is 0.5 + 2.58 * sqrt(1/6 / n).
# Say it now, so a weak verdict later is read as low power rather than no signal.
per_slice = n_paired // 2
bar = 0.5 + 2.58 * (0.16667 / max(per_slice, 1)) ** 0.5
print("\n  LIMIT=auto will give %d tiles per slice, so a candidate has to reach"
      % per_slice)
print("  AUROC %.3f to be certified. MIST cleared ~0.53-0.55 and BCI 0.558 with" % bar)
print("  1700+ tiles per slice; on %d a real effect of that size may not clear."
      % per_slice)

overlap = sorted(set(ta) & set(shared_cases))
if overlap:
    print("\n  Train and val share these case names: %s" % ", ".join(overlap))
    print("  %s" % ("EXCLUDING them from training (EXCLUDE_VAL_CASES=1)" if exclude
                    else "KEEPING them in training. Set EXCLUDE_VAL_CASES=1 to hold\n"
                         "  them out -- they are the only tiles that can ever give a\n"
                         "  paired metric against ground truth."))
else:
    print("\n  Train and val use different case names, so nothing overlaps.")

for sub in ("trainA", "trainB", "valA", "valB"):
    d = os.path.join(out, sub)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)


def link(src, dst):
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


counts = {}
# Training: everything, unpaired, no id matching required.
for idx, sub in ((ta, "trainA"), (tb, "trainB")):
    for case, tiles in idx.items():
        if exclude and case in shared_cases:
            continue
        for tile, src in tiles.items():
            # <case>_<id> keeps the case recoverable from the filename, which is
            # what --slide-regex '_[0-9]+$' groups on.
            name = "%s_%s%s" % (case, tile, os.path.splitext(src)[1])
            link(src, os.path.join(out, sub, name))
            counts[sub] = counts.get(sub, 0) + 1

# Validation: only ids present in BOTH domains, so valA[i] really pairs valB[i].
for case in shared_cases:
    for tile in paired[case]:
        for dom, idx in (("A", va), ("B", vb)):
            src = idx[case][tile]
            name = "%s_%s%s" % (case, tile, os.path.splitext(src)[1])
            link(src, os.path.join(out, "val" + dom, name))
            counts["val" + dom] = counts.get("val" + dom, 0) + 1

print()
for k in ("trainA", "trainB", "valA", "valB"):
    print("  %-7s %6d" % (k, counts.get(k, 0)))
PY

echo
echo "A metadata file, in case it carries tile coordinates -- with those you"
echo "could group by SOURCE TILE like MIST and BCI do, a stricter control than"
echo "grouping by case:"
meta=$(find "$VAL_A" -name tiles_metadata.csv 2>/dev/null | head -1)
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
