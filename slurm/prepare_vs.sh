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
# WSI tiling emits background. A blank tile is not harmless: deconvolution
# amplifies sensor and compression noise into the field, and persistence then
# finds hundreds of features in it -- on a near-empty BCI tile, 972 H0 and 1690
# H1, all noise. In the audit that dilutes real signal; in training it teaches
# the generator to invent tissue. MIN_TISSUE=0 keeps everything.
MIN_TISSUE=${MIN_TISSUE:-0.10}
WHITE_LEVEL=${WHITE_LEVEL:-220}

echo "train A (H&E, unregistered) ${TRAIN_A}"
echo "train B (SR,  unregistered) ${TRAIN_B}"
echo "val   A (H&E, REGISTERED)   ${VAL_A}"
echo "val   B (SR,  REGISTERED)   ${VAL_B}"
echo "out                         ${OUT}"
echo "exclude val cases from train: ${EXCLUDE_VAL_CASES}"
echo "min tissue fraction:          ${MIN_TISSUE} (below ${WHITE_LEVEL} grey)"
echo

python - "$TRAIN_A" "$TRAIN_B" "$VAL_A" "$VAL_B" "$OUT" "$COPY" "$EXCLUDE_VAL_CASES" \
         "$MIN_TISSUE" "$WHITE_LEVEL" <<'PY'
import csv, os, shutil, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from topo_i2i.crop import tissue_fraction

(train_a, train_b, val_a, val_b, out, copy, exclude,
 min_tissue, white_level) = sys.argv[1:]
copy, exclude = copy == "1", exclude == "1"
min_tissue, white_level = float(min_tissue), int(white_level)
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

def coords(root, case):
    """{tile id: (x, y, tile_size)} from tiles_metadata.csv, or None."""
    path = os.path.join(root, case, "tiles_metadata.csv")
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                out[row["tile_name"]] = (int(row["x"]), int(row["y"]),
                                         int(row["tile_size"]))
            except (KeyError, TypeError, ValueError):
                return None
    return out or None


# The decisive step, and it only applies to the validation set: a tile in A has
# to name the same tissue as its partner in B, or the pairing every AUROC rests
# on is fictional. Training is unpaired and needs none of this.
#
# Pair on the slide COORDINATE rather than the tile id. The coordinate is the
# physical fact; the id is a counter that happens to agree only if both domains
# were tiled in the same order. Where the metadata is present we use it and
# report whether id-pairing would have agreed -- a disagreement there is exactly
# the silent failure this whole check exists to catch.
shared_cases = sorted(set(va) & set(vb))
paired, by_pos, id_agree, id_total = {}, {}, 0, 0
for c in shared_cases:
    ca, cb = coords(val_a, c), coords(val_b, c)
    if ca and cb:
        pos_b = {(x, y): t for t, (x, y, _) in cb.items()}
        hits = []
        for tile, (x, y, size) in sorted(ca.items()):
            if tile not in va[c] or (x, y) not in pos_b:
                continue
            partner = pos_b[(x, y)]
            if partner not in vb[c]:
                continue
            hits.append((tile, partner))
            by_pos[(c, tile)] = (x // size, y // size)   # (col, row) on the grid
            id_total += 1
            id_agree += (tile == partner)
        paired[c] = hits
    else:
        # No usable metadata: fall back to the id, and say so.
        paired[c] = [(t, t) for t in sorted(set(va[c]) & set(vb[c]))]

n_paired = sum(len(v) for v in paired.values())
print("\nREGISTERED PAIRS (what the audit will use)")
print("  %d cases in both domains, %d tiles paired" % (len(shared_cases), n_paired))
for c in shared_cases:
    print("    case %-6s A %5d   B %5d   paired %5d"
          % (c, len(va[c]), len(vb[c]), len(paired[c])))
if id_total:
    print("  paired on slide coordinates from tiles_metadata.csv; the tile id "
          "agreed on %d of %d (%.1f%%)" % (id_agree, id_total,
                                           100.0 * id_agree / id_total))
    if id_agree < id_total:
        print("  [NOTE] the ids do NOT always agree, so pairing by filename alone "
              "would have\n         mismatched tiles. The coordinates are right.")
else:
    print("  [WARN] no usable tiles_metadata.csv -- paired on the tile id, which "
          "assumes\n         both domains were tiled in the same order.")

if n_paired == 0:
    sys.exit(
        "\nERROR: nothing paired between the two domains for any validation case.\n"
        "The audit pairs tiles BY NAME; with nothing to pair it would fall back to\n"
        "sorted order and every AUROC would be meaningless. Check that the two\n"
        "domains cover the same slide coordinates.")


overlap = sorted(set(ta) & set(shared_cases))
if overlap:
    print("\n  Train and val share these case names: %s" % ", ".join(overlap))
    print("  %s" % ("EXCLUDING them from training (EXCLUDE_VAL_CASES=1)" if exclude
                    else "KEEPING them in training. Set EXCLUDE_VAL_CASES=1 to hold\n"
                         "  them out -- they are the only tiles that can ever give a\n"
                         "  paired metric against ground truth."))
else:
    print("\n  Train and val use different case names, so nothing overlaps.")

# ---------------------------------------------------------------------------
# Tissue survey. Every tile that might be linked gets measured, so the threshold
# can be chosen against the actual distribution rather than guessed at.
# ---------------------------------------------------------------------------
def _tissue(path):
    try:
        with Image.open(path) as im:
            return path, tissue_fraction(im, white_level)
    except Exception:
        return path, 0.0


candidates = set()
for idx in (ta, tb):
    for case, tiles in idx.items():
        if exclude and case in shared_cases:
            continue
        candidates.update(tiles.values())
for case in shared_cases:
    for tile, partner in paired[case]:
        candidates.add(va[case][tile])
        candidates.add(vb[case][partner])

# Threads, not processes: PIL releases the GIL while decoding, so this scales,
# and a process pool would try to re-import __main__ -- which is stdin here.
print("\nmeasuring tissue in %d tiles..." % len(candidates))
with ThreadPoolExecutor(min((os.cpu_count() or 4) * 2, 32)) as pool:
    tissue = dict(pool.map(_tissue, sorted(candidates)))

vals = np.sort(np.fromiter(tissue.values(), dtype=float))
print("  tissue fraction deciles: %s"
      % "  ".join("%.2f" % v for v in np.percentile(vals, np.arange(0, 101, 10))))
for t in (0.01, 0.05, 0.10, 0.25, 0.50):
    print("    %d tiles (%.1f%%) fall below %.2f"
          % (int((vals < t).sum()), 100.0 * (vals < t).mean(), t))
print("  keeping tiles at or above %.2f" % min_tissue)

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


counts, dropped = {}, {}
# Training: everything with tissue in it, unpaired, no id matching required.
for idx, sub in ((ta, "trainA"), (tb, "trainB")):
    for case, tiles in idx.items():
        if exclude and case in shared_cases:
            continue
        for tile, src in tiles.items():
            if tissue.get(src, 1.0) < min_tissue:
                dropped[sub] = dropped.get(sub, 0) + 1
                continue
            # <case>_<id>: training is unpaired, so the id is only an identifier.
            name = "%s_%s%s" % (case, tile, os.path.splitext(src)[1])
            link(src, os.path.join(out, sub, name))
            counts[sub] = counts.get(sub, 0) + 1

# Validation: a matched pair gets the SAME name in both directories, built from
# its position on the slide grid:
#
#     <case>_<blockrow>_<blockcol>_r<row within block>c<col within block>
#
# Tiles here are 512 px at stride 512, so unlike MIST there are no quadrants of
# a larger tile to group. Pairing them into 2x2 blocks of the grid manufactures
# the same thing: stripping _r<r>c<c> leaves a group of four DIRECTLY ADJACENT
# tiles. That makes VS's within-slide control the same test as MIST's and BCI's,
# so its inflation number is comparable to theirs rather than to a weaker
# case-level control.
for case in shared_cases:
    for tile, partner in paired[case]:
        # A pair is only useful if BOTH sides show tissue: correspondence
        # between a tile and an empty one is not something the audit can test.
        if min(tissue.get(va[case][tile], 1.0),
               tissue.get(vb[case][partner], 1.0)) < min_tissue:
            dropped["val"] = dropped.get("val", 0) + 1
            continue
        pos = by_pos.get((case, tile))
        if pos is None:
            name_stem = "%s_%s" % (case, tile)          # no metadata: id only
        else:
            col, row = pos
            name_stem = "%s_%d_%d_r%dc%d" % (case, row // 2, col // 2,
                                             row % 2, col % 2)
        for dom, idx, t in (("A", va, tile), ("B", vb, partner)):
            src = idx[case][t]
            name = name_stem + os.path.splitext(src)[1]
            link(src, os.path.join(out, "val" + dom, name))
            counts["val" + dom] = counts.get("val" + dom, 0) + 1

print()
for k in ("trainA", "trainB", "valA", "valB"):
    note = ""
    if k.startswith("train") and dropped.get(k):
        note = "   (%d dropped as background)" % dropped[k]
    if k.startswith("val") and dropped.get("val"):
        note = "   (%d pairs dropped: one or both sides background)" % dropped["val"]
    print("  %-7s %6d%s" % (k, counts.get(k, 0), note))

kept_pairs = counts.get("valA", 0)
if kept_pairs:
    per_slice = kept_pairs // 2
    bar = 0.5 + 2.58 * (0.16667 / max(per_slice, 1)) ** 0.5
    print("\n  after filtering: %d registered pairs -> %d per slice, bar AUROC %.3f"
          % (kept_pairs, per_slice, bar))
PY

echo
echo "Tissue was measured from the images (fraction of pixels darker than"
echo "${WHITE_LEVEL} in grey). Each case also ships a masks/ directory -- if those are"
echo "tissue masks rather than annotations, measuring from them would be both"
echo "faster and more faithful than this greyscale rule. Check one and say so."

echo
echo "================================================================"
echo "Next, with VS's own tile root. No SLIDE_REGEX is needed: the names carry"
echo "_r<row>c<col>, so the DEFAULT regex groups four adjacent tiles, the same"
echo "strict control MIST and BCI get."
echo
echo "  export MARKERS=VS"
echo "  export TILES_VS=$(dirname "$(dirname "$OUT")")"
echo "  export AUDIT=/work2/bz66izin-TopoCG/field_audit_vs"
echo "  export FIX_FIELDS_B='sirius_red/hematoxylin hematoxylin/sirius_red hematoxylin+sirius_red'"
echo "  bash run_audit.sh"
echo
echo "NOTE: the sirius_red vector in the built-in table is a GUESS with no"
echo "      source behind it, so the audit's 'fixed' arm is only as good as that"
echo "      number on this dataset. The 'estimated' arm is the one to trust here."
echo "================================================================"
