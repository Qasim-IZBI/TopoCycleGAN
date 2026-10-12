#!/bin/bash
# Lay the 20 registered VS test slides out where every other script looks for
# a split -- ${TILES_VS}/VS/TrainValAB/{testA,testB} -- as SYMLINKS, nothing
# copied. Login node, seconds; no sbatch needed.
#
#   bash prepare_vs_test.sh
#
# The tiles stay per slide, <case>/images/<tile>.tif, as the UC tiling wrote
# them: the case number is what the test split is made on, and tile names
# repeat across cases. i2i-evaluate pairs real and generated tiles by that
# relative path, and topo-infer / i2i-inference write their predictions with
# it, so nothing needs renaming.
#
#   testA/<case>/images/<tile>  -> the H&E tiles. Images only: i2i-inference
#                                  walks the whole tree and would translate a
#                                  masks/ directory too.
#   testA/<case>/tiles_metadata.csv
#   testB/<case>/images/<tile>  -> the registered real SR tiles
#   testB/<case>/masks          -> the H&E case's masks/ DIRECTORY, linked whole.
#                                  The zoo's tissue filter finds a tile's mask by
#                                  swapping images/ for masks/ in its path, and
#                                  the real side is where it looks -- so the SR
#                                  tiles are filtered by the H&E tissue, the same
#                                  mask CPA cuts with. A directory link, because
#                                  os.walk does not follow one: FID's walk over
#                                  testB would otherwise take the masks for SR
#                                  images.
#
# First it checks the one thing the H&E masks and every paired metric rest
# on: that both stains were tiled on the SAME grid -- per case the same tile
# names at the same slide coordinates. It refuses to build anything otherwise.
#
# Then, from slurm/:
#
#   bash split_eval.sh split                                    # MARKERS=VS SPLIT=testA
#   sbatch --partition=paula --export=ALL,MARKER=VS,SPLIT=testA infer_baseline.sh
#   sbatch --partition=paula --export=ALL,MARKER=VS,SPLIT=testA,MIN_TISSUE=0.1 evaluate.sh
#
# FORCE=1 rebuilds a layout that already exists (it only ever removes links).

set -eo pipefail

TEST_A=${TEST_A:-/work2/bz66izin-UC_project/ID_HE/no_overlap/testA/tiles/testA}
TEST_B=${TEST_B:-/work2/bz66izin-UC_project/ID_SR/no_overlap/testB/tiles/testB}
TILES_VS=${TILES_VS:-/work2/bz66izin-TopoCG/VS_tiles}
OUT=${OUT:-${TILES_VS}/VS/TrainValAB}

echo "H&E (testA) ${TEST_A}"
echo "SR  (testB) ${TEST_B}"
echo "out         ${OUT}/{testA,testB}"
echo

python3 - "$TEST_A" "$TEST_B" "$OUT" "${FORCE:-0}" <<'PY'
import csv, os, shutil, sys

test_a, test_b, out, force = sys.argv[1:]
EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def grid(root, case):
    with open(os.path.join(root, case, "tiles_metadata.csv"), newline="") as f:
        return {r["tile_name"]: (int(r["x"]), int(r["y"]), int(r["tile_size"]))
                for r in csv.DictReader(f)}


def images(root, case):
    d = os.path.join(root, case, "images")
    return {os.path.splitext(f)[0]: os.path.join(d, f)
            for f in sorted(os.listdir(d)) if f.lower().endswith(EXTS)}


cases = sorted(c for c in os.listdir(test_a)
               if os.path.isfile(os.path.join(test_a, c, "tiles_metadata.csv")))
problems = []
for c in cases:
    if not os.path.isfile(os.path.join(test_b, c, "tiles_metadata.csv")):
        problems.append("%s: no SR tiling" % c)
        continue
    ga, gb = grid(test_a, c), grid(test_b, c)
    if ga != gb:
        moved = sum(1 for t in set(ga) & set(gb) if ga[t] != gb[t])
        problems.append("%s: grids differ -- %d tiles H&E only, %d SR only, %d at other coordinates"
                        % (c, len(set(ga) - set(gb)), len(set(gb) - set(ga)), moved))
    ia, ib = images(test_a, c), images(test_b, c)
    if set(ia) != set(ga) or set(ib) != set(gb):
        problems.append("%s: images and metadata disagree (H&E %d/%d, SR %d/%d)"
                        % (c, len(ia), len(ga), len(ib), len(gb)))
    if not os.path.isdir(os.path.join(test_a, c, "masks")):
        problems.append("%s: no H&E masks/" % c)
if problems:
    print("REFUSING -- the two stains are not on one grid:")
    for p in problems:
        print("  " + p)
    sys.exit(1)
print("%d cases, every one on the same grid in both stains" % len(cases))

for sub in ("testA", "testB"):
    d = os.path.join(out, sub)
    if os.path.lexists(d):
        if force != "1":
            sys.exit("%s exists -- FORCE=1 rebuilds it" % d)
        for dirpath, dirnames, filenames in os.walk(d):
            for f in filenames + [x for x in dirnames if os.path.islink(os.path.join(dirpath, x))]:
                if not os.path.islink(os.path.join(dirpath, f)) and f != "tiles_metadata.csv":
                    sys.exit("%s holds a real file (%s) -- not removing it" % (d, f))
        shutil.rmtree(d)

n = {"testA": 0, "testB": 0}
for c in cases:
    for sub, root in (("testA", test_a), ("testB", test_b)):
        d = os.path.join(out, sub, c, "images")
        os.makedirs(d)
        for stem, src in images(root, c).items():
            os.symlink(os.path.realpath(src), os.path.join(d, os.path.basename(src)))
            n[sub] += 1
        os.symlink(os.path.realpath(os.path.join(root, c, "tiles_metadata.csv")),
                   os.path.join(out, sub, c, "tiles_metadata.csv"))
    os.symlink(os.path.realpath(os.path.join(test_a, c, "masks")),
               os.path.join(out, "testB", c, "masks"))
print("linked %d H&E and %d SR tiles; testB/<case>/masks -> the H&E masks" % (n["testA"], n["testB"]))
PY

echo
echo "Done. Predictions made earlier from ${TEST_A} with SUBDIR=images"
echo "(<case>/images/<tile>) pair with this layout as they are."
