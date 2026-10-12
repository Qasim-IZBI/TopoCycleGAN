"""Split a validation set into a SELECTION half and a TEST half, by slide.

The sweep picks its configuration on one half and reports it on the other, so
the reported numbers are not the ones the choice was made on. The split is at
the SLIDE level -- every tile of a slide lands in the same half -- because
neighbouring tiles of one slide look alike, and splitting by tile would leak
the test half into the selection.

A tile stem's slide is the part before the first underscore:

    MIST  14M2102785_18_7_r1c1     -> 14M2102785     (slide)
    BCI   00509_train_2+_r0c1      -> 00509          (image; BCI ships no
                                                      patient ids, so the image
                                                      is the finest unit there)

A tiling kept per slide -- <case>/images/<tile>, as the VS test set is -- has
no slide in the tile name at all (every case starts at 0000000). There the
stem is the relative path and the slide its first component, the case:

    VS    007/images/0000123       -> 007            (whole-slide image)

Slides are shuffled with a fixed seed and dealt greedily, largest first, to
whichever half has fewer tiles, so the halves come out close to 50/50 by tile
count while the assignment stays a pure function of the seed. With
--balance slides they are dealt alternately instead, so the halves hold equal
NUMBERS of slides -- what matters when the slide is the unit a metric is
reported on (VS's CPA). That is the default for a per-slide tiling.

    python tools/split_val.py --tiles /work2/.../MIST_tiles/ER/TrainValAB/valA \\
        --out /work2/.../eval/ER/valA/val_test_split.csv

Output: a CSV of tile, slide, half (sel | test). Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from collections import Counter

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def slide_of(stem: str) -> str:
    if "/" in stem:
        return stem.split("/", 1)[0]
    return stem.split("_", 1)[0]


def list_stems(root: str):
    """Image stems under root: names for a flat directory, relative paths
    (<case>/images/<tile>) for a per-slide tree. masks/ is never a tile."""
    flat = sorted(os.path.splitext(f)[0] for f in os.listdir(root)
                  if f.lower().endswith(IMG_EXTS) and os.path.isfile(os.path.join(root, f)))
    if flat:
        return flat, False
    stems = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "masks"]
        for f in filenames:
            if f.lower().endswith(IMG_EXTS):
                rel = os.path.relpath(os.path.join(dirpath, f), root)
                stems.append(os.path.splitext(rel)[0].replace(os.sep, "/"))
    return sorted(stems), True


def split(stems, seed: int = 0, balance: str = "tiles"):
    """-> {stem: 'sel' | 'test'}, slides kept whole, halves ~equal by tiles
    (or, balance='slides', by number of slides)."""
    per_slide = Counter(slide_of(s) for s in stems)
    slides = sorted(per_slide)
    random.Random(seed).shuffle(slides)
    if balance == "slides":
        half_of = {s: ("sel", "test")[i % 2] for i, s in enumerate(slides)}
        return {st: half_of[slide_of(st)] for st in stems}
    # Largest first, so the greedy fill can balance; the shuffle above breaks
    # ties between equal-sized slides, which is where the seed enters.
    slides.sort(key=lambda s: -per_slide[s])
    size = {"sel": 0, "test": 0}
    half_of = {}
    for s in slides:
        h = "sel" if size["sel"] <= size["test"] else "test"
        half_of[s] = h
        size[h] += per_slide[s]
    return {st: half_of[slide_of(st)] for st in stems}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles", required=True, help="the H&E split directory (e.g. .../valA)")
    p.add_argument("--out", required=True, help="CSV to write")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--balance", choices=("auto", "tiles", "slides"), default="auto",
                   help="equal tiles or equal slides per half; auto = slides for a "
                        "per-slide tree, tiles for a flat directory")
    args = p.parse_args()

    stems, nested = list_stems(args.tiles)
    if not stems:
        raise SystemExit("no tiles under %s" % args.tiles)
    balance = args.balance if args.balance != "auto" else ("slides" if nested else "tiles")
    assign = split(stems, args.seed, balance)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        # Plain \n: the shell side reads this with awk, and csv's default \r\n
        # would leave "test\r" in the last column, matching nothing.
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["tile", "slide", "half"])
        for s in stems:
            w.writerow([s, slide_of(s), assign[s]])

    slides = {h: len({slide_of(s) for s in stems if assign[s] == h}) for h in ("sel", "test")}
    tiles = Counter(assign.values())
    print("%s: %d tiles, %d slides -> sel %d tiles / %d slides, test %d tiles / %d slides"
          % (args.tiles, len(stems), len({slide_of(s) for s in stems}),
             tiles["sel"], slides["sel"], tiles["test"], slides["test"]))
    if min(slides.values()) < 5:
        print("  WARNING: a half has fewer than 5 slides -- its numbers rest on very few "
              "independent samples")


if __name__ == "__main__":
    main()
