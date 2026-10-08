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

Slides are shuffled with a fixed seed and dealt greedily, largest first, to
whichever half has fewer tiles, so the halves come out close to 50/50 by tile
count while the assignment stays a pure function of the seed.

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
    return stem.split("_", 1)[0]


def split(stems, seed: int = 0):
    """-> {stem: 'sel' | 'test'}, slides kept whole, halves ~equal by tiles."""
    per_slide = Counter(slide_of(s) for s in stems)
    slides = sorted(per_slide)
    random.Random(seed).shuffle(slides)
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
    args = p.parse_args()

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(args.tiles)
                   if f.lower().endswith(IMG_EXTS))
    if not stems:
        raise SystemExit("no tiles under %s" % args.tiles)
    assign = split(stems, args.seed)

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
