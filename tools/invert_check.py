"""Does a model's generated IHC invert the H&E's tissue / lumen layout?

A third to a half of the deep-ensemble runs -- every model alike -- translate
H&E into IHC upside down in brightness: tissue rendered as lumen, lumen as
tissue, nuclei as small lumina. Nuclei recall drops to ~0.05, while FID, SSIM
and the training losses barely move. This measures it directly.

Per tile: the grey levels of the H&E and of the generated IHC, blurred so that
only the tissue / lumen layout is left, and their Pearson correlation r. Tissue
is darker than lumen in both stains, so an upright translation has r near +1 and
an inverted one r near -1. Tiles without layout contrast are skipped. A model
inverts all its tiles or none (on Ki67, every inverted member had r < 0 on
100% of its tiles, every upright one on 0%), so it is called inverted when its
median r is below 0.

Any prediction directories:

    python tools/invert_check.py --he <tiles>/Ki67/TrainValAB/valA \\
        seed4=<ens>/Ki67/topo/preds/seed4 seed6=<ens>/Ki67/topo/preds/seed6

Every deep-ensemble member, written for ensemble_report.py as
<eval-root>/<marker>/<split>/ensemble/<marker>_<model>_seed<k>/inversion.csv:

    python tools/invert_check.py --ensemble --root /work2/bz66izin-TopoCG
    python tools/invert_check.py --ensemble --root ... --ens-root .../screen_layout1 \\
        --eval-root .../eval_screen_layout1 --markers Ki67

Standard scientific Python only (numpy, scipy, PIL) -- runs on a login node.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from scipy import ndimage

BLUR = 4.0          # px at full resolution: only the tissue layout survives
MIN_STD = 0.03      # [0,1] grey levels: below this a tile has no layout
MODELS = ("topo", "cyclegan", "dclgan")
TILE_DIRS = {"BCI": "BCI_tiles", "VS": "VS_tiles"}     # everything else: MIST_tiles
EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def tiles(d: str) -> Dict[str, str]:
    """relative path without extension -> file, so nested layouts line up."""
    out = {}
    for p in glob.glob(os.path.join(d, "**", "*"), recursive=True):
        if p.lower().endswith(EXTS):
            out[os.path.splitext(os.path.relpath(p, d))[0]] = p
    return out


def grey(path: str, size=None):
    im = Image.open(path).convert("L")
    if size and im.size != size:
        im = im.resize(size, Image.BILINEAR)
    return ndimage.gaussian_filter(np.asarray(im, np.float64) / 255.0, BLUR), im.size


def tile_corrs(he: Dict[str, str], pred_dir: str, limit: int) -> np.ndarray:
    pred = tiles(pred_dir)
    common = sorted(set(he) & set(pred))
    if limit and len(common) > limit:          # evenly spread: every slide, not the first
        common = [common[int(i * len(common) / limit)] for i in range(limit)]
    r = []
    for s in common:
        a, size = grey(he[s])
        b, _ = grey(pred[s], size)
        if a.std() >= MIN_STD and b.std() >= MIN_STD:
            r.append(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    return np.array(r)


def summary(r: np.ndarray) -> Optional[dict]:
    if not len(r):
        return None
    med = float(np.median(r))
    return {"median_r": med, "frac_negative": float((r < 0).mean()),
            "n_tiles": len(r), "inverted": int(med < 0)}


def show(name: str, s: Optional[dict]) -> None:
    if s is None:
        print("  %-24s no usable tiles" % name)
    else:
        print("  %-24s median r %+.2f   tiles r<0 %3.0f%%   (%d tiles)%s"
              % (name, s["median_r"], 100 * s["frac_negative"], s["n_tiles"],
                 "   INVERTED" if s["inverted"] else ""))


def write(path: str, s: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["metric", "value"])
        for k in ("median_r", "frac_negative", "n_tiles", "inverted"):
            w.writerow([k, s[k]])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", metavar="NAME=PRED_DIR")
    p.add_argument("--he", help="the real H&E tiles the predictions were made from")
    p.add_argument("--ensemble", action="store_true", help="every deep-ensemble member")
    p.add_argument("--root", default="/work2/bz66izin-TopoCG")
    p.add_argument("--ens-root", default=None, help="default <root>/ensemble")
    p.add_argument("--eval-root", default=None, help="default <root>/eval")
    p.add_argument("--markers", nargs="+", default=["Ki67", "ER", "HER2", "PR", "BCI"])
    p.add_argument("--split", default="valA")
    p.add_argument("--limit", type=int, default=300, help="tiles per model (0 = all)")
    args = p.parse_args()

    if not args.ensemble:
        if not args.he or not args.runs:
            p.error("give --he and NAME=PRED_DIR pairs, or --ensemble")
        he = tiles(args.he)
        for spec in args.runs:
            name, d = spec.split("=", 1)
            show(name, summary(tile_corrs(he, d, args.limit)))
        return

    ens_root = args.ens_root or os.path.join(args.root, "ensemble")
    eval_root = args.eval_root or os.path.join(args.root, "eval")
    preds = "preds" if args.split == "valA" else "preds_" + args.split
    for marker in args.markers:
        he_dir = os.path.join(args.root, TILE_DIRS.get(marker, "MIST_tiles"),
                              marker, "TrainValAB", args.split)
        if not os.path.isdir(he_dir):
            print("[skip] %s: no H&E tiles at %s" % (marker, he_dir))
            continue
        he = tiles(he_dir)
        print("\n== %s / %s" % (marker, args.split))
        counts: Dict[str, List[int]] = {}
        for model in MODELS:
            for k in range(10):
                d = os.path.join(ens_root, marker, model, preds, "seed%d" % k)
                if not os.path.isdir(d):
                    continue
                name = "%s_%s_seed%d" % (marker, model, k)
                s = summary(tile_corrs(he, d, args.limit))
                show(name, s)
                if s is None:
                    continue
                write(os.path.join(eval_root, marker, args.split, "ensemble", name,
                                   "inversion.csv"), s)
                counts.setdefault(model, []).append(s["inverted"])
        for model, v in counts.items():
            print("  %-10s inverted %d/%d" % (model, sum(v), len(v)))
    print("\nwritten as inversion.csv under %s" % eval_root)


if __name__ == "__main__":
    main()
