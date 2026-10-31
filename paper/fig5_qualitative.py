"""The qualitative figure: one tile per dataset, the input, the real target and
every model's translation of it, side by side.

A row is a dataset (Ki67, ER, HER2, PR, BCI, VS); the columns are the H&E
input, the real neighbouring target section, then one column per model. Every
model shows the SAME tile. Each model entry is any of

    a prediction directory   the tile is found in it by its relative path /
                             stem (topo-infer and i2i-inference keep the input's
                             layout, written as .tif) -- what was scored
    a checkpoint             a .pt, a run directory or its checkpoints/: the
                             model is run on the row's H&E here (H&E -> target).
                             For a model without predictions yet, CycleDiffusion
                             say
    an image file            used as is

Each translation is labelled in summary.json with its layout r against the
input (topo_i2i/layout.py): the figure is meant to show upright runs, and an
r < 0 is warned about.

From Python:

    qualitative_grid(rows, models, "fig5/qual.pdf")
    rows   = [{"label": "Ki67", "he": ".../valA/t.png", "target": ".../valB/t.png",
               "models": {"TopoCycleGAN": ".../ensemble_layout1_stop20k/Ki67/topo/preds/seed0",
                          "CycleGAN": "...", "DCLGAN": "...", "CycleDiffusion": "<run dir>"}},
              ...]
    models = ["TopoCycleGAN", "CycleGAN", "DCLGAN", "CycleDiffusion"]   # column order

or from a JSON file with those two keys:  python paper/fig5_qualitative.py --config fig5.json --out fig5/qual.pdf
(paper/slurm/fig5_qualitative.sh writes that JSON from the cluster's directory layout).

--crop X Y SIZE shows the same window of every panel instead of the whole tile
(pixel coordinates on the --image-size tile), for a closer look at nuclei.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# Run by path (python paper/<script>.py): the sibling scripts, and the repo's
# topo_i2i even where the package is not installed.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [p for p in (_HERE, os.path.dirname(_HERE)) if p not in sys.path]

import numpy as np
import torch
from PIL import Image

EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def is_checkpoint(path: str) -> bool:
    return (path.endswith(".pt") or os.path.isdir(os.path.join(path, "checkpoints"))
            or bool(glob.glob(os.path.join(path, "step_*.pt"))))


def find_tile(pred_dir: str, he_path: str, he_root=None) -> str:
    """The prediction of `he_path` in `pred_dir`: same path relative to the H&E
    split directory when `he_root` is given, else the same stem anywhere."""
    stem = os.path.splitext(os.path.relpath(he_path, he_root) if he_root
                            else os.path.basename(he_path))[0]
    for ext in EXTS:
        p = os.path.join(pred_dir, stem + ext)
        if os.path.isfile(p):
            return p
    base = os.path.basename(stem)
    hits = [p for p in glob.glob(os.path.join(pred_dir, "**", base + ".*"), recursive=True)
            if p.lower().endswith(EXTS)]
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError("%s: %s of %s in %s" % ("ambiguous" if hits else "no prediction",
                                                     stem, he_path, pred_dir))


def load_rgb(path: str, size: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255.0


class _Models:
    """Checkpoints loaded once, however many rows use them."""

    def __init__(self, device=None, cd_steps=50):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cd_steps, self.cache = cd_steps, {}

    def translate(self, src: str, he_path: str, size: int) -> np.ndarray:
        from fig2_loss import find_checkpoint, load_tile, to_rgb
        from fig1_seeds import load_any
        if src not in self.cache:
            ckpt = find_checkpoint(src)
            self.cache[src] = load_any(ckpt, src, self.device, cd_steps=self.cd_steps)[0]
        with torch.no_grad():
            y = self.cache[src].forward_A2B(load_tile(he_path, size).to(self.device))
        return to_rgb(y)


def layout_r(he: np.ndarray, gen: np.ndarray) -> float:
    from topo_i2i.layout import layout_corr
    t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1
    r, valid = layout_corr(t(he), t(gen))
    return float(r[0]) if bool(valid[0]) else float("nan")


def collect(rows, models, size=256, device=None, cd_steps=50):
    """Every panel as an (H, W, 3) array in [0, 1], plus where it came from."""
    runner = _Models(device, cd_steps)
    out = []
    for row in rows:
        he = load_rgb(row["he"], size)
        cells = {"H&E": (he, row["he"]), "target": (load_rgb(row["target"], size), row["target"])}
        for name in models:
            src = (row.get("models") or {}).get(name)
            if not src:
                cells[name] = (None, None)
                continue
            if os.path.isfile(src) and not src.endswith(".pt"):
                cells[name] = (load_rgb(src, size), src)
            elif is_checkpoint(src):
                cells[name] = (runner.translate(src, row["he"], size), src)
            else:
                p = find_tile(src, row["he"], row.get("he_root"))
                cells[name] = (load_rgb(p, size), p)
        r = {n: layout_r(he, cells[n][0]) for n in models if cells[n][0] is not None}
        for n, v in r.items():
            if v < 0:
                print("[warn] %s / %s: layout r %+.2f < 0 -- check the panel is not inverted"
                      % (row["label"], n, v))
        out.append({"label": row["label"], "cells": cells, "r": r})
    return out


def qualitative_grid(rows, models, out_path, size=256, crop=None, target_label="Real target",
                     input_label="H\\&E input", device=None, cd_steps=50, panel_in=1.05,
                     save_panels=True):
    """Draw the grid; returns the collected panels. Missing cells stay blank."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})

    data = collect(rows, models, size, device, cd_steps)
    cols = ["H&E", "target"] + list(models)
    heads = [input_label.replace("\\&", "&"), target_label] + list(models)
    if crop:
        x0, y0, s = crop
        cut = lambda a: a[y0:y0 + s, x0:x0 + s]
    else:
        cut = lambda a: a

    fig, axes = plt.subplots(len(data), len(cols), squeeze=False,
                             figsize=(panel_in * len(cols) + 0.3, panel_in * len(data) + 0.25),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.03})
    for i, row in enumerate(data):
        for j, c in enumerate(cols):
            ax = axes[i][j]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            img = row["cells"][c][0]
            if img is None:
                blank = cut(np.full((size, size, 3), 0.93, np.float32))
                ax.imshow(blank)
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=7, color="0.5",
                        transform=ax.transAxes)
            else:
                ax.imshow(cut(img), interpolation="lanczos")
            if i == 0:
                ax.set_title(heads[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(row["label"], fontsize=8)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    base = os.path.splitext(out_path)[0]
    if save_panels:     # every panel on its own, to assemble elsewhere
        pdir = base + "_panels"
        os.makedirs(pdir, exist_ok=True)
        for row in data:
            for c in cols:
                img = row["cells"][c][0]
                if img is not None:
                    name = "%s_%s.png" % (row["label"], c)
                    Image.fromarray((cut(img) * 255).round().astype(np.uint8)).save(
                        os.path.join(pdir, name.replace(" ", "_").replace("&", "")))
    with open(base + "_summary.json", "w") as fh:
        json.dump({"figure": os.path.abspath(out_path), "crop": crop, "columns": heads,
                   "rows": [{"label": r["label"],
                             "sources": {c: r["cells"][c][1] for c in cols},
                             "layout_r": r["r"]} for r in data]}, fh, indent=2)
    return data


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                   help='JSON with "rows" and "models" (see qualitative_grid)')
    p.add_argument("--out", required=True, help="the figure; .pdf, .svg or .png")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--crop", type=int, nargs=3, metavar=("X", "Y", "SIZE"), default=None)
    p.add_argument("--target-label", default=None,
                   help='header of the real-target column (default: the config\'s, else "Real target")')
    p.add_argument("--panel-in", type=float, default=1.05, help="inches per panel")
    p.add_argument("--cd-steps", type=int, default=50)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    cfg = json.load(open(a.config))
    data = qualitative_grid(cfg["rows"], cfg["models"], a.out, a.image_size, a.crop,
                            a.target_label or cfg.get("target_label", "Real target"),
                            device=a.device, cd_steps=a.cd_steps, panel_in=a.panel_in)
    for r in data:
        print("%-8s %s" % (r["label"], "  ".join("%s %+.2f" % kv for kv in r["r"].items())))
    print("wrote %s" % os.path.abspath(a.out))


if __name__ == "__main__":
    main()
