"""Pilot for the nuclei evaluation: which StarDist model, at which scale?

Before nuclei on the real H&E can be matched against nuclei on the generated
IHC, two choices have to be made by looking, not guessing:

  * which segmenter reads IHC. StarDist ships no IHC model, so the IHC side is
    out of domain whichever way it is done. Two candidates:
      he_rgb     2D_versatile_he on the RGB tile -- the same learned detector
                 as the H&E side, but it may miss DAB-brown nuclei;
      fluo_hdab  2D_versatile_fluo on haematoxylin + DAB optical density
                 (skimage's rgb2hed) -- sees DAB-positive nuclei, which is
                 right for the nuclear markers (Ki67, ER, PR); for HER2 the
                 DAB is on the membrane and lights up whole cells instead;
      fluo_h     2D_versatile_fluo on haematoxylin alone -- right for HER2,
                 but a strongly DAB-positive nucleus can carry too little
                 haematoxylin to be found.
    Both fluo variants are closer to the deconvolved fields the topology loss
    trains on than he_rgb is.
  * the scale. The tiles are 512px crops resized to 256, so nuclei are half
    their native size in pixels, and the pretrained models expect roughly
    40x. predict_instances(scale=s) resizes by s internally and returns
    labels at the tile's own size.

For every sampled tile this writes one overlay sheet -- H&E, real IHC by each
candidate, and optionally one model's fake IHC -- at every scale, and a CSV of
nucleus counts and sizes. Real H&E and real IHC are adjacent sections, so
their nucleus counts per tile should be similar: the IHC candidate and scale
whose counts track the H&E ones, and whose outlines look right on the sheets,
is the one to use.

With --fake, the generated IHC for the same H&E tiles is segmented as well and
matched against the H&E nuclei at IoU 0.5, as a first look at the metric
itself. That number is a preview from a handful of tiles, not a result.

This is a standalone script, not part of the topo_i2i package: StarDist needs
TensorFlow, which lives in its own environment, and importing topo_i2i would
pull in torch. Run it by path:

    python tools/nuclei_pilot.py --tiles-root /work2/bz66izin-TopoCG/MIST_tiles \\
        --markers ER --out /work2/bz66izin-TopoCG/nuclei_pilot

Output: <out>/<marker>/{<tile>.png, counts.csv, summary.txt}
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import statistics as st
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"
OUTLINE = (255, 230, 0)          # yellow: reads on pink, blue and brown alike


def list_tiles(d: str) -> Dict[str, str]:
    """stem -> path for every image directly under d."""
    if not os.path.isdir(d):
        return {}
    return {os.path.splitext(f)[0]: os.path.join(d, f)
            for f in sorted(os.listdir(d)) if f.lower().endswith(IMG_EXTS)}


def load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def nuclear_od(rgb: np.ndarray, dab: bool = True) -> np.ndarray:
    """Haematoxylin (+ DAB) optical density, normalised the way the model sees
    it: bright where the stain is, which is what the fluorescence model expects."""
    from csbdeep.utils import normalize
    from skimage.color import rgb2hed
    # Floor the RGB first. A near-black pixel (saturated scanner pixel, dust)
    # has near-infinite optical density, and a sprinkle of them sets the 99.8th
    # percentile the normalisation divides by -- on dark DAB tissue that crushed
    # the nuclei to near black (PR in the first pilot). 10/255 still leaves the
    # darkest real stain distinct.
    hed = rgb2hed(np.maximum(rgb, 10))
    od = np.clip(hed[..., 0], 0, None)
    if dab:
        od = od + np.clip(hed[..., 2], 0, None)
    return np.clip(normalize(od.astype(np.float32), 1, 99.8), 0, 1)


MODELS = ("2D_versatile_he", "2D_versatile_fluo")


def fetch_models(models_dir: str) -> None:
    """Download both pretrained models once and copy them to models_dir.

    StarDist2D.from_pretrained() unpacks its archive into ~/.keras again on
    EVERY call, rewriting weights_best.h5 -- so two jobs starting together can
    read a half-written file ("bad local heap signature"). Loading from a copy
    nothing rewrites avoids that. Run this once, on the login node.
    """
    import shutil
    from csbdeep.models.pretrained import get_model_folder
    from stardist.models import StarDist2D
    os.makedirs(models_dir, exist_ok=True)
    for name in MODELS:
        src = get_model_folder(StarDist2D, name)
        dst = os.path.join(models_dir, name)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        StarDist2D(None, name=name, basedir=models_dir)     # loads, or raises
        print("[models] %s -> %s" % (name, dst))


class Segmenter:
    def __init__(self, models_dir: Optional[str] = None):
        from stardist.models import StarDist2D
        if models_dir:
            # Read-only load from the fixed copy -- nothing is unpacked, so any
            # number of jobs can do this at once.
            load = lambda name: StarDist2D(None, name=name, basedir=models_dir)
        else:
            load = StarDist2D.from_pretrained
        self.he = load("2D_versatile_he")
        self.fluo = load("2D_versatile_fluo")

    def he_rgb(self, rgb: np.ndarray, scale: float) -> np.ndarray:
        from csbdeep.utils import normalize
        x = normalize(rgb.astype(np.float32), 1, 99.8, axis=(0, 1, 2))
        labels, _ = self.he.predict_instances(x, scale=scale, show_tile_progress=False)
        return labels

    def fluo_hdab(self, rgb: np.ndarray, scale: float) -> np.ndarray:
        labels, _ = self.fluo.predict_instances(nuclear_od(rgb, dab=True), scale=scale,
                                                show_tile_progress=False)
        return labels

    def fluo_h(self, rgb: np.ndarray, scale: float) -> np.ndarray:
        labels, _ = self.fluo.predict_instances(nuclear_od(rgb, dab=False), scale=scale,
                                                show_tile_progress=False)
        return labels


def overlay(rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    from skimage.segmentation import find_boundaries
    out = rgb.copy()
    out[find_boundaries(labels, mode="inner")] = OUTLINE
    return out


def stats(labels: np.ndarray) -> Tuple[int, float]:
    """(count, median nucleus area in tile pixels)."""
    ids, areas = np.unique(labels[labels > 0], return_counts=True)
    return len(ids), (float(np.median(areas)) if len(ids) else 0.0)


def sheet(path: str, title: str, rows: List[Tuple[str, np.ndarray, List[Tuple[str, np.ndarray]]]]):
    """rows: (row label, the image as given, [(panel title, overlay), ...])."""
    ncol = 1 + max(len(r[2]) for r in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.6 * ncol, 2.75 * len(rows)),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for i, (label, image, panels) in enumerate(rows):
        for j in range(ncol):
            ax = axes[i][j]
            ax.axis("off")
            if j == 0:
                ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
                ax.set_title(label, fontsize=8.5, color=INK, loc="left")
            elif j - 1 < len(panels):
                t, im = panels[j - 1]
                ax.imshow(im)
                ax.set_title(t, fontsize=8, color=INK_2, loc="left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles-root",
                   help="holds <marker>/TrainValAB/<split>; pass BCI's root for BCI")
    p.add_argument("--markers", nargs="+")
    p.add_argument("--split", default="valA", help="H&E split; the IHC is its A->B partner")
    p.add_argument("--n", type=int, default=20, help="tiles per marker")
    p.add_argument("--seed", type=int, default=0)
    # x1 found almost nothing on any dataset in the first pilot -- nuclei are
    # half their native size in these tiles -- and x2 still missed nuclei in
    # dense H&E, so the range starts at 1.5 and goes past 2.
    p.add_argument("--scales", type=float, nargs="+", default=[1.5, 2.0, 2.5, 3.0])
    p.add_argument("--fake", action="append", default=[], metavar="MARKER=DIR",
                   help="one model's generated IHC for that marker, segmented too and "
                        "matched against the H&E nuclei (repeatable)")
    p.add_argument("--out")
    p.add_argument("--models-dir", default=None,
                   help="load the pretrained models from this fixed copy (made by "
                        "--fetch-models) instead of ~/.keras, which from_pretrained "
                        "rewrites on every call -- required when jobs run in parallel")
    p.add_argument("--fetch-models", action="store_true",
                   help="download both models into --models-dir and exit (login node)")
    args = p.parse_args()

    if args.fetch_models:
        if not args.models_dir:
            p.error("--fetch-models needs --models-dir")
        fetch_models(args.models_dir)
        return
    for flag in ("tiles_root", "markers", "out"):
        if not getattr(args, flag):
            p.error("--%s is required" % flag.replace("_", "-"))

    fakes = dict(f.split("=", 1) for f in args.fake)
    seg = Segmenter(args.models_dir)
    os.makedirs(args.out, exist_ok=True)

    rows_out = []
    match_rows = []
    for marker in args.markers:
        base = os.path.join(args.tiles_root, marker, "TrainValAB")
        he_tiles = list_tiles(os.path.join(base, args.split))
        ihc_tiles = list_tiles(os.path.join(base, args.split[:-1] + "B"))
        fake_tiles = list_tiles(fakes[marker]) if marker in fakes else {}
        stems = sorted(set(he_tiles) & set(ihc_tiles))
        if not stems:
            print("[skip] %s: no paired tiles under %s" % (marker, base))
            continue
        # Background tiles teach nothing about nuclei, so prefer tissue: draw a
        # random pool six times the size wanted and keep its darkest tiles by
        # mean H&E intensity -- cheap, and needs no mask.
        random.Random(args.seed).shuffle(stems)
        pool = stems[: args.n * 6]
        pool.sort(key=lambda s: load_rgb(he_tiles[s]).mean())
        chosen = sorted(pool[: args.n])

        out_dir = os.path.join(args.out, marker)
        os.makedirs(out_dir, exist_ok=True)
        print("[%s] %d tiles%s" % (marker, len(chosen),
                                   ", with fake IHC from %s" % fakes[marker] if fake_tiles else ""))

        for stem in chosen:
            he = load_rgb(he_tiles[stem])
            ihc = load_rgb(ihc_tiles[stem])
            fake = load_rgb(fake_tiles[stem]) if stem in fake_tiles else None
            # The pretrained models were trained at a fixed resolution, so every
            # image is segmented at the H&E tile's size.
            if ihc.shape[:2] != he.shape[:2]:
                ihc = np.asarray(Image.fromarray(ihc).resize(he.shape[1::-1], Image.BILINEAR))
            if fake is not None and fake.shape[:2] != he.shape[:2]:
                fake = np.asarray(Image.fromarray(fake).resize(he.shape[1::-1], Image.BILINEAR))

            # The H&E side by he_rgb (in domain) and by fluo_h (haematoxylin,
            # which H&E has too) -- the latter makes the comparison symmetric:
            # the same segmenter on both sides, so a difference is the image's.
            sheet_rows = []
            he_labels = {}
            for variant, fn, shown, note in (
                    ("he_rgb", seg.he_rgb, he, ""),
                    ("fluo_h", seg.fluo_h, nuclear_od(he, dab=False), "\nH density")):
                panels = []
                for s in args.scales:
                    lab = fn(he, s)
                    he_labels[(variant, s)] = lab
                    n, a = stats(lab)
                    rows_out.append((marker, stem, "he", variant, s, n, a))
                    panels.append(("%s  x%g  n=%d" % (variant, s, n), overlay(he, lab)))
                sheet_rows.append(("H&E (real)" + note, shown, panels))

            for name, img in (("IHC (real)", ihc), ("IHC (fake)", fake)):
                if img is None:
                    continue
                kind = "ihc_real" if "real" in name else "ihc_fake"
                for variant, fn, shown, note in (
                        ("he_rgb", seg.he_rgb, img, ""),
                        ("fluo_hdab", seg.fluo_hdab, nuclear_od(img, dab=True), "\nH+DAB density"),
                        ("fluo_h", seg.fluo_h, nuclear_od(img, dab=False), "\nH density")):
                    panels = []
                    for s in args.scales:
                        lab = fn(img, s)
                        n, a = stats(lab)
                        rows_out.append((marker, stem, kind, variant, s, n, a))
                        panels.append(("%s  x%g  n=%d" % (variant, s, n), overlay(img, lab)))
                        if kind == "ihc_fake":
                            # Like with like where the H&E side has the same
                            # segmenter (fluo_h); otherwise against he_rgb.
                            ref = "fluo_h" if variant == "fluo_h" else "he_rgb"
                            from stardist.matching import matching
                            m = matching(he_labels[(ref, s)], lab, thresh=0.5)
                            match_rows.append((marker, "%s vs H&E %s" % (variant, ref),
                                               s, m.tp, m.fp, m.fn))
                    sheet_rows.append((name + note, shown, panels))

            sheet(os.path.join(out_dir, stem + ".png"), "%s · %s" % (marker, stem), sheet_rows)

    # Everything for a marker lands in one folder: sheets, counts, summary.
    for marker in args.markers:
        rows = [r for r in rows_out if r[0] == marker]
        if not rows:
            continue
        out_dir = os.path.join(args.out, marker)
        with open(os.path.join(out_dir, "counts.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["marker", "tile", "image", "variant", "scale", "n_nuclei",
                        "median_area_px"])
            w.writerows(rows)
        text = summarise(marker, args.scales, rows,
                         [m for m in match_rows if m[0] == marker])
        print("\n" + text)
        with open(os.path.join(out_dir, "summary.txt"), "w") as f:
            f.write(text + "\n")
        print("\nsheets, counts.csv and summary.txt under %s/" % out_dir)


def summarise(marker, scales, rows, match_rows) -> str:
    """For each IHC candidate and scale, its per-tile count against the H&E
    count at the same scale. Adjacent sections, so a ratio near 1 is what a
    detector seeing the same nuclei would give."""
    by = defaultdict(dict)
    for _, stem, kind, variant, s, n, a in rows:
        by[s].setdefault(stem, {})[(kind, variant)] = (n, a)
    lines = ["== %s" % marker,
             "  %-22s %6s %10s %14s %20s" % ("image / variant", "scale", "median n",
                                              "median area", "n / n(H&E), median"),
             "  (n(H&E) by the same segmenter for fluo_h, by he_rgb otherwise)"]
    for kind, variant in (("he", "he_rgb"), ("he", "fluo_h"),
                          ("ihc_real", "he_rgb"), ("ihc_real", "fluo_hdab"), ("ihc_real", "fluo_h"),
                          ("ihc_fake", "he_rgb"), ("ihc_fake", "fluo_hdab"), ("ihc_fake", "fluo_h")):
        for s in scales:
            tiles = by.get(s, {})
            vals = [t[(kind, variant)] for t in tiles.values() if (kind, variant) in t]
            if not vals:
                continue
            # Against the H&E count by the same segmenter where there is one.
            ref = ("he", "fluo_h") if variant == "fluo_h" else ("he", "he_rgb")
            ratios = [t[(kind, variant)][0] / t[ref][0]
                      for t in tiles.values()
                      if (kind, variant) in t and ref in t and t[ref][0] > 0]
            lines.append("  %-22s %6g %10g %14.0f %20s" % (
                "%s / %s" % (kind, variant), s,
                st.median(v[0] for v in vals), st.median(v[1] for v in vals),
                "%.2f" % st.median(ratios) if ratios else "-"))
    if match_rows:
        lines += ["",
                  "  preview: fake IHC vs H&E nuclei, IoU 0.5, pooled over the sampled tiles",
                  "  (a handful of tiles from one model -- a sanity check, not a result)"]
        pooled = defaultdict(lambda: [0, 0, 0])
        for _, variant, s, tp, fp, fn in match_rows:
            acc = pooled[(variant, s)]
            acc[0] += tp
            acc[1] += fp
            acc[2] += fn
        for (variant, s), (tp, fp, fn) in sorted(pooled.items()):
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            lines.append("    %-24s x%-4g precision %.2f  recall %.2f  F1 %.2f"
                         % (variant, s, prec, rec, f1))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
