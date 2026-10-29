"""Nuclei evaluation: do the real H&E's nuclei reappear in the generated IHC?

The generated IHC is pixel-aligned with the H&E it was made from, so the two
can be compared nucleus by nucleus -- something real H&E and real IHC, adjacent
sections, never allow. Nuclei are segmented with StarDist's pretrained models
at the settings the pilot (nuclei_pilot.py) chose by eye:

    marker          H&E (reference)        IHC (generated)
    Ki67, ER, PR    2D_versatile_he x2.5   2D_versatile_fluo on H+DAB x2
    HER2            2D_versatile_he x2.5   2D_versatile_fluo on H     x2
    BCI (HER2)      2D_versatile_he x3     2D_versatile_fluo on H     x2

OTHER DETECTORS. --detector instanseg or cpsam (tools/nuclei_detectors.py)
reads both stains as RGB with one open-weights model instead -- InstanSeg's
brightfield model was trained on IHC too, which StarDist has none for. The
scales below are starting points; nuclei_pilot.py --detector picks them. Their
H&E reference is cached apart from StarDist's (the cache is keyed by variant
and scale), and evaluate_nuclei.sh writes their results to a nuclei_<detector>/
folder inside each model's directory, so they never overwrite StarDist's.

Matching is by CENTROID DISTANCE first: a generated nucleus whose centroid is
within r pixels of a real one, one-to-one (Hungarian). The two sides are read
by different segmenters, which draw boundaries differently, and IoU would
charge that to the model; a centroid does not depend on where a boundary is
drawn. IoU-0.5 matching (the counts stardist.matching gives) is reported alongside.

    recall     share of the H&E nuclei found again in the generated IHC --
               "the structure survived"; the headline number
    precision  share of the generated nuclei that have an H&E partner. The IHC
               segmenter finds more nuclei than the H&E one even on real data
               (about 1.6-2x in the pilot), so precision is low for EVERY
               model; it compares models, it is not an absolute
    f1         their harmonic mean

Run the identity row (the H&E itself as the prediction) beside the models: a
model that changes nothing keeps every nucleus, so recall alone cannot tell
staining from copying. FID can.

WHAT IS SAVED
  <ref-cache>/<config>/<tile>.npz  H&E label masks, every tile, once per
                                   marker -- every model is matched against
                                   the same reference
  <out>/nuclei.csv                 the pooled metrics (metric,value)
  <out>/nuclei_tiles.csv           per-tile counts and matches
  <out>/nuclei_centroids.npz       every generated nucleus: tile, y, x, area --
                                   enough to recompute any centroid metric
                                   without segmenting again
  <out>/figures/<tile>.png         H&E | generated IHC | match map, for a fixed
                                   set of tissue-rich tiles, the SAME tiles for
                                   every model -- paper panels side by side
  <out>/masks/<tile>.npz           generated-IHC label masks, with --save-masks
  <out>/nuclei_topo.csv            Betti number and Betti matching errors of the
  <out>/nuclei_topo_tiles.csv      H&E vs generated nuclei masks, pooled and per
                                   tile (tools/betti_eval.py: settings fixed there)

  <out>/nuclei_positivity.csv      share of DAB-positive nuclei, real vs generated
  <out>/nuclei_positivity_tiles.csv IHC, pooled and per tile, and every generated
  <out>/nuclei_dab.npz             nucleus's mean DAB (tools/positivity_eval.py)
  <ref-cache>/real_<config>/       the REAL IHC's label masks (same segmenter as
                                   the generated side) and dab_threshold_*.json,
                                   the marker's fixed positivity threshold

POSITIVITY. The real IHC of the split (--ihc; default the B directory beside
--he: valA -> valB) is segmented with the generated IHC's own segmenter, once
per marker, and its per-nucleus DAB sets one Otsu threshold every model is
scored against. Without a real IHC directory (VS) the positivity files are not
written.

ONE PART ONLY. --only topo / --only positivity segments the generated IHC again
and writes only that part's files, leaving nuclei.csv and everything else as it
was -- how models scored before those metrics existed get them without their
recall being recomputed. --topo-only is --only topo.

Like the pilot, this is a standalone script run by path from the stardist
environment; it imports the pilot's loading and segmentation code.

    python tools/nuclei_eval.py --marker ER --he .../ER/TrainValAB/valA \\
        --pred .../Outputs_er/preds/ER_lt0.002_cyc1_trans1 \\
        --ref-cache .../eval/ER/valA/nuclei_ref --out .../eval/ER/valA/main/<name> \\
        --models-dir /work2/bz66izin-TopoCG/stardist_models
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nuclei_pilot import (DETECTORS, INK, INK_2, SURFACE, iou_match,  # noqa: E402
                          list_tiles, load_rgb, make_segmenter)
import betti_eval  # noqa: E402
import positivity_eval as pos_eval  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# marker -> (H&E variant, H&E scale, IHC variant, IHC scale), from the pilot.
CONFIG = {
    "Ki67": ("he_rgb", 2.5, "fluo_hdab", 2.0),
    "ER": ("he_rgb", 2.5, "fluo_hdab", 2.0),
    "PR": ("he_rgb", 2.5, "fluo_hdab", 2.0),
    "HER2": ("he_rgb", 2.5, "fluo_h", 2.0),
    "BCI": ("he_rgb", 3.0, "fluo_h", 2.0),
}
# The other detectors: one model on the RGB of both stains, every marker alike.
# Unpiloted defaults -- nuclei_pilot.py --detector chooses the real ones.
DETECTOR_SCALES = {"instanseg": (1.5, 1.5), "cpsam": (2.0, 2.0)}


def settings(detector: str, marker: str) -> Optional[Tuple[str, float, str, float]]:
    """(H&E variant, H&E scale, IHC variant, IHC scale), or None if unknown."""
    if detector == "stardist":
        return CONFIG.get(marker)
    he_scale, ihc_scale = DETECTOR_SCALES[detector]
    return detector, he_scale, detector, ihc_scale
RADII = (3.0, 5.0, 8.0)
PRIMARY_RADIUS = 5.0

# Match map: status colours, each also told apart by line style in the legend.
FOUND = (12, 163, 12)       # good
MISSED = (208, 59, 59)      # critical
EXTRA = (250, 178, 25)      # warning


# ---------------------------------------------------------------- matching

def nuclei_table(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (label ids, centroids [n, 2] as y, x, areas) of a label image."""
    from scipy import ndimage
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return ids, np.zeros((0, 2)), np.zeros(0)
    cents = np.asarray(ndimage.center_of_mass(np.ones_like(labels), labels, ids))
    areas = np.asarray(ndimage.sum_labels(np.ones_like(labels), labels, ids))
    return ids, cents.reshape(-1, 2), areas


def centroid_match(ref: np.ndarray, pred: np.ndarray, r: float) -> List[Tuple[int, int]]:
    """One-to-one pairs (i_ref, j_pred) with centroids at most r apart,
    chosen to maximise the number of pairs, then minimise total distance."""
    from scipy.optimize import linear_sum_assignment
    if len(ref) == 0 or len(pred) == 0:
        return []
    d = np.sqrt(((ref[:, None, :] - pred[None, :, :]) ** 2).sum(-1))
    big = 1e6
    cost = np.where(d <= r, d, big)
    rows, cols = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(rows, cols) if d[i, j] <= r]


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return r, p, f


# ---------------------------------------------------------------- reference

def atomic_savez(path: str, **arrays) -> None:
    """Write then rename, so a parallel job never reads a half-written file."""
    tmp = "%s.%d.tmp.npz" % (path[:-4], os.getpid())
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# Tiles a previous run died on. StarDist's polygon library can abort the whole
# process from C++, which Python cannot catch; resubmitting would die on the
# same tile again. So each tile is marked in flight before it is segmented and
# unmarked after: a mark found at start-up is a tile that killed a run, and it
# gets an empty mask instead of a second attempt.
#
# The H&E reference is shared: array tasks started together build it at once
# and meet each other's marks. A FRESH mark there is another job segmenting
# that tile now, not a crash -- wait for its result (`done`) rather than cache
# an empty reference. Only a mark left STALE_S without a result is a crash.
FAILED: List[str] = []
STALE_S = 600


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass                          # another job got there first


def _mark_age(mark: str) -> Optional[float]:
    try:
        return time.time() - os.path.getmtime(mark)
    except FileNotFoundError:
        return None


def guarded(seg, variant: str, img: np.ndarray, scale: float,
            inflight_dir: str, stem: str, done: Optional[str] = None) -> np.ndarray:
    os.makedirs(inflight_dir, exist_ok=True)
    mark = os.path.join(inflight_dir, stem)
    age = _mark_age(mark)
    if done is not None:
        while age is not None and age < STALE_S:
            if os.path.isfile(done):
                try:
                    return np.load(done)["labels"]
                except Exception:
                    pass              # being renamed into place; look again
            time.sleep(2)
            age = _mark_age(mark)
        if age is None and os.path.isfile(done):
            try:
                return np.load(done)["labels"]
            except Exception:
                pass
    if age is not None:
        print("  [failed] %s killed a previous run; empty mask, not retried" % stem)
        FAILED.append(stem)
        _remove(mark)
        return np.zeros(img.shape[:2], dtype=np.int32)
    open(mark, "w").close()
    labels = getattr(seg, variant)(img, scale).astype(np.int32)
    _remove(mark)
    return labels


def reference(seg, he_path: str, cache_dir: str, variant: str,
              scale: float) -> np.ndarray:
    stem = os.path.splitext(os.path.basename(he_path))[0]
    path = os.path.join(cache_dir, stem + ".npz")
    if os.path.isfile(path):
        try:
            return np.load(path)["labels"]
        except Exception:
            pass                      # unreadable: recompute and overwrite
    labels = guarded(seg, variant, load_rgb(he_path), scale,
                     os.path.join(cache_dir, ".inflight"), stem, done=path)
    atomic_savez(path, labels=labels)
    return labels


def figure_tiles(he: Dict[str, str], n: int, seed: int, seg,
                 cache_dir: str, variant: str, scale: float) -> List[str]:
    """A fixed set of tissue-rich tiles: the same for every model, because it
    depends only on the H&E tiles, the seed and the cached reference."""
    if n <= 0:
        return []
    stems = sorted(he)
    random.Random(seed).shuffle(stems)
    pool = stems[: n * 5]
    counts = {}
    for s in pool:
        ids = np.unique(reference(seg, he[s], cache_dir, variant, scale))
        counts[s] = int((ids > 0).sum())
    med = float(np.median(list(counts.values()))) if counts else 0
    return sorted([s for s in pool if counts[s] >= med][:n])


def draw_figure(path: str, title: str, he_rgb: np.ndarray, ihc_rgb: np.ndarray,
                ref: np.ndarray, pred: np.ndarray, matched_ref: set,
                matched_pred: set, ref_ids, pred_ids, counts: str) -> None:
    from skimage.segmentation import find_boundaries

    def outline(img, labels, ids, colour):
        out = img.copy()
        if len(ids):
            mask = np.isin(labels, list(ids))
            out[find_boundaries(np.where(mask, labels, 0), mode="inner")] = colour
        return out

    left = outline(he_rgb, ref, ref_ids, (255, 230, 0))
    mid = outline(ihc_rgb, pred, pred_ids, (255, 230, 0))
    right = outline(ihc_rgb, pred, matched_pred, FOUND)
    right = outline(right, pred, [i for i in pred_ids if i not in matched_pred], EXTRA)
    right = outline(right, ref, [i for i in ref_ids if i not in matched_ref], MISSED)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
    fig.patch.set_facecolor(SURFACE)
    for ax, im, t in zip(axes, (left, mid, right),
                         ("H&E · reference nuclei", "Generated IHC · nuclei",
                          "Match map")):
        ax.imshow(im)
        ax.set_title(t, fontsize=9, color=INK, loc="left")
        ax.axis("off")
    handles = [plt.Line2D([], [], color=np.array(c) / 255, linewidth=2, label=l)
               for c, l in ((FOUND, "found (H&E nucleus reappears)"),
                            (MISSED, "missed (H&E nucleus lost)"),
                            (EXTRA, "extra (no H&E partner)"))]
    fig.legend(handles=handles, loc="lower right", ncol=3, frameon=False,
               fontsize=8, labelcolor=INK_2)
    fig.suptitle("%s   %s" % (title, counts), x=0.01, ha="left", fontsize=9.5,
                 color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- topology

def write_topo(out: str, topo_rows, marker: str) -> None:
    """nuclei_topo_tiles.csv (per tile) and nuclei_topo.csv (pooled)."""
    cols = betti_eval.TILE_COLUMNS
    with open(os.path.join(out, "nuclei_topo_tiles.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("tile",) + cols)
        for s, t in topo_rows:
            t = dict(t, **betti_eval.tile_errors(t))
            w.writerow([s] + [t[c] for c in cols])
    pooled = betti_eval.pooled([t for _, t in topo_rows])
    with open(os.path.join(out, "nuclei_topo.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(sorted(pooled.items()))
        w.writerow(["topo_settings", betti_eval.SETTINGS])
    if pooled:
        print("[%s] topology: BM0 %.2f/tile (norm %.3f)  BM1 %.2f/tile  "
              "Betti-0 err %.2f  Betti-1 err %.2f"
              % (marker, pooled["bm0_mean"], pooled["bm0_norm"], pooled["bm1_mean"],
                 pooled["betti0_err_mean"], pooled["betti1_err_mean"]))


# ---------------------------------------------------------------- positivity

def real_ihc_dir(he_dir: str) -> Optional[str]:
    """.../valA -> .../valB (testA -> testB), if it exists."""
    d = os.path.normpath(he_dir)
    if not d.endswith("A"):
        return None
    b = d[:-1] + "B"
    return b if os.path.isdir(b) else None


def dab_threshold(seg, real: Dict[str, str], stems: List[str], cache_dir: str,
                  variant: str, scale: float, marker: str, grow: int) -> float:
    """The marker's fixed threshold: Otsu of the real IHC's per-nucleus DAB.
    Computed once and stored beside the real masks; every model reads it."""
    tag = pos_eval.COMPARTMENT.get(marker, "nucleus")
    path = os.path.join(cache_dir, "dab_threshold_%s%s.json"
                        % (tag, "_g%d" % grow if tag == "cell" else ""))
    if os.path.isfile(path):
        with open(path) as fh:
            return float(json.load(fh)["threshold"])
    vals = []
    t0 = time.time()
    for k, s in enumerate(stems, 1):
        lab = reference(seg, real[s], cache_dir, variant, scale)
        vals.append(pos_eval.measure(load_rgb(real[s]), lab, marker, grow))
        if k % 200 == 0 or k == len(stems):
            print("[real IHC] %d/%d  %.2fs/tile" % (k, len(stems), (time.time() - t0) / k))
    thr = pos_eval.otsu(vals)
    allv = np.concatenate(vals)
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "w") as fh:
        json.dump({"threshold": thr, "method": "otsu", "marker": marker, "compartment": tag,
                   "cell_grow": grow, "n_tiles": len(stems), "n_nuclei": int(len(allv)),
                   "positive_rate": float((allv > thr).mean())}, fh, indent=2)
    os.replace(tmp, path)
    print("[real IHC] %s DAB threshold %.4f (Otsu over %d nuclei, %.1f%% positive) -> %s"
          % (marker, thr, len(allv), 100 * (allv > thr).mean(), path))
    return thr


def write_positivity(out: str, stems, real, gen, gen_on_he, thr: float, marker: str) -> None:
    with open(os.path.join(out, "nuclei_positivity_tiles.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("tile",) + pos_eval.TILE_COLUMNS)
        for s, r, g, h in zip(stems, real, gen, gen_on_he):
            row = pos_eval.tile_row(r, g, thr, h)
            w.writerow([s] + [row[c] for c in pos_eval.TILE_COLUMNS])
    pooled = pos_eval.pooled(real, gen, thr, gen_on_he)
    with open(os.path.join(out, "nuclei_positivity.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(sorted(pooled.items()))
        w.writerow(["compartment", pos_eval.COMPARTMENT.get(marker, "nucleus")])
    np.savez_compressed(os.path.join(out, "nuclei_dab.npz"), tiles=np.array(stems),
                        tile=np.repeat(np.arange(len(gen)), [len(g) for g in gen]).astype(np.int32),
                        dab=np.concatenate(gen) if gen else np.zeros(0, np.float32),
                        threshold=thr)
    print("[%s] %s" % (marker, pos_eval.summary_line(pooled)))


# ---------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--marker", required=True, help="picks the segmentation settings")
    p.add_argument("--he", required=True, help="the real H&E split (e.g. .../valA)")
    p.add_argument("--pred", help="the generated IHC for those tiles")
    p.add_argument("--out", help="where this model's results go")
    p.add_argument("--ref-cache", required=True,
                   help="H&E reference masks, shared by every model of this marker")
    p.add_argument("--detector", choices=DETECTORS, default="stardist")
    p.add_argument("--models-dir", default=None,
                   help="fixed copy of the detector's models (nuclei_pilot.py --fetch-models)")
    p.add_argument("--ref-only", action="store_true",
                   help="build the H&E reference cache and exit -- run once per "
                        "marker before the model array, so its tasks do not all "
                        "segment the same H&E at once")
    p.add_argument("--he-scale", type=float, default=None)
    p.add_argument("--ihc-variant", choices=("he_rgb", "fluo_hdab", "fluo_h") + DETECTORS[1:],
                   default=None)
    p.add_argument("--ihc-scale", type=float, default=None)
    p.add_argument("--radii", type=float, nargs="+", default=list(RADII))
    p.add_argument("--figures", type=int, default=12,
                   help="overlay figures for this many fixed tissue-rich tiles")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-masks", action="store_true",
                   help="also keep every generated-IHC label mask (~10-20 MB per model)")
    p.add_argument("--limit", type=int, default=0, help="first N tiles only (testing)")
    p.add_argument("--only", choices=("topo", "positivity"), default=None,
                   help="write only that part's files (nuclei_topo*.csv or "
                        "nuclei_positivity*.csv); leave nuclei.csv and the rest untouched")
    p.add_argument("--topo-only", action="store_true", help="the same as --only topo")
    p.add_argument("--ihc", default=None,
                   help="the REAL IHC split for positivity (default: --he with A -> B)")
    p.add_argument("--no-positivity", action="store_true")
    p.add_argument("--dab-threshold", type=float, default=None,
                   help="instead of the Otsu threshold of the real IHC")
    p.add_argument("--cell-grow", type=int, default=pos_eval.CELL_GROW,
                   help="px a nucleus is grown by for the membranous markers (HER2, BCI)")
    args = p.parse_args()
    if args.topo_only:
        args.only = "topo"

    found = settings(args.detector, args.marker)
    if found is None and None in (args.he_scale, args.ihc_variant, args.ihc_scale):
        p.error("no settings for marker %r -- give --he-scale, --ihc-variant and "
                "--ihc-scale" % args.marker)
    he_variant, he_scale, ihc_variant, ihc_scale = found or ("he_rgb", 0, "", 0)
    he_scale = args.he_scale or he_scale
    ihc_variant = args.ihc_variant or ihc_variant
    ihc_scale = args.ihc_scale or ihc_scale
    primary = PRIMARY_RADIUS if PRIMARY_RADIUS in args.radii else args.radii[0]

    # The cache is keyed by its settings, so changing them cannot silently
    # reuse masks made with the old ones.
    cache_dir = os.path.join(args.ref_cache, "%s_x%g" % (he_variant, he_scale))
    os.makedirs(cache_dir, exist_ok=True)

    he = list_tiles(args.he)
    if not he:
        sys.exit("no tiles under %s" % args.he)
    seg = make_segmenter(args.detector, args.models_dir)

    # The real IHC, for positivity: segmented like the generated IHC, cached
    # per marker beside the H&E reference.
    real: Dict[str, str] = {}
    want_pos = not args.no_positivity and args.only != "topo"
    ihc_dir = args.ihc or real_ihc_dir(args.he)
    if want_pos and ihc_dir:
        real = list_tiles(ihc_dir)
    elif want_pos:
        print("[positivity] no real IHC beside %s (give --ihc): not scored" % args.he)
    real_cache = os.path.join(args.ref_cache, "real_%s_x%g" % (ihc_variant, ihc_scale))
    real_stems = sorted(set(he) & set(real))[: args.limit or None]
    if real and not real_stems:
        print("[positivity] no real IHC tile in %s shares a name with %s: not scored"
              % (ihc_dir, args.he))
    thr = None
    if real_stems:
        os.makedirs(real_cache, exist_ok=True)
        thr = args.dab_threshold
        if thr is None:
            thr = dab_threshold(seg, real, real_stems, real_cache, ihc_variant, ihc_scale,
                                args.marker, args.cell_grow)

    if args.ref_only:
        stems = sorted(he)[: args.limit or None]
        t0 = time.time()
        for k, s in enumerate(stems, 1):
            reference(seg, he[s], cache_dir, he_variant, he_scale)
            if k % 200 == 0 or k == len(stems):
                print("[ref] %d/%d  %.2fs/tile" % (k, len(stems), (time.time() - t0) / k))
        print("[ref] cache complete: %s  (%d blank tiles given an empty mask%s)"
              % (cache_dir, seg.blank,
                 ", %d that crashed the detector: %s" % (len(FAILED), " ".join(FAILED))
                 if FAILED else ""))
        return
    if args.only == "positivity" and thr is None:
        sys.exit("--only positivity, but there is no real IHC to compare with")

    if not (args.pred and args.out):
        p.error("--pred and --out are required unless --ref-only")
    pred = list_tiles(args.pred)
    stems = sorted(set(he) & set(pred))[: args.limit or None]
    if not stems:
        sys.exit("no tiles under %s share a name with %s" % (args.pred, args.he))
    missing = len(he) - len(stems) if not args.limit else 0
    os.makedirs(args.out, exist_ok=True)
    fig_dir = os.path.join(args.out, "figures")
    mask_dir = os.path.join(args.out, "masks")
    if args.save_masks:
        os.makedirs(mask_dir, exist_ok=True)

    fig_set = set() if args.only else set(
        figure_tiles(he, args.figures, args.seed, seg, cache_dir, he_variant, he_scale))
    if fig_set:
        os.makedirs(fig_dir, exist_ok=True)

    print("[%s] %d tiles  H&E %s x%g  |  IHC %s x%g  |  radii %s px%s"
          % (args.marker, len(stems), he_variant, he_scale, ihc_variant, ihc_scale,
             " ".join("%g" % r for r in args.radii),
             "  (%d H&E tiles have no prediction)" % missing if missing else ""))

    tot = {r: [0, 0, 0] for r in args.radii}          # tp, fp, fn
    iou = [0, 0, 0]
    iou_scores: List[float] = []
    n_ref = n_pred = 0
    tile_rows = []
    topo_rows = []
    pos_stems, pos_real, pos_gen, pos_he = [], [], [], []
    cent_tile, cent_yx, cent_area = [], [], []
    t0 = time.time()

    for k, s in enumerate(stems, 1):
        ref = reference(seg, he[s], cache_dir, he_variant, he_scale)
        he_img = load_rgb(he[s])
        img = load_rgb(pred[s])
        if img.shape[:2] != he_img.shape[:2]:
            img = np.asarray(Image.fromarray(img).resize(he_img.shape[1::-1], Image.BILINEAR))
        lab = guarded(seg, ihc_variant, img, ihc_scale,
                      os.path.join(args.out, ".inflight"), s)
        if args.only != "positivity":
            topo_rows.append((s, betti_eval.topo_compare(ref, lab)))
        if thr is not None and s in real:
            rlab = reference(seg, real[s], real_cache, ihc_variant, ihc_scale)
            pos_stems.append(s)
            pos_real.append(pos_eval.measure(load_rgb(real[s]), rlab, args.marker, args.cell_grow))
            pos_gen.append(pos_eval.measure(img, lab, args.marker, args.cell_grow))
            pos_he.append(pos_eval.measure(img, ref, args.marker, args.cell_grow)
                          if ref.shape == img.shape[:2] else np.zeros(0, np.float32))
        if args.only:
            if k % 200 == 0 or k == len(stems):
                print("  %d/%d  %.2fs/tile" % (k, len(stems), (time.time() - t0) / k))
            continue

        rid, rc, _ = nuclei_table(ref)
        pid, pc, pa = nuclei_table(lab)
        n_ref += len(rid)
        n_pred += len(pid)
        cent_tile += [k - 1] * len(pid)
        cent_yx.append(pc)
        cent_area.append(pa)

        row = [s, len(rid), len(pid)]
        primary_pairs = []
        for r in args.radii:
            pairs = centroid_match(rc, pc, r)
            tp = len(pairs)
            tot[r][0] += tp
            tot[r][1] += len(pid) - tp
            tot[r][2] += len(rid) - tp
            row.append(tp)
            if r == primary:
                primary_pairs = pairs

        tp, fp, fn, mean_iou = iou_match(ref, lab)
        iou[0] += tp
        iou[1] += fp
        iou[2] += fn
        if tp:
            iou_scores += [mean_iou] * tp
        row.append(tp)
        tile_rows.append(row)

        if args.save_masks:
            atomic_savez(os.path.join(mask_dir, s + ".npz"), labels=lab)
        if s in fig_set:
            mr = {int(rid[i]) for i, _ in primary_pairs}
            mp = {int(pid[j]) for _, j in primary_pairs}
            draw_figure(os.path.join(fig_dir, s + ".png"), "%s · %s" % (args.marker, s),
                        he_img, img, ref, lab, mr, mp, set(map(int, rid)),
                        set(map(int, pid)),
                        "H&E %d · generated %d · found %d (r=%g px)"
                        % (len(rid), len(pid), len(primary_pairs), primary))

        if k % 200 == 0 or k == len(stems):
            print("  %d/%d  %.2fs/tile" % (k, len(stems), (time.time() - t0) / k))

    if args.only != "positivity":
        write_topo(args.out, topo_rows, args.marker)
    if thr is not None:
        write_positivity(args.out, pos_stems, pos_real, pos_gen, pos_he, thr, args.marker)
    if args.only:
        return

    rec, prec, f1 = prf(*tot[primary])
    rows = [("recall", rec), ("precision", prec), ("f1", f1)]
    for r in args.radii:
        rr, pp, ff = prf(*tot[r])
        rows += [("recall_r%g" % r, rr), ("precision_r%g" % r, pp), ("f1_r%g" % r, ff)]
    ir, ip, if1 = prf(*iou)
    rows += [("iou50_recall", ir), ("iou50_precision", ip), ("iou50_f1", if1),
             ("iou50_mean_matched_iou", float(np.mean(iou_scores)) if iou_scores else 0.0),
             ("n_tiles", len(stems)), ("n_ref_nuclei", n_ref), ("n_pred_nuclei", n_pred),
             ("blank_tiles_skipped", seg.blank),
             ("failed_tiles", len(FAILED)),
             ("radius_px", primary), ("detector", args.detector), ("he_segmenter", "%s x%g" % (he_variant, he_scale)),
             ("ihc_segmenter", "%s x%g" % (ihc_variant, ihc_scale))]
    with open(os.path.join(args.out, "nuclei.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(rows)
    with open(os.path.join(args.out, "nuclei_tiles.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile", "n_ref", "n_pred"] + ["tp_r%g" % r for r in args.radii] + ["tp_iou50"])
        w.writerows(tile_rows)
    np.savez_compressed(os.path.join(args.out, "nuclei_centroids.npz"),
                        tiles=np.array(stems), tile=np.array(cent_tile, dtype=np.int32),
                        yx=np.concatenate(cent_yx) if cent_yx else np.zeros((0, 2)),
                        area=np.concatenate(cent_area) if cent_area else np.zeros(0))

    print("\n[%s] r=%g px  recall %.3f  precision %.3f  F1 %.3f   |   IoU 0.5  "
          "recall %.3f  precision %.3f  F1 %.3f" % (args.marker, primary, rec, prec, f1,
                                                     ir, ip, if1))
    print("  %d H&E nuclei, %d generated, over %d tiles -> %s"
          % (n_ref, n_pred, len(stems), args.out))
    if seg.blank:
        print("  %d blank tile(s) given an empty mask without segmenting" % seg.blank)
    if FAILED:
        print("  %d tile(s) crashed the detector in an earlier run and were left empty: %s"
              % (len(FAILED), " ".join(FAILED)))


if __name__ == "__main__":
    main()
