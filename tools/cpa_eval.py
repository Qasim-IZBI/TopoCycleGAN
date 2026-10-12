"""Collagen proportionate area (CPA) of a whole slide, computed tile by tile.

The VS task metric. The zoo's pipeline gets it by stitching every tile back
into a whole slide (i2i-reconstruct), segmenting the slide with the frozen
nnU-Net Dataset314_SR_light, zeroing everything outside the H&E tissue mask
(i2i-apply-he-mask), filling enclosed holes in the tissue (i2i-fill-tissue-
holes), and taking  CPA = SR / (tissue + SR)  per slide (i2i-compare-psr). The
stitched slides, the masks and nnU-Net's whole-slide logits are what made it
cost hundreds of GB.

This computes the same number without ever holding or writing a slide. A
slide's CPA is a ratio of PIXEL COUNTS, so it is enough to know, per tile, how
many pixels are tissue and how many SR -- summed over the slide, not averaged.
Every step of the old pipeline is reproduced at tile level:

  stitch      tiles are upscaled saved_size -> tile_size with Lanczos, as
              reconstruct does, onto the same non-overlapping grid. A grid
              cell without a tile is zeros, as on the stitched canvas.
  nnU-Net     nnU-Net 2D predicts with a sliding window over the WHOLE image,
              so a bare tile would be predicted without its surroundings.
              Here the window walks the slide exactly as nnU-Net's would --
              same crop to the nonzero bounding box, same steps (step 0.5),
              same Gaussian weighting, same mirroring -- but the image is
              streamed from the tiles one band of rows at a time, and the
              logits are kept for only one band plus a patch. Normalisation
              is nnU-Net's ZScoreNormalization over the whole (cropped)
              image, so the slide's mean and std come from a first pass over
              its tiles; per-tile statistics would change the prediction.
  H&E mask    pixelwise, so per tile is exact: the slide-level H&E mask was
              itself stitched from <case>/masks/<tile>.
  fill holes  a hole can span tiles, so it cannot be filled inside one. Each
              tile labels its background (4-connected, as binary_fill_holes)
              and keeps only its four edge rows; a union-find joins regions
              across tile edges, and any joined region that never reaches the
              slide border is a hole. Filling turns background into tissue,
              so only the tissue count changes -- by the hole's pixel count.
              Identical to binary_fill_holes on the stitched mask.

What can differ from the old run is arithmetic only: the old run accumulated
logits in half precision on the CPU; this accumulates in float32. --check-old
measures the difference on slides the old pipeline segmented.

    python tools/cpa_eval.py \\
        --he-tiles /work2/.../ID_HE/no_overlap/testA/tiles/testA \\
        --pred     /work2/.../Outputs_vs/preds_testA/VS_lt0.002_cyc1_trans1 \\
        --model    /work2/.../nnUNet_results/Dataset314_SR_light/nnUNetTrainer__nnUNetPlans__2d \\
        --out      /work2/.../eval/VS/testA/main/VS_lt0.002_cyc1_trans1 \\
        --ref      /work2/.../eval/VS/testA/cpa_ref

--pred is any tree of <case>/images/<tile>.<ext> on the H&E tiling's grid:
generated SR, or the real SR tiles themselves (the reference). The H&E mask
always comes from --he-tiles -- real and generated SR are cut out by the same
tissue, as in the old pipeline.

WHAT IS SAVED (per model; a few MB)
  slides/<case>.json          the slide's counts, CPA, timing, normalisation
  slides/<case>_tiles.csv     per tile: tissue, SR and filled-hole pixels
  thumbs/<case>.png / .npy    the final label map at 1/--thumb scale
  cpa_wsi.csv                 one row per slide
  cpa_tiles.csv               every tile of every slide
  cpa.csv                     with --ref: MAE, signed bias, Pearson, Spearman
                              against the reference slides (metric,value)

A slide whose slides/<case>.json exists is skipped (--force redoes it), so a
job that ran out of time resumes where it stopped.

--check-old DIR compares a finished output (normally the real-SR reference)
with the old pipeline's final whole-slide masks: CPA per slide, and label
agreement on the thumbnail grid.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = None
EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


# --------------------------------------------------------------------- tiling

class Slide:
    """One case of the H&E tiling: its grid and where each tile lives."""

    def __init__(self, he_root: str, case: str):
        self.case = case
        self.he_dir = os.path.join(he_root, case)
        meta = os.path.join(self.he_dir, "tiles_metadata.csv")
        with open(meta, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError("%s is empty" % meta)
        self.source = rows[0]["source_file"].strip("'\"")
        sizes = {int(r["tile_size"]) for r in rows}
        if len(sizes) != 1:
            raise ValueError("%s: mixed tile sizes %s" % (meta, sorted(sizes)))
        self.ts = sizes.pop()
        self.tiles: Dict[Tuple[int, int], str] = {}          # (row, col) -> tile name
        self.xy: Dict[str, Tuple[int, int]] = {}
        for r in rows:
            x, y = int(r["x"]), int(r["y"])
            if x % self.ts or y % self.ts:
                raise ValueError("%s: tile %s at (%d, %d) is off the %d px grid"
                                 % (meta, r["tile_name"], x, y, self.ts))
            self.tiles[(y // self.ts, x // self.ts)] = r["tile_name"]
            self.xy[r["tile_name"]] = (x, y)
        # The stitched canvas ends at the last tile, as in reconstruct.
        self.R = max(rc[0] for rc in self.tiles) + 1
        self.C = max(rc[1] for rc in self.tiles) + 1
        self.H, self.W = self.R * self.ts, self.C * self.ts

    def check_grid(self, pred_root: str) -> None:
        """If the prediction side carries its own tiling metadata (the real SR
        does), it must put every tile where the H&E does -- else the H&E mask
        cuts out the wrong tissue."""
        meta = os.path.join(pred_root, self.case, "tiles_metadata.csv")
        if not os.path.isfile(meta):
            return
        with open(meta, newline="") as f:
            other = {r["tile_name"]: (int(r["x"]), int(r["y"])) for r in csv.DictReader(f)}
        bad = [t for t, xy in self.xy.items() if other.get(t) != xy]
        if bad or len(other) != len(self.xy):
            raise ValueError("%s: grid differs from the H&E's (%d of %d tiles disagree, %d vs %d "
                             "tiles)" % (meta, len(bad), len(self.xy), len(other), len(self.xy)))


def find_file(directory: str, stem: str, ext_hint: List[str]) -> Optional[str]:
    """<directory>/<stem><ext>, trying the extension that worked last first."""
    for ext in ext_hint + [e for e in EXTS if e not in ext_hint]:
        p = os.path.join(directory, stem + ext)
        if os.path.isfile(p):
            if ext_hint[:1] != [ext]:
                ext_hint[:] = [ext]
            return p
    return None


class TileSource:
    """Loads a slide's tiles at stitched resolution, as reconstruct places them."""

    def __init__(self, slide: Slide, pred_root: str, workers: int):
        self.slide = slide
        self.img_dir = os.path.join(pred_root, slide.case, "images")
        self.mask_dir = os.path.join(slide.he_dir, "masks")
        self.pool = ThreadPoolExecutor(workers)
        self._img_ext: List[str] = []
        self._mask_ext: List[str] = []
        self.missing_pred: List[str] = []
        self.missing_mask = 0

    def rgb(self, rc: Tuple[int, int]) -> Optional[np.ndarray]:
        name = self.slide.tiles.get(rc)
        if name is None:
            return None
        path = find_file(self.img_dir, name, self._img_ext)
        if path is None:
            self.missing_pred.append(name)
            return None
        with Image.open(path) as im:
            im = im.convert("RGB")
            ts = self.slide.ts
            if im.size != (ts, ts):
                im = im.resize((ts, ts), Image.LANCZOS)
            return np.asarray(im, dtype=np.uint8)

    def tissue(self, rc: Tuple[int, int]) -> Optional[np.ndarray]:
        name = self.slide.tiles.get(rc)
        if name is None:
            return None
        path = find_file(self.mask_dir, name, self._mask_ext)
        if path is None:
            self.missing_mask += 1
            return None
        with Image.open(path) as im:
            ts = self.slide.ts
            if im.size != (ts, ts):
                im = im.resize((ts, ts), Image.NEAREST)
            a = np.asarray(im)
        return (a[..., 0] if a.ndim == 3 else a) > 0

    def row(self, r: int) -> List[Optional[np.ndarray]]:
        return list(self.pool.map(self.rgb, [(r, c) for c in range(self.slide.C)]))


# ------------------------------------------------------- pass 1: crop + z-score

def slide_stats(src: TileSource) -> Tuple[Tuple[int, int, int, int], np.ndarray, np.ndarray]:
    """nnU-Net crops to the bounding box of nonzero pixels, then z-scores each
    channel over that box. Every nonzero pixel is inside the box, so the sums
    over the whole canvas ARE the sums over the box; only the pixel count
    needs the box. -> (y0, y1, x0, x1), mean[3], std[3]."""
    s = np.zeros(3)
    q = np.zeros(3)
    y0 = x0 = 1 << 60
    y1 = x1 = -1
    ts = src.slide.ts
    for r in range(src.slide.R):
        for c, a in enumerate(src.row(r)):
            if a is None:
                continue
            f = a.reshape(-1, 3).astype(np.float64)
            s += f.sum(0)
            q += (f * f).sum(0)
            nz = a.any(axis=2)
            rows, cols = np.flatnonzero(nz.any(1)), np.flatnonzero(nz.any(0))
            if rows.size:
                y0, y1 = min(y0, r * ts + rows[0]), max(y1, r * ts + rows[-1] + 1)
                x0, x1 = min(x0, c * ts + cols[0]), max(x1, c * ts + cols[-1] + 1)
    if y1 < 0:
        raise ValueError("slide %s is entirely zero" % src.slide.case)
    n = float((y1 - y0) * (x1 - x0))
    mean = s / n
    std = np.sqrt(np.maximum(q / n - mean * mean, 0.0))
    return (int(y0), int(y1), int(x0), int(x1)), mean, std


# ------------------------------------------------------ pass 2: sliding window

def stream_slide(src: TileSource, bbox, mean, std, predict: Callable, to_seg: Callable,
                 patch: int, steps: Tuple[List[int], List[int]], gaussian, batch: int,
                 device, on_band: Callable) -> None:
    """Walk nnU-Net's windows over the slide, top to bottom, holding only the
    logits of one tile row plus a patch. When the next window row starts
    below a tile row, nothing can add to that row any more: it is turned
    into labels and handed to on_band(row, labels[ts, W]) -- outside the
    crop box (no window ever covered it) the label is 0, as nnU-Net leaves it
    when it reverts the crop."""
    import torch

    ts, W = src.slide.ts, src.slide.W
    y0, y1, x0, x1 = bbox
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(3, 1, 1)
    std_t = torch.tensor(np.maximum(std, 1e-8), dtype=torch.float32, device=device).view(3, 1, 1)
    gauss = gaussian.to(device=device, dtype=torch.float32)

    hb = ts + patch
    logits = weight = None
    buf_y0 = 0
    cache: Dict[int, np.ndarray] = {}

    def tile_row(r: int) -> np.ndarray:
        if r not in cache:
            out = np.zeros((ts, W, 3), dtype=np.uint8)
            for c, a in enumerate(src.row(r)):
                if a is not None:
                    out[:, c * ts:(c + 1) * ts] = a
            cache[r] = out
        return cache[r]

    def finalize() -> None:
        nonlocal buf_y0
        r = buf_y0 // ts
        if logits is None:
            on_band(r, np.zeros((ts, W), dtype=np.uint8))
        else:
            w = weight[:ts]
            band = to_seg(logits[:, :ts] / w.clamp_min(1e-12))
            band = torch.where(w > 0, band, torch.zeros_like(band))
            on_band(r, band.to("cpu").numpy().astype(np.uint8))
            logits[:, :hb - ts] = logits[:, ts:].clone()
            logits[:, hb - ts:] = 0
            weight[:hb - ts] = weight[ts:].clone()
            weight[hb - ts:] = 0
        for k in [k for k in cache if k <= r]:
            del cache[k]
        buf_y0 += ts

    for sy in steps[0]:
        y = y0 + sy
        while buf_y0 + ts <= y:
            finalize()
        # The window rows [y, y + patch) of the crop box, normalised.
        r0, r1 = y // ts, (y + patch - 1) // ts
        strip = np.concatenate([tile_row(r) for r in range(r0, r1 + 1)], axis=0)
        strip = strip[y - r0 * ts: y - r0 * ts + patch, x0:x1]
        img = torch.from_numpy(np.ascontiguousarray(strip)).to(device)
        img = (img.permute(2, 0, 1).float() - mean_t) / std_t
        for i in range(0, len(steps[1]), batch):
            xs = steps[1][i:i + batch]
            out = predict(torch.stack([img[:, :, sx:sx + patch] for sx in xs]))
            if logits is None:
                logits = torch.zeros((out.shape[1], hb, W), dtype=torch.float32, device=device)
                weight = torch.zeros((hb, W), dtype=torch.float32, device=device)
            yy = y - buf_y0
            for j, sx in enumerate(xs):
                xx = x0 + sx
                logits[:, yy:yy + patch, xx:xx + patch] += out[j].float() * gauss
                weight[yy:yy + patch, xx:xx + patch] += gauss
    while buf_y0 < src.slide.H:
        finalize()


# ---------------------------------------------------------- per-tile counting

class Counter:
    """H&E mask, counts and the hole bookkeeping, one tile at a time.

    A tile keeps its tissue / SR counts, the size of each 4-connected
    background region and the region labels along its four edges -- a few KB
    -- instead of its mask."""

    def __init__(self, src: TileSource, thumb: int):
        self.src = src
        s = src.slide
        self.k = thumb
        self.thumb = np.zeros((math.ceil(s.H / thumb), math.ceil(s.W / thumb)), dtype=np.uint8)
        self.counts: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.sizes: Dict[Tuple[int, int], np.ndarray] = {}
        self.edges: Dict[Tuple[int, int], Tuple[np.ndarray, ...]] = {}

    def band(self, r: int, labels: np.ndarray) -> None:
        ts, k = self.src.slide.ts, self.k
        for c in range(self.src.slide.C):
            seg = labels[:, c * ts:(c + 1) * ts].copy()
            tissue = self.src.tissue((r, c))
            if tissue is None:
                seg[:] = 0                       # no H&E mask tile: zero, as on the canvas
            else:
                seg[~tissue] = 0
            self.add((r, c), seg)
            self.thumb[(r * ts) // k:((r + 1) * ts) // k,
                       (c * ts) // k:((c + 1) * ts) // k] = seg[::k, ::k]

    def add(self, rc: Tuple[int, int], seg: np.ndarray) -> None:
        self.counts[rc] = (int((seg == 1).sum()), int((seg == 2).sum()))
        lab, n = ndimage.label(seg == 0)         # default structure: 4-connected
        self.sizes[rc] = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(np.int64)
        self.edges[rc] = (lab[0].copy(), lab[-1].copy(), lab[:, 0].copy(), lab[:, -1].copy())

    def holes(self) -> Dict[Tuple[int, int], int]:
        """Filled-hole pixels per tile: background regions, joined across tile
        edges, that never touch the slide border."""
        R, C = self.src.slide.R, self.src.slide.C
        offset, total = {}, 0
        for rc in sorted(self.sizes):
            offset[rc] = total
            total += len(self.sizes[rc])
        parent = np.arange(total)

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def join(a: np.ndarray, oa: int, b: np.ndarray, ob: int) -> None:
            keep = (a > 0) & (b > 0)
            for u, v in set(zip(a[keep].tolist(), b[keep].tolist())):
                ru, rv = find(oa + u - 1), find(ob + v - 1)
                if ru != rv:
                    parent[ru] = rv

        for (r, c), (top, bottom, left, right) in self.edges.items():
            if c + 1 < C:
                join(right, offset[(r, c)], self.edges[(r, c + 1)][2], offset[(r, c + 1)])
            if r + 1 < R:
                join(bottom, offset[(r, c)], self.edges[(r + 1, c)][0], offset[(r + 1, c)])
        outside = set()
        for (r, c), (top, bottom, left, right) in self.edges.items():
            o = offset[(r, c)]
            for edge, on_border in ((top, r == 0), (bottom, r == R - 1),
                                    (left, c == 0), (right, c == C - 1)):
                if on_border:
                    outside.update(find(o + v - 1) for v in np.unique(edge[edge > 0]).tolist())
        out = {}
        for rc, sizes in self.sizes.items():
            o = offset[rc]
            hole = [i for i in range(len(sizes)) if find(o + i) not in outside]
            out[rc] = int(sizes[hole].sum()) if hole else 0
        return out


# ------------------------------------------------------------------- nnU-Net

def load_nnunet(model_dir: str, fold: str, checkpoint: str, device, mirror: bool, amp: bool):
    """The trained network and the pieces of nnU-Net's own inference this
    reuses: patch size, mirroring axes, the Gaussian, the window steps and
    the logits -> labels conversion (sigmoid regions here)."""
    import itertools

    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.inference.sliding_window_prediction import (compute_gaussian,
                                                              compute_steps_for_sliding_window)

    pr = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=mirror,
                         perform_everything_on_device=True, device=device, verbose=False,
                         verbose_preprocessing=False, allow_tqdm=False)
    pr.initialize_from_trained_model_folder(model_dir, use_folds=(int(fold) if fold.isdigit() else fold,),
                                            checkpoint_name=checkpoint)
    cm = pr.configuration_manager
    norm = list(cm.normalization_schemes)
    if any(n != "ZScoreNormalization" for n in norm) or any(cm.use_mask_for_norm):
        sys.exit("this reproduces ZScoreNormalization without a mask only; the plans say %s / %s"
                 % (norm, cm.use_mask_for_norm))
    if list(pr.plans_manager.transpose_forward) != [0, 1, 2]:
        sys.exit("plans transpose the image (%s); not supported" % pr.plans_manager.transpose_forward)
    patch = list(cm.patch_size)
    if len(patch) != 2 or patch[0] != patch[1]:
        sys.exit("expected a square 2D patch, got %s" % patch)

    net = pr.network
    getattr(net, "_orig_mod", net).load_state_dict(pr.list_of_parameters[0])
    net = net.to(device).eval()
    axes = [m + 2 for m in (pr.allowed_mirroring_axes or ())] if mirror else []
    combos = [c for i in range(len(axes)) for c in itertools.combinations(axes, i + 1)]
    autocast = amp and device.type == "cuda"

    @torch.inference_mode()
    def predict(x):
        # nnU-Net's _internal_maybe_mirror_and_predict, batched.
        with torch.autocast(device.type, enabled=autocast):
            p = net(x).float()
            for ax in combos:
                p += torch.flip(net(torch.flip(x, ax)), ax).float()
        return p / (len(combos) + 1)

    @torch.inference_mode()
    def to_seg(logits):
        return pr.label_manager.convert_logits_to_segmentation(logits)

    gaussian = compute_gaussian(tuple(patch), sigma_scale=1. / 8, value_scaling_factor=10,
                                device=device)
    steps = lambda h, w: compute_steps_for_sliding_window((h, w), tuple(patch), 0.5)
    info = {"patch": patch, "mirror_axes": list(pr.allowed_mirroring_axes or ()) if mirror else [],
            "normalization": norm, "regions": bool(pr.label_manager.has_regions)}
    return predict, to_seg, patch[0], gaussian, steps, info


# --------------------------------------------------------------------- driver

def run_slide(slide: Slide, args, predict, to_seg, patch, gaussian, steps_fn, device, info) -> dict:
    t0 = time.time()
    slide.check_grid(args.pred)
    src = TileSource(slide, args.pred, args.workers)
    bbox, mean, std = slide_stats(src)
    if src.missing_pred:
        raise FileNotFoundError("%s: %d of %d tiles have no prediction under %s (first: %s)"
                                % (slide.case, len(src.missing_pred), len(slide.tiles),
                                   src.img_dir, src.missing_pred[0]))
    y0, y1, x0, x1 = bbox
    if y1 - y0 < patch or x1 - x0 < patch:
        raise ValueError("%s: the image is smaller than one %d px patch" % (slide.case, patch))
    t1 = time.time()
    counter = Counter(src, args.thumb)
    steps = steps_fn(y1 - y0, x1 - x0)
    stream_slide(src, bbox, mean, std, predict, to_seg, patch, steps, gaussian, args.batch,
                 device, counter.band)
    holes = counter.holes()
    t2 = time.time()

    rows = []
    tot = [0, 0, 0]
    for rc in sorted(counter.counts):
        n1, n2 = counter.counts[rc]
        h = holes.get(rc, 0)
        tot[0] += n1
        tot[1] += n2
        tot[2] += h
        name = slide.tiles.get(rc)
        if name is not None or n1 or n2 or h:
            rows.append([slide.case, name or "", rc[1] * slide.ts, rc[0] * slide.ts, n1, n2, h])
    denom = tot[0] + tot[1] + tot[2]
    return {
        "case": slide.case, "source_file": slide.source, "tiles": len(slide.tiles),
        "grid": [slide.R, slide.C], "tile_size": slide.ts, "crop_bbox": [y0, y1, x0, x1],
        "mean": mean.tolist(), "std": std.tolist(), "windows": len(steps[0]) * len(steps[1]),
        "tissue_px": tot[0], "sr_px": tot[1], "hole_px": tot[2],
        "cpa": tot[1] / denom if denom else None,
        "missing_mask_tiles": src.missing_mask,
        "seconds": {"stats": round(t1 - t0, 1), "segment": round(t2 - t1, 1)},
        "nnunet": info, "_rows": rows, "_thumb": counter.thumb,
    }


def write_slide(out: str, res: dict, thumb_k: int) -> None:
    os.makedirs(os.path.join(out, "slides"), exist_ok=True)
    os.makedirs(os.path.join(out, "thumbs"), exist_ok=True)
    case = res["case"]
    with open(os.path.join(out, "slides", case + "_tiles.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["case", "tile", "x", "y", "tissue_px", "sr_px", "hole_px"])
        w.writerows(res.pop("_rows"))
    thumb = res.pop("_thumb")
    np.save(os.path.join(out, "thumbs", case + ".npy"), thumb)
    palette = np.array([[255, 255, 255], [225, 205, 205], [180, 20, 40]], dtype=np.uint8)
    Image.fromarray(palette[np.minimum(thumb, 2)]).save(os.path.join(out, "thumbs", case + ".png"))
    res["thumb_scale"] = thumb_k
    tmp = os.path.join(out, "slides", case + ".json.tmp")
    with open(tmp, "w") as f:
        json.dump(res, f, indent=2)
    os.replace(tmp, os.path.join(out, "slides", case + ".json"))   # last: marks the slide done


def read_slides(out: str) -> Dict[str, dict]:
    res = {}
    for p in sorted(glob.glob(os.path.join(out, "slides", "*.json"))):
        with open(p) as f:
            d = json.load(f)
        res[d["case"]] = d
    return res


def pearson(a: List[float], b: List[float]) -> Optional[float]:
    n = len(a)
    if n < 3:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb) if sa and sb else None


def ranks(v: List[float]) -> List[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def aggregate(out: str, ref: Optional[str]) -> None:
    slides = read_slides(out)
    with open(os.path.join(out, "cpa_wsi.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["case", "source_file", "tiles", "tissue_px", "sr_px", "hole_px", "cpa"])
        for c, d in sorted(slides.items()):
            w.writerow([c, d["source_file"], d["tiles"], d["tissue_px"], d["sr_px"], d["hole_px"],
                        "" if d["cpa"] is None else "%.6f" % d["cpa"]])
    with open(os.path.join(out, "cpa_tiles.csv"), "w", newline="") as f:
        f.write("case,tile,x,y,tissue_px,sr_px,hole_px\n")
        for c in sorted(slides):
            with open(os.path.join(out, "slides", c + "_tiles.csv")) as g:
                next(g)
                f.writelines(g)
    print("\n%-6s %-34s %9s" % ("case", "slide", "CPA"))
    for c, d in sorted(slides.items()):
        print("%-6s %-34s %9s" % (c, d["source_file"][:34],
                                 "-" if d["cpa"] is None else "%.4f" % d["cpa"]))
    if not ref:
        return
    refs = read_slides(ref)
    common = sorted(c for c in set(slides) & set(refs)
                    if slides[c]["cpa"] is not None and refs[c]["cpa"] is not None)
    if not common:
        print("\nno slide in common with the reference %s yet" % ref)
        return
    g = [slides[c]["cpa"] for c in common]
    r = [refs[c]["cpa"] for c in common]
    d = [x - y for x, y in zip(g, r)]
    mae = sum(abs(x) for x in d) / len(d)
    bias = sum(d) / len(d)
    pr, sr = pearson(g, r), pearson(ranks(g), ranks(r))
    with open(os.path.join(out, "cpa.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["metric", "value"])
        w.writerow(["cpa_mae", "%.6f" % mae])
        w.writerow(["cpa_bias", "%.6f" % bias])
        w.writerow(["cpa_pearson", "" if pr is None else "%.4f" % pr])
        w.writerow(["cpa_spearman", "" if sr is None else "%.4f" % sr])
        w.writerow(["n_slides", len(common)])
    print("\nagainst %s, %d slides: CPA MAE %.4f  bias %+.4f  Pearson %s  Spearman %s"
          % (ref, len(common), mae, bias, "-" if pr is None else "%.3f" % pr,
             "-" if sr is None else "%.3f" % sr))
    missing = sorted(set(refs) - set(slides))
    if missing:
        print("  (reference slides not done here yet: %s)" % " ".join(missing))


def check_old(out: str, old_dir: str) -> None:
    """The tile pipeline against the old whole-slide masks (labels 0/1/2 after
    H&E masking and hole filling): CPA per slide, and the labels on the
    thumbnail grid."""
    import tifffile

    slides = read_slides(out)
    if not slides:
        sys.exit("nothing computed under %s" % out)
    files = {os.path.splitext(os.path.basename(p))[0].strip("'\""): p
             for p in glob.glob(os.path.join(old_dir, "*")) if p.lower().endswith((".tif", ".tiff"))}
    print("%-6s %-32s %9s %9s %9s %8s %8s" % ("case", "slide", "CPA old", "CPA new", "diff",
                                            "agree", "SR Dice"))
    diffs = []
    for c, d in sorted(slides.items()):
        stem = os.path.splitext(d["source_file"])[0].strip("'\"")
        path = files.get(stem) or files.get(stem + "_mask")
        if path is None:
            print("%-6s %-32s  no old mask in %s" % (c, stem[:32], old_dir))
            continue
        m = tifffile.imread(path)
        if m.ndim > 2:
            m = m[..., 0]
        H, W = d["grid"][0] * d["tile_size"], d["grid"][1] * d["tile_size"]
        n1, n2 = int((m == 1).sum()), int((m == 2).sum())
        old = n2 / (n1 + n2) if n1 + n2 else float("nan")
        thumb = np.load(os.path.join(out, "thumbs", c + ".npy"))
        k = d.get("thumb_scale", 16)
        o = m[::k, ::k][:thumb.shape[0], :thumb.shape[1]]
        t = thumb[:o.shape[0], :o.shape[1]]
        agree = float((o == t).mean())
        a, b = o == 2, t == 2
        dice = 2.0 * (a & b).sum() / max(a.sum() + b.sum(), 1)
        note = "" if m.shape == (H, W) else "  (old mask %dx%d, grid %dx%d)" % (m.shape[0], m.shape[1], H, W)
        diffs.append(d["cpa"] - old)
        print("%-6s %-32s %9.4f %9.4f %+9.4f %8.4f %8.4f%s"
              % (c, stem[:32], old, d["cpa"], d["cpa"] - old, agree, dice, note))
    if diffs:
        print("\n%d slides: mean |diff| %.5f, max |diff| %.5f"
              % (len(diffs), sum(map(abs, diffs)) / len(diffs), max(map(abs, diffs))))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--he-tiles", help="H&E tiling root: <case>/{images,masks,tiles_metadata.csv}")
    p.add_argument("--pred", help="SR tiles to segment: <case>/images/<tile>.<ext>")
    p.add_argument("--model", help="nnU-Net trained model folder (…/nnUNetTrainer__nnUNetPlans__2d)")
    p.add_argument("--out", required=True)
    p.add_argument("--ref", default=None, help="the real-SR output of this script, for cpa.csv")
    p.add_argument("--cases", nargs="*", default=None, help="only these cases (default: all)")
    p.add_argument("--fold", default="0")
    p.add_argument("--checkpoint", default="checkpoint_final.pth")
    p.add_argument("--batch", type=int, default=8, help="windows per forward pass")
    p.add_argument("--no-mirror", action="store_true",
                   help="skip nnU-Net's mirroring (4x faster; the old pipeline mirrored)")
    p.add_argument("--amp", action="store_true",
                   help="autocast on CUDA. The old pipeline ran on the CPU in fp32")
    p.add_argument("--workers", type=int, default=8, help="threads loading tiles")
    p.add_argument("--thumb", type=int, default=16, help="thumbnail downscale")
    p.add_argument("--force", action="store_true", help="redo slides already done")
    p.add_argument("--aggregate-only", action="store_true", help="only rebuild the CSVs")
    p.add_argument("--check-old", default=None, metavar="DIR",
                   help="compare --out with the old pipeline's whole-slide masks in DIR")
    args = p.parse_args()

    if args.check_old:
        check_old(args.out, args.check_old)
        return
    if args.aggregate_only:
        aggregate(args.out, args.ref)
        return
    for need in ("he_tiles", "pred", "model"):
        if not getattr(args, need):
            p.error("--%s is required" % need.replace("_", "-"))

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = args.cases or sorted(d for d in os.listdir(args.he_tiles)
                                 if os.path.isfile(os.path.join(args.he_tiles, d, "tiles_metadata.csv")))
    done = read_slides(args.out) if not args.force else {}
    todo = [c for c in cases if c not in done]
    print("pred   %s\nH&E    %s\nmodel  %s\nout    %s\ndevice %s  mirror %s  amp %s"
          % (args.pred, args.he_tiles, args.model, args.out, device, not args.no_mirror, args.amp))
    print("%d cases, %d done, %d to do" % (len(cases), len(cases) - len(todo), len(todo)))
    if todo:
        predict, to_seg, patch, gaussian, steps_fn, info = load_nnunet(
            args.model, args.fold, args.checkpoint, device, not args.no_mirror, args.amp)
        print("nnU-Net: patch %s, mirror axes %s, %s, regions %s"
              % (info["patch"], info["mirror_axes"], info["normalization"][0], info["regions"]))
    failed = []
    for c in todo:
        try:
            slide = Slide(args.he_tiles, c)
            res = run_slide(slide, args, predict, to_seg, patch, gaussian, steps_fn, device, info)
        except (FileNotFoundError, ValueError) as e:
            print("[FAIL] %s: %s" % (c, e))
            failed.append(c)
            continue
        write_slide(args.out, res, args.thumb)
        print("%s  %-34s %5d tiles  CPA %.4f  (holes %d px, %d masks missing)  %ss + %ss"
              % (c, res["source_file"][:34], res["tiles"], res["cpa"] or float("nan"),
                 res["hole_px"], res["missing_mask_tiles"],
                 res["seconds"]["stats"], res["seconds"]["segment"]), flush=True)
    aggregate(args.out, args.ref)
    if failed:
        sys.exit("failed: %s" % " ".join(failed))


if __name__ == "__main__":
    main()
