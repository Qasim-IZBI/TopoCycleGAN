"""IHC positivity: the share of DAB-positive nuclei, real IHC vs generated IHC.

A nucleus is positive when the mean DAB optical density over it is above a
threshold. The real IHC is an adjacent section, so nuclei cannot be paired one
by one; what is compared is the positive RATE per tile (the Ki67 index, the
share of ER / PR positive cells) and over the whole split.

Fixed before any model was scored:

  DAB          skimage's rgb2hed (Ruifrok & Johnston vectors) on the RGB
               floored at 10/255, as nuclei_pilot.nuclear_od; negative OD is 0
  compartment  Ki67, ER, PR are nuclear stains: DAB is averaged over the
               nucleus. HER2 (MIST and BCI) is membranous: the nucleus is grown
               by CELL_GROW px (skimage expand_labels, no overlap between
               neighbours) and DAB averaged over that cell
  threshold    one per marker: Otsu's threshold of the per-nucleus DAB of the
               REAL IHC of the split, so it comes from the data, never from a
               model, and every model is scored against the same one
  nuclei       real and generated IHC are segmented by the same IHC segmenter
               at the same scale, so both rates are read the same way
  tiles        the tile-wise metrics use tiles with at least MIN_NUCLEI nuclei
               on both sides -- a rate over 2 nuclei is noise

    pos_rate_real / _gen   positive nuclei / all nuclei, pooled over tiles
    pos_rate_abs_diff      |gen - real| of those
    tile_mae               mean over tiles of |rate_gen - rate_real|
    tile_pearson/_spearman correlation of the two rates across tiles -- does the
                           model put positivity in the tiles where it belongs
    *_lo / *_hi            pooled rates at 0.75x and 1.25x the threshold: how
                           much the conclusion depends on where it was drawn
    pos_rate_gen_on_he     the generated IHC's DAB read inside the H&E nuclei
                           (pixel-aligned, so no second segmentation): the same
                           rate without the IHC segmenter in the loop
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import ndimage

COMPARTMENT = {"Ki67": "nucleus", "ER": "nucleus", "PR": "nucleus",
               "HER2": "cell", "BCI": "cell"}
CELL_GROW = 4
MIN_NUCLEI = 5
SWEEP = (0.75, 1.25)


def dab_od(rgb: np.ndarray) -> np.ndarray:
    """DAB optical density of an RGB uint8 tile, >= 0."""
    from skimage.color import rgb2hed
    return np.clip(rgb2hed(np.maximum(rgb, 10))[..., 2], 0, None).astype(np.float32)


def compartment(labels: np.ndarray, marker: str, grow: int = CELL_GROW) -> np.ndarray:
    if COMPARTMENT.get(marker, "nucleus") == "cell" and grow > 0:
        from skimage.segmentation import expand_labels
        return expand_labels(labels, grow)
    return labels


def nucleus_dab(labels: np.ndarray, dab: np.ndarray) -> np.ndarray:
    """Mean DAB over each labelled object, in label order."""
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return np.zeros(0, np.float32)
    if dab.shape != labels.shape:
        raise ValueError("DAB %s and labels %s differ in shape" % (dab.shape, labels.shape))
    return np.asarray(ndimage.mean(dab, labels, ids), np.float32)


def measure(rgb: np.ndarray, labels: np.ndarray, marker: str,
            grow: int = CELL_GROW) -> np.ndarray:
    return nucleus_dab(compartment(labels, marker, grow), dab_od(rgb))


def otsu(values: Iterable[np.ndarray]) -> float:
    from skimage.filters import threshold_otsu
    v = np.concatenate([np.asarray(x, np.float64) for x in values] or [np.zeros(0)])
    if len(v) < 2 or v.min() == v.max():
        raise ValueError("too few distinct nuclei (%d) for a threshold" % len(v))
    return float(threshold_otsu(v))


def _rate(v: np.ndarray, thr: float) -> float:
    return float((v > thr).mean()) if len(v) else float("nan")


def tile_row(real: np.ndarray, gen: np.ndarray, thr: float,
             gen_on_he: np.ndarray = None) -> Dict[str, float]:
    row = {"n_real": len(real), "n_gen": len(gen),
           "pos_real": int((real > thr).sum()), "pos_gen": int((gen > thr).sum()),
           "rate_real": _rate(real, thr), "rate_gen": _rate(gen, thr)}
    if gen_on_he is not None:
        row.update(n_he=len(gen_on_he), pos_gen_on_he=int((gen_on_he > thr).sum()))
    return row


TILE_COLUMNS = ("n_real", "n_gen", "pos_real", "pos_gen", "rate_real", "rate_gen",
                "n_he", "pos_gen_on_he")


def pooled(real: List[np.ndarray], gen: List[np.ndarray], thr: float,
           gen_on_he: List[np.ndarray] = None, min_nuclei: int = MIN_NUCLEI
           ) -> Dict[str, float]:
    """Split-level metrics from the per-tile DAB values of both sides (same tile order)."""
    from scipy.stats import pearsonr, spearmanr
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)
    R, G = cat(real), cat(gen)
    out = {"dab_threshold": thr, "n_tiles": len(real),
           "n_real_nuclei": len(R), "n_gen_nuclei": len(G),
           "pos_rate_real": _rate(R, thr), "pos_rate_gen": _rate(G, thr)}
    out["pos_rate_abs_diff"] = abs(out["pos_rate_gen"] - out["pos_rate_real"])
    for f, tag in zip(SWEEP, ("lo", "hi")):
        out["pos_rate_real_" + tag] = _rate(R, thr * f)
        out["pos_rate_gen_" + tag] = _rate(G, thr * f)
    if gen_on_he is not None:
        out["pos_rate_gen_on_he"] = _rate(cat(gen_on_he), thr)

    pairs = [(_rate(r, thr), _rate(g, thr)) for r, g in zip(real, gen)
             if len(r) >= min_nuclei and len(g) >= min_nuclei]
    out["tile_n"] = len(pairs)
    if pairs:
        a, b = np.array(pairs).T
        out["tile_mae"] = float(np.abs(a - b).mean())
        varies = len(pairs) > 2 and a.std() > 0 and b.std() > 0
        out["tile_pearson"] = float(pearsonr(a, b)[0]) if varies else float("nan")
        out["tile_spearman"] = float(spearmanr(a, b)[0]) if varies else float("nan")
    return out


def summary_line(p: Dict[str, float]) -> str:
    return ("positive rate real %.3f  generated %.3f  (|diff| %.3f)  tile MAE %.3f  "
            "r %.2f  rho %.2f over %d tiles  [threshold %.3f]"
            % (p["pos_rate_real"], p["pos_rate_gen"], p["pos_rate_abs_diff"],
               p.get("tile_mae", float("nan")), p.get("tile_pearson", float("nan")),
               p.get("tile_spearman", float("nan")), p.get("tile_n", 0), p["dab_threshold"]))
