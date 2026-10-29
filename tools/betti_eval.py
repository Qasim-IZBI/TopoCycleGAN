"""Topological agreement of two nuclei masks: Betti number and Betti matching errors.

Compares the nuclei of the real H&E with those of the IHC generated from it --
the two are pixel-aligned, which Betti MATCHING needs (it asks whether a
feature has a counterpart at the same place, not only whether the counts
agree). The real IHC is an adjacent section and is never compared this way.

Fixed before anything was computed, as Berger et al. (2024, "Pitfalls of
topology-aware image segmentation") ask -- a metric's connectivity and its
artifact rule are dataset choices, not tuning knobs:

  instances   a label image becomes a binary mask with every pixel removed
              that touches a DIFFERENT nucleus (4-neighbourhood), so touching
              nuclei stay separate components instead of merging into one
  connectivity  foreground 4-connected, background 8-connected (their "D"),
              the pair under which separated nuclei stay separate and the
              two are dual, so holes are well defined
  specks      foreground components under MIN_SIZE px are removed from BOTH
              masks before counting
  holes       background components that do not touch the tile border

Dimension 0 (nuclei). Betti matching (Stucki et al. 2023) matches a feature of
L to one of G when both are matched, through induced matchings, to the same
feature of the comparison image C. For binary masks under a superlevel
filtration, C is the union L | G, every component of L or G lies in exactly one
component of C, and each component of C can be matched to at most one
component of L and one of G. So the matched pairs are the components of the
union that contain foreground of both masks, m0, and

    BM0 = n0(L) + n0(G) - 2 m0       (features without a spatial counterpart)

Dimension 1 (holes, e.g. rings of nuclei closing around a lumen). Dually, a
hole of L or G corresponds to the holes of C's background (the intersection of
the two backgrounds) that it contains. A hole of L and a hole of G are paired
when they share such a hole, one-to-one, as many as possible (maximum
bipartite matching):  BM1 = n1(L) + n1(G) - 2 m1.

These are closed forms for binary masks; they were not checked against the
reference implementation (Betti-matching-3D). On these masks dimension 0 is the
informative one: it is a matched count of nuclei by overlap, the topological
counterpart of the centroid recall. Dimension 1 on nuclei masks mostly counts
rings of nuclei and is reported, not relied on.

    beta-error_d = |n_d(L) - n_d(G)|     (Betti number error, no matching)
    BM_d / (n_d(L) + n_d(G))             normalised: 0 = every feature matched,
                                         1 = none; 1 - this is a Dice over features
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

MIN_SIZE = 5
FG = ndimage.generate_binary_structure(2, 1)     # 4-connected foreground
BG = ndimage.generate_binary_structure(2, 2)     # 8-connected background
SETTINGS = "fg4/bg8, instance borders removed, components < %d px removed" % MIN_SIZE


def binary_nuclei(labels: np.ndarray, min_size: int = MIN_SIZE) -> np.ndarray:
    """Label image -> binary mask with touching nuclei kept apart and specks gone."""
    lab = np.asarray(labels)
    fg = lab > 0
    p = np.pad(lab, 1, mode="constant")
    border = np.zeros_like(fg)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nb = p[1 + dy:p.shape[0] - 1 + dy, 1 + dx:p.shape[1] - 1 + dx]
        border |= (nb > 0) & (nb != lab)
    mask = fg & ~border
    if min_size > 1 and mask.any():
        comp, n = ndimage.label(mask, structure=FG)
        sizes = np.bincount(comp.ravel(), minlength=n + 1)
        small = sizes < min_size
        small[0] = False
        mask &= ~small[comp]
    return mask


def components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    return ndimage.label(mask, structure=FG)


def holes(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Bounded background components (8-connected), relabelled 1..n."""
    bg, n = ndimage.label(~mask, structure=BG)
    if n == 0:
        return bg, 0
    edge = np.unique(np.concatenate([bg[0], bg[-1], bg[:, 0], bg[:, -1]]))
    keep = np.ones(n + 1, bool)
    keep[edge] = False
    keep[0] = False
    remap = np.zeros(n + 1, np.int64)
    remap[keep] = np.arange(1, keep.sum() + 1)
    return remap[bg], int(keep.sum())


def _matched_dim0(l_comp, g_comp, union):
    """Union components containing foreground of both masks."""
    u, nu = union
    if nu == 0:
        return 0
    has_l = np.zeros(nu + 1, bool)
    has_g = np.zeros(nu + 1, bool)
    has_l[np.unique(u[l_comp[0] > 0])] = True
    has_g[np.unique(u[g_comp[0] > 0])] = True
    has_l[0] = has_g[0] = False
    return int((has_l & has_g).sum())


def _matched_dim1(l_holes, g_holes, c_holes):
    """Holes of L and of G that share a hole of C, paired one-to-one."""
    lh, nl = l_holes
    gh, ng = g_holes
    ch, nc = c_holes
    if 0 in (nl, ng, nc):
        return 0
    inside = ch > 0
    pairs = np.unique(np.stack([lh[inside], gh[inside]], 1), axis=0)
    pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0)]
    if len(pairs) == 0:
        return 0
    adj = np.zeros((nl, ng))
    adj[pairs[:, 0] - 1, pairs[:, 1] - 1] = 1
    r, c = linear_sum_assignment(-adj)
    return int(adj[r, c].sum())


def topo_compare(ref_labels: np.ndarray, pred_labels: np.ndarray,
                 min_size: int = MIN_SIZE) -> Dict[str, int]:
    """Betti numbers of both masks and their matched counts, for one tile."""
    L = binary_nuclei(ref_labels, min_size)
    G = binary_nuclei(pred_labels, min_size)
    lc, gc = components(L), components(G)
    lh, gh = holes(L), holes(G)
    m0 = _matched_dim0(lc, gc, components(L | G))
    m1 = _matched_dim1(lh, gh, holes(L | G))
    return {"b0_ref": lc[1], "b0_pred": gc[1], "m0": m0,
            "b1_ref": lh[1], "b1_pred": gh[1], "m1": m1}


def tile_errors(t: Dict[str, int]) -> Dict[str, int]:
    return {"bm0": t["b0_ref"] + t["b0_pred"] - 2 * t["m0"],
            "bm1": t["b1_ref"] + t["b1_pred"] - 2 * t["m1"],
            "betti0_err": abs(t["b0_ref"] - t["b0_pred"]),
            "betti1_err": abs(t["b1_ref"] - t["b1_pred"])}


TILE_COLUMNS = ("b0_ref", "b0_pred", "m0", "b1_ref", "b1_pred", "m1",
                "bm0", "bm1", "betti0_err", "betti1_err")


def pooled(tiles) -> Dict[str, float]:
    """Means per tile, and the feature-normalised matching errors over all tiles."""
    if not tiles:
        return {}
    rows = [dict(t, **tile_errors(t)) for t in tiles]
    n = len(rows)
    out = {"topo_n_tiles": n}
    for k in ("bm0", "bm1", "betti0_err", "betti1_err"):
        out["%s_mean" % k] = sum(r[k] for r in rows) / n
    for d in (0, 1):
        feats = sum(r["b%d_ref" % d] + r["b%d_pred" % d] for r in rows)
        out["bm%d_norm" % d] = (sum(r["bm%d" % d] for r in rows) / feats) if feats else 0.0
    return out
