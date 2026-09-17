"""Estimate stain vectors from tiles, so nothing is hard-coded per stain pair.

Macenko et al. (2009), the method behind QuPath's "Estimate stain vectors ->
Auto": convert to optical density, keep the pixels that carry stain, take the
plane of the two leading eigenvectors of the OD cloud, and read the two stain
directions off the angular extremes.

Estimate from TRAINING tiles, write the result to JSON, and reuse that file for
every downstream step -- re-estimating per run (or worse, per batch) makes the
loss non-stationary and puts a discontinuity in the middle of a requeued job.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def angle_between(a, b) -> float:
    """Angle in degrees -- the natural unit for comparing stain directions."""
    return float(np.degrees(np.arccos(np.clip(_unit(a) @ _unit(b), -1.0, 1.0))))


def rgb_to_od(rgb, background=255.0):
    v = np.maximum(np.asarray(rgb, dtype=np.float64), 1.0)
    return np.maximum(-np.log10(v / background), 0.0)


def estimate_stains(od_pixels, min_od=0.05, max_od=1.0, ignore_percentage=1.0):
    """Macenko on an (N, 3) array of optical densities. Returns two unit vectors."""
    strength = od_pixels.max(axis=1)
    keep = (strength >= min_od) & (strength <= max_od)
    od = od_pixels[keep]
    if od.shape[0] < 1000:
        raise ValueError("only %d stained pixels passed the OD thresholds -- "
                         "sample more tiles or loosen min_od/max_od" % od.shape[0])

    # Non-centred second moment: the OD cloud radiates from the origin.
    eigvals, eigvecs = np.linalg.eigh((od.T @ od) / od.shape[0])
    order = np.argsort(eigvals)[::-1]
    e1, e2 = eigvecs[:, order[0]], eigvecs[:, order[1]]
    if e1.sum() < 0:
        e1 = -e1
    if e2.sum() < 0:
        e2 = -e2

    phi = np.arctan2(od @ e2, od @ e1)
    lo = np.percentile(phi, ignore_percentage)
    hi = np.percentile(phi, 100.0 - ignore_percentage)
    v1 = _unit(np.clip(e1 * np.cos(lo) + e2 * np.sin(lo), 0, None))
    v2 = _unit(np.clip(e1 * np.cos(hi) + e2 * np.sin(hi), 0, None))
    return v1, v2


def sample_od(directory: str, limit: int = 200, max_side: int = 256, seed: int = 0,
              pixels_per_tile: int = 20000):
    """Pool optical densities from a random sample of tiles in a directory.

    `pixels_per_tile` caps how many pixels each tile contributes. The estimate
    needs a 3x3 second-moment matrix and two angular percentiles, which converge
    after a few hundred thousand pixels -- what actually varies between runs is
    which SLIDES were drawn, not how many pixels came from each. Capping the
    pixels lets `limit` cover far more of the cohort at the same memory: pooling
    whole 256px tiles costs ~1.6 MB each, so the full training set would need
    tens of GB, while 2000 capped tiles fit in under a GB.
    """
    files = sorted(f for f in os.listdir(directory) if f.lower().endswith(IMAGE_EXTS))
    if not files:
        raise ValueError("no images in %s" % directory)
    rng = np.random.default_rng(seed)
    if limit and len(files) > limit:
        files = [files[i] for i in rng.permutation(len(files))[:limit]]
    chunks = []
    for f in files:
        img = Image.open(os.path.join(directory, f)).convert("RGB")
        if max(img.size) > max_side:
            img = img.resize((max_side, max_side), Image.BILINEAR)
        od = rgb_to_od(np.asarray(img)).reshape(-1, 3)
        if pixels_per_tile and od.shape[0] > pixels_per_tile:
            od = od[rng.choice(od.shape[0], pixels_per_tile, replace=False)]
        chunks.append(od)
    return np.concatenate(chunks, axis=0), len(files)


def order_like(pair, reference=None):
    """Put the vector closest to `reference` first.

    With no reference, fall back to the larger red component -- hematoxylin
    absorbs red most strongly, so this puts it first for H&E and H-DAB. The
    reference form is what generalises: it keeps domain B's channels aligned
    with domain A's rather than letting them silently swap between runs.
    """
    v1, v2 = pair
    if reference is None:
        return (v1, v2) if v1[0] > v2[0] else (v2, v1)
    return (v1, v2) if angle_between(v1, reference) <= angle_between(v2, reference) else (v2, v1)


def estimate_for_run(data_a, data_b, limit=200, min_od=0.05, max_od=1.0,
                     ignore_percentage=1.0, seed=0, pin_shared=False,
                     min_separation=15.0, pixels_per_tile=20000):
    """Estimate both domains' stain pairs and return a JSON-ready dict."""
    out = {"meta": {"method": "macenko", "limit": limit, "min_od": min_od,
                    "max_od": max_od, "ignore_percentage": ignore_percentage,
                    "seed": seed, "pin_shared": pin_shared,
                    "pixels_per_tile": pixels_per_tile}}
    warnings = []

    od_a, n_a = sample_od(data_a, limit, seed=seed, pixels_per_tile=pixels_per_tile)
    a1, a2 = order_like(estimate_stains(od_a, min_od, max_od, ignore_percentage))

    od_b, n_b = sample_od(data_b, limit, seed=seed, pixels_per_tile=pixels_per_tile)
    # Order B's pair against A's first stain so the two domains' channel 1 mean
    # the same thing, rather than depending on a per-domain heuristic.
    b1, b2 = order_like(estimate_stains(od_b, min_od, max_od, ignore_percentage),
                        reference=a1)
    if pin_shared:
        warnings.append("domain B stain1 pinned to domain A stain1 (was %.1f deg away)"
                        % angle_between(b1, a1))
        b1 = a1

    for dom, (s1, s2) in (("A", (a1, a2)), ("B", (b1, b2))):
        sep = angle_between(s1, s2)
        if sep < min_separation:
            warnings.append("domain %s stains are only %.1f deg apart (< %.1f): the "
                            "cohort may lack pixels of one stain, making the "
                            "deconvolution unstable" % (dom, sep, min_separation))
        out[dom] = {"stain1": [float(x) for x in s1],
                    "stain2": [float(x) for x in s2],
                    "separation_deg": sep}
    out["meta"]["tiles_A"], out["meta"]["tiles_B"] = n_a, n_b
    out["meta"]["shared_stain_angle_deg"] = angle_between(a1, b1)
    out["meta"]["warnings"] = warnings
    return out


def load_vectors(path: str):
    """Read a stains JSON into {'A': {...}, 'B': {...}} name -> vector maps."""
    with open(path) as fh:
        data = json.load(fh)
    return {dom: {"stain1": tuple(data[dom]["stain1"]),
                  "stain2": tuple(data[dom]["stain2"])} for dom in ("A", "B")}, data


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataA", required=True, help="domain A tiles (estimate from TRAIN)")
    p.add_argument("--dataB", required=True, help="domain B tiles")
    p.add_argument("--out", required=True, help="where to write the JSON")
    p.add_argument("--limit", type=int, default=2000, help="tiles to pool per domain")
    p.add_argument("--pixels-per-tile", type=int, default=20000, metavar="N",
                   help="cap each tile's contribution. Coverage across slides "
                        "matters more than pixels per slide, and pooling whole "
                        "tiles would need tens of GB for a full training set")
    p.add_argument("--min-od", type=float, default=0.05)
    p.add_argument("--max-od", type=float, default=1.0)
    p.add_argument("--ignore", type=float, default=1.0, help="angular percentile to trim")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pin-shared", action="store_true",
                   help="force domain B's first stain to equal domain A's. Keeps the "
                        "two domains' channel 1 exactly comparable, at the cost of "
                        "fidelity to domain B's own colours")
    p.add_argument("--min-separation", type=float, default=15.0,
                   help="warn if a domain's two stains are closer than this")
    return p


def main() -> None:
    args = build_parser().parse_args()
    result = estimate_for_run(args.dataA, args.dataB, args.limit, args.min_od,
                              args.max_od, args.ignore, args.seed, args.pin_shared,
                              args.min_separation, args.pixels_per_tile)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)

    for dom, label in (("A", args.dataA), ("B", args.dataB)):
        d = result[dom]
        print("domain %s  (%s)" % (dom, label))
        print("  stain1  %s" % np.round(d["stain1"], 4))
        print("  stain2  %s" % np.round(d["stain2"], 4))
        print("  separation %.1f deg" % d["separation_deg"])
    print("\nshared stain differs between domains by %.1f deg"
          % result["meta"]["shared_stain_angle_deg"])
    for w in result["meta"]["warnings"]:
        print("[WARN] %s" % w)
    print("\nwrote %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
