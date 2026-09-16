"""Score field choices against MIST's registered pairs, without training anything.

MIST ships H&E and IHC tiles of the same tissue, so for a validation tile there
is a known correct counterpart. That gives the topological distance a ground
truth it can be tested against: if the loss measures nuclear correspondence at
all, a TRUE pair must score lower than a RANDOM pair.

    d_true_i     = d( field_A(A_i), field_B(B_i) )         same tissue
    d_shuffled   = d( field_A(A_i), field_B(B_perm(i)) )   unrelated tissue

The headline number is AUROC: the probability that a randomly chosen true pair
scores below a randomly chosen shuffled pair. 0.5 means the distance carries no
information about correspondence and no amount of lambda tuning will help; 1.0
means it separates them perfectly.

Pass several specs to each of --field-A / --field-B to score the whole cross
product and pick the pair to train ph_trans on:

    topo-validate-fields --dataA val/valA --dataB val/valB \\
        --field-A hematoxylin/eosin gray \\
        --field-B hematoxylin/dab dab/hematoxylin dab+hematoxylin
"""

from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import torch
from PIL import Image

from topo_i2i.fields import make_field
from topo_i2i.losses import diagram_distance
from topo_i2i.persistence import persistence_diagram

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMAGE_EXTS))


def matched_pairs(dir_a: str, dir_b: str, limit: int = 0, offset: int = 0):
    """Pair tiles by filename stem; fall back to sorted order if names differ."""
    a, b = list_images(dir_a), list_images(dir_b)
    stem = lambda f: os.path.splitext(f)[0]
    common = sorted(set(map(stem, a)) & set(map(stem, b)))
    if common:
        by_a = {stem(f): f for f in a}
        by_b = {stem(f): f for f in b}
        pairs = [(by_a[s], by_b[s]) for s in common]
        how = "matched %d tiles by filename" % len(pairs)
    else:
        n = min(len(a), len(b))
        pairs = list(zip(a[:n], b[:n]))
        how = ("WARNING: no filenames in common -- falling back to sorted order "
               "for %d tiles. Check that these directories really are registered." % n)
    if offset:
        pairs = pairs[offset:]
        how += ", skipping the first %d" % offset
    if limit:
        pairs = pairs[:limit]
    return pairs, how


def load(path: str, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1


def diagrams_for(paths, spec, combine, size, downsample, dims, invert):
    """One diagram per image for a given field spec (the expensive step)."""
    field = make_field(spec, combine)
    out = []
    for p in paths:
        x = load(p, size)
        if downsample > 1:
            x = torch.nn.functional.avg_pool2d(x, downsample)
        f = field(x)[0].double()
        out.append(persistence_diagram(-f if invert else f, dims))
    return out


def auroc(true_scores, shuffled_scores) -> float:
    """P(a true pair scores below a shuffled one), ties counted as half."""
    t = np.asarray(true_scores)[:, None]
    s = np.asarray(shuffled_scores)[None, :]
    return float(((t < s).sum() + 0.5 * (t == s).sum()) / (t.size * s.size))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataA", required=True, help="H&E tiles (registered with dataB)")
    p.add_argument("--dataB", required=True, help="IHC tiles of the same tissue")
    p.add_argument("--field-A", nargs="+", default=["hematoxylin/eosin"],
                   metavar="SPEC", help="one or more domain-A specs to score")
    p.add_argument("--field-B", nargs="+",
                   default=["hematoxylin/dab", "dab/hematoxylin", "dab+hematoxylin"],
                   metavar="SPEC", help="one or more domain-B specs to score")
    p.add_argument("--field-combine", default="sum", choices=("max", "sum", "mean"),
                   help="merge rule for any 'a+b' spec")
    p.add_argument("--limit", type=int, default=128, help="tiles to use (0 = all)")
    p.add_argument("--offset", type=int, default=0, metavar="N",
                   help="skip the first N tiles. Screen the whole grid on the "
                        "first slice, then re-score the leaders with an --offset "
                        "past it -- a held-out sample is what protects the "
                        "ranking from selection noise")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--downsample", type=int, nargs="+", default=[1], metavar="N",
                   help="one or more pooling factors to score. Persistence cost "
                        "falls roughly with the pixel count, so if a coarser "
                        "field keeps the same AUROC it is free signal: train at "
                        "that --topo-downsample instead")
    # Diagrams always carry both homology dimensions, and the projection only
    # changes how they are compared -- so these two axes reuse the cached
    # diagrams and cost essentially nothing to sweep. Only the field specs and
    # --downsample change the diagrams themselves.
    p.add_argument("--dims-set", nargs="+", default=["0,1"], metavar="SET",
                   help="homology dimensions to score, comma-joined: 0 1 0,1")
    p.add_argument("--topo-projection", nargs="+", default=["auto"],
                   choices=("auto", "birth", "lifetime", "death"),
                   help="'auto' (lifetime for H0, birth for H1) is what training "
                        "uses -- keep it in the sweep or the winner cannot be "
                        "compared against your current setting")
    p.add_argument("--shuffles", type=int, default=5,
                   help="random permutations to average the shuffled baseline over")
    p.add_argument("--no-invert", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    dim_sets = [tuple(int(x) for x in ds.split(",")) for ds in args.dims_set]
    all_dims = tuple(sorted({d for ds in dim_sets for d in ds}))
    invert = not args.no_invert

    pairs, how = matched_pairs(args.dataA, args.dataB, args.limit, args.offset)
    if not pairs:
        raise SystemExit("no tiles to compare")
    print(how)
    paths_a = [os.path.join(args.dataA, f) for f, _ in pairs]
    paths_b = [os.path.join(args.dataB, f) for _, f in pairs]
    n = len(pairs)

    rng = np.random.default_rng(args.seed)
    perms = []
    while len(perms) < args.shuffles:
        p = rng.permutation(n)
        if n < 2 or (p != np.arange(n)).all():     # a derangement: no true pair survives
            perms.append(p)

    # diagrams are the expensive part -- compute each (spec, downsample) once
    cache = {}
    for ds in args.downsample:
        for spec in args.field_A:
            cache[("A", spec, ds)] = diagrams_for(paths_a, spec, args.field_combine,
                                                  args.image_size, ds, all_dims, invert)
        for spec in args.field_B:
            cache[("B", spec, ds)] = diagrams_for(paths_b, spec, args.field_combine,
                                                  args.image_size, ds, all_dims, invert)

    combos = list(itertools.product(args.field_A, args.field_B, args.downsample,
                                    dim_sets, args.topo_projection))
    print("\n%d tiles, %d combinations\n" % (n, len(combos)))
    header = "%-20s %-18s %3s %5s %-9s %9s %9s %7s %7s" % (
        "field_A", "field_B", "ds", "dims", "proj", "true", "shuffled", "ratio", "AUROC")
    print(header); print("-" * len(header))

    rows = []
    for fa, fb, ds, dims, projname in combos:
        proj = None if projname == "auto" else projname
        da, db = cache[("A", fa, ds)], cache[("B", fb, ds)]
        true = [float(diagram_distance(da[i], db[i], dims, proj)) for i in range(n)]
        shuf = [float(diagram_distance(da[i], db[p[i]], dims, proj))
                for p in perms for i in range(n)]
        mt, ms = float(np.mean(true)), float(np.mean(shuf))
        row = (fa, fb, ds, ",".join(map(str, dims)), projname,
               mt, ms, ms / mt if mt else float("inf"), auroc(true, shuf))
        rows.append(row)
        print("%-20s %-18s %3d %5s %-9s %9.2f %9.2f %7.2f %7.3f" % row)

    best = max(rows, key=lambda r: r[8])
    print("\nbest:     %s / %s  ds=%d dims=%s proj=%s  (AUROC %.3f)"
          % (best[0], best[1], best[2], best[3], best[4], best[8]))
    # Raw distances are not comparable across downsample factors -- the sum runs
    # over more diagram points at finer resolution -- but AUROC is.
    near = [r for r in rows if r[8] >= best[8] - 0.01]
    cheapest = max(near, key=lambda r: r[2])
    if cheapest[2] > best[2]:
        print("cheapest within 0.01 AUROC: %s / %s ds=%d dims=%s proj=%s (AUROC %.3f)"
              % (cheapest[0], cheapest[1], cheapest[2], cheapest[3], cheapest[4], cheapest[8]))
    if len(rows) > 10:
        print("\nNOTE: %d combinations on %d tiles. At this sample size the leading "
              "rows are\n      statistically tied (95%% band is roughly +/-%.3f), so "
              "the top one is partly\n      luck. Re-score the leaders with "
              "--offset %d on a disjoint slice."
              % (len(rows), n, 1.96 * (0.5 / max(n, 1)) ** 0.5, args.offset + n))
    print("AUROC 0.5 means the distance says nothing about which tiles belong together;")
    print("a field pair that cannot separate true from random here will not teach")
    print("ph_trans anything during training.")


if __name__ == "__main__":
    main()
