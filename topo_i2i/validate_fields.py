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


def matched_pairs(dir_a: str, dir_b: str, limit: int = 0):
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
    p.add_argument("--limit", type=int, default=64, help="tiles to use (0 = all)")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--downsample", type=int, default=1)
    p.add_argument("--topo-dims", type=int, nargs="+", default=[0, 1])
    p.add_argument("--topo-projection", default="auto",
                   choices=("auto", "birth", "lifetime", "death"))
    p.add_argument("--shuffles", type=int, default=5,
                   help="random permutations to average the shuffled baseline over")
    p.add_argument("--no-invert", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    dims = tuple(args.topo_dims)
    proj = None if args.topo_projection == "auto" else args.topo_projection
    invert = not args.no_invert

    pairs, how = matched_pairs(args.dataA, args.dataB, args.limit)
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

    # diagrams are the expensive part -- compute each spec once
    cache = {}
    for spec in args.field_A:
        cache[("A", spec)] = diagrams_for(paths_a, spec, args.field_combine,
                                          args.image_size, args.downsample, dims, invert)
    for spec in args.field_B:
        cache[("B", spec)] = diagrams_for(paths_b, spec, args.field_combine,
                                          args.image_size, args.downsample, dims, invert)

    print("\n%d tiles, dims=%s, projection=%s\n" % (n, list(dims), args.topo_projection))
    header = "%-20s %-18s %9s %9s %7s %7s" % (
        "field_A", "field_B", "true", "shuffled", "ratio", "AUROC")
    print(header); print("-" * len(header))

    rows = []
    for fa, fb in itertools.product(args.field_A, args.field_B):
        da, db = cache[("A", fa)], cache[("B", fb)]
        true = [float(diagram_distance(da[i], db[i], dims, proj)) for i in range(n)]
        shuf = [float(diagram_distance(da[i], db[p[i]], dims, proj))
                for p in perms for i in range(n)]
        mt, ms = float(np.mean(true)), float(np.mean(shuf))
        row = (fa, fb, mt, ms, ms / mt if mt else float("inf"), auroc(true, shuf))
        rows.append(row)
        print("%-20s %-18s %9.2f %9.2f %7.2f %7.3f" % row)

    best = max(rows, key=lambda r: r[5])
    print("\nbest separation: %s / %s  (AUROC %.3f)" % (best[0], best[1], best[5]))
    print("AUROC 0.5 means the distance says nothing about which tiles belong together;")
    print("a field pair that cannot separate true from random here will not teach")
    print("ph_trans anything during training.")


if __name__ == "__main__":
    main()
