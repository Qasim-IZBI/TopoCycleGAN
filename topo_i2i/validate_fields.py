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
import json
import os
import re

import numpy as np
import torch
from PIL import Image

from topo_i2i.fields import make_field
from topo_i2i.stains import load_vectors
from topo_i2i.losses import diagram_distance
from topo_i2i.persistence import persistence_diagram

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMAGE_EXTS))


def matched_pairs(dir_a: str, dir_b: str, limit: int = 0, offset: int = 0,
                  sample: str = "random", seed: int = 0):
    """Pair tiles by filename stem; fall back to sorted order if names differ.

    Tiles are named <source image>_r<row>c<col>, so sorted order groups every
    crop of one slide together and `--limit N` would otherwise sample only the
    first few slides. `sample="random"` permutes deterministically by `seed`
    first, so a slice spans the whole directory -- and because the permutation
    depends only on the seed, `offset` still carves out an exactly disjoint
    second sample as long as the same seed is used.
    """
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
    if sample == "random":
        order = np.random.default_rng(seed).permutation(len(pairs))
        pairs = [pairs[i] for i in order]
        how += ", sampled at random (seed %d)" % seed
    elif sample != "head":
        raise ValueError("sample must be 'random' or 'head', got %r" % (sample,))
    if offset:
        pairs = pairs[offset:]
        how += ", skipping the first %d" % offset
    if limit:
        pairs = pairs[:limit]
    return pairs, how


DEFAULT_SLIDE_REGEX = r"_r\d+c\d+$"


def slide_of(name: str, pattern: str = DEFAULT_SLIDE_REGEX) -> str:
    """Group a crop belongs to. `pattern` is stripped from the filename stem.

    The default strips topo-crop's `_r<row>c<col>`, so a group is one SOURCE
    TILE -- and its members are the directly adjacent quadrants of that tile.
    That makes the within-slide control a hard test: it asks the distance to
    tell a tile from the tissue right next to it. A coarser pattern (grouping by
    slide or case, if the filenames encode it) gives a weaker, often fairer
    control.
    """
    return re.sub(pattern, "", os.path.splitext(name)[0])


def shuffled_partners(pairs, rng, within_slide: bool,
                      pattern: str = DEFAULT_SLIDE_REGEX):
    """A derangement of the B side.

    within_slide=True draws each wrong partner from the SAME source image. That
    controls for staining intensity, section thickness and scanner -- a true pair
    shares all of those with its partner, so an unstratified shuffle lets the
    distance score well by recognising the specimen rather than the tissue.
    """
    n = len(pairs)
    perm = np.arange(n)
    if not within_slide:
        while True:
            perm = rng.permutation(n)
            if n < 2 or (perm != np.arange(n)).all():
                return perm
    groups = {}
    for i, (a, _) in enumerate(pairs):
        groups.setdefault(slide_of(a, pattern), []).append(i)
    usable = []
    for idx in groups.values():
        if len(idx) < 2:
            continue                      # no alternative partner on this slide
        usable.extend(idx)
        # Rotate by a random non-zero amount: always a derangement, and unlike a
        # fixed shift it actually differs between --shuffles repetitions.
        k = int(rng.integers(1, len(idx)))
        rolled = idx[k:] + idx[:k]
        for src, dst in zip(idx, rolled):
            perm[src] = dst
    # Tiles with no alternative keep perm[i] = i. They MUST be excluded from the
    # comparison: scoring them would put true-pair distances into the shuffled
    # set and drag the AUROC toward 0.5.
    return perm, sorted(usable)


def load(path: str, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1


def diagrams_for(paths, spec, combine, size, downsample, dims, invert, vectors=None):
    """One diagram per image for a given field spec (the expensive step)."""
    field = make_field(spec, combine, vectors=vectors)
    out = []
    for p in paths:
        x = load(p, size)
        if downsample > 1:
            x = torch.nn.functional.avg_pool2d(x, downsample)
        f = field(x)[0].double()
        out.append(persistence_diagram(-f if invert else f, dims))
    return out


def auroc_se(a: float, n1: int, n2: int) -> float:
    """Hanley-McNeil standard error of an AUROC.

    The naive sqrt(0.5/n) is far too pessimistic here: it ignores the size of
    the shuffled sample and assumes the worst case at A=0.5.
    """
    if n1 < 2 or n2 < 2:
        return float("nan")
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2)
    return float(max(var, 0.0) ** 0.5)


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
    p.add_argument("--stains", default=None, metavar="JSON",
                   help="stain vectors from topo-estimate-stains. With this, use "
                        "specs built from 'stain1'/'stain2' -- each domain gets "
                        "its own estimated pair")
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
    p.add_argument("--shuffle-within-slide", action="store_true",
                   help="draw each wrong partner from the same source image. "
                        "Without this, a true pair also shares staining and "
                        "scanner with its partner, so the distance can score "
                        "well by recognising the specimen rather than the tissue")
    p.add_argument("--slide-regex", default=DEFAULT_SLIDE_REGEX, metavar="RE",
                   help="with --shuffle-within-slide, the pattern stripped from a "
                        "filename to get its group. The default strips "
                        "topo-crop's _r<row>c<col>, so groups are single source "
                        "tiles and partners are ADJACENT tissue -- a deliberately "
                        "hard control. Use a coarser pattern to group by slide")
    p.add_argument("--shuffles", type=int, default=5,
                   help="random permutations to average the shuffled baseline over")
    p.add_argument("--no-invert", action="store_true")
    p.add_argument("--json", default=None, metavar="PATH",
                   help="also write the table and the settings that produced it "
                        "as JSON, which is what topo-audit reads")
    p.add_argument("--sample", choices=("random", "head"), default="random",
                   help="'random' permutes the tile list by --seed before "
                        "slicing, so a --limit sample spans every slide; 'head' "
                        "takes them in filename order, which groups all crops of "
                        "the first few slides")
    p.add_argument("--seed", type=int, default=0,
                   help="controls both the tile permutation and the shuffled "
                        "baseline. Keep it FIXED between the screen and the "
                        "held-out re-score, or --offset will not be disjoint")
    return p


def main() -> None:
    args = build_parser().parse_args()
    dim_sets = [tuple(int(x) for x in ds.split(",")) for ds in args.dims_set]
    all_dims = tuple(sorted({d for ds in dim_sets for d in ds}))
    invert = not args.no_invert

    vectors = {"A": None, "B": None}
    if args.stains:
        vecs, raw = load_vectors(args.stains)
        vectors = vecs
        print("stain vectors from %s" % args.stains)
        for dom in ("A", "B"):
            print("  %s stain1 %s  stain2 %s  (%.1f deg apart)"
                  % (dom, np.round(vecs[dom]["stain1"], 4),
                     np.round(vecs[dom]["stain2"], 4), raw[dom]["separation_deg"]))
        for w in raw.get("meta", {}).get("warnings", []):
            print("  [WARN] %s" % w)

    pairs, how = matched_pairs(args.dataA, args.dataB, args.limit, args.offset,
                               args.sample, args.seed)
    if not pairs:
        raise SystemExit("no tiles to compare")
    print(how)
    paths_a = [os.path.join(args.dataA, f) for f, _ in pairs]
    paths_b = [os.path.join(args.dataB, f) for _, f in pairs]
    n = len(pairs)

    rng = np.random.default_rng(args.seed)
    perms = []
    index = list(range(n))
    for _ in range(args.shuffles):
        out = shuffled_partners(pairs, rng, args.shuffle_within_slide,
                                args.slide_regex)
        if args.shuffle_within_slide:
            perm, usable = out
            perms.append(perm)
            index = usable
        else:
            perms.append(out)
    if args.shuffle_within_slide:
        groups = len({slide_of(a, args.slide_regex) for a, _ in pairs})
        kind = ("source tile (adjacent quadrants -- a strict control)"
                if args.slide_regex == DEFAULT_SLIDE_REGEX
                else "custom grouping")
        print("shuffling within group: regex %r -> %d groups over %d tiles, %s"
              % (args.slide_regex, groups, n, kind))
        print("  %d of %d tiles have an alternative partner in their group; the "
              "other %d are excluded from BOTH sides of the comparison"
              % (len(index), n, n - len(index)))
        if not index:
            raise SystemExit("no tile has a same-image alternative -- nothing to compare")

    # diagrams are the expensive part -- compute each (spec, downsample) once
    cache = {}
    for ds in args.downsample:
        for spec in args.field_A:
            cache[("A", spec, ds)] = diagrams_for(paths_a, spec, args.field_combine,
                                                  args.image_size, ds, all_dims, invert,
                                                  vectors["A"])
        for spec in args.field_B:
            cache[("B", spec, ds)] = diagrams_for(paths_b, spec, args.field_combine,
                                                  args.image_size, ds, all_dims, invert,
                                                  vectors["B"])

    combos = list(itertools.product(args.field_A, args.field_B, args.downsample,
                                    dim_sets, args.topo_projection))
    print("\n%d tiles compared, %d combinations\n" % (len(index), len(combos)))
    header = "%-20s %-18s %3s %5s %-9s %9s %9s %7s %7s" % (
        "field_A", "field_B", "ds", "dims", "proj", "true", "shuffled", "ratio", "AUROC")
    print(header); print("-" * len(header))

    rows = []
    for fa, fb, ds, dims, projname in combos:
        proj = None if projname == "auto" else projname
        da, db = cache[("A", fa, ds)], cache[("B", fb, ds)]
        true = [float(diagram_distance(da[i], db[i], dims, proj)) for i in index]
        shuf = [float(diagram_distance(da[i], db[p[i]], dims, proj))
                for p in perms for i in index]
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
        # Shuffled comparisons reuse the same B tiles, so they are not
        # n * shuffles independent samples; n is the honest denominator.
        se = auroc_se(best[8], len(index), len(index))
        print("\nNOTE: %d combinations on %d tiles. The 95%% band on the best AUROC is "
              "+/-%.3f,\n      and taking the maximum over many correlated "
              "combinations inflates it further,\n      so treat the top row as an "
              "upper bound rather than a measurement. What to\n      trust is a "
              "setting that ranks well CONSISTENTLY -- across markers, and on a\n"
              "      disjoint slice: re-score the leaders with --offset %d --seed %d."
              % (len(rows), n, 1.96 * se, args.offset + n, args.seed))
    print("AUROC 0.5 means the distance says nothing about which tiles belong together;")
    print("a field pair that cannot separate true from random here will not teach")
    print("ph_trans anything during training.")

    if args.json:
        # One row per combination, each carrying its own standard error, plus
        # every setting needed to reproduce it. topo-audit consumes this; the
        # printed table above is for humans and is not parsed by anything.
        se_of = lambda a: auroc_se(a, len(index), len(index))
        payload = {
            "rows": [{"field_A": r[0], "field_B": r[1], "downsample": r[2],
                      "dims": r[3], "projection": r[4], "true": r[5],
                      "shuffled": r[6], "ratio": r[7], "auroc": r[8],
                      "auroc_se": se_of(r[8])} for r in rows],
            "n_tiles": len(index), "n_pairs": n, "combinations": len(combos),
            "within_slide": bool(args.shuffle_within_slide),
            "slide_regex": args.slide_regex if args.shuffle_within_slide else None,
            "groups": (len({slide_of(a, args.slide_regex) for a, _ in pairs})
                       if args.shuffle_within_slide else None),
            "stains": args.stains, "limit": args.limit, "offset": args.offset,
            "seed": args.seed, "sample": args.sample, "shuffles": args.shuffles,
            "image_size": args.image_size, "field_combine": args.field_combine,
            "invert": invert, "dataA": args.dataA, "dataB": args.dataB,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
