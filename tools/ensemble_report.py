"""Seed spread of the deep ensembles, and the chosen model against the baselines
over all ten seeds rather than one.

Reads what evaluate.sh and evaluate_nuclei.sh wrote for ARM=ensemble -- per
tile nuclei counts, per-tile patch SSIM and LPIPS, FID -- and reports, per
marker and model (TopoCycleGAN, CycleGAN small, DCLGAN small):

MEMBERS   every seed's nuclei recall / precision / F1, SSIM, LPIPS, FID, and
          whether it collapsed (recall below --fail-below).
SPREAD    mean, standard deviation, min and max over the seeds, and how many
          of them collapsed. Collapsed seeds are counted at their real value:
          how often a model fails to train is part of what it is.
INVERTED  how many seeds invert the tissue / lumen layout -- the cause of the
          collapse -- read from the inversion.csv tools/invert_check.py
          --ensemble writes beside each member's results. Sharper than the
          recall cut-off, and the one to use where recall is not bimodal (BCI).
VERSUS    mean recall of TopoCycleGAN minus that of each baseline, with a 95%
          interval from a two-level bootstrap: resample the slides (paired --
          one resample feeds every model), then each model's seeds. It covers
          both what slides were drawn and which seeds happened to train well.

The half is the one select_config.py reports on -- the "test" half of the val
split, since the configuration was chosen on "sel". A split without a
val_test_split.csv (BCI's held-out testA) is reported whole.

    python tools/ensemble_report.py --eval-root /work2/bz66izin-TopoCG/eval
    python tools/ensemble_report.py --eval-root ... --markers BCI --split testA
    python tools/ensemble_report.py --eval-root ... --detector instanseg

--detector reads the nuclei results of another detector (evaluate_nuclei.sh
DETECTOR=..., in each member's nuclei_<detector>/) instead of StarDist's; SSIM,
LPIPS, FID and the inversion flag are the same either way.

Output under <eval-root>/ensemble_report/<split>_<half>[_<detector>]/:
members.csv, spread.csv, versus.csv. Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import statistics as st
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_config import Model, read_rows  # noqa: E402

MODELS = (("topo", "TopoCycleGAN"), ("cyclegan", "CycleGAN small"), ("dclgan", "DCLGAN small"))
MEMBER_RE = re.compile(r"^(?P<m>.+)_(?P<model>topo|cyclegan|dclgan)_seed(?P<k>\d+)$")
METRICS = ("recall", "precision", "f1", "patch_ssim", "lpips", "fid")


class WholeSplit(dict):
    """Every tile in one half, "all" -- for a split that was never halved."""

    def get(self, key, default=None):
        return "all"


def fid_of(mod: Model, half: str) -> Optional[float]:
    if half != "all":
        return mod.fid(half)
    path = os.path.join(mod.dir, "fid.csv")
    try:
        return float(read_rows(path)[1][1]) if os.path.isfile(path) else None
    except (IndexError, ValueError):
        return None


def two_level_diff(a: List[Dict[str, Tuple[int, int]]], b: List[Dict[str, Tuple[int, int]]],
                   n_boot: int, seed: int) -> Tuple[float, float, float]:
    """mean-over-seeds recall(a) - that of b, and its 95% interval: slides
    resampled once per draw and shared by both models, then each model's
    seeds resampled on their own."""
    slides = sorted(set.intersection(*(set(m) for m in a + b)))
    if not slides or not a or not b:
        return float("nan"), float("nan"), float("nan")

    def recall(m, sample):
        n = sum(m[s][0] for s in sample)
        return sum(m[s][1] for s in sample) / n if n else 0.0

    def stat(sample, sa, sb):
        return st.mean(recall(m, sample) for m in sa) - st.mean(recall(m, sample) for m in sb)

    rng = random.Random(seed)
    boots = sorted(stat([rng.choice(slides) for _ in slides],
                        [rng.choice(a) for _ in a], [rng.choice(b) for _ in b])
                   for _ in range(n_boot))
    return stat(slides, a, b), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def inverted_of(member_dir: str) -> Optional[int]:
    """1 / 0 from tools/invert_check.py's inversion.csv, None if not checked."""
    path = os.path.join(member_dir, "inversion.csv")
    if not os.path.isfile(path):
        return None
    for row in read_rows(path)[1:]:
        if row and row[0] == "inverted":
            return int(float(row[1]))
    return None


def fmt(v, d=4):
    return ("%." + str(d) + "f") % v if v is not None else "-"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-root", required=True)
    p.add_argument("--markers", nargs="+", default=["Ki67", "ER", "HER2", "PR", "BCI"])
    p.add_argument("--split", default="valA")
    p.add_argument("--half", default=None,
                   help="test, sel or all; default test for valA, all for any other split")
    p.add_argument("--radius", default="5", help="centroid radius the recall is read at")
    p.add_argument("--fail-below", type=float, default=0.4)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--detector", default="stardist",
                   help="whose nuclei results: stardist, or instanseg / cpsam")
    args = p.parse_args()
    nuclei_dir = "" if args.detector == "stardist" else "nuclei_" + args.detector
    if args.half is None:
        args.half = "test" if args.split == "valA" else "all"

    out_dir = os.path.join(args.eval_root, "ensemble_report", "%s_%s%s" % (
        args.split, args.half, "_" + args.detector if nuclei_dir else ""))
    os.makedirs(out_dir, exist_ok=True)
    member_rows, spread_rows, versus_rows = [], [], []

    for marker in args.markers:
        eval_dir = os.path.join(args.eval_root, marker, args.split)
        ens_dir = os.path.join(eval_dir, "ensemble")
        if not os.path.isdir(ens_dir):
            print("[skip] %s/%s: nothing under %s" % (marker, args.split, ens_dir))
            continue
        split_csv = os.path.join(eval_dir, "val_test_split.csv")
        half = args.half
        if os.path.isfile(split_csv):
            split = {r[0]: r[2] for r in read_rows(split_csv)[1:]}
        else:
            split, half = WholeSplit(), "all"
            if args.half != "all":
                print("[note] %s/%s has no val_test_split.csv -- reported whole" % (marker, args.split))

        members: Dict[str, List[Tuple[int, Model, dict]]] = {m: [] for m, _ in MODELS}
        for name in sorted(os.listdir(ens_dir)):
            mm = MEMBER_RE.match(name)
            if not mm or mm.group("m") != marker:
                continue
            mod = Model(os.path.join(ens_dir, name), split, args.radius, nuclei_dir)
            nuc = mod.nuclei(half) or {}
            vals = {"recall": nuc.get("recall"), "precision": nuc.get("precision"),
                    "f1": nuc.get("f1"), "patch_ssim": mod.mean("patch_ssim", half),
                    "lpips": mod.mean("lpips", half), "fid": fid_of(mod, half),
                    "inverted": inverted_of(os.path.join(ens_dir, name))}
            members[mm.group("model")].append((int(mm.group("k")), mod, vals))

        print("\n== %s / %s, %s half   (%s nuclei, recall at r=%s px; collapsed = recall < %g)"
              % (marker, args.split, half, args.detector, args.radius, args.fail_below))
        print("  %-15s %5s  %-15s %6s  %-15s %-15s %-15s %-17s %-15s %-9s %s"
              % ("model", "seeds", "recall", "min", "precision", "F1", "SSIM", "LPIPS", "FID",
                 "collapsed", "inverted"))
        for key, nice in MODELS:
            ms = sorted(members[key], key=lambda t: t[0])
            for k, mod, vals in ms:
                member_rows.append([marker, args.split, half, key, k,
                                    *(fmt(vals[m], 5) for m in METRICS),
                                    int(vals["recall"] is not None and vals["recall"] < args.fail_below),
                                    "" if vals["inverted"] is None else vals["inverted"]])
            cells, row = [], [marker, args.split, half, key, len(ms)]
            for m in METRICS:
                v = [vals[m] for _, _, vals in ms if vals[m] is not None]
                if v:
                    sd = st.stdev(v) if len(v) > 1 else 0.0
                    d = {"fid": 1, "lpips": 4}.get(m, 3)
                    cells.append(("%." + str(d) + "f ± %." + str(d) + "f") % (st.mean(v), sd))
                    if m == "recall":
                        cells.append("%.3f" % min(v))
                    row += [fmt(st.mean(v), 5), fmt(sd, 5), fmt(min(v), 5), fmt(max(v), 5), len(v)]
                else:
                    cells += ["-", "-"] if m == "recall" else ["-"]
                    row += ["", "", "", "", 0]
            recs = [vals["recall"] for _, _, vals in ms if vals["recall"] is not None]
            n_fail = sum(r < args.fail_below for r in recs)
            inv = [vals["inverted"] for _, _, vals in ms if vals["inverted"] is not None]
            row += [n_fail, sum(inv) if inv else "", len(inv)]
            spread_rows.append(row)
            print("  %-15s %5d  %-15s %6s  %-15s %-15s %-15s %-17s %-15s %-9s %s"
                  % (nice, len(ms), *cells, "%d/%d" % (n_fail, len(recs)),
                     "%d/%d" % (sum(inv), len(inv)) if inv else "-"))

        topo = [mod.slide_sums(half) for _, mod, vals in members["topo"] if vals["recall"] is not None]
        for key, nice in MODELS[1:]:
            base = [mod.slide_sums(half) for _, mod, vals in members[key] if vals["recall"] is not None]
            if not topo or not base:
                continue
            d, lo, hi = two_level_diff(topo, base, args.boot, args.seed)
            print("  recall TopoCycleGAN - %-14s %+.4f  [95%% CI %+.4f, %+.4f]%s   (%d vs %d seeds)"
                  % (nice, d, lo, hi, "" if lo > 0 or hi < 0 else "   (interval includes 0)",
                     len(topo), len(base)))
            versus_rows.append([marker, args.split, half, key, len(topo), len(base),
                                fmt(d, 5), fmt(lo, 5), fmt(hi, 5)])

    def write(name, head, rows):
        with open(os.path.join(out_dir, name), "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(head)
            w.writerows(rows)

    write("members.csv", ["marker", "split", "half", "model", "seed", *METRICS, "collapsed",
                          "inverted"], member_rows)
    write("spread.csv", ["marker", "split", "half", "model", "seeds",
                         *("%s_%s" % (m, s) for m in METRICS
                           for s in ("mean", "sd", "min", "max", "n")), "collapsed",
                          "inverted", "inverted_n"], spread_rows)
    write("versus.csv", ["marker", "split", "half", "baseline", "topo_seeds", "baseline_seeds",
                         "recall_diff", "ci_low", "ci_high"], versus_rows)
    print("\nwritten under %s" % out_dir)


if __name__ == "__main__":
    main()
