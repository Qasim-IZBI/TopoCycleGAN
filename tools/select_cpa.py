"""Choose the VS TopoCycleGAN configuration on its task metric, CPA.

VS (H&E -> Sirius Red) has no nuclei to recover, so it is not chosen by
select_config.py's nuclei recall. Its task metric is the collagen
proportionate area of each whole test slide against the real Sirius Red of the
registered serial section -- the zoo paper's CPA MAE -- which evaluate_cpa.sh
computes per slide. This reads those results and split_val.py's split of the
20 registered test slides into a selection and a test half (by slide, so no
slide is in both), and:

SELECTION (sel half): every TopoCycleGAN cell evaluated in the main (merged
    ph_cyc) and cycsplit arms, ranked by CPA MAE over the sel slides; ties
    broken by the smaller absolute bias. A cell missing any sel slide is not
    ranked. Nothing here looks at the test half.

TEST (test half): the chosen cell against CycleGAN small, DCLGAN small, the
    sweep's own vanilla CycleGAN cell and the identity row, on CPA MAE / bias /
    Pearson / Spearman, patch SSIM, LPIPS and -- once computed per half by
    `split_eval.sh fid` -- FID. The MAE difference to each baseline gets a 95%
    bootstrap interval over the test SLIDES, paired: one resample of slides
    feeds both models.

With 10 slides per half the intervals are wide; read them as such.

REGIONS (--region N): instead of one CPA per whole slide, CPA per N x N px
    region of the stitched slide (N a multiple of the 512 px tile; 2048 = 4x4
    tiles), summed from the per-tile counts evaluate_cpa.sh already wrote --
    nothing is segmented again, and holes were filled on the whole slide, so
    the counts are exact at any N. A slide's error is the tissue-weighted mean
    |CPA_gen - CPA_real| over its regions; the MAE is the mean of that over
    slides, so every slide weighs the same and the slide stays the unit of the
    bootstrap (regions of one slide are not independent). The regions scored
    are fixed by the REAL SR -- those with at least --min-tissue of their area
    in tissue -- so every model is scored on the same ones. A generated region
    with no tissue at all counts as CPA 0. Unlike the whole-slide MAE, too much
    collagen in one place no longer cancels too little in another.

    Whatever --region is, a scan over --scan (whole slide and 1024 / 2048 /
    4096 px by default) prints and writes, per scale, how far the sel-half
    ranking of the cells carries over to the test half (Spearman over the
    cells) and which cell each scale would choose -- the evidence for fixing
    one scale for the paper. Fix it on that, not on the test result.

    python tools/select_cpa.py --eval-dir /work2/bz66izin-TopoCG/eval/VS/testA
    python tools/select_cpa.py --eval-dir ... --region 2048

Output under <eval-dir>/selection_cpa/ (whole slide) or selection_cpa_r<N>/:
selection.json, sel_ranking.csv, test_report.csv, all_models_by_half.csv
(every model, sel / test / all), scale_scan.csv. Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics as st
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_config import ARMS, CELL_RE, grid_task, read_rows  # noqa: E402

HALVES = ("sel", "test", "all")


def case_of(stem: str) -> str:
    return stem.split("/", 1)[0] if "/" in stem else stem.split("_", 1)[0]


def read_cpa(path: str) -> Dict[str, float]:
    if not os.path.isfile(path):
        return {}
    rows = read_rows(path)
    head = rows[0]
    ic, iv = head.index("case"), head.index("cpa")
    return {r[ic]: float(r[iv]) for r in rows[1:] if len(r) > iv and r[iv] != ""}


def pearson(a: List[float], b: List[float]) -> Optional[float]:
    if len(a) < 3:
        return None
    ma, mb = st.mean(a), st.mean(b)
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


WSI = 0          # "scale" of the whole slide


def scale_name(scale: int) -> str:
    return "wsi" if scale == WSI else "r%d" % scale


class Model:
    def __init__(self, model_dir: str):
        self.dir = model_dir
        self.cpa = read_cpa(os.path.join(model_dir, "cpa_wsi.csv"))
        self.regions: Dict[int, Dict[str, Dict[Tuple[int, int], List[int]]]] = {}
        self._errors: Dict[Tuple[int, float], Dict[str, Tuple[float, float, list]]] = {}
        self.per_tile: Dict[str, Dict[str, float]] = {}
        for metric in ("patch_ssim", "lpips"):
            path = os.path.join(model_dir, metric + ".csv")
            if os.path.isfile(path):
                # i2i-evaluate names a tile by its path relative to the truth
                # root -- <case>/images/<tile>.tif for this set.
                self.per_tile[metric] = {
                    os.path.splitext(r[0].replace("\\", "/"))[0]: float(r[1])
                    for r in read_rows(path)[1:] if r[0] != "MEAN"}

    def load_regions(self, scales: List[int]) -> None:
        """Sum cpa_tiles.csv's per-tile counts into regions, every scale in one read:
        scale -> case -> (row, col) -> [tissue + SR + hole px, SR px]."""
        want = [s for s in scales if s != WSI and s not in self.regions]
        if not want:
            return
        path = os.path.join(self.dir, "cpa_tiles.csv")
        for s in want:
            self.regions[s] = {}
        if not os.path.isfile(path):
            return
        with open(path, newline="") as f:
            rd = csv.reader(f)
            head = next(rd)
            ic, ix, iy = head.index("case"), head.index("x"), head.index("y")
            it, isr, ih = head.index("tissue_px"), head.index("sr_px"), head.index("hole_px")
            for r in rd:
                if not r:
                    continue
                x, y, sr = int(r[ix]), int(r[iy]), int(r[isr])
                tot = int(r[it]) + sr + int(r[ih])
                if not tot:
                    continue
                for s in want:
                    acc = self.regions[s].setdefault(r[ic], {}).setdefault((y // s, x // s), [0, 0])
                    acc[0] += tot
                    acc[1] += sr

    def case_errors(self, ref: "Model", scale: int, min_tissue: float
                    ) -> Dict[str, Tuple[float, float, list]]:
        """case -> (|error|, signed error, [(gen, real) CPA per scored unit]). Whole
        slide: one unit. Regions: the real side's regions with at least min_tissue
        of their area in tissue, errors weighted by the real tissue."""
        key = (scale, min_tissue)
        if key in self._errors:
            return self._errors[key]
        out = {}
        if scale == WSI:
            for c in self.cpa:
                if c in ref.cpa:
                    g, r = self.cpa[c], ref.cpa[c]
                    out[c] = (abs(g - r), g - r, [(g, r)])
        else:
            floor = min_tissue * scale * scale
            mine_all = self.regions.get(scale, {})
            for c, regs in ref.regions.get(scale, {}).items():
                if c not in self.cpa:            # slide not done: no partial scoring
                    continue
                mine = mine_all.get(c, {})
                wsum = ea = eb = 0.0
                pairs = []
                for k, (rt, rs) in regs.items():
                    if rt < floor:
                        continue
                    gt, gs = mine.get(k, (0, 0))
                    g, r = (gs / gt if gt else 0.0), rs / rt
                    wsum += rt
                    ea += rt * abs(g - r)
                    eb += rt * (g - r)
                    pairs.append((g, r))
                if wsum:
                    out[c] = (ea / wsum, eb / wsum, pairs)
        self._errors[key] = out
        return out

    def cpa_stats(self, ref: "Model", cases: List[str], scale: int = WSI,
                  min_tissue: float = 0.0) -> Optional[dict]:
        """MAE and bias: mean over slides of each slide's error. Pearson and
        Spearman: over every scored unit (slides, or the regions pooled)."""
        err = self.case_errors(ref, scale, min_tissue)
        common = [c for c in cases if c in err]
        if not common:
            return None
        pairs = [p for c in common for p in err[c][2]]
        g, r = [p[0] for p in pairs], [p[1] for p in pairs]
        return {"mae": st.mean(err[c][0] for c in common),
                "bias": st.mean(err[c][1] for c in common),
                "pearson": pearson(g, r), "spearman": pearson(ranks(g), ranks(r)),
                "n": len(common), "units": len(pairs)}

    def mean(self, metric: str, cases: set) -> Optional[float]:
        vals = [v for s, v in self.per_tile.get(metric, {}).items() if case_of(s) in cases]
        return st.mean(vals) if vals else None

    def fid(self, half: str) -> Optional[float]:
        path = os.path.join(self.dir, "fid.csv" if half == "all" else "fid_%s.csv" % half)
        if not os.path.isfile(path):
            return None
        try:
            return float(read_rows(path)[1][1])
        except (IndexError, ValueError):
            return None


def bootstrap_mae_diff(a: Model, b: Model, ref: Model, cases: List[str], scale: int,
                       min_tissue: float, n_boot: int, seed: int) -> Tuple[float, float, float]:
    """MAE(a) - MAE(b) over the cases, and its 95% interval, resampling cases
    -- paired: one resample feeds both models. At a region scale a slide is
    resampled with all its regions."""
    ea, eb = a.case_errors(ref, scale, min_tissue), b.case_errors(ref, scale, min_tissue)
    cases = [c for c in cases if c in ea and c in eb]
    if not cases:
        return float("nan"), float("nan"), float("nan")
    da = {c: ea[c][0] for c in cases}
    db = {c: eb[c][0] for c in cases}

    def diff(sample):
        return st.mean(da[c] for c in sample) - st.mean(db[c] for c in sample)

    rng = random.Random(seed)
    boots = sorted(diff([rng.choice(cases) for _ in cases]) for _ in range(n_boot))
    return diff(cases), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def fmt(v, d=4):
    return ("%." + str(d) + "f") % v if v is not None else "-"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", required=True, help="<eval-root>/VS/testA")
    p.add_argument("--marker", default="VS")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--region", default="wsi",
                   help="scale the selection and report use: wsi (whole slide) or a region "
                        "size in px, a multiple of the tile (e.g. 2048)")
    p.add_argument("--min-tissue", type=float, default=0.25,
                   help="a region is scored when at least this fraction of it is tissue "
                        "in the real SR")
    p.add_argument("--scan", nargs="+", default=["wsi", "1024", "2048", "4096"],
                   help="scales the sel-vs-test agreement scan covers")
    args = p.parse_args()

    def parse_scale(v: str) -> int:
        return WSI if v == "wsi" else int(v)

    scale = parse_scale(args.region)
    scan = list(dict.fromkeys([parse_scale(v) for v in args.scan] + [scale]))
    mt = args.min_tissue

    split_csv = os.path.join(args.eval_dir, "val_test_split.csv")
    if not os.path.isfile(split_csv):
        raise SystemExit("no split at %s -- run `MARKERS=VS SPLIT=testA bash split_eval.sh split`"
                         % split_csv)
    case_half = {r[1]: r[2] for r in read_rows(split_csv)[1:]}
    ref = Model(os.path.join(args.eval_dir, "cpa_ref"))
    if not ref.cpa:
        raise SystemExit("no real-SR reference under %s/cpa_ref -- run `evaluate_cpa.sh ref`"
                         % args.eval_dir)
    tile = set()
    for c in ref.cpa:
        sj = os.path.join(ref.dir, "slides", c + ".json")
        if os.path.isfile(sj):
            with open(sj) as f:
                tile.add(int(json.load(f)["tile_size"]))
    for sc in scan:
        if sc != WSI and any(sc % t for t in tile):
            raise SystemExit("region %d px is not a multiple of the tile size (%s px)"
                             % (sc, "/".join(map(str, sorted(tile)))))
    cases = {"sel": sorted(c for c in ref.cpa if case_half.get(c) == "sel"),
             "test": sorted(c for c in ref.cpa if case_half.get(c) == "test"),
             "all": sorted(ref.cpa)}
    unsplit = sorted(set(ref.cpa) - set(case_half))
    if unsplit:
        print("WARNING: slides in the reference but not in the split: %s" % " ".join(unsplit))
    print("reference: %d slides -- sel %s | test %s"
          % (len(ref.cpa), " ".join(cases["sel"]), " ".join(cases["test"])))

    models: Dict[Tuple[str, str], Model] = {}
    for arm in ARMS:
        arm_dir = os.path.join(args.eval_dir, arm)
        if not os.path.isdir(arm_dir):
            continue
        for name in sorted(os.listdir(arm_dir)):
            d = os.path.join(arm_dir, name)
            if os.path.isdir(d):
                models[(arm, name)] = Model(d)
    for mod in [ref, *models.values()]:
        mod.load_regions(scan)
    if scale != WSI:
        print("scale: %d x %d px regions with >= %g tissue in the real SR -- %d regions "
              "on %d slides" % (scale, scale, mt,
                                sum(len(e[2]) for e in ref.case_errors(ref, scale, mt).values()),
                                len(ref.case_errors(ref, scale, mt))))

    out_dir = os.path.join(args.eval_dir, "selection_cpa"
                           + ("" if scale == WSI else "_r%d" % scale))
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "all_models_by_half.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["arm", "name", "half", "slides", "cpa_mae", "cpa_bias", "cpa_pearson",
                    "cpa_spearman", "patch_ssim", "lpips", "fid", "scale", "units"])
        for (arm, name), mod in sorted(models.items()):
            for half in HALVES:
                c = mod.cpa_stats(ref, cases[half], scale, mt) or {}
                w.writerow([arm, name, half, c.get("n", ""),
                            *(fmt(c.get(k)) for k in ("mae", "bias", "pearson", "spearman")),
                            fmt(mod.mean("patch_ssim", set(cases[half]))),
                            fmt(mod.mean("lpips", set(cases[half])), 5), fmt(mod.fid(half), 3),
                            scale_name(scale), c.get("units", "")])

    # ------------------------------------------------ the scale scan
    def rank_cells(sc: int, half: str) -> List[dict]:
        out = []
        for (arm, name), mod in models.items():
            cm = CELL_RE.match(name)
            if arm not in ("main", "cycsplit") or not cm:
                continue
            c = mod.cpa_stats(ref, cases[half], sc, mt)
            if c and c["n"] == len(cases[half]):
                out.append({"arm": arm, "name": name, "lambda_topo": cm.group("lt"),
                            "ph_cyc": cm.group("c"), "ph_trans": cm.group("t"), **c})
        out.sort(key=lambda r: (r["mae"], abs(r["bias"])))
        return out

    print("\nSCALE SCAN -- does the sel-half ranking of the cells hold on the test half?")
    print("  %-7s %5s %9s %9s  %-30s %8s %8s"
          % ("scale", "cells", "spearman", "top3 hit", "sel would choose", "its sel", "its test"))
    scan_rows = []
    for sc in scan:
        rs, rt = rank_cells(sc, "sel"), rank_cells(sc, "test")
        both = {(r["arm"], r["name"]) for r in rs} & {(r["arm"], r["name"]) for r in rt}
        ms = {(r["arm"], r["name"]): r["mae"] for r in rs if (r["arm"], r["name"]) in both}
        mtst = {(r["arm"], r["name"]): r["mae"] for r in rt if (r["arm"], r["name"]) in both}
        keys = sorted(both)
        rho = pearson(ranks([ms[k] for k in keys]), ranks([mtst[k] for k in keys])) \
            if len(keys) >= 3 else None
        top_s = [(r["arm"], r["name"]) for r in rs if (r["arm"], r["name"]) in both][:3]
        top_t = [(r["arm"], r["name"]) for r in rt if (r["arm"], r["name"]) in both][:3]
        hit = len(set(top_s) & set(top_t))
        pick = top_s[0] if top_s else None
        print("  %-7s %5d %9s %7d/3  %-30s %8s %8s"
              % (scale_name(sc), len(keys), fmt(rho, 3), hit, pick[1] if pick else "-",
                 fmt(ms.get(pick)), fmt(mtst.get(pick))))
        scan_rows.append([scale_name(sc), mt, len(keys), fmt(rho), hit,
                          "%s/%s" % pick if pick else "", fmt(ms.get(pick), 5),
                          fmt(mtst.get(pick), 5)])
    with open(os.path.join(out_dir, "scale_scan.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["scale", "min_tissue", "cells", "sel_test_spearman", "top3_overlap",
                    "sel_choice", "choice_sel_mae", "choice_test_mae"])
        w.writerows(scan_rows)

    # ------------------------------------------------ selection, sel half only
    for (arm, name), mod in models.items():
        c = mod.cpa_stats(ref, cases["sel"], scale, mt)
        if arm in ("main", "cycsplit") and CELL_RE.match(name) and c \
                and c["n"] < len(cases["sel"]):
            print("[skip] %s/%s: %d of %d sel slides done" % (arm, name, c["n"], len(cases["sel"])))
    ranking = rank_cells(scale, "sel")
    if not ranking:
        raise SystemExit("no TopoCycleGAN cell has CPA on every sel slide yet")
    best = ranking[0]

    print("\nSELECTION -- sel half (%d slides), CPA MAE vs real SR, %s"
          % (len(cases["sel"]), "whole slide" if scale == WSI else
             "%d px regions (tissue-weighted per slide, then mean over slides)" % scale))
    print("  %-10s %-30s %9s %9s %8s %8s" % ("arm", "cell", "MAE", "bias", "pearson", "spearman"))
    for r in ranking:
        print("  %-10s %-30s %9.4f %+9.4f %8s %8s%s"
              % (r["arm"], r["name"], r["mae"], r["bias"], fmt(r["pearson"], 3),
                 fmt(r["spearman"], 3), "   <- chosen" if r is best else ""))
    with open(os.path.join(out_dir, "sel_ranking.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["rank", "arm", "name", "lambda_topo", "ph_cyc", "ph_trans", "slides",
                    "cpa_mae", "cpa_bias", "cpa_pearson", "cpa_spearman"])
        for i, r in enumerate(ranking, 1):
            w.writerow([i, r["arm"], r["name"], r["lambda_topo"], r["ph_cyc"], r["ph_trans"],
                        r["n"], fmt(r["mae"], 5), fmt(r["bias"], 5), fmt(r["pearson"]),
                        fmt(r["spearman"])])

    # ------------------------------------------------ test half
    m = args.marker

    def find(name: str, arms=("main", "baselines")) -> Optional[Model]:
        for arm in arms:
            if (arm, name) in models:
                return models[(arm, name)]
        return None

    contenders = [("TopoCycleGAN (chosen)", models[(best["arm"], best["name"])]),
                  ("CycleGAN small", find("%s_cyclegan_small" % m)),
                  ("DCLGAN small", find("%s_dclgan_small" % m)),
                  ("sweep cell 0 (vanilla)", find("%s_baseline_cyclegan" % m, ("main",))),
                  ("identity (H&E unchanged)", find("%s_identity" % m, ("identity",)))]
    test = set(cases["test"])
    print("\nTEST -- test half (%d slides); FID only once `split_eval.sh fid` has run for it"
          % len(cases["test"]))
    print("    %-26s%9s%9s%9s%9s%8s%9s%8s" % ("model", "MAE", "bias", "pearson", "spearman",
                                            "SSIM", "LPIPS", "FID"))
    rows = []
    for nice, mod in contenders:
        if mod is None:
            print("    %-26s  (not evaluated)" % nice)
            continue
        c = mod.cpa_stats(ref, cases["test"], scale, mt) or {}
        ssim, lp, fid = mod.mean("patch_ssim", test), mod.mean("lpips", test), mod.fid("test")
        print("    %-26s%9s%9s%9s%9s%8s%9s%8s"
              % (nice, fmt(c.get("mae")), fmt(c.get("bias")), fmt(c.get("pearson"), 3),
                 fmt(c.get("spearman"), 3), fmt(ssim, 3), fmt(lp, 4), fmt(fid, 1)))
        rows.append([nice, mod.dir, c.get("n", ""),
                     *(fmt(c.get(k), 5) for k in ("mae", "bias", "pearson", "spearman")),
                     fmt(ssim), fmt(lp, 5), fmt(fid, 3), "", ""])
    chosen = contenders[0][1]
    for nice, base in contenders[1:4]:
        if base is None:
            continue
        d, lo, hi = bootstrap_mae_diff(chosen, base, ref, cases["test"], scale, mt,
                                       args.boot, args.seed)
        print("    MAE vs %-19s %+.4f  [95%% CI %+.4f, %+.4f]%s"
              % (nice, d, lo, hi, "" if lo > 0 or hi < 0 else "   (interval includes 0)"))
        rows.append(["MAE diff vs " + nice, "", "", fmt(d, 5), "", "", "", "", "", "",
                     fmt(lo, 5), fmt(hi, 5)])

    with open(os.path.join(out_dir, "test_report.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["model", "dir", "slides", "cpa_mae (or diff)", "cpa_bias", "cpa_pearson",
                    "cpa_spearman", "patch_ssim", "lpips", "fid", "ci_low", "ci_high"])
        w.writerows(rows)
    with open(os.path.join(out_dir, "selection.json"), "w") as f:
        json.dump({"rule": "lowest CPA MAE vs real SR over the sel slides; ties by |bias|",
                   "scale": scale_name(scale), "min_tissue": mt if scale != WSI else None,
                   "sel_slides": cases["sel"], "test_slides": cases["test"],
                   "chosen": {k: best[k] for k in ("arm", "name", "lambda_topo", "ph_cyc",
                                                   "ph_trans", "mae", "bias", "n")},
                   "ranking": ranking}, f, indent=2)

    task = grid_task(best["lambda_topo"], best["ph_cyc"], best["ph_trans"])
    print("\nchosen: %s/%s  (task %s)" % (best["arm"], best["name"], task))
    if chosen.fid("test") is None:
        print("\nFID on the test half is still missing -- from slurm/:")
        base = "--export=ALL,MARKER=%s,SPLIT=testA,HALF=test,MIN_TISSUE=0.1" % m
        print("  sbatch --partition=paula --array=%s %s,ARM=%s split_eval.sh fid"
              % (task, base, best["arm"]))
        print("  sbatch --partition=paula --array=0,13,14 %s,ARM=main split_eval.sh fid" % base)
        print("  sbatch --partition=paula --array=0 %s,ARM=identity split_eval.sh fid" % base)
        print("  then run this again to fill the FID column.")
    print("\nwritten under %s" % out_dir)


if __name__ == "__main__":
    main()
