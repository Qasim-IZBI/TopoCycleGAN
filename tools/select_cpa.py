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

    python tools/select_cpa.py --eval-dir /work2/bz66izin-TopoCG/eval/VS/testA

Output under <eval-dir>/selection_cpa/: selection.json, sel_ranking.csv,
test_report.csv, all_models_by_half.csv (every model, sel / test / all).
Standard library only.
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


class Model:
    def __init__(self, model_dir: str):
        self.dir = model_dir
        self.cpa = read_cpa(os.path.join(model_dir, "cpa_wsi.csv"))
        self.per_tile: Dict[str, Dict[str, float]] = {}
        for metric in ("patch_ssim", "lpips"):
            path = os.path.join(model_dir, metric + ".csv")
            if os.path.isfile(path):
                # i2i-evaluate names a tile by its path relative to the truth
                # root -- <case>/images/<tile>.tif for this set.
                self.per_tile[metric] = {
                    os.path.splitext(r[0].replace("\\", "/"))[0]: float(r[1])
                    for r in read_rows(path)[1:] if r[0] != "MEAN"}

    def cpa_stats(self, ref: Dict[str, float], cases: List[str]) -> Optional[dict]:
        common = [c for c in cases if c in self.cpa and c in ref]
        if not common:
            return None
        g = [self.cpa[c] for c in common]
        r = [ref[c] for c in common]
        d = [x - y for x, y in zip(g, r)]
        return {"mae": st.mean(abs(x) for x in d), "bias": st.mean(d),
                "pearson": pearson(g, r), "spearman": pearson(ranks(g), ranks(r)),
                "n": len(common)}

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


def bootstrap_mae_diff(a: Model, b: Model, ref: Dict[str, float], cases: List[str],
                       n_boot: int, seed: int) -> Tuple[float, float, float]:
    """MAE(a) - MAE(b) over the cases, and its 95% interval, resampling cases
    -- paired: one resample feeds both models."""
    cases = [c for c in cases if c in a.cpa and c in b.cpa and c in ref]
    if not cases:
        return float("nan"), float("nan"), float("nan")
    da = {c: abs(a.cpa[c] - ref[c]) for c in cases}
    db = {c: abs(b.cpa[c] - ref[c]) for c in cases}

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
    args = p.parse_args()

    split_csv = os.path.join(args.eval_dir, "val_test_split.csv")
    if not os.path.isfile(split_csv):
        raise SystemExit("no split at %s -- run `MARKERS=VS SPLIT=testA bash split_eval.sh split`"
                         % split_csv)
    case_half = {r[1]: r[2] for r in read_rows(split_csv)[1:]}
    ref = read_cpa(os.path.join(args.eval_dir, "cpa_ref", "cpa_wsi.csv"))
    if not ref:
        raise SystemExit("no real-SR reference under %s/cpa_ref -- run `evaluate_cpa.sh ref`"
                         % args.eval_dir)
    cases = {"sel": sorted(c for c in ref if case_half.get(c) == "sel"),
             "test": sorted(c for c in ref if case_half.get(c) == "test"),
             "all": sorted(ref)}
    unsplit = sorted(set(ref) - set(case_half))
    if unsplit:
        print("WARNING: slides in the reference but not in the split: %s" % " ".join(unsplit))
    print("reference: %d slides -- sel %s | test %s"
          % (len(ref), " ".join(cases["sel"]), " ".join(cases["test"])))

    models: Dict[Tuple[str, str], Model] = {}
    for arm in ARMS:
        arm_dir = os.path.join(args.eval_dir, arm)
        if not os.path.isdir(arm_dir):
            continue
        for name in sorted(os.listdir(arm_dir)):
            d = os.path.join(arm_dir, name)
            if os.path.isdir(d):
                models[(arm, name)] = Model(d)

    out_dir = os.path.join(args.eval_dir, "selection_cpa")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "all_models_by_half.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["arm", "name", "half", "slides", "cpa_mae", "cpa_bias", "cpa_pearson",
                    "cpa_spearman", "patch_ssim", "lpips", "fid"])
        for (arm, name), mod in sorted(models.items()):
            for half in HALVES:
                c = mod.cpa_stats(ref, cases[half]) or {}
                w.writerow([arm, name, half, c.get("n", ""),
                            *(fmt(c.get(k)) for k in ("mae", "bias", "pearson", "spearman")),
                            fmt(mod.mean("patch_ssim", set(cases[half]))),
                            fmt(mod.mean("lpips", set(cases[half])), 5), fmt(mod.fid(half), 3)])

    # ------------------------------------------------ selection, sel half only
    ranking = []
    for (arm, name), mod in models.items():
        cm = CELL_RE.match(name)
        if arm not in ("main", "cycsplit") or not cm:
            continue
        c = mod.cpa_stats(ref, cases["sel"])
        if not c:
            continue
        if c["n"] < len(cases["sel"]):
            print("[skip] %s/%s: %d of %d sel slides done" % (arm, name, c["n"], len(cases["sel"])))
            continue
        ranking.append({"arm": arm, "name": name, "lambda_topo": cm.group("lt"),
                        "ph_cyc": cm.group("c"), "ph_trans": cm.group("t"), **c})
    if not ranking:
        raise SystemExit("no TopoCycleGAN cell has CPA on every sel slide yet")
    ranking.sort(key=lambda r: (r["mae"], abs(r["bias"])))
    best = ranking[0]

    print("\nSELECTION -- sel half (%d slides), CPA MAE vs real SR" % len(cases["sel"]))
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
        c = mod.cpa_stats(ref, cases["test"]) or {}
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
        d, lo, hi = bootstrap_mae_diff(chosen, base, ref, cases["test"], args.boot, args.seed)
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
