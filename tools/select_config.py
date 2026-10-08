"""Choose the TopoCycleGAN configuration on one half of val, report it on the other.

Reads what evaluate.sh and evaluate_nuclei.sh already wrote -- per-tile nuclei
counts, per-tile patch SSIM and LPIPS -- and the split from split_val.py, and
recomputes every metric per half without segmenting or inferring again. FID is
a set statistic and cannot be split after the fact; split_eval.sh fid computes
it per half where it is needed.

SELECTION (the "sel" half, the selection markers -- MIST by default):
    candidates  every best-field configuration with ph_trans on: merged ph_cyc
                with cyc0/cyc1 x 4 lambda_topo, and split ph_cyc cyc1 x 4
    rule        highest nuclei recall averaged over the selection markers,
                failed runs counted at their real value (a configuration that
                works on some datasets only is not one to recommend); ties
                broken by mean nuclei F1. Nothing here looks at the test half.

TEST (the "test" half, every report marker): the chosen configuration against
CycleGAN small and DCLGAN small, with the identity row as reference, on nuclei
recall / precision / F1, patch SSIM, LPIPS and -- once computed -- FID. The
recall difference to each baseline gets a 95% bootstrap interval over SLIDES
(the unit the split was made on): slides are resampled with replacement and
the pooled recall recomputed, paired, since every model is matched against
the same H&E nuclei.

    python tools/select_config.py --eval-root /work2/bz66izin-TopoCG/eval

Output under <eval-root>/selection/: selection.json, sel_ranking.csv,
test_report.csv, all_models_by_half.csv (every model, both halves -- for the
supplementary tables). Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import statistics as st
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

CELL_RE = re.compile(r"^(?P<m>[^_]+)_lt(?P<lt>[0-9.eE+-]+)_cyc(?P<c>\d)_trans(?P<t>\d)$")
ARMS = ("main", "cycsplit", "worstfield_cycmerged", "worstfield_cycsplit", "baselines",
        "identity", "custom")


def read_rows(path: str) -> List[List[str]]:
    with open(path, newline="") as f:
        return [r for r in csv.reader(f) if r]


def load_split(eval_dir: str) -> Dict[str, str]:
    path = os.path.join(eval_dir, "val_test_split.csv")
    if not os.path.isfile(path):
        raise SystemExit("no split at %s -- run `split_eval.sh split` first" % path)
    return {r[0]: r[2] for r in read_rows(path)[1:]}


class Model:
    """One evaluated model: its per-tile results, reduced per half on demand."""

    def __init__(self, model_dir: str, split: Dict[str, str], radius: str):
        self.dir = model_dir
        self.tiles: Dict[str, Tuple[int, int, int]] = {}      # stem -> n_ref, n_pred, tp
        nt = os.path.join(model_dir, "nuclei_tiles.csv")
        if os.path.isfile(nt):
            rows = read_rows(nt)
            col = rows[0].index("tp_r%s" % radius)
            for r in rows[1:]:
                self.tiles[r[0]] = (int(r[1]), int(r[2]), int(r[col]))
        self.per_tile = {}                                     # metric -> {stem: v}
        for metric in ("patch_ssim", "lpips"):
            path = os.path.join(model_dir, metric + ".csv")
            if os.path.isfile(path):
                self.per_tile[metric] = {
                    os.path.splitext(os.path.basename(r[0]))[0]: float(r[1])
                    for r in read_rows(path)[1:] if r[0] != "MEAN"}
        self.split = split

    def nuclei(self, half: str) -> Optional[Dict[str, float]]:
        rows = [v for s, v in self.tiles.items() if self.split.get(s) == half]
        if not rows:
            return None
        n_ref = sum(r[0] for r in rows)
        n_pred = sum(r[1] for r in rows)
        tp = sum(r[2] for r in rows)
        rec = tp / n_ref if n_ref else 0.0
        prec = tp / n_pred if n_pred else 0.0
        f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0.0
        return {"recall": rec, "precision": prec, "f1": f1, "tiles": len(rows)}

    def mean(self, metric: str, half: str) -> Optional[float]:
        vals = [v for s, v in self.per_tile.get(metric, {}).items() if self.split.get(s) == half]
        return st.mean(vals) if vals else None

    def fid(self, half: str) -> Optional[float]:
        path = os.path.join(self.dir, "fid_%s.csv" % half)
        if not os.path.isfile(path):
            return None
        try:
            return float(read_rows(path)[1][1])
        except (IndexError, ValueError):
            return None

    def slide_sums(self, half: str) -> Dict[str, Tuple[int, int]]:
        """slide -> (n_ref, tp) over this half -- the bootstrap's resampling unit."""
        out: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        for s, (n_ref, _, tp) in self.tiles.items():
            if self.split.get(s) == half:
                out[s.split("_", 1)[0]][0] += n_ref
                out[s.split("_", 1)[0]][1] += tp
        return {k: (v[0], v[1]) for k, v in out.items()}


def bootstrap_diff(a: Dict[str, Tuple[int, int]], b: Dict[str, Tuple[int, int]],
                   n_boot: int, seed: int) -> Tuple[float, float, float]:
    """Recall(a) - recall(b) and its 95% interval, resampling shared slides --
    paired: one resample of slides feeds both models. Each recall uses its own
    n_ref; on real data they are equal (one H&E reference for every model),
    but nothing here depends on it."""
    slides = sorted(set(a) & set(b))
    if not slides:
        return float("nan"), float("nan"), float("nan")

    def recall(m, sample):
        n = sum(m[s][0] for s in sample)
        return sum(m[s][1] for s in sample) / n if n else 0.0

    def diff(sample):
        return recall(a, sample) - recall(b, sample)

    rng = random.Random(seed)
    boots = sorted(diff([rng.choice(slides) for _ in slides]) for _ in range(n_boot))
    return diff(slides), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def label(arm: str, lt: str, c: str, t: str) -> str:
    return "%s  lt=%s  cyc%s trans%s" % ("merged" if arm == "main" else "split ", lt, c, t)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-root", required=True)
    p.add_argument("--split", default="valA")
    p.add_argument("--select-markers", nargs="+", default=["Ki67", "ER", "HER2", "PR"])
    p.add_argument("--report-markers", nargs="+", default=["Ki67", "ER", "HER2", "PR", "BCI"])
    p.add_argument("--radius", default="5", help="centroid radius the recall is read at")
    p.add_argument("--fail-below", type=float, default=0.4,
                   help="a run below this recall is counted as failed (collapsed)")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    markers = list(dict.fromkeys(args.select_markers + args.report_markers))
    models: Dict[Tuple[str, str, str], Model] = {}            # (marker, arm, name)
    for m in markers:
        eval_dir = os.path.join(args.eval_root, m, args.split)
        if not os.path.isdir(eval_dir):
            print("[skip] %s: nothing under %s" % (m, eval_dir))
            continue
        split = load_split(eval_dir)
        for arm in ARMS:
            arm_dir = os.path.join(eval_dir, arm)
            if not os.path.isdir(arm_dir):
                continue
            for name in sorted(os.listdir(arm_dir)):
                if os.path.isdir(os.path.join(arm_dir, name)):
                    models[(m, arm, name)] = Model(os.path.join(arm_dir, name), split, args.radius)

    out_dir = os.path.join(args.eval_root, "selection")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------ every model, both halves
    with open(os.path.join(out_dir, "all_models_by_half.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["marker", "arm", "name", "half", "tiles", "nuc_recall", "nuc_precision",
                    "nuc_f1", "patch_ssim", "lpips", "fid"])
        for (m, arm, name), mod in sorted(models.items()):
            for half in ("sel", "test"):
                n = mod.nuclei(half) or {}
                w.writerow([m, arm, name, half, n.get("tiles", ""),
                            *("%.4f" % n[k] if k in n else "" for k in ("recall", "precision", "f1")),
                            *("%.4f" % v if v is not None else ""
                              for v in (mod.mean("patch_ssim", half), mod.mean("lpips", half),
                                        mod.fid(half)))])

    # ------------------------------------------------ selection, sel half only
    configs: Dict[Tuple[str, str, str, str], Dict[str, Dict[str, float]]] = defaultdict(dict)
    for (m, arm, name), mod in models.items():
        if m not in args.select_markers or arm not in ("main", "cycsplit"):
            continue
        cm = CELL_RE.match(name)
        if not cm or cm.group("t") != "1":
            continue
        n = mod.nuclei("sel")
        if n:
            configs[(arm, cm.group("lt"), cm.group("c"), cm.group("t"))][m] = n

    ranking = []
    for key, per_marker in configs.items():
        if set(per_marker) != set(args.select_markers):
            print("[skip] %s: not evaluated on %s" % (label(*key),
                  ", ".join(sorted(set(args.select_markers) - set(per_marker)))))
            continue
        recs = [per_marker[m]["recall"] for m in args.select_markers]
        f1s = [per_marker[m]["f1"] for m in args.select_markers]
        ranking.append({"arm": key[0], "lambda_topo": key[1], "ph_cyc": key[2],
                        "ph_trans": key[3], "label": label(*key),
                        "mean_recall": st.mean(recs), "mean_f1": st.mean(f1s),
                        "failed": sum(r < args.fail_below for r in recs),
                        "recall": {m: per_marker[m]["recall"] for m in args.select_markers}})
    if not ranking:
        raise SystemExit("no complete candidate configuration -- are the nuclei results there?")
    ranking.sort(key=lambda r: (-r["mean_recall"], -r["mean_f1"]))
    best = ranking[0]

    print("\nSELECTION -- sel half, %s, recall at r=%s px; failed = recall < %g"
          % (" ".join(args.select_markers), args.radius, args.fail_below))
    head = "  %-34s" % "configuration" + "".join("%8s" % m for m in args.select_markers) \
        + "%10s%9s%8s" % ("mean rec", "mean F1", "failed")
    print(head)
    for r in ranking:
        print("  %-34s" % r["label"]
              + "".join("%8.3f" % r["recall"][m] for m in args.select_markers)
              + "%10.3f%9.3f%6d/%d" % (r["mean_recall"], r["mean_f1"], r["failed"],
                                         len(args.select_markers))
              + ("   <- chosen" if r is best else ""))
    with open(os.path.join(out_dir, "sel_ranking.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "configuration", *args.select_markers, "mean_recall", "mean_f1", "failed"])
        for i, r in enumerate(ranking, 1):
            w.writerow([i, r["label"].replace("  ", " "),
                        *("%.4f" % r["recall"][m] for m in args.select_markers),
                        "%.4f" % r["mean_recall"], "%.4f" % r["mean_f1"], r["failed"]])

    # ------------------------------------------------ test half: best vs small baselines
    def name_of(m: str, which: str) -> Tuple[str, str]:
        if which == "best":
            return best["arm"], "%s_lt%s_cyc%s_trans%s" % (m, best["lambda_topo"],
                                                          best["ph_cyc"], best["ph_trans"])
        if which == "identity":
            return "identity", "%s_identity" % m
        return "main", "%s_%s_small" % (m, which)

    contenders = (("best", "TopoCycleGAN (chosen)"), ("cyclegan", "CycleGAN small"),
                  ("dclgan", "DCLGAN small"), ("identity", "identity (H&E unchanged)"))
    print("\nTEST -- test half; FID only once `split_eval.sh fid` has run for it")
    rows = []
    for m in args.report_markers:
        print("\n  %s" % m)
        print("    %-26s%8s%8s%8s%8s%9s%8s" % ("model", "recall", "prec", "F1", "SSIM", "LPIPS", "FID"))
        for which, nice in contenders:
            mod = models.get((m,) + name_of(m, which))
            if mod is None:
                print("    %-26s  (not evaluated)" % nice)
                continue
            n = mod.nuclei("test") or {}
            ssim, lp, fid = mod.mean("patch_ssim", "test"), mod.mean("lpips", "test"), mod.fid("test")
            fmt = lambda v, d=3: ("%." + str(d) + "f") % v if v is not None else "-"
            print("    %-26s%8s%8s%8s%8s%9s%8s" % (nice, fmt(n.get("recall")), fmt(n.get("precision")),
                                                   fmt(n.get("f1")), fmt(ssim), fmt(lp, 4), fmt(fid, 1)))
            rows.append([m, nice, mod.dir, n.get("tiles", ""),
                         *(fmt(n.get(k), 4) for k in ("recall", "precision", "f1")),
                         fmt(ssim, 4), fmt(lp, 5), fmt(fid, 3)])
        best_mod = models.get((m,) + name_of(m, "best"))
        for which, nice in contenders[1:3]:
            base = models.get((m,) + name_of(m, which))
            if best_mod and base:
                d, lo, hi = bootstrap_diff(best_mod.slide_sums("test"), base.slide_sums("test"),
                                           args.boot, args.seed)
                print("    recall vs %-15s %+.3f  [95%% CI %+.3f, %+.3f]%s"
                      % (nice, d, lo, hi, "" if lo > 0 or hi < 0 else "   (interval includes 0)"))
                rows.append([m, "recall diff vs " + nice, "", "", "%.4f" % d, "%.4f" % lo,
                             "%.4f" % hi, "", "", ""])

    with open(os.path.join(out_dir, "test_report.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["marker", "model", "dir", "tiles", "nuc_recall (or diff)", "nuc_precision (or CI low)",
                    "nuc_f1 (or CI high)", "patch_ssim", "lpips", "fid"])
        w.writerows(rows)
    with open(os.path.join(out_dir, "selection.json"), "w") as f:
        json.dump({"rule": "highest mean nuclei recall (r=%s px) over %s on the sel half; "
                           "ties by mean F1; failed (< %g) runs counted at their value"
                           % (args.radius, ", ".join(args.select_markers), args.fail_below),
                   "chosen": {k: best[k] for k in ("arm", "lambda_topo", "ph_cyc", "ph_trans", "label",
                                                   "mean_recall", "mean_f1", "failed")},
                   "ranking": ranking}, f, indent=2)

    task = grid_task(best["lambda_topo"], best["ph_cyc"], best["ph_trans"])
    print("\nchosen: %s  (arm=%s, task %s)" % (best["label"], best["arm"], task))
    if any(models.get((m,) + name_of(m, "best")) and
           models[(m,) + name_of(m, "best")].fid("test") is None for m in args.report_markers):
        print("\nFID on the test half is still missing -- from slurm/:")
        for m in args.report_markers:
            print("  sbatch --array=%s --export=ALL,MARKER=%s,ARM=%s,HALF=test split_eval.sh fid"
                  % (task, m, best["arm"]))
            print("  sbatch --array=13,14 --export=ALL,MARKER=%s,ARM=main,HALF=test split_eval.sh fid" % m)
        print("  then run this again to fill the FID column.")
    print("\nwritten under %s" % out_dir)


# _grid.sh's cell order: 0 is the baseline, then per lambda_topo (ascending)
# the three PH settings cyc0/trans1, cyc1/trans0, cyc1/trans1.
GRID_LAMBDAS = ("0.0002", "0.002", "0.02", "0.2")
GRID_PH = (("0", "1"), ("1", "0"), ("1", "1"))


def grid_task(lt: str, c: str, t: str) -> str:
    try:
        return str(1 + 3 * GRID_LAMBDAS.index(lt) + GRID_PH.index((c, t)))
    except ValueError:
        return "?"


if __name__ == "__main__":
    main()
