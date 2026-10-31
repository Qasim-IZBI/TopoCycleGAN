"""The main paper's results figure: every seed of every model, per dataset.

Six panels, one per metric, datasets along x, models side by side within a
dataset; each dot is a seed, the bar its mean over seeds:

    SSIM (patch, vs the neighbouring section)  up
    LPIPS                                       down
    FID                                         down
    nuclei recall (H&E nuclei found again)      up      IHC datasets
    task MAE, percentage points                 down    IHC: |positive-nuclei %
                                                        generated - real| per
                                                        tile; VS: slide CPA
    BM0 / nuclei (Betti matching, dim 0)        down    IHC datasets

Reads the members.csv tools/ensemble_report.py writes -- one or more, e.g. the
valA report of the MIST markers, BCI's testA one and VS's -- and a member that
inverted (inverted = 1) is drawn as a cross, so the figure never hides one.

    python tools/results_figure.py \\
        --members $R/eval/ensemble_report/valA_test_cpsam/members.csv \\
                  $R/eval/ensemble_report/testA_all_cpsam/members.csv \\
                  $R/eval/ensemble_report/testA_all/members.csv \\
        --out results.pdf

The detailed numbers behind it are the supplementary tables (fill_results.py).
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

DATASETS = ("Ki67", "ER", "HER2", "PR", "BCI", "VS")
MODELS = (("topo", "TopoCycleGAN", "#1f5fa8"), ("cyclegan", "CycleGAN", "#d1782a"),
          ("dclgan", "DCLGAN", "#3f9a4a"), ("cyclediffusion", "CycleDiffusion", "#8a5bb0"))
PANELS = (("patch_ssim", "SSIM", True), ("lpips", "LPIPS", False), ("fid", "FID", False),
          ("recall", "Nuclei recall", True),
          ("task_mae", "Positive-nuclei % / CPA MAE (pp)", False),
          ("bm0_norm", "BM$_0$ / nuclei", False))


def read_members(paths):
    """{(dataset, model): [(value dict, inverted)]}, the first file to have a cell wins."""
    out = {}
    for path in paths:
        cells = defaultdict(list)
        with open(path) as fh:
            for r in csv.DictReader(fh):
                vals = {}
                for k, v in r.items():
                    try:
                        vals[k] = float(v)
                    except (TypeError, ValueError):
                        pass
                vals["task_mae"] = vals.get("cpa_mae") if r["marker"] == "VS" else vals.get("pos_mae")
                cells[(r["marker"], r["model"])].append((vals, r.get("inverted") == "1"))
        for key, rows in cells.items():
            out.setdefault(key, rows)
    return out


def draw(members, out_path, datasets=DATASETS, ncols=3, width=6.9):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    present = [d for d in datasets if any(k[0] == d for k in members)]
    models = [m for m in MODELS if any(k[1] == m[0] for k in members)]
    nrows = -(-len(PANELS) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, 1.55 * nrows + 0.3), squeeze=False)
    rng = np.random.default_rng(0)
    step = 0.8 / max(len(models), 1)
    for ax, (metric, title, up) in zip(axes.flat, PANELS):
        used = []
        for i, d in enumerate(present):
            had = False
            for j, (key, _, colour) in enumerate(models):
                rows = [(v[metric], inv) for v, inv in members.get((d, key), [])
                        if v.get(metric) is not None]
                if not rows:
                    continue
                had = True
                x = i - 0.4 + step * (j + 0.5)
                ys = np.array([y for y, _ in rows])
                xs = x + rng.uniform(-0.28, 0.28, len(ys)) * step
                ok = np.array([not inv for _, inv in rows])
                ax.scatter(xs[ok], ys[ok], s=5, color=colour, alpha=0.75, linewidths=0, zorder=3)
                if (~ok).any():
                    ax.scatter(xs[~ok], ys[~ok], s=12, marker="x", color=colour, linewidths=0.8,
                               zorder=3)
                ax.plot([x - 0.4 * step, x + 0.4 * step], [ys.mean()] * 2, color=colour, lw=1.4,
                        zorder=4)
            if had:
                used.append(i)
        ax.set_title("%s %s" % (title, "$\\uparrow$" if up else "$\\downarrow$"), fontsize=7.5)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present)
        for i in range(len(present)):
            if i not in used:
                ax.text(i, 0.5, "n/a", transform=ax.get_xaxis_transform(), ha="center",
                        va="center", fontsize=6, color="0.6")
        ax.set_xlim(-0.5, len(present) - 0.5)
        ax.tick_params(length=2, pad=1.5)
        ax.grid(axis="y", lw=0.3, color="0.88", zorder=0)
    for ax in list(axes.flat)[len(PANELS):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=c, marker="o", ms=3, lw=1.4, label=n) for _, n, c in models]
    if any(inv for rows in members.values() for _, inv in rows):
        handles.append(plt.Line2D([], [], color="0.3", marker="x", lw=0, ms=4, label="inverted seed"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=7, handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.6, w_pad=0.8)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--members", nargs="+", required=True,
                   help="ensemble_report.py members.csv files; earlier ones win a shared cell")
    p.add_argument("--out", required=True, help=".pdf, .svg or .png")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--ncols", type=int, default=3, help="3: 2 x 3 panels, 6: one row")
    p.add_argument("--width", type=float, default=6.9, help="inches (6.9 = CVPR text width)")
    a = p.parse_args()
    members = read_members(a.members)
    draw(members, a.out, a.datasets, a.ncols, a.width)
    for (d, m), rows in sorted(members.items()):
        print("%-5s %-15s %2d seeds" % (d, m, len(rows)))
    print("wrote %s" % os.path.abspath(a.out))


if __name__ == "__main__":
    main()
