"""Plot what evaluate.sh scored: one figure per marker and metric.

Reads the per-model CSVs i2i-evaluate wrote under

    <eval_root>/<marker>/<split>/<arm>/<name>/{fid,patch_ssim,lpips}.csv

and draws, for each metric, five panels on one shared y-axis so the arms can be
read against each other directly:

    best field, merged | best field, split | worst field, merged |
    worst field, split | baselines by size

In the four sweep panels the x-axis is lambda_topo and each PH configuration
(cyc0 trans1, cyc1 trans0, cyc1 trans1) is one line. The split arms train only
the ph_cyc=1 cells, so they have two lines, not three. The baselines are grey
everywhere -- horizontal reference lines in the sweep panels (the sweep's own
vanilla CycleGAN cell 0 and the small zoo baselines), and the capacity ladder
in the last panel -- so a baseline never shares a colour with a PH
configuration.

A collapsed run (FID in the hundreds) would flatten every other line, so the
y-axis stops at a robust upper bound and anything beyond it is drawn as a
triangle on the top edge with its value printed beside it.

    python -m topo_i2i.plot_eval --eval-root /work2/bz66izin-TopoCG/eval \\
        --markers ER PR HER2 Ki67 BCI --split valA

Output: <eval_root>/<marker>/<split>/plots/<metric>.{png,pdf}
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

METRICS = {
    # name: (axis label, lower is better)
    "fid": ("FID", True),
    "patch_ssim": ("Patch SSIM", False),
    "lpips": ("LPIPS (zoo, unweighted VGG16)", True),
    # From evaluate_nuclei.sh: StarDist nuclei on the H&E matched by centroid
    # to nuclei on the generated IHC. Plotted only where it has been run.
    "nuc_recall": ("Nuclei recall (H&E nuclei found again)", False),
    "nuc_f1": ("Nuclei F1 (centroid match)", False),
}

# Which file each metric is read from, and the row holding its value.
SOURCES = {
    "fid": ("fid.csv", None),
    "patch_ssim": ("patch_ssim.csv", "MEAN"),
    "lpips": ("lpips.csv", "MEAN"),
    "nuc_recall": ("nuclei.csv", "recall"),
    "nuc_f1": ("nuclei.csv", "f1"),
}

SWEEP_PANELS = [
    ("main", "Best field · merged ph_cyc"),
    ("cycsplit", "Best field · split ph_cyc"),
    ("worstfield_cycmerged", "Worst field · merged ph_cyc"),
    ("worstfield_cycsplit", "Worst field · split ph_cyc"),
]

# PH configuration -> (label, colour, marker). Colour follows the configuration
# in every panel and every figure; the marker repeats the identity for readers
# who cannot separate the hues, and for print.
CONFIGS = {
    (0, 1): ("cyc0 trans1", "#2a78d6", "o"),
    (1, 0): ("cyc1 trans0", "#eb6834", "s"),
    (1, 1): ("cyc1 trans1", "#1baf7a", "D"),
}

# Baselines: greys, told apart by dash and marker, never by a series hue.
REFS = {
    "cell0": ("Cell 0: vanilla CycleGAN (topo-train)", "#8a8984", (0, (1, 2)), None),
    "cyclegan": ("CycleGAN (zoo)", "#3d3c39", "-", "P"),
    "dclgan": ("DCLGAN (zoo)", "#8a8984", "--", "X"),
    "identity": ("Identity: H&E unchanged", "#b5b4ae", (0, (6, 2, 1, 2)), None),
}
SIZES = ["small", "medium", "large"]

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

CELL_RE = re.compile(r"_lt([0-9.eE+-]+)_cyc(\d)_trans(\d)$")
BASE_RE = re.compile(r"_(cyclegan|dclgan)_(small|medium|large)$")


def read_metric(model_dir: str, metric: str) -> Optional[float]:
    """The headline value of a metric for one model, or None if not there yet."""
    fname, key = SOURCES[metric]
    csv_path = os.path.join(model_dir, fname)
    if not os.path.isfile(csv_path):
        return None
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    try:
        if key is None:                         # fid: metric,value,n_real,n_fake
            return float(rows[1][1])
        for r in rows:                          # MEAN row, or a metric,value row
            if r and r[0] == key:
                return float(r[1])
    except (IndexError, ValueError):
        pass
    return None


def load(split_dir: str, metric: str):
    """-> (sweep, refs, ladder).

    sweep[arm][(cyc, trans)] = [(lambda_topo, value), ...]
    refs[key] = value              for cell0 / cyclegan / dclgan (small)
    ladder[model][size] = value
    """
    sweep: Dict[str, Dict[Tuple[int, int], List[Tuple[float, float]]]] = {}
    refs: Dict[str, float] = {}
    ladder: Dict[str, Dict[str, float]] = {"cyclegan": {}, "dclgan": {}}

    for arm in sorted(os.listdir(split_dir)) if os.path.isdir(split_dir) else []:
        arm_dir = os.path.join(split_dir, arm)
        if not os.path.isdir(arm_dir) or arm in ("plots", "nuclei_ref"):
            continue
        for name in sorted(os.listdir(arm_dir)):
            v = read_metric(os.path.join(arm_dir, name), metric)
            if v is None:
                continue
            if name.endswith("_identity"):
                refs["identity"] = v
                continue
            if name.endswith("_baseline_cyclegan"):
                if arm == "main":
                    refs["cell0"] = v
                continue
            m = CELL_RE.search(name)
            if m:
                key = (int(m.group(2)), int(m.group(3)))
                sweep.setdefault(arm, {}).setdefault(key, []).append((float(m.group(1)), v))
                continue
            m = BASE_RE.search(name)
            if m:
                model, size = m.group(1), m.group(2)
                ladder[model][size] = v
                if size == "small":
                    refs[model] = v
    for arm in sweep.values():
        for pts in arm.values():
            pts.sort()
    return sweep, refs, ladder


def robust_top(values: List[float]) -> float:
    """Upper y-limit that keeps a collapsed run from flattening the rest.

    Capped at four times the median: a damaged-but-trained run (a bad field's
    FID of 150-200 against a bulk near 50) is a result and stays on-scale; only
    a collapse far beyond it (FID 340-390) is cut. Higher-is-better metrics
    never reach the cap, so nothing of theirs is clipped.
    """
    vs = sorted(values)
    med = vs[len(vs) // 2]
    top = max(v for v in vs if v <= 4.0 * abs(med)) if med else vs[-1]
    bottom = vs[0]
    return top + 0.08 * max(top - bottom, abs(top) * 0.02)


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def clip_point(ax, x, y, top, color, marker):
    """Draw y, or a triangle on the top edge with the value if it is off-scale.

    The value goes inside the panel, just under the edge, and stacks downward
    when several series leave the scale at the same x -- above the edge it
    would run into the panel title.
    """
    if y <= top:
        ax.plot([x], [y], marker=marker, color=color, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
        return
    ax.plot([x], [top], marker="^", color=color, markersize=7, clip_on=False,
            markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=5)
    stack = ax.__dict__.setdefault("_offscale", {})
    k = stack.get(x, 0)
    stack[x] = k + 1
    ax.annotate("%.0f" % y, (x, top), xytext=(7, -3 - 10 * k),
                textcoords="offset points", ha="left", va="top", fontsize=7,
                color=INK_2, fontweight="bold", annotation_clip=False)


def plot_metric(marker: str, split: str, metric: str, split_dir: str,
                out_dir: str) -> Optional[str]:
    label, lower_better = METRICS[metric]
    sweep, refs, ladder = load(split_dir, metric)
    all_vals = ([v for arm in sweep.values() for pts in arm.values() for _, v in pts]
                + list(refs.values())
                + [v for m in ladder.values() for v in m.values()])
    if not all_vals:
        return None

    top = robust_top(all_vals)
    lo = min(all_vals)
    bottom = lo - 0.08 * (top - lo)

    fig, axes = plt.subplots(1, 5, figsize=(17, 4.2), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.75]})
    fig.patch.set_facecolor(SURFACE)

    lambdas = sorted({lt for arm in sweep.values() for pts in arm.values() for lt, _ in pts})
    for ax, (arm, title) in zip(axes[:4], SWEEP_PANELS):
        style_axes(ax)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=10)
        ax.set_xscale("log")
        if lambdas:
            ax.set_xticks(lambdas)
            ax.set_xticklabels(["%g" % l for l in lambdas])
            ax.set_xlim(lambdas[0] / 1.8, lambdas[-1] * 1.8)
        ax.minorticks_off()
        ax.set_xlabel("λ_topo", fontsize=8.5, color=INK_2)

        # Reference lines first, so the sweep draws over them.
        for key in ("cell0", "cyclegan", "dclgan", "identity"):
            if key in refs and refs[key] <= top:
                _, col, ls, _ = REFS[key]
                ax.axhline(refs[key], color=col, linestyle=ls, linewidth=1.2, zorder=2)

        arm_data = sweep.get(arm, {})
        if not arm_data:
            ax.text(0.5, 0.5, "not evaluated", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color=INK_2)
        for key, (_, col, mk) in CONFIGS.items():
            pts = arm_data.get(key)
            if not pts:
                continue
            xs = [x for x, _ in pts]
            ys = [min(y, top) for _, y in pts]
            ax.plot(xs, ys, color=col, linewidth=2, zorder=3)
            for x, y in pts:
                clip_point(ax, x, y, top, col, mk)

    # Capacity ladder.
    ax = axes[4]
    style_axes(ax)
    ax.set_title("Baselines by size", fontsize=9.5, color=INK, loc="left", pad=8)
    xs = list(range(len(SIZES)))
    for model in ("cyclegan", "dclgan"):
        _, col, ls, mk = REFS[model]
        pts = [(i, ladder[model][s]) for i, s in enumerate(SIZES) if s in ladder[model]]
        if not pts:
            continue
        ax.plot([i for i, _ in pts], [min(v, top) for _, v in pts],
                color=col, linestyle=ls, linewidth=1.6, zorder=3)
        for i, v in pts:
            clip_point(ax, i, v, top, col, mk)
    ax.set_xticks(xs)
    ax.set_xticklabels(SIZES)
    ax.set_xlim(-0.4, len(SIZES) - 0.6)
    ax.set_xlabel("generator width (ngf 64 / 136 / 192)", fontsize=8.5, color=INK_2)

    axes[0].set_ylim(bottom, top)
    axes[0].set_ylabel("%s  (%s is better)" % (label, "lower" if lower_better else "higher"),
                       fontsize=9, color=INK)

    handles = [Line2D([], [], color=c, marker=m, linewidth=2, markersize=6,
                      markeredgecolor=SURFACE, label=l)
               for l, c, m in CONFIGS.values()]
    handles += [Line2D([], [], color=c, linestyle=ls, linewidth=1.4,
                       marker=mk, markersize=6 if mk else 0, label=l)
                for key, (l, c, ls, mk) in REFS.items()
                if key != "identity" or "identity" in refs]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, 0.995),
               ncol=len(handles), frameon=False, fontsize=8.5, labelcolor=INK_2,
               handlelength=2.6, columnspacing=1.6)
    fig.suptitle("%s · %s · %s" % (marker, split, label.split(" (")[0]),
                 x=0.055, y=1.075, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.text(0.055, 0.005,
             "Grey lines in the sweep panels: the small baselines, the sweep's own cell 0 and, "
             "where run, the identity row (H&E unchanged). ▲ on the top edge: off-scale, "
             "value printed. One seed per model.",
             fontsize=7.5, color=INK_2)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.86, bottom=0.17, wspace=0.08)

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, metric)
    fig.savefig(stem + ".png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(stem + ".pdf", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return stem + ".png"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-root", required=True,
                   help="EVAL_ROOT of evaluate.sh, holding <marker>/<split>/<arm>/...")
    p.add_argument("--markers", nargs="+", required=True)
    p.add_argument("--split", default="valA")
    p.add_argument("--metrics", nargs="+", default=list(METRICS), choices=list(METRICS))
    args = p.parse_args()

    for marker in args.markers:
        split_dir = os.path.join(args.eval_root, marker, args.split)
        out_dir = os.path.join(split_dir, "plots")
        for metric in args.metrics:
            path = plot_metric(marker, args.split, metric, split_dir, out_dir)
            print("[%s] %-10s %s" % (marker, metric, path or "nothing evaluated yet"))


if __name__ == "__main__":
    main()
