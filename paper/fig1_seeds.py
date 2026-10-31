"""One tile through every seed of a deep ensemble: the inversion figure.

Given an H&E (or IHC) tile and an ensemble directory, this translates the tile
with the first N seeds' models into the other domain and writes one image per
seed, each labelled with the layout correlation r between input and output --
the same Lambda the layout term uses (topo_i2i/layout.py): r near +1 upright,
near -1 inverted.

Run it on the original ensemble to show the inverted seeds, and on the
ensemble trained with the layout term to show them gone; several ensembles in
one call become one row each of a single grid:

    python paper/fig1_seeds.py --image he.tif --domain he --n-seeds 10 --outdir fig1/ki67 \\
        --ens "no layout term=$R/ensemble/Ki67/cyclegan" \\
        --ens "layout term=$R/ensemble_layout1_stop20k/Ki67/cyclegan"

An ensemble directory is <ENS_ROOT>/<marker>/<model> as train_ensemble.sh
writes it (its results/seed<k>/, seed 0 a link to the existing run), its
results/ directory, or any directory of seed<k>/ run directories. The model
(topo, cyclegan, dclgan, cyclediffusion) is read from the checkpoint config
where it can be (a TopoCycleGAN config carries its topo block), else from the
directory name, else --model.

Each seed uses its newest step_<N>.pt (or --step). A seed whose newest
checkpoint is short of the others' is still drawn but flagged in summary.json:
it is still training.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

# Run by path (python paper/<script>.py): the sibling scripts, and the repo's
# topo_i2i even where the package is not installed.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [p for p in (_HERE, os.path.dirname(_HERE)) if p not in sys.path]

import numpy as np
import torch
from PIL import Image

from fig2_loss import find_checkpoint, load_tile, to_rgb
from topo_i2i.layout import layout_corr

MODELS = ("topo", "cyclegan", "dclgan", "cyclediffusion")


def seed_dirs(ens_dir: str) -> dict:
    """{seed: run directory} under an ensemble directory."""
    for d in (os.path.join(ens_dir, "results"), ens_dir):
        if os.path.isdir(d):
            found = {}
            for name in os.listdir(d):
                m = re.fullmatch(r"seed(\d+)", name)
                if m and os.path.isdir(os.path.join(d, name)):
                    found[int(m.group(1))] = os.path.join(d, name)
            if found:
                return dict(sorted(found.items()))
    # Anything else (an archive, say): seed<k>/ directories holding checkpoints/,
    # up to three levels down -- but never two of the same seed.
    found = {}
    base = os.path.normpath(ens_dir)
    for d, subdirs, _ in os.walk(ens_dir, followlinks=True):
        if d[len(base):].count(os.sep) >= 3:
            subdirs[:] = []
        m = re.fullmatch(r"seed(\d+)", os.path.basename(d))
        if m and os.path.isdir(os.path.join(d, "checkpoints")):
            k = int(m.group(1))
            if k in found:
                raise ValueError("seed%d twice under %s: %s and %s -- point at one "
                                 "ensemble" % (k, ens_dir, found[k], d))
            found[k] = d
            subdirs[:] = []
    if found:
        return dict(sorted(found.items()))
    raise FileNotFoundError("no seed<k>/ run directories in %s" % ens_dir)


def model_kind(ckpt: dict, ens_dir: str, override=None) -> str:
    if override:
        return override
    cfg = ckpt.get("config") or {}
    if "topo" in cfg:
        return "topo"
    if "sample_steps" in cfg:
        return "cyclediffusion"
    for part in reversed(os.path.normpath(os.path.abspath(ens_dir)).split(os.sep)):
        for m in MODELS:
            if part == m or part.startswith(m + "_") or part.endswith("_" + m):
                return m
    raise ValueError("cannot tell the model of %s -- pass model=/--model (%s)"
                     % (ens_dir, ", ".join(MODELS)))


def load_any(path: str, ens_dir: str, device, model=None, cd_steps: int = 50):
    """Rebuild a topo / zoo model from its checkpoint. Returns (model, kind, step)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    kind = model_kind(ckpt, ens_dir, model)
    step = ckpt.get("global_step")
    if kind == "topo":
        from topo_i2i.inference import load_model
        return load_model(path, device), kind, step
    from i2i_stain_zoo.models import (CycleDiffusion, CycleDiffusionConfig, CycleGAN,
                                      CycleGANConfig, DCLGAN, DCLGANConfig)
    saved = dict(ckpt.get("config") or {})
    if kind == "cyclegan":
        net = CycleGAN(CycleGANConfig(**saved))
    elif kind == "dclgan":
        net = DCLGAN(DCLGANConfig(**saved))
    elif kind == "cyclediffusion":
        saved["sample_steps"] = cd_steps
        net = CycleDiffusion(CycleDiffusionConfig(**saved))
    else:
        raise ValueError("unknown model %r" % kind)
    net.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
    return net.to(device).eval(), kind, step


def seed_translations(image: str, ens_dir: str, n_seeds: int, outdir: str,
                      domain: str = "he", seeds=None, step=None, model=None,
                      image_size: int = 256, device=None, cd_steps: int = 50) -> dict:
    """Translate `image` with the first `n_seeds` seeds of the ensemble.

    `domain` is the input's: 'he' translates H&E -> IHC (A2B), 'ihc' IHC -> H&E
    (B2A). `seeds` picks seeds explicitly instead of the first n. Writes
    <outdir>/input.png, <outdir>/seed<k>.png and summary.json; returns
    {"input", "model", "seeds": [{seed, image (H,W,3 in [0,1]), r, inverted,
    checkpoint, step}]}.
    """
    if domain not in ("he", "ihc"):
        raise ValueError("domain must be 'he' or 'ihc', got %r" % domain)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    runs = seed_dirs(ens_dir)
    if seeds is None:
        seeds = list(runs)[:n_seeds]
        if len(seeds) < n_seeds:
            print("[warn] %s has %d seeds, not %d" % (ens_dir, len(seeds), n_seeds))
    missing = [s for s in seeds if s not in runs]
    if missing:
        raise FileNotFoundError("no seed%s in %s" % (missing, ens_dir))

    os.makedirs(outdir, exist_ok=True)
    x = load_tile(image, image_size).to(device)
    Image.fromarray((to_rgb(x) * 255).round().astype(np.uint8)).save(
        os.path.join(outdir, "input.png"))

    out, kind = [], None
    for s in seeds:
        try:
            ckpt = find_checkpoint(runs[s], step)
        except FileNotFoundError as e:
            print("[skip] seed%d: %s" % (s, e))
            continue
        net, kind, got = load_any(ckpt, ens_dir, device, model, cd_steps)
        with torch.no_grad():
            y = net.forward_A2B(x) if domain == "he" else net.forward_B2A(x)
            r, valid = layout_corr(x, y.to(x.dtype))
        r = float(r[0]) if bool(valid[0]) else float("nan")
        img = to_rgb(y)
        Image.fromarray((img * 255).round().astype(np.uint8)).save(
            os.path.join(outdir, "seed%d.png" % s))
        out.append({"seed": s, "image": img, "r": r,
                    "inverted": bool(r < 0) if r == r else None,
                    "checkpoint": os.path.abspath(ckpt), "step": got})
        print("  seed%-2d r %+.3f %s  (%s)" % (s, r, "INVERTED" if r < 0 else "upright",
                                             os.path.basename(ckpt)))
        del net

    steps = [o["step"] for o in out if isinstance(o["step"], int)]
    for o in out:
        o["still_training"] = bool(steps) and isinstance(o["step"], int) and o["step"] < max(steps)
    res = {"input": image, "domain": domain, "ens_dir": ens_dir, "model": kind, "seeds": out}
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        record = {k: v for k, v in res.items() if k != "seeds"}
        record["n_inverted"] = sum(bool(o["inverted"]) for o in out)
        record["seeds"] = [{k: v for k, v in o.items() if k != "image"} for o in out]
        json.dump(record, fh, indent=2)
    return res


def grid(rows, input_rgb, path, ncol=None, show_r=True):
    """One row per ensemble: the input, then its seeds; inverted ones framed red."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    ncol = ncol or max(len(r["seeds"]) for _, r in rows)
    fig, axes = plt.subplots(len(rows), ncol + 1, figsize=(1.25 * (ncol + 1), 1.65 * len(rows)),
                             squeeze=False)
    for i, (label, res) in enumerate(rows):
        ax = axes[i][0]
        ax.imshow(input_rgb)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(label, fontsize=8)
        if i == 0:
            ax.set_title("input", fontsize=8)
        for j in range(ncol):
            ax = axes[i][j + 1]
            ax.set_xticks([]); ax.set_yticks([])
            if j >= len(res["seeds"]):
                ax.axis("off")
                continue
            o = res["seeds"][j]
            ax.imshow(o["image"])
            bad = bool(o["inverted"])
            for sp in ax.spines.values():
                sp.set_edgecolor("#d62728" if bad else "0.6")
                sp.set_linewidth(2.0 if bad else 0.5)
            ax.set_title("seed %d" % o["seed"] + ("\n$r$ = %+.2f" % o["r"] if show_r else ""),
                         fontsize=7, color="#d62728" if bad else "k")
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, help="the input tile")
    p.add_argument("--domain", required=True, choices=("he", "ihc"),
                   help="the input's domain; it is translated into the other one")
    p.add_argument("--ens", required=True, action="append", metavar="[LABEL=]DIR",
                   help="an ensemble directory; repeat for one grid row each")
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="these seeds instead of the first --n-seeds")
    p.add_argument("--step", type=int, default=None, help="default: each seed's newest")
    p.add_argument("--model", choices=MODELS, default=None,
                   help="only if it cannot be read from the checkpoint or the path")
    p.add_argument("--outdir", required=True)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--cd-steps", type=int, default=50, help="DDIM steps, CycleDiffusion only")
    p.add_argument("--format", default="pdf", choices=("pdf", "svg", "png"))
    p.add_argument("--no-r", action="store_true", help="leave r out of the grid titles")
    p.add_argument("--device", default=None)
    return p


def main() -> None:
    a = build_parser().parse_args()
    rows = []
    for i, spec in enumerate(a.ens):
        label, _, path = spec.rpartition("=") if "=" in spec else ("", "", spec)
        if not label:      # <marker>/<model>, whether given the model dir or its results/
            parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
            parts = parts[:-1] if parts[-1] == "results" else parts
            label = "/".join(parts[-2:])
        sub = os.path.join(a.outdir, "row%d_%s" % (i, re.sub(r"[^A-Za-z0-9._-]+", "_", label)))
        print("%s  (%s)" % (label, path))
        rows.append((label, seed_translations(a.image, path, a.n_seeds, sub, a.domain, a.seeds,
                                              a.step, a.model, a.image_size, a.device,
                                              a.cd_steps)))
    path = grid(rows, to_rgb(load_tile(a.image, a.image_size)), os.path.join(a.outdir, "grid.%s" % a.format), show_r=not a.no_r)
    for label, res in rows:
        n_inv = sum(bool(o["inverted"]) for o in res["seeds"])
        print("%-30s %s  inverted %d/%d" % (label, res["model"], n_inv, len(res["seeds"])))
    print("wrote %s" % os.path.abspath(path))


if __name__ == "__main__":
    main()
