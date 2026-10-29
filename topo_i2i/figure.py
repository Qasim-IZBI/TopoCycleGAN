"""The panels of the loss figure: one H&E / IHC pair through a trained model.

Given an H&E tile x, an IHC tile y and a trained TopoCycleGAN, this writes

    images      x, y and the two translations G(x) (H&E -> IHC) and F(y)
                (IHC -> H&E)
    fields      the stain field of each of the four images: the field the
                model's config selects for its domain (field_A for x and F(y),
                field_B for y and G(x)), with the stain vectors, combine and
                downsampling stored in the checkpoint -- the exact field the
                PH terms filter
    diagrams    the persistence diagram of each field, in the homology
                dimensions the model was trained with
    projections each diagram reduced to the 1-D axis the loss compares
                (birth / lifetime / death per dimension, as configured), and,
                for each ph_trans term, the two sorted projections it pairs:
                the area between them is the term's value

Every image is written on its own (PNG, plus .npy for the fields) and every
plot as its own vector file, so the figure can be assembled elsewhere;
overview.png puts them all on one page.

The fields are filtered negated (sublevel of -f, i.e. superlevel of the stain
concentration). The plots show them in concentration units: birth is the
concentration at which a feature appears, death the lower one at which it
merges or fills. Negating every value changes no distance -- sorting and
pairing commute with it, and the zero padding stays zero -- so the numbers
printed are the loss's own. --sublevel plots the filtered values as they are.

    topo-figure --he he.png --ihc ihc.png \\
        --ckpt-dir runs/results/Ki67_lt0.01_cyc0_trans1 --outdir fig2/ki67
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import torch
from PIL import Image

from topo_i2i.losses import _project, _resolve_projection, diagram_distance

# name -> (domain whose field it gets, label). The order is the figure's.
IMAGES = (
    ("he", "A", "H&E $x$"),
    ("he2ihc", "B", "$G(x)$"),
    ("ihc", "B", "IHC $y$"),
    ("ihc2he", "A", "$F(y)$"),
)
# The two ph_trans terms: source vs its own translation.
TERMS = (("trans_H", "he", "he2ihc"), ("trans_I", "ihc", "ihc2he"))
DIM_COLOURS = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c"}


def find_checkpoint(path: str, step=None) -> str:
    """A checkpoint file from a .pt path, a run directory or its checkpoints/.

    The newest step_<N>.pt unless `step` is given.
    """
    if os.path.isfile(path):
        return path
    d = os.path.join(path, "checkpoints") if os.path.isdir(os.path.join(path, "checkpoints")) \
        else path
    if not os.path.isdir(d):
        raise FileNotFoundError("no such checkpoint or directory: %s" % path)
    if step is not None:
        f = os.path.join(d, "step_%d.pt" % int(step))
        if not os.path.isfile(f):
            raise FileNotFoundError("no step_%d.pt in %s" % (int(step), d))
        return f
    steps = {}
    for f in glob.glob(os.path.join(d, "step_*.pt")):
        m = re.fullmatch(r"step_(\d+)\.pt", os.path.basename(f))
        if m:
            steps[int(m.group(1))] = f
    if steps:
        return steps[max(steps)]
    pts = sorted(glob.glob(os.path.join(d, "*.pt")), key=os.path.getmtime)
    if not pts:
        raise FileNotFoundError("no checkpoint (*.pt) in %s" % d)
    return pts[-1]


def load_tile(path: str, size: int) -> torch.Tensor:
    """(1, 3, size, size) in [-1, 1], exactly as topo-infer prepares a tile."""
    from torchvision import transforms
    t = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                            transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    return t(Image.open(path).convert("RGB"))[None]


def to_rgb(x: torch.Tensor) -> np.ndarray:
    return ((x[0].detach().cpu().float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()


def compute_panels(model, x: torch.Tensor, y: torch.Tensor) -> dict:
    """Everything the figure shows, from the model's own field and diagram code.

    Returns {"images", "fields", "diagrams", "distances", "dims", "projection"}:
    images and fields keyed by IMAGES' names, fields as the filtered values
    (negated when the config inverts), diagrams as {dim: (n, 2) array of
    [birth, death]} in those same filtered values.
    """
    cfg = model.topo_cfg
    dev = next(model.parameters()).device
    x, y = x.to(dev), y.to(dev)
    with torch.no_grad():
        images = {"he": x, "ihc": y,
                  "he2ihc": model.forward_A2B(x), "ihc2he": model.forward_B2A(y)}
        fields, diagrams = {}, {}
        for name, dom, _ in IMAGES:
            fields[name] = model._to_field(images[name], dom)[0].double().cpu().numpy()
            dgm = model._diagrams(images[name], dom, detach=True)[0]
            diagrams[name] = {k: v.double().cpu() for k, v in dgm.items()}
    dims = tuple(cfg.dims)
    proj = None if cfg.projection == "auto" else cfg.projection
    distances = {}
    for term, a, b in TERMS:
        per = {str(k): float(diagram_distance(diagrams[b], diagrams[a], (k,), proj))
               for k in dims}
        per["total"] = float(diagram_distance(diagrams[b], diagrams[a], dims, proj))
        distances[term] = per
    return {"images": {k: to_rgb(v) for k, v in images.items()},
            "fields": fields,
            "diagrams": {k: {d: v.numpy() for d, v in dg.items()} for k, dg in diagrams.items()},
            "distances": distances, "dims": dims, "projection": proj,
            "invert": bool(cfg.invert)}


def sorted_pair(dgm_a, dgm_b, dim, how):
    """The two projections as diagram_distance pairs them: zero-padded to a
    common length, then each sorted. Their |difference| summed is the term."""
    pa = _project({dim: torch.as_tensor(dgm_a[dim])}, dim, how).numpy()
    pb = _project({dim: torch.as_tensor(dgm_b[dim])}, dim, how).numpy()
    n = max(len(pa), len(pb))
    pa = np.sort(np.concatenate([pa, np.zeros(n - len(pa))]))
    pb = np.sort(np.concatenate([pb, np.zeros(n - len(pb))]))
    return pa, pb


def _display(values, how, flip):
    """Filtered values -> plotted ones: concentration units when `flip`.
    A lifetime is a difference and keeps its sign."""
    return -values if flip and how != "lifetime" else values


# ---------------------------------------------------------------- plotting

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def save_field_png(field, path, lo, hi, cmap):
    import matplotlib
    v = np.clip((field - lo) / max(hi - lo, 1e-12), 0, 1)
    rgba = matplotlib.colormaps[cmap](v)
    Image.fromarray((rgba[..., :3] * 255).round().astype(np.uint8)).save(path)


def draw_diagram(ax, dgm, dims, flip, lim):
    for k in dims:
        pts = dgm.get(k, np.zeros((0, 2)))
        if len(pts):
            p = -pts if flip else pts
            ax.scatter(p[:, 0], p[:, 1], s=5, alpha=0.6, lw=0, color=DIM_COLOURS.get(k),
                       label="$H_%d$ (%d)" % (k, len(pts)))
    ax.plot(lim, lim, "k-", lw=0.6)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("birth"); ax.set_ylabel("death")
    ax.legend(loc="upper left" if flip else "lower right", frameon=False, handletextpad=0.1)


def draw_projection(ax, dgm, dims, proj, flip, xlim):
    """The diagram reduced to one axis: each point dropped onto the value the
    loss compares, one row per homology dimension."""
    for row, k in enumerate(dims):
        how = _resolve_projection(proj, k)
        v = _display(_project({k: torch.as_tensor(dgm.get(k, np.zeros((0, 2))))}, k, how)
                     .numpy(), how, flip)
        ax.scatter(v, np.full_like(v, -row) + np.random.default_rng(k).uniform(-.18, .18, len(v)),
                   s=4, alpha=0.6, lw=0, color=DIM_COLOURS.get(k))
    ax.set_yticks([-r for r in range(len(dims))])
    ax.set_yticklabels(["$H_%d$\n%s" % (k, _resolve_projection(proj, k)) for k in dims])
    ax.set_ylim(-len(dims) + 0.5, 0.5)
    ax.set_xlim(xlim)
    ax.set_xlabel("projected value")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_pair(axes, pan, term, a, b, labels, flip):
    """One ph_trans term per dimension: the two sorted, padded projections."""
    for ax, k in zip(axes, pan["dims"]):
        how = _resolve_projection(pan["projection"], k)
        pa, pb = sorted_pair(pan["diagrams"][a], pan["diagrams"][b], k, how)
        pa, pb = _display(pa, how, flip), _display(pb, how, flip)
        # Display order only: flipping reverses both sorts, the pairing is unchanged.
        if flip and how != "lifetime":
            pa, pb = pa[::-1], pb[::-1]
        r = np.arange(len(pa))
        ax.fill_between(r, pa, pb, color="0.8", lw=0, step="mid")
        ax.step(r, pa, where="mid", lw=0.9, color="k", label=labels[a])
        ax.step(r, pb, where="mid", lw=0.9, color=DIM_COLOURS.get(k), label=labels[b])
        ax.set_title("$H_%d$ %s:  $d$ = %.3g" % (k, how, pan["distances"][term][str(k)]))
        ax.set_xlabel("rank"); ax.set_ylabel(how)
        ax.legend(frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)


def _span(values, pad=0.04):
    values = np.concatenate([np.ravel(v) for v in values if np.size(v)] or [np.zeros(1)])
    lo, hi = float(values.min()), float(values.max())
    m = (hi - lo) * pad or 1e-3
    return lo - m, hi + m


def write_figures(pan, outdir, fmt="pdf", flip=True, cmap="magma"):
    plt = _plt()
    labels = {n: lab for n, _, lab in IMAGES}
    dims, proj = pan["dims"], pan["projection"]
    flip = flip and pan["invert"]
    written = []

    def save(fig, name):
        path = os.path.join(outdir, "%s.%s" % (name, fmt))
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        written.append(path)

    for name, im in pan["images"].items():
        path = os.path.join(outdir, "image_%s.png" % name)
        Image.fromarray((im * 255).round().astype(np.uint8)).save(path)
        written.append(path)

    # Fields: concentration, one colour scale per domain so x vs F(y) and
    # y vs G(x) are comparable; the .npy holds the filtered values exactly.
    shown = {n: (-f if flip else f) for n, f in pan["fields"].items()}
    for dom in ("A", "B"):
        names = [n for n, d, _ in IMAGES if d == dom]
        both = np.concatenate([shown[n].ravel() for n in names])
        lo, hi = np.percentile(both, 0.5), np.percentile(both, 99.5)
        for n in names:
            save_field_png(shown[n], os.path.join(outdir, "field_%s.png" % n), lo, hi, cmap)
            np.save(os.path.join(outdir, "field_%s.npy" % n), pan["fields"][n])
            written.append(os.path.join(outdir, "field_%s.png" % n))

    # One axis range for all four diagrams, and one for all four projections.
    allpts = [(-p if flip else p) for dg in pan["diagrams"].values()
              for k, p in dg.items() if k in dims]
    lim = _span(allpts)
    projected = []
    for dg in pan["diagrams"].values():
        for k in dims:
            how = _resolve_projection(proj, k)
            projected.append(_display(_project({k: torch.as_tensor(dg.get(k, np.zeros((0, 2))))},
                                               k, how).numpy(), how, flip))
    xlim = _span(projected)

    for name, _, label in IMAGES:
        fig, ax = plt.subplots(figsize=(2.2, 2.2))
        draw_diagram(ax, pan["diagrams"][name], dims, flip, lim)
        ax.set_title(label)
        save(fig, "diagram_%s" % name)
        fig, ax = plt.subplots(figsize=(2.6, 0.45 + 0.45 * len(dims)))
        draw_projection(ax, pan["diagrams"][name], dims, proj, flip, xlim)
        ax.set_title(label)
        save(fig, "projection_%s" % name)
    for term, a, b in TERMS:
        fig, axes = plt.subplots(1, len(dims), figsize=(2.6 * len(dims), 2.0), squeeze=False)
        draw_pair(axes[0], pan, term, a, b, labels, flip)
        save(fig, "pair_%s" % term)

    # Everything on one page, for a look.
    fig, ax = plt.subplots(5, 4, figsize=(11, 12),
                           gridspec_kw={"height_ratios": [1, 1, 1, 0.55, 0.8]})
    for c, (name, _, label) in enumerate(IMAGES):
        ax[0][c].imshow(pan["images"][name]); ax[0][c].set_title(label)
        ax[1][c].imshow(shown[name], cmap=cmap); ax[1][c].set_title("field " + label)
        ax[0][c].axis("off"); ax[1][c].axis("off")
        draw_diagram(ax[2][c], pan["diagrams"][name], dims, flip, lim)
        draw_projection(ax[3][c], pan["diagrams"][name], dims, proj, flip, xlim)
    for c, (term, a, b) in enumerate(TERMS):
        k_axes = [ax[4][2 * c + j] for j in range(2)][:len(dims)]
        draw_pair(k_axes, pan, term, a, b, labels, flip)
        for j in range(len(dims), 2):
            ax[4][2 * c + j].axis("off")
    fig.tight_layout()
    path = os.path.join(outdir, "overview.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)
    return written


def loss_figure(he_path: str, ihc_path: str, ckpt_dir: str, outdir: str, step=None,
                image_size: int = 256, fmt: str = "pdf", superlevel: bool = True,
                cmap: str = "magma", device=None) -> dict:
    """Write the loss figure's panels for one H&E / IHC pair; return them.

    `ckpt_dir` is a run directory, its checkpoints/ directory or a .pt file;
    the newest step_<N>.pt is used unless `step` is given. Returns the dict of
    compute_panels plus "checkpoint" and "files".
    """
    from topo_i2i.inference import load_model

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = find_checkpoint(ckpt_dir, step)
    model = load_model(ckpt, device)
    pan = compute_panels(model, load_tile(he_path, image_size), load_tile(ihc_path, image_size))
    os.makedirs(outdir, exist_ok=True)
    files = write_figures(pan, outdir, fmt, superlevel, cmap)

    cfg = model.topo_cfg
    summary = {
        "checkpoint": os.path.abspath(ckpt), "he": he_path, "ihc": ihc_path,
        "field_A": cfg.field_A, "field_B": cfg.field_B, "combine": cfg.combine,
        "vectors_A": cfg.vectors_A, "vectors_B": cfg.vectors_B,
        "downsample": cfg.downsample, "invert": cfg.invert,
        "dims": list(pan["dims"]), "projection": cfg.projection,
        "projection_per_dim": {str(k): _resolve_projection(pan["projection"], k)
                               for k in pan["dims"]},
        "lambda_topo": cfg.lambda_topo, "lambda_ph_trans": cfg.lambda_ph_trans,
        "lambda_ph_cyc": cfg.lambda_ph_cyc,
        "points": {n: {str(k): int(len(v)) for k, v in dg.items()}
                   for n, dg in pan["diagrams"].items()},
        "distances": pan["distances"],
        "plotted_as": "superlevel (concentration)" if superlevel and cfg.invert
                      else "filtered values",
    }
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    files.append(os.path.join(outdir, "summary.json"))
    pan.update(checkpoint=ckpt, files=files)
    return pan


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--he", required=True, help="H&E tile (domain A)")
    p.add_argument("--ihc", required=True, help="IHC tile (domain B)")
    p.add_argument("--ckpt-dir", required=True,
                   help="run directory, its checkpoints/ directory, or a .pt file")
    p.add_argument("--step", type=int, default=None, help="default: the newest step_<N>.pt")
    p.add_argument("--outdir", required=True)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--format", default="pdf", choices=("pdf", "svg", "png"),
                   help="for the plots; images and fields are always PNG")
    p.add_argument("--cmap", default="magma", help="colour map for the fields")
    p.add_argument("--sublevel", action="store_true",
                   help="plot the filtered (negated) values instead of concentrations")
    p.add_argument("--device", default=None)
    return p


def main() -> None:
    a = build_parser().parse_args()
    pan = loss_figure(a.he, a.ihc, a.ckpt_dir, a.outdir, a.step, a.image_size, a.format,
                      not a.sublevel, a.cmap, a.device)
    print("checkpoint %s" % pan["checkpoint"])
    for term, d in pan["distances"].items():
        print("  %-8s %s" % (term, "  ".join("%s=%.4g" % kv for kv in d.items())))
    print("wrote %d files to %s" % (len(pan["files"]), os.path.abspath(a.outdir)))


if __name__ == "__main__":
    main()
