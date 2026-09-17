"""Walk one image pair through the whole loss, writing every intermediate out.

Given an image from each domain, this produces the deconvolved stain channels,
the scalar field each one is reduced to, the persistence diagram of each field,
and the diagram distance between them -- the exact quantity ph_trans minimises.

For a merged spec like `stain1+stain2` the combined image is written too, since
that is what the diagram is actually built from. `<tag>_field.npy` holds the
exact values; the matching PNG is contrast stretched for viewing, which is
monotone and so leaves the topology unchanged.

It is a diagnostic, not part of training: use it to see what the loss is actually
looking at on a tile you recognise, and to produce figures.

    topo-inspect --imageA he.png --imageB ihc.png \\
        --stains stains_ER.json --field-A stain1/stain2 --field-B stain1+stain2 \\
        --outdir inspect/ER_tile00

With no --stains the vectors are estimated from the two images themselves, which
is noisy from a single tile -- fine for a quick look, not for anything reported.
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import torch
from PIL import Image

from topo_i2i.fields import STAIN_VECTORS, make_field, split_specs
from topo_i2i.losses import _project, _resolve_projection, diagram_distance
from topo_i2i.persistence import persistence_diagram
from topo_i2i.stains import estimate_stains, load_vectors, order_like, rgb_to_od


def load_image(path: str, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1


def concentration_png(conc: np.ndarray, path: str) -> None:
    """Ruifrok/ImageJ convention: dark where the stain is strong."""
    img = 255.0 * np.power(10.0, -np.maximum(conc, 0.0))
    Image.fromarray(np.clip(np.rint(img), 0, 255).astype(np.uint8)).save(path)


def field_png(field: np.ndarray, path: str) -> None:
    """Contrast-stretched view of the filtered field, for looking at only."""
    lo, hi = np.percentile(field, 0.5), np.percentile(field, 99.5)
    if hi <= lo:
        hi = lo + 1e-6
    img = 255.0 * np.clip((field - lo) / (hi - lo), 0, 1)
    Image.fromarray(np.rint(img).astype(np.uint8)).save(path)


def estimate_from_images(img_a, img_b, size):
    """Fallback when no --stains is given: estimate from these two tiles alone."""
    out = {}
    for tag, path in (("A", img_a), ("B", img_b)):
        arr = np.asarray(Image.open(path).convert("RGB").resize((size, size)))
        v1, v2 = order_like(estimate_stains(rgb_to_od(arr).reshape(-1, 3)))
        out[tag] = {"stain1": tuple(v1), "stain2": tuple(v2)}
    return out


def specs_of(args, tag: str) -> str:
    return args.field_A if tag == "A" else args.field_B


def write_diagram_csv(dgm, path) -> None:
    with open(path, "w") as fh:
        fh.write("dim,birth,death,lifetime\n")
        for dim in sorted(dgm):
            for b, d in dgm[dim].tolist():
                fh.write("%d,%.6f,%.6f,%.6f\n" % (dim, b, d, d - b))


def make_figure(panels, dgms, dims, proj, outdir):
    """Optional: one panel figure. Skipped if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[note] matplotlib not available -- skipping the figure")
        return None

    ncol = max(len(panels["A"]), len(panels["B"]))
    fig, axes = plt.subplots(3, ncol, figsize=(3.7 * ncol, 11), squeeze=False)
    for row, tag in enumerate(("A", "B")):
        for col, (title, arr) in enumerate(panels[tag]):
            ax = axes[row][col]
            ax.imshow(arr, cmap=None if arr.ndim == 3 else "gray")
            ax.set_title("%s: %s" % (tag, title), fontsize=9)
            ax.axis("off")
        for col in range(len(panels[tag]), ncol):
            axes[row][col].axis("off")

    ax = axes[2][0]
    for tag, marker in (("A", "o"), ("B", "x")):
        for dim in dims:
            pts = dgms[tag][dim]
            if pts.shape[0]:
                ax.scatter(pts[:, 0], pts[:, 1], s=6, marker=marker, alpha=0.5,
                           label="%s H%d" % (tag, dim))
    lims = ax.get_xlim()
    ax.plot(lims, lims, "k--", lw=0.5)
    ax.set_xlabel("birth"); ax.set_ylabel("death")
    ax.set_title("persistence diagrams", fontsize=9)
    ax.legend(fontsize=6)

    # What the distance actually compares: the sorted projections, zero-padded.
    ax = axes[2][1]
    for dim in dims:
        how = _resolve_projection(proj, dim)
        pa = np.sort(_project(dgms["A"], dim, how).numpy())
        pb = np.sort(_project(dgms["B"], dim, how).numpy())
        n = max(len(pa), len(pb))
        pa = np.pad(pa, (n - len(pa), 0)); pb = np.pad(pb, (n - len(pb), 0))
        ax.plot(pa, label="A H%d (%s)" % (dim, how), lw=1)
        ax.plot(pb, label="B H%d (%s)" % (dim, how), lw=1, ls="--")
    ax.set_xlabel("rank"); ax.set_ylabel("projected value")
    ax.set_title("sorted projections -- the distance is the area between", fontsize=9)
    ax.legend(fontsize=6)
    for col in range(2, ncol):
        axes[2][col].axis("off")

    fig.tight_layout()
    path = os.path.join(outdir, "overview.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--imageA", required=True, help="domain A image (H&E)")
    p.add_argument("--imageB", required=True, help="domain B image (IHC)")
    p.add_argument("--stains", default=None,
                   help="stain vectors from topo-estimate-stains; estimated from "
                        "the two images if omitted")
    p.add_argument("--literature", action="store_true",
                   help="use the built-in fields.STAIN_VECTORS table instead of "
                        "estimating or loading anything. This is what training "
                        "does when the audit's fixed-vector arm wins, so it is "
                        "how you inspect that setting faithfully.")
    p.add_argument("--field-A", default="stain1/stain2")
    p.add_argument("--field-B", default="stain1+stain2")
    p.add_argument("--field-combine", default="sum", choices=("max", "sum", "mean"))
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--topo-dims", type=int, nargs="+", default=[0, 1])
    p.add_argument("--topo-projection", default="birth",
                   choices=("auto", "birth", "lifetime", "death"))
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--no-invert", action="store_true")
    p.add_argument("--outdir", required=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    dims = tuple(args.topo_dims)
    proj = None if args.topo_projection == "auto" else args.topo_projection
    invert = not args.no_invert
    os.makedirs(args.outdir, exist_ok=True)

    if args.literature and args.stains:
        raise SystemExit("--literature and --stains are mutually exclusive")
    if args.literature:
        # None means make_field falls through to fields.STAIN_VECTORS, exactly
        # as training does with no --stains.
        unknown = sorted({n for tag in ("A", "B")
                          for n in re.split(r"[/+]", specs_of(args, tag))
                          if n and n != "gray" and n not in STAIN_VECTORS})
        if unknown:
            raise SystemExit(
                "--literature needs specs named after real stains, but %s "
                "%s no entry in the literature table.\nNames like stain1/stain2 "
                "are positional and only mean something with estimated vectors, "
                "so pass --stains for those.\nKnown stains: %s"
                % (", ".join(unknown), "have" if len(unknown) > 1 else "has",
                   ", ".join(sorted(STAIN_VECTORS))))
        vectors = {"A": None, "B": None}
        src = "the built-in literature table"
    elif args.stains:
        vectors, _ = load_vectors(args.stains)
        src = args.stains
    else:
        vectors = estimate_from_images(args.imageA, args.imageB, args.image_size)
        src = "estimated from these two images (noisy -- one tile per domain)"
    print("stain vectors: %s" % src)
    for tag in ("A", "B"):
        if vectors[tag] is None:
            used = [n for n in re.split(r"[/+]", specs_of(args, tag)) if n]
            print("  %s %s" % (tag, "  ".join(
                "%s %s" % (n, np.round(STAIN_VECTORS[n], 4))
                for n in used if n in STAIN_VECTORS)))
        else:
            print("  %s stain1 %s  stain2 %s"
                  % (tag, np.round(vectors[tag]["stain1"], 4),
                     np.round(vectors[tag]["stain2"], 4)))

    images = {"A": args.imageA, "B": args.imageB}
    specs = {"A": args.field_A, "B": args.field_B}
    panels, dgms, summary = {}, {}, {}

    for tag in ("A", "B"):
        x = load_image(images[tag], args.image_size)
        rgb = ((x[0].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        panels[tag] = [("input", rgb)]

        # Both deconvolved channels, whatever the field spec uses.
        try:
            first, second = split_specs(specs[tag])
        except ValueError:
            first = second = None
        if first:
            for name, spec in (("stain1", first), ("stain2", second)):
                c = make_field(spec, vectors=vectors[tag])(x)[0].numpy()
                concentration_png(c, os.path.join(args.outdir, "%s_%s.png" % (tag, name)))
                panels[tag].append(("%s concentration" % name, c))

        # When the spec merges two stains, the merged concentration is what the
        # diagram is built from -- save it in the same dark-is-strong convention
        # as the individual channels, so the three are directly comparable.
        if "+" in specs[tag]:
            merged = make_field(specs[tag], args.field_combine,
                                vectors=vectors[tag])(x)[0].numpy()
            concentration_png(merged, os.path.join(args.outdir, "%s_combined.png" % tag))
            panels[tag].append(("%s (%s)" % (specs[tag], args.field_combine), merged))

        # The field the loss actually filters, after combine / downsample / invert.
        xd = torch.nn.functional.avg_pool2d(x, args.downsample) if args.downsample > 1 else x
        f = make_field(specs[tag], args.field_combine, vectors=vectors[tag])(xd)[0].double()
        f = -f if invert else f
        arr = f.numpy()
        field_png(arr, os.path.join(args.outdir, "%s_field.png" % tag))
        # The exact values the diagram came from -- the PNG above is contrast
        # stretched for viewing, which is monotone and so leaves the topology
        # unchanged, but is not the numbers.
        np.save(os.path.join(args.outdir, "%s_field.npy" % tag), arr)
        panels[tag].append(("filtered field%s" % (" (negated)" if invert else ""), arr))

        d = persistence_diagram(f, dims)
        dgms[tag] = d
        write_diagram_csv(d, os.path.join(args.outdir, "%s_diagram.csv" % tag))
        summary[tag] = {"image": images[tag], "field": specs[tag],
                        "field_shape": list(arr.shape),
                        "points": {str(k): int(v.shape[0]) for k, v in d.items()}}

    print("\nfield %s -> diagram points" % ("negated" if invert else "as-is"))
    for tag in ("A", "B"):
        print("  %s %s  %s" % (tag, summary[tag]["field_shape"], summary[tag]["points"]))

    print("\ndiagram distance (projection=%s)" % args.topo_projection)
    per_dim = {}
    for dim in dims:
        d = float(diagram_distance(dgms["A"], dgms["B"], (dim,), proj))
        per_dim[str(dim)] = d
        print("  H%d  %10.4f   (%s)" % (dim, d, _resolve_projection(proj, dim)))
    total = float(diagram_distance(dgms["A"], dgms["B"], dims, proj))
    print("  %-4s %10.4f   <- the value ph_trans minimises" % ("all", total))

    summary.update({"distance_total": total, "distance_per_dim": per_dim,
                    "dims": list(dims), "projection": args.topo_projection,
                    "downsample": args.downsample, "combine": args.field_combine,
                    "invert": invert,
                    "vector_source": src,
                    "vectors": {t: (None if vectors[t] is None
                                    else {k: list(map(float, v))
                                          for k, v in vectors[t].items()})
                                for t in ("A", "B")}})
    with open(os.path.join(args.outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    fig = make_figure(panels, dgms, dims, proj, args.outdir)
    print("\nwrote %s" % os.path.abspath(args.outdir))
    for f in sorted(os.listdir(args.outdir)):
        print("  %s" % f)


if __name__ == "__main__":
    main()
