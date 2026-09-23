"""Build one contact sheet per tile: the input, the ground truth, every cell.

Inference writes each sweep cell's predictions into its own directory, which
makes any comparison between cells a matter of opening thirteen folders at the
same filename. This puts that tile's thirteen answers on one sheet, next to the
H&E it was given and the real IHC/SR where that exists, so the cells can be
read against each other and against the truth in one look.

Like topo-inspect this is a diagnostic: nothing downstream reads its output.

    topo-compare --input tiles/ER/TrainValAB/valA \\
        --truth tiles/ER/TrainValAB/valB \\
        --pred "baseline=preds/ER_baseline_cyclegan" \\
        --pred "lt0.02 cyc1=preds/ER_lt0.02_cyc1_trans0" \\
        --outdir compare/ER

The panels are laid out in the order the --pred flags are given, so the caller
controls the arrangement. compare.sh passes them in grid order, which at the
default three columns puts one lambda_topo decade on each row.

A cell that has not finished training has no predictions yet; its panel says so
rather than the sheet silently losing a column and misaligning every row after
it. The same goes for a tile with no registered partner in the truth domain.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

# Panel furniture, in pixels at the default panel size. Scaled with --panel-size
# so a bigger sheet does not end up with a caption too small to read.
CAPTION_H = 22
PAD = 8
TITLE_H = 30

INK = (32, 32, 32)
PAPER = (255, 255, 255)
MUTED = (120, 120, 120)
BLANK = (238, 238, 238)
# The input and the truth are not predictions; a rule under them says so
# without a legend.
ACCENT = (40, 90, 160)


def _font(px: int):
    """A real font if one can be found -- the bitmap default is tiny."""
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def list_tiles(root: str, subdir: str = ""):
    """Every image under root, as paths relative to it, sorted.

    Relative rather than absolute because that is the key a prediction is found
    by: topo-infer writes its output at the input's path relative to --data.
    """
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(IMG_EXTS):
                continue
            if subdir and os.path.basename(dirpath) != subdir:
                continue
            out.append(os.path.relpath(os.path.join(dirpath, fn), root))
    return sorted(out)


def index_by_stem(root: str):
    """{stem: path} for a directory, keyed both by relative path and basename.

    The two keys are the same thing in a flat TrainValAB layout. They come apart
    for a per-case tiling, where the truth may be filed under a different case
    directory than the input -- so try the full relative stem first, which is
    unambiguous, and fall back to the basename.
    """
    by_rel, by_base = {}, {}
    if not root or not os.path.isdir(root):
        return by_rel, by_base
    for rel in list_tiles(root):
        stem = os.path.splitext(rel)[0]
        by_rel[stem] = os.path.join(root, rel)
        by_base.setdefault(os.path.basename(stem), os.path.join(root, rel))
    return by_rel, by_base


def find_match(stem: str, by_rel, by_base):
    """The file for this tile in another directory, or None."""
    if stem in by_rel:
        return by_rel[stem]
    return by_base.get(os.path.basename(stem))


def load_panel(path: str, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return img


def blank_panel(size: int, note: str, font) -> Image.Image:
    """A labelled placeholder, so a missing cell keeps its slot in the grid."""
    img = Image.new("RGB", (size, size), BLANK)
    d = ImageDraw.Draw(img)
    box = d.textbbox((0, 0), note, font=font)
    d.text(((size - (box[2] - box[0])) / 2, (size - (box[3] - box[1])) / 2),
           note, fill=MUTED, font=font)
    return img


def compose(panels, cols: int, size: int, title: str) -> Image.Image:
    """Lay labelled panels out in a grid, `cols` wide, under one title.

    `panels` is a list of (label, PIL image or None, accent) -- a None image
    draws the placeholder, and accent marks the reference panels.
    """
    scale = size / 256.0
    cap_h = max(14, int(CAPTION_H * scale))
    pad = max(4, int(PAD * scale))
    title_h = max(18, int(TITLE_H * scale))
    cap_font = _font(max(9, int(13 * scale)))
    title_font = _font(max(11, int(16 * scale)))

    rows = (len(panels) + cols - 1) // cols
    W = cols * size + (cols + 1) * pad
    H = title_h + rows * (size + cap_h) + (rows + 1) * pad

    sheet = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, max(2, int(6 * scale))), title, fill=INK, font=title_font)

    for i, (label, img, accent) in enumerate(panels):
        r, c = divmod(i, cols)
        x = pad + c * (size + pad)
        y = title_h + pad + r * (size + cap_h + pad)
        sheet.paste(img if img is not None
                    else blank_panel(size, "not found", cap_font), (x, y))
        # A hairline under a reference panel; predictions get none. Cheaper to
        # read than a legend, and it survives being printed in greyscale.
        colour = ACCENT if accent else None
        if colour:
            draw.rectangle([x, y + size, x + size - 1, y + size + 2], fill=colour)
        draw.text((x + 2, y + size + max(3, int(4 * scale))), label,
                  fill=INK if accent else MUTED, font=cap_font)
    return sheet


def parse_pred(spec: str):
    """`label=dir`, split on the LAST `=`.

    The last rather than the first because the label is the part likely to
    contain one -- `lt=0.02 cyc1 trans0` is exactly how these cells want to be
    captioned -- while the directory is a path, where `=` is rare. A path that
    does contain one has to be passed with a label that does not.
    """
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            "--pred wants label=directory, got %r" % (spec,))
    label, path = spec.rsplit("=", 1)
    label, path = label.strip(), path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError(
            "--pred wants a non-empty label and directory, got %r" % (spec,))
    return label, path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
                   help="directory of input tiles -- the H&E the models were given")
    p.add_argument("--truth", default="",
                   help="directory of real target-domain tiles, if there are any")
    p.add_argument("--pred", action="append", default=[], metavar="LABEL=DIR",
                   help="a prediction directory and the caption for it; repeat "
                        "once per sweep cell, in the order they should appear")
    p.add_argument("--outdir", required=True)
    p.add_argument("--tiles", type=int, default=8,
                   help="how many tiles to sheet (0 = every one)")
    p.add_argument("--tile", action="append", default=[],
                   help="sheet this tile by name, instead of sampling; repeatable")
    p.add_argument("--sample", choices=("random", "head"), default="random",
                   help="random draws from the whole directory rather than the "
                        "first few slides, which sorted order would give")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subdir", default="",
                   help="only use inputs whose parent directory has this name, "
                        "e.g. 'images' for a <case>/images/<id>.tif layout")
    p.add_argument("--panel-size", type=int, default=256)
    p.add_argument("--cols", type=int, default=3,
                   help="panels per row; the default puts the input, the truth "
                        "and the baseline on the first row and one lambda_topo "
                        "decade on each row after it")
    p.add_argument("--ext", default=".png", help="sheet file extension")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not args.pred:
        raise SystemExit("nothing to compare: pass at least one --pred label=dir")
    preds = [parse_pred(s) for s in args.pred]

    tiles = list_tiles(args.input, args.subdir)
    if not tiles:
        raise SystemExit("no images under %s%s"
                         % (args.input,
                            " in a %r directory" % args.subdir if args.subdir else ""))

    if args.tile:
        # Names may be given with or without the extension, and with or without
        # the leading directories.
        wanted = {os.path.splitext(t)[0] for t in args.tile}
        chosen = [t for t in tiles
                  if os.path.splitext(t)[0] in wanted
                  or os.path.basename(os.path.splitext(t)[0]) in wanted]
        missing = wanted - {os.path.splitext(t)[0] for t in chosen} \
                         - {os.path.basename(os.path.splitext(t)[0]) for t in chosen}
        if missing:
            raise SystemExit("not under %s: %s"
                             % (args.input, ", ".join(sorted(missing))))
    else:
        chosen = list(tiles)
        if args.sample == "random":
            # Deterministic by seed, and drawn across the whole directory:
            # sorted order groups every crop of one slide together, so the head
            # of it is a couple of slides rather than a cross-section.
            order = np.random.default_rng(args.seed).permutation(len(chosen))
            chosen = [chosen[i] for i in order]
        if args.tiles:
            chosen = chosen[:args.tiles]

    truth_rel, truth_base = index_by_stem(args.truth)
    pred_index = [(label, index_by_stem(d)) for label, d in preds]

    os.makedirs(args.outdir, exist_ok=True)
    cap_font = _font(max(9, int(13 * args.panel_size / 256.0)))
    written, missing_counts = 0, {label: 0 for label, _ in preds}
    no_truth = 0

    for rel in chosen:
        stem = os.path.splitext(rel)[0]
        panels = [("input (H&E)", load_panel(os.path.join(args.input, rel),
                                             args.panel_size), True)]

        if args.truth:
            match = find_match(stem, truth_rel, truth_base)
            if match:
                panels.append(("ground truth",
                               load_panel(match, args.panel_size), True))
            else:
                no_truth += 1
                panels.append(("ground truth: none for this tile",
                               blank_panel(args.panel_size, "unpaired", cap_font),
                               True))
        else:
            panels.append(("no ground truth for this set",
                           blank_panel(args.panel_size, "none", cap_font), True))

        for label, (by_rel, by_base) in pred_index:
            match = find_match(stem, by_rel, by_base)
            if match:
                panels.append((label, load_panel(match, args.panel_size), False))
            else:
                missing_counts[label] += 1
                panels.append((label + " -- not inferred",
                               blank_panel(args.panel_size, "no prediction",
                                           cap_font), False))

        sheet = compose(panels, args.cols, args.panel_size, stem)
        # Flatten the relative path: a sheet is for flipping through, and a
        # nested tree of one file per directory is not.
        out = os.path.join(args.outdir,
                           stem.replace(os.sep, "_").replace("/", "_") + args.ext)
        sheet.save(out)
        written += 1

    print("[compare] wrote %d sheets to %s" % (written, os.path.abspath(args.outdir)))
    print("[compare] %d panels per sheet: input, ground truth, %d cells"
          % (2 + len(preds), len(preds)))
    if no_truth:
        print("[compare] %d of %d tiles had no partner in %s"
              % (no_truth, written, args.truth))
    absent = {k: v for k, v in missing_counts.items() if v}
    if absent:
        print("[compare] cells with tiles not yet inferred:")
        for label in sorted(absent):
            print("    %-40s %d of %d missing" % (label, absent[label], written))


if __name__ == "__main__":
    main()
