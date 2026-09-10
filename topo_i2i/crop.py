"""Cut pre-tiled images into a grid of smaller tiles.

For datasets that already ship tiles (trainA/trainB/valA/valB of, say,
1024x1024) rather than whole slides, where the zoo's WSI `tile.py` does not
apply. Mirrors the input directory tree, so the output drops straight into
`topo-train --dataA .../trainA --dataB .../trainB`.

Flag names follow the zoo's tile.py (`--tile_size`, `--resize_to`,
`--tissue_threshold`, `--num_workers`) so the two are interchangeable in a
pipeline.

    topo-crop --input data/raw --output data/tiles --tile_size 512 --resize_to 256

A note on scale: `--resize_to` changes the microns per pixel of the result.
Cutting 512 and resizing to 256 halves the resolution relative to cutting 256
directly. Match whatever your other tiles were produced at -- for persistent
homology in particular, resolution decides whether adjacent nuclei stay separate
connected components.
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from PIL import Image

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
DEFAULT_SUBDIRS = ("trainA", "trainB", "valA", "valB")


def list_images(directory: str):
    return sorted(
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.lower().endswith(IMAGE_EXTS)
    )


def tissue_fraction(tile: Image.Image, white_level: int = 220) -> float:
    """Fraction of pixels darker than `white_level` in grayscale -- a crude but
    dependable stand-in for tissue on brightfield, where background is white."""
    gray = np.asarray(tile.convert("L"))
    return float((gray < white_level).mean())


def crop_one(path: str, out_dir: str, tile_size: int, resize_to=None,
             overlap: int = 0, tissue_threshold: float = 0.0,
             white_level: int = 220, ext: str = ".png"):
    """Cut one image into a grid. Returns (written, skipped)."""
    img = Image.open(path).convert("RGB")
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("--overlap must be smaller than --tile_size")

    stem = os.path.splitext(os.path.basename(path))[0]
    written = skipped = 0

    # Any remainder on the right/bottom edge is dropped rather than padded or
    # unevenly overlapped, so every emitted tile covers real tissue at the
    # same scale.
    for row, top in enumerate(range(0, img.height - tile_size + 1, stride)):
        for col, left in enumerate(range(0, img.width - tile_size + 1, stride)):
            tile = img.crop((left, top, left + tile_size, top + tile_size))
            if tissue_threshold > 0 and tissue_fraction(tile, white_level) < tissue_threshold:
                skipped += 1
                continue
            if resize_to and resize_to != tile_size:
                tile = tile.resize((resize_to, resize_to), Image.LANCZOS)
            tile.save(os.path.join(out_dir, "%s_r%dc%d%s" % (stem, row, col, ext)))
            written += 1
    return written, skipped


def _worker(path, **kwargs):
    try:
        return crop_one(path, **kwargs)
    except Exception as exc:  # a corrupt tile should not kill the whole run
        print("[WARN] %s: %s" % (path, exc))
        return (0, 0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="root holding the subfolders")
    p.add_argument("--output", required=True, help="mirrored output root")
    p.add_argument("--subdirs", nargs="+", default=list(DEFAULT_SUBDIRS),
                   help="subfolders to process (default: %s)" % " ".join(DEFAULT_SUBDIRS))
    p.add_argument("--tile_size", type=int, default=512,
                   help="crop size taken from the source image")
    p.add_argument("--resize_to", type=int, default=None,
                   help="resample each crop to this size (default: keep tile_size)")
    p.add_argument("--overlap", type=int, default=0,
                   help="overlap between neighbouring crops, in source pixels")
    p.add_argument("--tissue_threshold", type=float, default=0.0,
                   help="drop crops whose tissue fraction is below this "
                        "(0 = keep everything, the default: nothing is discarded "
                        "unless you ask)")
    p.add_argument("--white_level", type=int, default=220,
                   help="grayscale value above which a pixel counts as background")
    p.add_argument("--ext", default=".png", help="output file extension")
    p.add_argument("--num_workers", type=int, default=os.cpu_count())
    p.add_argument("--dry_run", action="store_true",
                   help="report what would be written without writing it")
    return p


def main() -> None:
    args = build_parser().parse_args()
    resize_to = args.resize_to or args.tile_size

    if args.resize_to and args.resize_to != args.tile_size:
        ratio = args.tile_size / args.resize_to
        print("[scale] %d -> %d: each output pixel covers %.2gx the tissue of a "
              "%d-px crop taken directly" % (args.tile_size, args.resize_to, ratio,
                                             args.resize_to))

    total_w = total_s = 0
    for sub in args.subdirs:
        src = os.path.join(args.input, sub)
        if not os.path.isdir(src):
            print("[skip] %s does not exist" % src)
            continue
        paths = list_images(src)
        if not paths:
            print("[skip] %s has no images" % src)
            continue

        with Image.open(paths[0]) as probe:
            w, h = probe.size
        per = max(0, (h - args.tile_size) // (args.tile_size - args.overlap) + 1) * \
              max(0, (w - args.tile_size) // (args.tile_size - args.overlap) + 1)
        rem = (w % args.tile_size, h % args.tile_size) if args.overlap == 0 else None
        note = "" if not rem or rem == (0, 0) else "  (dropping %dx%d edge remainder)" % rem
        # Only the first image is probed, so this is an estimate when the source
        # tiles differ in size; each image is still gridded to its own extent.
        print("[%s] %d images, first is %dx%d -> up to %d crops each%s"
              % (sub, len(paths), w, h, per, note))

        if args.dry_run:
            total_w += len(paths) * per
            continue

        dst = os.path.join(args.output, sub)
        os.makedirs(dst, exist_ok=True)
        fn = partial(_worker, out_dir=dst, tile_size=args.tile_size,
                     resize_to=resize_to, overlap=args.overlap,
                     tissue_threshold=args.tissue_threshold,
                     white_level=args.white_level, ext=args.ext)
        if args.num_workers and args.num_workers > 1:
            with Pool(args.num_workers) as pool:
                results = pool.map(fn, paths)
        else:
            results = [fn(p) for p in paths]

        w_sum = sum(r[0] for r in results)
        s_sum = sum(r[1] for r in results)
        total_w += w_sum
        total_s += s_sum
        print("[%s] wrote %d tiles%s" % (sub, w_sum,
              ", skipped %d below the tissue threshold" % s_sum if s_sum else ""))

    verb = "would write" if args.dry_run else "wrote"
    print("\n%s %d tiles%s" % (verb, total_w,
          ", skipped %d" % total_s if total_s else ""))


if __name__ == "__main__":
    main()
