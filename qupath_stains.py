"""
QuPath-style automatic stain-vector estimation and colour deconvolution.

Reimplements what QuPath does when you press "Auto" in
  Analyze -> Estimate stain vectors...
and then applies the resulting stain matrix to split an RGB brightfield image
into its Hematoxylin, Eosin and Residual channels.

Algorithm (Macenko et al. 2009, as used by QuPath):
  1. optionally median-filter the (downsampled) RGB image to suppress noise
  2. convert to optical density:  OD = -log10(max(v, 1) / background), clipped at 0
  3. keep only "stain-like" pixels: min_od <= max(ODr, ODg, ODb) <= max_od
  4. eigen-decompose the (non-centred) 3x3 second-moment matrix of the OD cloud
  5. project the OD values onto the plane of the two leading eigenvectors and
     take the angular percentiles [alpha, 100 - alpha] as the two stain vectors
  6. the third ("Residual") vector is the normalised cross product of the two,
     exactly as QuPath's StainVector.makeResidualStainVector does
  7. deconvolve: concentrations = inv(M) . OD, with M's columns the stain vectors

Dependencies: numpy, Pillow.

Usage:
    python qupath_stains.py slide.png --outdir out
    python qupath_stains.py slide.tif --min-od 0.05 --max-od 1.0 --ignore 1.0
    python qupath_stains.py slide.png --outdir out --normalize
    python qupath_stains.py slide.png --outdir out --combine
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

# QuPath's built-in "H&E default" vectors (used as reference for naming/ordering)
DEFAULT_HEMATOXYLIN = np.array([0.65, 0.70, 0.29])
DEFAULT_EOSIN = np.array([0.2159, 0.8012, 0.5581])


# --------------------------------------------------------------------------- #
# stain container
# --------------------------------------------------------------------------- #
@dataclass
class Stains:
    """Three unit-length stain vectors in RGB optical-density space."""

    stain1: np.ndarray  # Hematoxylin
    stain2: np.ndarray  # Eosin
    stain3: np.ndarray  # Residual
    background: np.ndarray  # max/white values per channel
    names: tuple = ("Hematoxylin", "Eosin", "Residual")

    @property
    def matrix(self) -> np.ndarray:
        """3x3 matrix whose *columns* are the stain vectors."""
        return np.stack([self.stain1, self.stain2, self.stain3], axis=1)

    def __str__(self) -> str:
        lines = ["Background (max) values: R=%.1f G=%.1f B=%.1f" % tuple(self.background)]
        for name, v in zip(self.names, (self.stain1, self.stain2, self.stain3)):
            lines.append("%-12s R=%.4f  G=%.4f  B=%.4f" % (name, v[0], v[1], v[2]))
        return "\n".join(lines)

    def to_qupath_script(self) -> str:
        """Groovy snippet that sets these stains on the current image in QuPath."""
        def fmt(v):
            return " ".join("%.4f" % x for x in v)

        return (
            'setColorDeconvolutionStains(\'{"Name" : "Estimated stains", '
            '"Stain 1" : "%s", "Values 1" : "%s", '
            '"Stain 2" : "%s", "Values 2" : "%s", '
            '"Background" : "%s"}\')'
            % (
                self.names[0], fmt(self.stain1),
                self.names[1], fmt(self.stain2),
                " ".join("%.0f" % x for x in self.background),
            )
        )


def _normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def make_residual(stain1: np.ndarray, stain2: np.ndarray) -> np.ndarray:
    """Normalised cross product -- QuPath's residual (third) stain vector."""
    return _normalise(np.cross(stain1, stain2))


# --------------------------------------------------------------------------- #
# optical density
# --------------------------------------------------------------------------- #
def rgb_to_od(rgb: np.ndarray, background=(255.0, 255.0, 255.0)) -> np.ndarray:
    """QuPath's ColorDeconvolutionHelper.makeOD: max(0, -log10(max(v,1)/max))."""
    bg = np.asarray(background, dtype=np.float64).reshape(1, 1, 3)
    v = np.maximum(np.asarray(rgb, dtype=np.float64), 1.0)
    return np.maximum(-np.log10(v / bg), 0.0)


def estimate_background(rgb: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Per-channel white point from the brightest pixels (QuPath's 'Auto' background)."""
    flat = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    bg = np.percentile(flat, percentile, axis=0)
    return np.clip(bg, 1.0, 255.0)


# --------------------------------------------------------------------------- #
# stain estimation ("Auto")
# --------------------------------------------------------------------------- #
def estimate_stains(
    rgb: np.ndarray,
    background=(255.0, 255.0, 255.0),
    min_od: float = 0.05,
    max_od: float = 1.0,
    ignore_percentage: float = 1.0,
    od_stat: str = "max",
    median_filter: bool = True,
    max_estimation_dim: int = 1024,
    clip_negative: bool = True,
) -> Stains:
    """Estimate H, E and residual stain vectors from an RGB brightfield image.

    Parameters mirror QuPath's "Estimate stain vectors" dialog:
        min_od              "Min OD"                       (default 0.05)
        max_od              "Max OD"                       (default 1.0)
        ignore_percentage   "Ignore extrema (%)"           (default 1.0)
        median_filter       3x3 median pre-filter, as QuPath applies
        od_stat             which OD statistic the min/max thresholds test
                            ('max' per-channel, or 'mean'/'sum' over channels)
        max_estimation_dim  downsample so the longest side is at most this many
                            pixels before estimating (0 disables); the stain
                            matrix is scale-invariant, this only bounds cost
    """
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8)[..., :3], mode="RGB")

    if max_estimation_dim and max(img.size) > max_estimation_dim:
        scale = max_estimation_dim / max(img.size)
        new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        img = img.resize(new_size, Image.BILINEAR)

    if median_filter:
        img = img.filter(ImageFilter.MedianFilter(3))

    sample = np.asarray(img)
    od = rgb_to_od(sample, background).reshape(-1, 3)

    if od_stat == "max":
        strength = od.max(axis=1)
    elif od_stat == "mean":
        strength = od.mean(axis=1)
    elif od_stat == "sum":
        strength = od.sum(axis=1)
    else:
        raise ValueError("od_stat must be 'max', 'mean' or 'sum'")

    keep = (strength >= min_od) & (strength <= max_od)
    od = od[keep]
    if od.shape[0] < 100:
        raise ValueError(
            "Only %d pixels passed the OD thresholds -- loosen min_od/max_od "
            "or check the background values." % od.shape[0]
        )

    # Non-centred second-moment matrix (the OD cloud radiates from the origin).
    cov = (od.T @ od) / od.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    e1, e2 = eigvecs[:, order[0]], eigvecs[:, order[1]]

    # Fix the sign so the plane basis points into the positive OD octant.
    if e1.sum() < 0:
        e1 = -e1
    if e2.sum() < 0:
        e2 = -e2

    # Angles within the plane spanned by the two leading eigenvectors.
    phi = np.arctan2(od @ e2, od @ e1)
    alpha = ignore_percentage
    phi_min = np.percentile(phi, alpha)
    phi_max = np.percentile(phi, 100.0 - alpha)

    v1 = _normalise(e1 * np.cos(phi_min) + e2 * np.sin(phi_min))
    v2 = _normalise(e1 * np.cos(phi_max) + e2 * np.sin(phi_max))

    if clip_negative:
        v1 = _normalise(np.clip(v1, 0.0, None))
        v2 = _normalise(np.clip(v2, 0.0, None))

    # Hematoxylin absorbs red most strongly, so it has the larger red component.
    hematoxylin, eosin = (v1, v2) if v1[0] > v2[0] else (v2, v1)

    return Stains(
        stain1=hematoxylin,
        stain2=eosin,
        stain3=make_residual(hematoxylin, eosin),
        background=np.asarray(background, dtype=np.float64),
    )


# --------------------------------------------------------------------------- #
# deconvolution
# --------------------------------------------------------------------------- #
def deconvolve(rgb: np.ndarray, stains: Stains) -> np.ndarray:
    """Split an RGB image into stain concentrations. Returns float32 HxWx3."""
    od = rgb_to_od(rgb[..., :3], stains.background)
    inv = np.linalg.inv(stains.matrix)
    conc = np.einsum("ij,hwj->hwi", inv, od)
    return conc.astype(np.float32)


def concentration_to_uint8(conc: np.ndarray, background=255.0) -> np.ndarray:
    """Ruifrok/ImageJ-style 8-bit channel: dark where the stain is strong."""
    out = background * np.power(10.0, -np.maximum(conc, 0.0))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def combine_stains(conc: np.ndarray, method: str = "sum") -> np.ndarray:
    """Merge the two stain concentration maps (H+E, or H+DAB) into one channel.

    'sum'  total stain load in OD units (H + E) -- the physically meaningful
           combination, and the default
    'max'  strongest of the two stains at each pixel
    'mean' average of the two

    The residual channel is deliberately excluded, so the result is the part of
    the image explained by the two stains.
    """
    h, e = conc[..., 0], conc[..., 1]
    if method == "sum":
        return h + e
    if method == "max":
        return np.maximum(h, e)
    if method == "mean":
        return (h + e) / 2.0
    raise ValueError("method must be 'sum', 'max' or 'mean'")


def concentration_to_uint8_normalized(conc: np.ndarray, low_pct: float = 0.1,
                                      high_pct: float = 99.9):
    """Contrast-stretched 8-bit channel, for visual inspection only.

    Unlike concentration_to_uint8 this is *not* an absolute scale: the range
    [percentile(low_pct), percentile(high_pct)] of this particular channel is
    mapped onto the full 0-255 range, so values are not comparable between
    channels or between images. The polarity is kept the same (dark = strong
    stain). Returns (uint8 image, (lo, hi)) so the stretch can be reported.
    """
    lo = float(np.percentile(conc, low_pct))
    hi = float(np.percentile(conc, high_pct))
    if hi <= lo:
        # Flat channel: nothing to stretch, fall back to plain white.
        return np.full(conc.shape, 255, dtype=np.uint8), (lo, hi)
    scaled = np.clip((conc - lo) / (hi - lo), 0.0, 1.0)
    return np.clip(np.rint(255.0 * (1.0 - scaled)), 0, 255).astype(np.uint8), (lo, hi)


def concentration_to_pseudocolour(conc: np.ndarray, vector: np.ndarray,
                                  background=255.0) -> np.ndarray:
    """Reconstruct the RGB appearance of a single stain (QuPath's colour preview)."""
    od = np.maximum(conc, 0.0)[..., None] * vector.reshape(1, 1, 3)
    out = background * np.power(10.0, -od)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# paired-image intensity matching
# --------------------------------------------------------------------------- #
_SHORT_NAMES = {"hematoxylin": "H", "eosin": "E", "dab": "DAB", "residual": "residual"}


def short_name(name: str) -> str:
    return _SHORT_NAMES.get(name.lower(), name.replace(" ", "_"))


def shift_to_match_mean(a: np.ndarray, b: np.ndarray, apply: bool = True):
    """Shift whichever of two 8-bit images has the higher mean down by the
    difference of the means, then clamp to 0-255.

    With apply=False the images are returned untouched, but the shift that would
    have been applied is still reported -- use this to inspect the difference, or
    when a downstream step (e.g. a grayscale filtration) must not see an intensity
    offset.

    Returns (a_out, b_out, stats). Because of the clamp the means afterwards are
    close but not exactly equal, so the stats report what was actually achieved.
    """
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    mean_a, mean_b = af.mean(), bf.mean()
    delta = mean_a - mean_b

    if not apply:
        shifted = "none"
    elif delta >= 0:
        af = af - delta
        shifted = "A"
    else:
        bf = bf + delta
        shifted = "B"

    a_out = np.clip(np.rint(af), 0, 255).astype(np.uint8)
    b_out = np.clip(np.rint(bf), 0, 255).astype(np.uint8)
    stats = {
        "mean_a_before": mean_a,
        "mean_b_before": mean_b,
        "delta": abs(delta),
        "shifted": shifted,
        "mean_a_after": a_out.mean(),
        "mean_b_after": b_out.mean(),
    }
    return a_out, b_out, stats


def shifted_label(stats, stem_a: str, stem_b: str) -> str:
    """Name of the image that was shifted, or 'none'."""
    return {"A": stem_a, "B": stem_b}.get(stats["shifted"], "none")


def match_pair(conc_a: np.ndarray, conc_b: np.ndarray, norm_percentiles=(0.1, 99.9),
               match_channel: str = "all", apply_shift: bool = True):
    """Normalize both images' channels, then mean-match them.

    match_channel:
        'all'  each channel is matched against its counterpart independently
        0/1/2  the shift is derived from that channel alone and applied to every
               channel of whichever image has the higher mean there
    """
    norm_a, norm_b, ranges_a, ranges_b = [], [], [], []
    for i in range(3):
        na, ra = concentration_to_uint8_normalized(conc_a[..., i], *norm_percentiles)
        nb, rb = concentration_to_uint8_normalized(conc_b[..., i], *norm_percentiles)
        norm_a.append(na)
        norm_b.append(nb)
        ranges_a.append(ra)
        ranges_b.append(rb)

    out_a, out_b, all_stats = [], [], []
    if match_channel == "all":
        for i in range(3):
            a, b, st = shift_to_match_mean(norm_a[i], norm_b[i], apply_shift)
            out_a.append(a)
            out_b.append(b)
            all_stats.append(st)
    else:
        ref = int(match_channel)
        _, _, st = shift_to_match_mean(norm_a[ref], norm_b[ref], apply_shift)
        signed = st["mean_a_before"] - st["mean_b_before"]
        delta = signed if apply_shift else 0.0
        for i in range(3):
            af = norm_a[i].astype(np.float64) - max(delta, 0.0)
            bf = norm_b[i].astype(np.float64) - max(-delta, 0.0)
            a = np.clip(np.rint(af), 0, 255).astype(np.uint8)
            b = np.clip(np.rint(bf), 0, 255).astype(np.uint8)
            out_a.append(a)
            out_b.append(b)
            all_stats.append({
                "mean_a_before": norm_a[i].mean(), "mean_b_before": norm_b[i].mean(),
                "delta": abs(signed), "shifted": st["shifted"],
                "mean_a_after": a.mean(), "mean_b_after": b.mean(),
            })

    return out_a, out_b, all_stats, (ranges_a, ranges_b)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="RGB brightfield image (png/jpg/tif)")
    p.add_argument("--pair", metavar="IMAGE2", default=None,
                   help="second image (e.g. Ki67) to deconvolve and mean-match "
                        "against the first; enables paired mode")
    p.add_argument("--stain2-name", default="DAB",
                   help="name of the second stain in the --pair image (default DAB)")
    p.add_argument("--match-channel", default="combined",
                   choices=("combined", "all", "0", "1", "2"),
                   help="'combined' (default) matches the H channel of the first "
                        "image against the combined H+stain2 channel of the --pair "
                        "image, and writes only those two; 'all' matches every "
                        "channel to its counterpart; 0/1/2 derives one shift from "
                        "that channel (0=Hematoxylin) and applies it to all")
    p.add_argument("--no-mean-shift", action="store_true",
                   help="normalize and report the means, but do not apply the "
                        "mean shift (paired mode only)")
    p.add_argument("--outdir", default="stains_out")
    p.add_argument("--prefix-a", default=None, metavar="NAME",
                   help="filename prefix for the first image "
                        "(paired mode default: HE; single mode default: the filename)")
    p.add_argument("--prefix-b", default=None, metavar="NAME",
                   help="filename prefix for the --pair image (default: Ki)")
    p.add_argument("--min-od", type=float, default=0.05, help="QuPath 'Min OD'")
    p.add_argument("--max-od", type=float, default=1.0, help="QuPath 'Max OD'")
    p.add_argument("--ignore", type=float, default=1.0,
                   help="QuPath 'Ignore extrema (%%)'")
    p.add_argument("--od-stat", choices=("max", "mean", "sum"), default="max")
    p.add_argument("--no-median", action="store_true", help="skip the 3x3 median filter")
    p.add_argument("--auto-background", action="store_true",
                   help="estimate the white point instead of assuming 255")
    p.add_argument("--background", type=float, nargs=3, metavar=("R", "G", "B"),
                   default=None, help="fixed background/max values")
    p.add_argument("--save-float", action="store_true",
                   help="also write float32 concentration TIFFs")
    p.add_argument("--pseudocolour", action="store_true",
                   help="also write per-stain colour reconstructions")
    p.add_argument("--combine", choices=("sum", "max", "mean"), nargs="?",
                   const="sum", default=None,
                   help="also write H and E merged into one channel (_HE.png); "
                        "defaults to 'sum' when given without a value")
    p.add_argument("--normalize", action="store_true",
                   help="also write per-channel contrast-stretched PNGs (_norm.png); "
                        "for viewing only, not an absolute scale")
    p.add_argument("--norm-percentiles", type=float, nargs=2, metavar=("LOW", "HIGH"),
                   default=(0.1, 99.9), help="stretch limits for --normalize")
    return p


def resolve_background(rgb, args) -> np.ndarray:
    if args.background is not None:
        return np.asarray(args.background, dtype=np.float64)
    if args.auto_background:
        return estimate_background(rgb)
    return np.array([255.0, 255.0, 255.0])


def process_image(path: str, args, stain2_name: str = "Eosin"):
    """Load, estimate stains and deconvolve one image."""
    rgb = np.asarray(Image.open(path).convert("RGB"))
    stains = estimate_stains(
        rgb,
        background=resolve_background(rgb, args),
        min_od=args.min_od,
        max_od=args.max_od,
        ignore_percentage=args.ignore,
        od_stat=args.od_stat,
        median_filter=not args.no_median,
    )
    stains.names = ("Hematoxylin", stain2_name, "Residual")
    conc = deconvolve(rgb, stains)
    print("\n=== %s ===" % os.path.basename(path))
    print(stains)
    return rgb, stains, conc


def write_single(path, stains, conc, args) -> None:
    stem = args.prefix_a or os.path.splitext(os.path.basename(path))[0]
    shorts = [short_name(n) for n in stains.names]
    for i, short in enumerate(shorts):
        base = os.path.join(args.outdir, "%s_%s" % (stem, short))
        Image.fromarray(concentration_to_uint8(conc[..., i])).save(base + ".png")
        if args.normalize:
            norm, (lo, hi) = concentration_to_uint8_normalized(
                conc[..., i], *args.norm_percentiles)
            Image.fromarray(norm).save(base + "_norm.png")
            print("%-9s stretched OD [%.3f, %.3f] -> 0-255" % (short, lo, hi))
        if args.pseudocolour:
            vector = (stains.stain1, stains.stain2, stains.stain3)[i]
            Image.fromarray(
                concentration_to_pseudocolour(conc[..., i], vector)
            ).save(base + "_colour.png")
        if args.save_float:
            Image.fromarray(conc[..., i]).save(base + ".tif")

    if args.combine:
        combined = combine_stains(conc, args.combine)
        base = os.path.join(args.outdir, "%s_HE" % stem)
        Image.fromarray(concentration_to_uint8(combined)).save(base + ".png")
        if args.normalize:
            norm, (lo, hi) = concentration_to_uint8_normalized(
                combined, *args.norm_percentiles)
            Image.fromarray(norm).save(base + "_norm.png")
            print("%-9s stretched OD [%.3f, %.3f] -> 0-255" % ("H+E", lo, hi))
        if args.save_float:
            Image.fromarray(combined.astype(np.float32)).save(base + ".tif")
        print("Combined H and E using '%s'" % args.combine)


def run_match_combined(conc_a, conc_b, stem_a, stem_b, shorts_a, shorts_b, args) -> None:
    """Match the H channel of image A against the combined H+stain2 channel of B.

    Both are contrast-stretched to 0-255 first; whichever has the higher mean is
    then shifted down by the difference of the means and clamped.
    """
    method = args.combine or "sum"
    label_a = shorts_a[0]
    label_b = "%s+%s" % (shorts_b[0], shorts_b[1])

    norm_a, range_a = concentration_to_uint8_normalized(
        conc_a[..., 0], *args.norm_percentiles)
    combined_b = combine_stains(conc_b, method)
    norm_b, range_b = concentration_to_uint8_normalized(
        combined_b, *args.norm_percentiles)

    out_a, out_b, st = shift_to_match_mean(norm_a, norm_b, not args.no_mean_shift)

    print("\nMean matching: %s %s  vs  %s %s (combined with '%s')"
          % (stem_a, label_a, stem_b, label_b, method))
    print("  %s %-8s stretched OD [%.3f, %.3f] -> 0-255"
          % (stem_a, label_a, range_a[0], range_a[1]))
    print("  %s %-8s stretched OD [%.3f, %.3f] -> 0-255"
          % (stem_b, label_b, range_b[0], range_b[1]))

    header = "%-10s %-10s %8s %8s %7s %-8s %8s %8s" % (
        "A", "B", "meanA", "meanB", "shift", "shifted", "meanA'", "meanB'")
    print(header)
    print("-" * len(header))
    print("%-10s %-10s %8.2f %8.2f %7.2f %-8s %8.2f %8.2f" % (
        label_a, label_b,
        st["mean_a_before"], st["mean_b_before"], st["delta"],
        shifted_label(st, stem_a, stem_b),
        st["mean_a_after"], st["mean_b_after"]))

    suffix = "norm" if args.no_mean_shift else "matched"
    Image.fromarray(out_a).save(
        os.path.join(args.outdir, "%s_%s_%s.png" % (stem_a, label_a, suffix)))
    Image.fromarray(out_b).save(
        os.path.join(args.outdir, "%s_%s_%s.png" % (stem_b, label_b, suffix)))


def run_pair(args) -> None:
    """Deconvolve two images, normalize, then mean-match them channel by channel."""
    _, stains_a, conc_a = process_image(args.image, args, stain2_name="Eosin")
    _, stains_b, conc_b = process_image(args.pair, args, stain2_name=args.stain2_name)

    stem_a = args.prefix_a or "HE"
    stem_b = args.prefix_b or "Ki"
    shorts_a = [short_name(n) for n in stains_a.names]
    shorts_b = [short_name(n) for n in stains_b.names]

    if args.match_channel == "combined":
        run_match_combined(conc_a, conc_b, stem_a, stem_b, shorts_a, shorts_b, args)
        return

    out_a, out_b, stats, _ = match_pair(
        conc_a, conc_b, args.norm_percentiles, args.match_channel,
        not args.no_mean_shift)

    if args.match_channel == "all":
        print("\nMean matching: each channel against its counterpart")
    else:
        ref = int(args.match_channel)
        print("\nMean matching: shift taken from channel %d (%s / %s), "
              "applied to all channels" % (ref, shorts_a[ref], shorts_b[ref]))

    header = "%-14s %-14s %8s %8s %7s %-8s %8s %8s" % (
        "A channel", "B channel", "meanA", "meanB", "shift", "shifted", "meanA'", "meanB'")
    print(header)
    print("-" * len(header))
    for i, st in enumerate(stats):
        print("%-14s %-14s %8.2f %8.2f %7.2f %-8s %8.2f %8.2f" % (
            shorts_a[i], shorts_b[i],
            st["mean_a_before"], st["mean_b_before"], st["delta"],
            shifted_label(st, stem_a, stem_b),
            st["mean_a_after"], st["mean_b_after"]))

        suffix = "norm" if args.no_mean_shift else "matched"
        Image.fromarray(out_a[i]).save(
            os.path.join(args.outdir, "%s_%s_%s.png" % (stem_a, shorts_a[i], suffix)))
        Image.fromarray(out_b[i]).save(
            os.path.join(args.outdir, "%s_%s_%s.png" % (stem_b, shorts_b[i], suffix)))

    if args.combine:
        for stem, conc, shorts in ((stem_a, conc_a, shorts_a), (stem_b, conc_b, shorts_b)):
            combined = combine_stains(conc, args.combine)
            norm, _ = concentration_to_uint8_normalized(combined, *args.norm_percentiles)
            Image.fromarray(norm).save(
                os.path.join(args.outdir, "%s_%s+%s_norm.png"
                             % (stem, shorts[0], shorts[1])))


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.pair:
        run_pair(args)
    else:
        _, stains, conc = process_image(args.image, args)
        print("\nQuPath equivalent:\n" + stains.to_qupath_script())
        write_single(args.image, stains, conc, args)

    print("\nWrote channels to %s" % os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
