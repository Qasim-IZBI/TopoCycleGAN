"""Layout consistency: keep the translation's tissue / lumen layout upright.

A third to a half of the deep-ensemble runs -- TopoCycleGAN, CycleGAN and
DCLGAN alike -- settle in their first few thousand steps on an INVERTED
translation: tissue rendered as lumen, lumen as tissue, nuclei as small lumina.
Every loss lets it through. An intensity inversion is invertible, so the cycle
reconstructs perfectly; each ~70 px patch still looks like IHC, so the PatchGAN
discriminator accepts it; and it is a whole-image flip, so no term acting at
nuclear scale -- PH included -- can turn a run back once it has chosen it.

The term here forbids exactly that and little else: one minus the Pearson
correlation between the source's and the translation's grey levels, after
pooling and blurring away everything finer than the tissue layout. Tissue is
darker than lumen in H&E and in IHC alike, so an upright translation correlates
near +1 and an inverted one near -1. Correlation ignores colour, brightness and
contrast, so it does not pull the IHC towards looking like H&E, and at this
scale it says nothing about nuclei -- the PH terms keep that job.

    L_layout = (1 - r(real_A, fake_B)) + (1 - r(real_B, fake_A))

Tiles with no layout to speak of (all tissue, all background) are left out of
the mean: their correlation is noise.

It is attached to a model instance rather than built into one, so the same code
serves TopoCycleGAN (topo-train) and the zoo's CycleGAN and DCLGAN
(topo-baseline) without touching their configs -- an existing run's checkpoint
config is unchanged, and so is its resume. The weight is recorded instead in
<output>/layout.json, which also stops a run from being resumed with a
different weight.
"""

from __future__ import annotations

import json
import math
import os

import torch
import torch.nn.functional as F

DOWNSAMPLE = 4      # 256 px tile -> 64 px layout map
BLUR = 1.0          # Gaussian sigma on the layout map, i.e. ~4 px at full size
MIN_STD = 0.06      # source contrast below this ([-1,1] units) -> tile skipped
RECORD = "layout.json"


def _layout_map(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] NCHW RGB -> N x (h*w) coarse grey levels."""
    g = x.mean(dim=1, keepdim=True)
    if DOWNSAMPLE > 1:
        g = F.avg_pool2d(g, DOWNSAMPLE)
    radius = max(1, int(math.ceil(3 * BLUR)))
    t = torch.arange(-radius, radius + 1, dtype=g.dtype, device=g.device)
    k = torch.exp(-0.5 * (t / BLUR) ** 2)
    k = k / k.sum()
    g = F.pad(g, (radius, radius, radius, radius), mode="reflect")
    g = F.conv2d(g, k.view(1, 1, 1, -1))
    g = F.conv2d(g, k.view(1, 1, -1, 1))
    return g.flatten(1)


def layout_corr(src: torch.Tensor, gen: torch.Tensor):
    """Per-image correlation of the two layouts, and which images have a layout."""
    a = _layout_map(src)
    b = _layout_map(gen)
    valid = a.std(dim=1) > MIN_STD
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    r = (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)
    return r, valid


def layout_loss(src: torch.Tensor, gen: torch.Tensor) -> torch.Tensor:
    """Mean of 1 - r over the images with a layout; 0 when none has one."""
    r, valid = layout_corr(src, gen.to(src.dtype))
    if not bool(valid.any()):
        return (gen * 0).sum()          # zero, but still part of the graph
    return (1 - r[valid]).mean()


def layout_terms(visuals) -> torch.Tensor:
    """Both translation directions, from a zoo model's generator visuals."""
    missing = [k for k in ("real_A", "fake_B", "real_B", "fake_A") if k not in visuals]
    if missing:
        raise KeyError("the model's generator visuals lack %s -- the layout term "
                       "needs both translations" % ", ".join(missing))
    return (layout_loss(visuals["real_A"], visuals["fake_B"])
            + layout_loss(visuals["real_B"], visuals["fake_A"]))


def check_record(output: str, weight: float) -> None:
    """Refuse to resume a run with another layout weight than it started with;
    record the weight of a new run that uses one."""
    path = os.path.join(output, RECORD)
    if os.path.isfile(path):
        with open(path) as f:
            saved = float(json.load(f)["lambda_layout"])
        if saved != weight:
            raise SystemExit("%s says this run trains with lambda_layout=%g, not %g -- "
                             "resume it with the same weight, or use a new --output"
                             % (path, saved, weight))
        return
    if weight == 0:
        return
    os.makedirs(output, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"lambda_layout": weight, "downsample": DOWNSAMPLE, "blur": BLUR,
                   "min_std": MIN_STD}, f, indent=2)


def attach(model, weight: float, output: str):
    """Add weight * L_layout to `model`'s generator loss, from step 0."""
    check_record(output, weight)
    if weight == 0:
        return model
    inner = model.compute_generator_loss

    def compute_generator_loss(batch):
        loss_G, logs, visuals = inner(batch)
        term = layout_terms(visuals)
        loss_G = loss_G + weight * term
        logs["loss_layout"] = float(term.detach().cpu())
        logs["loss_G"] = float(loss_G.detach().cpu())
        return loss_G, logs, visuals

    model.compute_generator_loss = compute_generator_loss
    print("[layout] lambda_layout=%g (downsample %d, blur %g, min_std %g)"
          % (weight, DOWNSAMPLE, BLUR, MIN_STD))
    return model
