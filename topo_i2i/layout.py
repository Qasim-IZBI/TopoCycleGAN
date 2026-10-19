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

Its only job is to pick the upright basin, which every run settles into within
its first few thousand steps; left on for a whole run (weight 1, 400k steps) it
also ties the IHC's texture to the H&E's and costs nuclei precision and SSIM.
So it can be switched off after `stop` generator steps. From then on it is
still computed and logged as loss_layout, but no longer trained on -- the log
shows whether a run stays upright on its own.

It is attached to a model instance rather than built into one, so the same code
serves TopoCycleGAN (topo-train) and the zoo's CycleGAN and DCLGAN
(topo-baseline) without touching their configs -- an existing run's checkpoint
config is unchanged, and so is its resume. The weight is recorded instead in
<output>/layout.json, which also stops a run from being resumed with a
different weight or stop step. The record also carries how many steps the term
has run (written every PROGRESS_EVERY steps until the stop), so a resumed run
switches it off on schedule rather than counting from 0 again.
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
PROGRESS_EVERY = 1000   # generator steps between progress writes to the record


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


def _write_record(path: str, record: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, path)


def check_record(output: str, weight: float, stop: int = 0) -> int:
    """Refuse to resume a run with another layout weight or stop step than it
    started with; record the settings of a new run that uses the term. Returns
    the generator steps the run has already taken with the term (0 if new)."""
    path = os.path.join(output, RECORD)
    if os.path.isfile(path):
        with open(path) as f:
            saved = json.load(f)
        w, s = float(saved["lambda_layout"]), int(saved.get("stop_step", 0))
        if (w, s) != (weight, stop):
            raise SystemExit("%s says this run trains with lambda_layout=%g, layout_stop=%d, "
                             "not %g, %d -- resume it with the same settings, or use a new "
                             "--output" % (path, w, s, weight, stop))
        return int(saved.get("steps_done", 0))
    if weight == 0:
        if stop:
            raise SystemExit("a layout stop step needs a layout weight")
        return 0
    os.makedirs(output, exist_ok=True)
    _write_record(path, {"lambda_layout": weight, "stop_step": stop, "steps_done": 0,
                         "downsample": DOWNSAMPLE, "blur": BLUR, "min_std": MIN_STD})
    return 0


def attach(model, weight: float, output: str, stop: int = 0):
    """Add weight * L_layout to `model`'s generator loss, from step 0 up to and
    including generator step `stop` (0 = the whole run)."""
    done = check_record(output, weight, stop)
    if weight == 0:
        return model
    inner = model.compute_generator_loss
    path = os.path.join(output, RECORD)
    state = {"step": done}

    def progress(step: int) -> None:
        with open(path) as f:
            record = json.load(f)
        record["steps_done"] = step
        _write_record(path, record)

    def compute_generator_loss(batch):
        loss_G, logs, visuals = inner(batch)
        state["step"] += 1
        step = state["step"]
        if stop and step > stop:
            with torch.no_grad():               # watched, not trained on
                term = layout_terms(visuals)
        else:
            term = layout_terms(visuals)
            loss_G = loss_G + weight * term
            logs["loss_G"] = float(loss_G.detach().cpu())
            if stop and (step % PROGRESS_EVERY == 0 or step == stop):
                progress(step)
                if step == stop:
                    print("[layout] step %d: switched off, still logged" % step)
        logs["loss_layout"] = float(term.detach().cpu())
        return loss_G, logs, visuals

    model.compute_generator_loss = compute_generator_loss
    print("[layout] lambda_layout=%g, %s (downsample %d, blur %g, min_std %g)%s"
          % (weight, "until step %d" % stop if stop else "whole run",
             DOWNSAMPLE, BLUR, MIN_STD,
             "; resumed at step %d" % done if done else ""))
    return model
