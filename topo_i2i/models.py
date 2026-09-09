"""Zoo models with a topological term added, by subclassing rather than forking.

BaseTrainer takes a model *instance* and only calls compute_generator_loss,
compute_discriminator_loss and generator_parameters on it, so a subclass that
adds terms to the generator loss trains unmodified. Nothing in I2I-Stain-Zoo is
patched.

The full CycleGAN objective here is

    L = L_adv + lambda_cycle * L_cyc + lambda_identity * L_idt
        + lambda_topo * (lambda_ph_cyc  * (L_PH-cyc,H  + L_PH-cyc,I)
                       + lambda_ph_trans * (L_PH-trans,H + L_PH-trans,I))

with persistent homology computed on *stain-specific* channels, never on raw
RGB: hematoxylin for the H&E domain, and by default DAB+hematoxylin (deconvolved,
merged with max) for the IHC domain. Domain A is H&E, domain B is IHC throughout, matching the zoo's
`batch["A"]` / `batch["B"]`.

Writing x = real_A (H&E), y_hat = fake_B, x_hat = rec_A for the forward pass and
y = real_B (IHC), x_hat' = fake_A, y_hat' = rec_B for the reverse one, and H_A /
H_B for the two stain fields:

  L_PH-cyc,H   d_PH(H_A(x), H_A(x_hat))       source vs its reconstruction
  L_PH-cyc,I   d_PH(H_B(y), H_B(y_hat'))      source vs its reconstruction
  L_PH-trans,H  d_PH(H_A(x), H_B(y_hat))       source vs its own translation
  L_PH-trans,I  d_PH(H_B(y), H_A(x_hat'))      source vs its own translation

All four are *paired*: term i compares image i with something generated from
image i, so they use the diagram distance directly and no matching is needed.
They are named `trans` (translation consistency) rather than `dist`: the
reference is the image's own source, not an unpaired real tile from the target
domain, so this is not TopoGAN's distribution matching. TopoGAN's set-level OT loss is still available as
losses.topological_loss / matched_diagram_loss if you want to compare.

Note the two `trans` terms deliberately read *different* stain fields on their two
sides -- H&E nuclei against IHC nuclei-plus-chromogen -- which is the point: the
nuclear topology visible in the source must survive translation.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Tuple

import torch

from i2i_stain_zoo.models import CycleGAN, CycleGANConfig

from topo_i2i.fields import make_field
from topo_i2i.losses import DEFAULT_PROJECTION, paired_diagram_loss
from topo_i2i.persistence import batch_diagrams


@dataclass
class TopoConfig:
    """Everything the topological terms add on top of a zoo model config."""

    # Overall scale, then the split between the two families of terms.
    lambda_topo: float = 1.0
    lambda_ph_cyc: float = 1.0     # cycle topology (self-consistency)
    lambda_ph_trans: float = 1.0    # distribution topology (TopoGAN-style OT)

    # Stain-specific scalar fields, one per domain. The defaults in
    # fields.STAIN_VECTORS are literature values; estimate the real vectors from
    # your own slides (QuPath: Analyze > Estimate stain vectors > Auto) first.
    # A single name projects onto that stain vector; 'a+b' deconvolves the pair
    # properly and merges the two channels with `combine`. For IHC, DAB alone
    # sees only the positive nuclei, so DAB+hematoxylin (all nuclei, positive
    # and negative) is usually the structure you want to compare topologically.
    field_A: str = "hematoxylin"        # H&E domain -> nuclei
    field_B: str = "dab+hematoxylin"    # IHC domain -> chromogen + counterstain
    combine: str = "max"                # how 'a+b' fields merge: max/sum/mean

    # 0 = connected components, 1 = loops. TopoGAN focuses on 1; for nuclei,
    # 0 usually carries more signal. Both to start.
    dims: Tuple[int, ...] = (0, 1)

    # Which 1-D projection of the diagrams to compare. 'auto' uses
    # losses.DEFAULT_PROJECTION -- lifetime for H0, birth for H1 -- while
    # 'birth', 'lifetime' or 'death' force one for every dimension.
    projection: str = "auto"

    # Delay the PH terms until the generator produces something worth measuring.
    # Early outputs are noise, and noise maximises the number of critical points,
    # so the term is at its largest, least meaningful and most expensive exactly
    # when it can do least good. `warmup_steps` then ramps the weight linearly
    # from 0 to 1 rather than switching it on in one step, which avoids a jump in
    # the loss that Adam's moment estimates have to absorb.
    start_step: int = 0
    warmup_steps: int = 0

    # Cost control -- persistence is CPU-bound and unbatched, and there are now
    # six fields per step to diagram.
    every_n_steps: int = 1
    max_images: int = 0            # 0 = whole batch
    downsample: int = 1            # average-pool the field first

    # Sublevel-set filtration treats LOW values as foreground; stain channels
    # are high where stain is strong, so negate by default.
    invert: bool = True


@dataclass
class TopoCycleGANConfig(CycleGANConfig):
    topo: TopoConfig = dc_field(default_factory=TopoConfig)


class TopoLossMixin:
    """Adds the four persistent-homology terms to any CycleGAN-shaped model."""

    topo_cfg: TopoConfig

    def _init_topo(self, cfg: TopoConfig) -> None:
        self.topo_cfg = cfg
        # A buffer, not a plain int, so it is saved in the checkpoint: a requeued
        # job must not restart the warmup from zero.
        self.register_buffer("_topo_step", torch.zeros((), dtype=torch.long))
        self._field_mods = {
            "A": make_field(cfg.field_A, cfg.combine),
            "B": make_field(cfg.field_B, cfg.combine),
        }

    def _to_field(self, rgb: torch.Tensor, domain: str) -> torch.Tensor:
        """RGB -> the stain channel this domain's topology is computed on."""
        cfg = self.topo_cfg
        if cfg.downsample > 1:
            rgb = torch.nn.functional.avg_pool2d(rgb, cfg.downsample)
        f = self._field_mods[domain].to(rgb.device)(rgb)
        return -f if cfg.invert else f

    def _diagrams(self, rgb: torch.Tensor, domain: str, detach: bool):
        cfg = self.topo_cfg
        if cfg.max_images:
            rgb = rgb[: cfg.max_images]
        if detach:
            rgb = rgb.detach()
        return batch_diagrams(self._to_field(rgb, domain), tuple(cfg.dims))

    def topo_schedule(self, step: int) -> float:
        """Weight multiplier for the PH terms at this step: 0 before
        `start_step`, then ramping to 1 over `warmup_steps`."""
        cfg = self.topo_cfg
        if step < cfg.start_step:
            return 0.0
        if cfg.warmup_steps <= 0:
            return 1.0
        return min(1.0, (step - cfg.start_step + 1) / cfg.warmup_steps)

    def topo_terms(self, batch: Dict[str, torch.Tensor],
                   visuals: Dict[str, torch.Tensor]):
        """Returns (weighted total, logs) for the four PH terms.

        Each source domain's diagrams are computed once and shared by both terms
        anchored on it (H_A(x) feeds L_PH-cyc,H and L_PH-trans,H; H_B(y) feeds
        the other two), which is why this is one method rather than four
        independent loss calls.
        """
        cfg = self.topo_cfg
        self._topo_step += 1
        step = int(self._topo_step)
        scale = self.topo_schedule(step)

        zero = visuals["fake_B"].sum() * 0.0
        logs = {"loss_ph_cyc_H": 0.0, "loss_ph_cyc_I": 0.0,
                "loss_ph_trans_H": 0.0, "loss_ph_trans_I": 0.0,
                "loss_topo": 0.0, "topo_scale": scale}
        # Skip the persistence computation entirely when it would not be used.
        if cfg.lambda_topo == 0 or scale == 0.0 or (step % cfg.every_n_steps) != 0:
            return zero, logs

        dims = tuple(cfg.dims)
        proj = None if cfg.projection == "auto" else cfg.projection
        # Each real-domain diagram set is used twice; compute it once.
        dgm_real_A = self._diagrams(batch["A"], "A", detach=True)
        dgm_real_B = self._diagrams(batch["B"], "B", detach=True)

        total = zero

        if cfg.lambda_ph_cyc != 0:
            dgm_rec_A = self._diagrams(visuals["rec_A"], "A", detach=False)
            dgm_rec_B = self._diagrams(visuals["rec_B"], "B", detach=False)
            cyc_H = paired_diagram_loss(dgm_rec_A, dgm_real_A, dims, projection=proj)
            cyc_I = paired_diagram_loss(dgm_rec_B, dgm_real_B, dims, projection=proj)
            total = total + cfg.lambda_ph_cyc * (cyc_H + cyc_I)
            logs["loss_ph_cyc_H"] = float(cyc_H.detach().cpu())
            logs["loss_ph_cyc_I"] = float(cyc_I.detach().cpu())

        if cfg.lambda_ph_trans != 0:
            # Forward: the H&E source anchors its own translation into IHC.
            dgm_fake_B = self._diagrams(visuals["fake_B"], "B", detach=False)
            trans_H = paired_diagram_loss(dgm_fake_B, dgm_real_A, dims, projection=proj)
            # Reverse: the IHC source anchors its own translation into H&E.
            dgm_fake_A = self._diagrams(visuals["fake_A"], "A", detach=False)
            trans_I = paired_diagram_loss(dgm_fake_A, dgm_real_B, dims, projection=proj)
            total = total + cfg.lambda_ph_trans * (trans_H + trans_I)
            logs["loss_ph_trans_H"] = float(trans_H.detach().cpu())
            logs["loss_ph_trans_I"] = float(trans_I.detach().cpu())

        total = total * scale
        logs["loss_topo"] = float(total.detach().cpu())
        return total, logs


class TopoCycleGAN(TopoLossMixin, CycleGAN):
    """CycleGAN + adversarial + cycle + the four persistent-homology terms."""

    def __init__(self, cfg: TopoCycleGANConfig):
        super().__init__(cfg)
        self._init_topo(cfg.topo)

    def compute_generator_loss(self, batch: Dict[str, torch.Tensor]):
        loss_G, logs, visuals = super().compute_generator_loss(batch)

        loss_topo, topo_logs = self.topo_terms(batch, visuals)
        loss_G = loss_G + self.topo_cfg.lambda_topo * loss_topo

        logs.update(topo_logs)
        logs["loss_G"] = float(loss_G.detach().cpu())
        return loss_G, logs, visuals
