"""Turn a generated/real RGB image into the scalar field that gets filtered.

TopoGAN filters the distance transform of a binary mask. Our generators emit
3-channel stain images instead, so we need a differentiable RGB -> scalar map.
Two are provided:

  StainField      colour-deconvolution onto one stain vector (recommended):
                  the optical density explained by e.g. Sirius Red collagen or
                  hematoxylin. Linear in log-space RGB, so gradients flow.
  'gray'          plain luminance, if you want no stain assumption.

The paper's footnote 1 permits an arbitrary scalar function here, but note the
semantics change: birth times become intensity values rather than the "gap to
close" distances you get from a distance transform.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# QuPath's built-in vectors, in RGB optical-density space.
STAIN_VECTORS = {
    "hematoxylin": (0.65, 0.70, 0.29),
    "eosin": (0.2159, 0.8012, 0.5581),
    "dab": (0.27, 0.57, 0.78),
    # Sirius Red picks up collagen. These are literature values -- estimate the
    # real vectors from your own slides (QuPath: Analyze > Estimate stain
    # vectors > Auto) and replace them.
    "sirius_red": (0.36, 0.66, 0.66),
}

_LUMA = (0.299, 0.587, 0.114)


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


class StainField(nn.Module):
    """Project an RGB image onto one stain vector in optical-density space.

    Input is assumed to be in [-1, 1] (the zoo's normalisation). The map is
        OD   = -log10(clamp(rgb01, eps, 1))
        conc = <OD, v> / <v, v>
    which is the projection onto the stain vector -- a differentiable stand-in
    for a full 3x3 deconvolution when only one channel is needed.
    """

    def __init__(self, stain="hematoxylin", eps: float = 1e-3, in_range=(-1.0, 1.0)):
        super().__init__()
        vec = STAIN_VECTORS[stain] if isinstance(stain, str) else stain
        self.register_buffer("vec", torch.from_numpy(_unit(vec)).view(1, 3, 1, 1))
        self.eps = eps
        self.in_range = in_range

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        lo, hi = self.in_range
        rgb01 = ((rgb - lo) / (hi - lo)).clamp(self.eps, 1.0)
        od = -torch.log10(rgb01)
        return (od * self.vec).sum(dim=1)  # (B, H, W)


def rgb_to_scalar_field(rgb: torch.Tensor, mode="gray", in_range=(-1.0, 1.0)) -> torch.Tensor:
    """Convenience wrapper: 'gray' or any key of STAIN_VECTORS."""
    if mode == "gray":
        lo, hi = in_range
        rgb01 = ((rgb - lo) / (hi - lo)).clamp(0.0, 1.0)
        w = torch.tensor(_LUMA, device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
        return (rgb01 * w).sum(dim=1)
    return StainField(mode, in_range=in_range).to(rgb.device)(rgb)

class DeconvolutionField(nn.Module):
    """True colour deconvolution onto a stain pair, then combine the channels.

    Unlike StainField (which projects onto one vector and so leaves the stains
    correlated), this inverts the full 3x3 stain matrix -- the two named vectors
    plus their normalised cross product as the residual, exactly as QuPath's
    StainVector.makeResidualStainVector does. The channels it returns are
    therefore separated concentrations, which is what makes combining them with
    'max' meaningful.

    The inverse is a fixed linear map computed once at construction, so the whole
    thing stays differentiable.
    """

    def __init__(self, stains=("hematoxylin", "dab"), combine: str = "max",
                 eps: float = 1e-3, in_range=(-1.0, 1.0)):
        super().__init__()
        if len(stains) != 2:
            raise ValueError("expected exactly two stains, got %r" % (stains,))
        v1, v2 = (_unit(STAIN_VECTORS[s] if isinstance(s, str) else s) for s in stains)
        residual = np.cross(v1, v2)
        residual = residual / np.linalg.norm(residual)
        matrix = np.stack([v1, v2, residual], axis=1)          # columns = stains
        inverse = np.linalg.inv(matrix).astype(np.float32)
        # Only the two stain rows are needed; the residual row is discarded.
        self.register_buffer("inv", torch.from_numpy(inverse[:2]))  # (2, 3)
        if combine not in ("max", "sum", "mean"):
            raise ValueError("combine must be 'max', 'sum' or 'mean'")
        self.combine = combine
        self.eps = eps
        self.in_range = in_range

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        lo, hi = self.in_range
        rgb01 = ((rgb - lo) / (hi - lo)).clamp(self.eps, 1.0)
        od = -torch.log10(rgb01)                                # (B, 3, H, W)
        conc = torch.einsum("cj,bjhw->bchw", self.inv, od)      # (B, 2, H, W)
        if self.combine == "max":
            return conc.max(dim=1).values
        if self.combine == "sum":
            return conc.sum(dim=1)
        return conc.mean(dim=1)


def make_field(spec: str, combine: str = "max", in_range=(-1.0, 1.0)) -> nn.Module:
    """Build the scalar-field module named by `spec`.

        'gray'              luminance, no stain assumption
        'dab'               projection onto one stain vector (approximate:
                            the stains stay correlated)
        'dab+hematoxylin'   full deconvolution of the pair, channels combined
                            with `combine` -- the accurate way to use more than
                            one stain
    """
    if spec == "gray":
        return _GrayField(in_range)
    if "+" in spec:
        return DeconvolutionField(tuple(spec.split("+")), combine=combine, in_range=in_range)
    return StainField(spec, in_range=in_range)


class _GrayField(nn.Module):
    def __init__(self, in_range=(-1.0, 1.0)):
        super().__init__()
        self.in_range = in_range

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return rgb_to_scalar_field(rgb, "gray", self.in_range)


# Field pairs for the translation tasks, named by the two domains. `--preset`
# selects one; --field-A / --field-B / --field-combine override any part of it.
#
#   he-ki67  Ki67 is DAB with a hematoxylin counterstain, so domain B merges
#            both to get *all* nuclei rather than only the positive ones, and
#            domain A uses hematoxylin to match.
#   he-sr    Sirius Red marks collagen, and eosin is the H&E channel that picks
#            up the same collagen-rich stroma, so the pair is E <-> the target's
#            chromogen channel.
FIELD_PRESETS = {
    "he-ki67": {"field_A": "hematoxylin", "field_B": "dab+hematoxylin", "combine": "max"},
    "he-sr":   {"field_A": "eosin",       "field_B": "dab",             "combine": "max"},
}


def resolve_fields(preset: str, field_A=None, field_B=None, combine=None) -> dict:
    """Preset values, with any explicitly-given field overriding them."""
    if preset not in FIELD_PRESETS:
        raise ValueError("unknown preset %r; choose from %s"
                         % (preset, ", ".join(sorted(FIELD_PRESETS))))
    resolved = dict(FIELD_PRESETS[preset])
    for key, value in (("field_A", field_A), ("field_B", field_B), ("combine", combine)):
        if value is not None:
            resolved[key] = value
    return resolved
