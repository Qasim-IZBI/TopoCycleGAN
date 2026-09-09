"""Cubical persistent homology, differentiable w.r.t. the input image.

The trick that makes this differentiable without a custom autograd.Function:
persistent homology of a *sublevel-set filtration* does not invent new values.
Every birth and death time is the value of the scalar field at one specific
pixel (the critical cell). So we let gudhi tell us *which* pixels those are --
a combinatorial, non-differentiable step done on detached numpy -- and then
gather those pixels straight out of the live torch tensor. Autograd then routes
each diagram point's gradient back to its critical pixel for free.

gudhi is imported lazily so the rest of the package works without it.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

_GUDHI_HINT = (
    "gudhi is required for persistent homology. Install it with "
    "`pip install gudhi` (or `pip install -e '.[dev]'` once it is in your deps)."
)


def _gudhi():
    try:
        import gudhi
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(_GUDHI_HINT) from exc
    return gudhi


def critical_indices(field: np.ndarray, dims=(0, 1)) -> Dict[int, np.ndarray]:
    """Flat pixel indices of the critical cells, per homology dimension.

    Returns {dim: array of shape (n_points, 2)} holding (birth_idx, death_idx)
    into `field.ravel()`. Essential features (those that never die) are dropped:
    their death is +inf, which has no gradient and no meaning in a birth-only
    distance.
    """
    gudhi = _gudhi()
    field = np.ascontiguousarray(field, dtype=np.float64)
    cc = gudhi.CubicalComplex(top_dimensional_cells=field)
    cc.compute_persistence(homology_coeff_field=2)
    regular, _essential = cc.cofaces_of_persistence_pairs()

    out = {}
    for dim in dims:
        if dim < len(regular) and len(regular[dim]) > 0:
            out[dim] = np.asarray(regular[dim], dtype=np.int64).reshape(-1, 2)
        else:
            out[dim] = np.zeros((0, 2), dtype=np.int64)
    return out


def persistence_diagram(field: torch.Tensor, dims=(0, 1)) -> Dict[int, torch.Tensor]:
    """Persistence diagram of one 2-D scalar field, differentiable in `field`.

    Args:
        field: (H, W) tensor. Sublevel-set filtration, so *low* values are the
               foreground -- invert beforehand if strong stain is high in yours.

    Returns {dim: (n_points, 2) tensor of [birth, death]} that carries gradient
    back to the critical pixels of `field`.
    """
    if field.dim() != 2:
        raise ValueError("expected a single (H, W) field, got shape %s" % (tuple(field.shape),))

    idx = critical_indices(field.detach().cpu().numpy(), dims)
    flat = field.reshape(-1)

    out = {}
    for dim, pairs in idx.items():
        if pairs.shape[0] == 0:
            out[dim] = field.new_zeros((0, 2))
            continue
        t = torch.from_numpy(pairs).to(field.device)
        out[dim] = torch.stack([flat[t[:, 0]], flat[t[:, 1]]], dim=1)
    return out


def batch_diagrams(fields: torch.Tensor, dims=(0, 1)):
    """persistence_diagram over a (B, H, W) batch. Returns a list of dicts.

    Note this is a Python loop over CPU-bound gudhi calls -- it is the cost
    centre of the whole loss. See TopoLossMixin for the every-k-steps and
    subsample knobs that keep it affordable.
    """
    return [persistence_diagram(fields[i], dims) for i in range(fields.shape[0])]
