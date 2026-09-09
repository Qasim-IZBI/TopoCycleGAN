"""The TopoGAN loss (Wang et al., ECCV 2020), Eqs. 3-5.

Eq. 3  diagram_distance   1-Wasserstein between two diagrams, on *birth times
                          only* -- death times are dropped because the quantity
                          of interest is the gap that must be closed to complete
                          an almost-hole, not the size of the hole. With the
                          diagonal slice taken at d = 0, unmatched points cost
                          their own birth time, and the optimal matching in 1-D
                          is simply "sort both lists and pair them in order".

Eq. 4  topological_loss   the loss between a *set* of synthetic diagrams and a
                          *set* of real ones: match the two sets, then sum the
                          per-pair diagram distances. Because the matching is
                          set-to-set, the loss needs no paired data, which is
                          what makes it usable in an unpaired staining setting.

Eq. 5  the matching is Monge-Kantorovich optimal transport with uniform
       marginals. For equal-sized batches the LP solution is a permutation, so
       we solve the equivalent assignment problem with scipy's Hungarian solver
       and skip the POT dependency. For unequal batch sizes this is a
       rectangular assignment, which matches each synthetic diagram to a
       distinct real one rather than to argmax_j gamma_ij; with the batch sizes
       used in training the two agree.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from scipy.optimize import linear_sum_assignment

from topo_i2i.persistence import batch_diagrams

Diagram = Dict[int, torch.Tensor]


def _births(diagram: Diagram, dim: int) -> torch.Tensor:
    pts = diagram.get(dim)
    if pts is None or pts.shape[0] == 0:
        return torch.zeros(0)
    return pts[:, 0]


def diagram_distance(dgm_a: Diagram, dgm_b: Diagram, dims=(0, 1)) -> torch.Tensor:
    """Eq. 3, summed over the requested homology dimensions.

    Both diagrams are projected onto the birth axis, zero-padded to a common
    length (padding = matching to the diagonal at d = 0) and paired in sorted
    order, which is the exact 1-D optimal matching.
    """
    total = None
    for dim in dims:
        ba, bb = _births(dgm_a, dim), _births(dgm_b, dim)
        n = max(ba.numel(), bb.numel())
        if n == 0:
            continue
        ref = ba if ba.numel() else bb
        pad_a = torch.zeros(n - ba.numel(), device=ref.device, dtype=ref.dtype)
        pad_b = torch.zeros(n - bb.numel(), device=ref.device, dtype=ref.dtype)
        ba = torch.cat([ba.to(ref.device), pad_a]) if ba.numel() < n else ba
        bb = torch.cat([bb.to(ref.device), pad_b]) if bb.numel() < n else bb
        d = (torch.sort(ba)[0] - torch.sort(bb)[0]).abs().sum()
        total = d if total is None else total + d
    if total is None:
        return torch.zeros((), requires_grad=False)
    return total


def paired_diagram_loss(dgm_a: Sequence[Diagram], dgm_b: Sequence[Diagram],
                        dims=(0, 1), reduction: str = "mean") -> torch.Tensor:
    """Diagram distance between *corresponding* images, index by index.

    This is the cycle-topology term: image i and its own reconstruction are the
    same tissue, so no matching is needed -- unlike the distribution term, which
    compares unpaired sets and must solve an assignment first.
    """
    if len(dgm_a) != len(dgm_b):
        raise ValueError("paired loss needs equal-length diagram lists, got %d and %d"
                         % (len(dgm_a), len(dgm_b)))
    if not dgm_a:
        return torch.zeros(())
    terms = torch.stack([diagram_distance(a, b, dims) for a, b in zip(dgm_a, dgm_b)])
    return terms.mean() if reduction == "mean" else terms.sum()


def matched_diagram_loss(dgm_syn: Sequence[Diagram], dgm_real: Sequence[Diagram],
                         dims=(0, 1), reduction: str = "mean") -> torch.Tensor:
    """Eq. 4 on precomputed diagrams: assign the two sets, then sum the pairs."""
    if not dgm_syn or not dgm_real:
        return torch.zeros(())
    pairs, _ = match_diagram_sets(dgm_syn, dgm_real, dims)
    terms = torch.stack([diagram_distance(dgm_syn[i], dgm_real[j], dims) for i, j in pairs])
    return terms.mean() if reduction == "mean" else terms.sum()


def match_diagram_sets(syn: Sequence[Diagram], real: Sequence[Diagram], dims=(0, 1)):
    """Eq. 5: optimal assignment between the two diagram sets.

    Returns (pairs, cost_matrix) with pairs a list of (i, j) index tuples.
    """
    cost = torch.zeros(len(syn), len(real))
    for i, ds in enumerate(syn):
        for j, dr in enumerate(real):
            with torch.no_grad():
                cost[i, j] = diagram_distance(ds, dr, dims)
    rows, cols = linear_sum_assignment(cost.numpy())
    return list(zip(rows.tolist(), cols.tolist())), cost


def topological_loss(fake: torch.Tensor, real: torch.Tensor, dims=(0, 1),
                     reduction: str = "mean") -> torch.Tensor:
    """Eq. 4 between a batch of generated and a batch of real scalar fields.

    Args:
        fake: (B, H, W) scalar fields from the generator; carries gradient.
        real: (B', H, W) scalar fields from the target domain; detached.

    The real side is detached: the loss should move the generator, never the
    data. Only the matched synthetic diagrams contribute gradient.
    """
    dgm_syn = batch_diagrams(fake, dims)
    dgm_real = batch_diagrams(real.detach(), dims)
    if not dgm_syn or not dgm_real:
        return fake.sum() * 0.0
    return matched_diagram_loss(dgm_syn, dgm_real, dims, reduction)


def paired_topological_loss(fake: torch.Tensor, real: torch.Tensor, dims=(0, 1),
                            reduction: str = "mean") -> torch.Tensor:
    """Cycle-topology term: image i against its own reconstruction, no matching."""
    dgm_a = batch_diagrams(fake, dims)
    dgm_b = batch_diagrams(real.detach(), dims)
    if not dgm_a:
        return fake.sum() * 0.0
    return paired_diagram_loss(dgm_a, dgm_b, dims, reduction)
