"""Adapter spectral analysis — an ENGINEERING DIAGNOSTIC, not ``C_resid(D)``.

This is a post-hoc SVD of a **single trained adapter's** ``A`` and ``B``
matrices. It is *not* the research plan's compressibility metric
``C_resid(D) = dD(r_small) / dD(r_large)``, which compares external task
performance across ranks and only becomes computable after a rank sweep — see
:mod:`neurosymbolic.evaluation.compressibility`. Do not conflate the two.

The model is never modified by this analysis.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from neurosymbolic.models.base import ResidualAdaptedModel

DEFAULT_ENERGY_RANKS: tuple[int, ...] = (1, 2, 4, 8, 16)


def effective_rank(singular_values: torch.Tensor, *, eps: float = 1e-12) -> float:
    """Spectral-entropy effective rank ``exp(-sum p_i log p_i)`` with ``p = s / sum(s)``.

    Equals ``k`` for a flat spectrum of ``k`` equal values and ``1`` for a rank-1
    spectrum, so it reads as a continuous "how many directions are in use".
    """
    values = singular_values.detach().float().clamp_min(0.0)
    total = values.sum()
    if float(total) <= eps:
        return 0.0
    probabilities = values / total
    nonzero = probabilities[probabilities > eps]
    entropy = -(nonzero * nonzero.log()).sum()
    return float(entropy.exp().item())


def cumulative_energy(
    singular_values: torch.Tensor, ranks: Sequence[int] = DEFAULT_ENERGY_RANKS
) -> dict[str, float]:
    """Fraction of squared-singular-value energy captured by the top-``k`` directions.

    Ranks above the matrix's actual rank are skipped, per the A1 spec.
    """
    values = singular_values.detach().float()
    squared = values.pow(2)
    total = float(squared.sum().item())
    matrix_rank = int(values.numel())
    out: dict[str, float] = {}
    for rank in ranks:
        if rank > matrix_rank:
            continue
        captured = float(squared[:rank].sum().item())
        out[f"top_{rank}"] = captured / total if total > 0 else 0.0
    return out


def analyze_matrix(
    weight: torch.Tensor,
    *,
    energy_ranks: Sequence[int] = DEFAULT_ENERGY_RANKS,
    max_singular_values: int = 64,
) -> dict[str, Any]:
    """Shape, Frobenius norm, singular values, effective rank and energy curve."""
    matrix = weight.detach().float().cpu()
    singular_values = torch.linalg.svdvals(matrix)
    return {
        "shape": list(matrix.shape),
        "frobenius_norm": float(matrix.norm().item()),
        "spectral_norm": (
            float(singular_values.max().item()) if singular_values.numel() else 0.0
        ),
        "num_singular_values": int(singular_values.numel()),
        "singular_values": singular_values[:max_singular_values].tolist(),
        "effective_rank": effective_rank(singular_values),
        "cumulative_energy": cumulative_energy(singular_values, energy_ranks),
    }


def analyze_adapter_spectra(
    model: ResidualAdaptedModel,
    *,
    energy_ranks: Sequence[int] = DEFAULT_ENERGY_RANKS,
) -> dict[str, Any]:
    """Run :func:`analyze_matrix` on every adapter's ``A`` and ``B``.

    Labelled ``adapter_spectral_analysis`` in all saved artifacts to keep it
    distinguishable from ``C_resid(D)``.
    """
    per_layer: dict[str, Any] = {}
    for layer_index, adapter in model.adapters.adapters():
        per_layer[str(layer_index)] = {
            "rank": adapter.rank,
            "alpha": adapter.alpha,
            "scaling": adapter.scaling,
            "activation": adapter.activation_name,
            "A": analyze_matrix(adapter.A.weight, energy_ranks=energy_ranks),
            "B": analyze_matrix(adapter.B.weight, energy_ranks=energy_ranks),
        }
    return {
        "analysis": "adapter_spectral_analysis",
        "note": (
            "Post-hoc SVD of one trained adapter's matrices. NOT the research "
            "plan's C_resid(D) compressibility metric."
        ),
        "energy_ranks_requested": list(energy_ranks),
        "per_layer": per_layer,
    }


__all__ = [
    "DEFAULT_ENERGY_RANKS",
    "analyze_adapter_spectra",
    "analyze_matrix",
    "cumulative_energy",
    "effective_rank",
]
