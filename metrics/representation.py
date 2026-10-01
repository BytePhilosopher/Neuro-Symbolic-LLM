"""Internal knowledge-representation metrics.

Residual compressibility compares the improvement a low-rank residual
achieves against a high-rank one on the same domain. Values near 1 mean the
base already contains directions that a small rank can exploit.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp


def loss_improvement(
    base_loss: jax.Array | float, loss: jax.Array | float
) -> jax.Array:
    """Improvement ``Δ_D(r) = L_base - L_adapted``; positive when the residual helps."""
    return jnp.asarray(base_loss, jnp.float32) - jnp.asarray(loss, jnp.float32)


def residual_compressibility(
    delta_small: jax.Array | float,
    delta_large: jax.Array | float,
    eps: float = 1e-8,
) -> jax.Array:
    """``C_resid(D) = Δ_D(r_small) / (Δ_D(r_large) + ε)``.

    Only meaningful when ``delta_large > 0``: if the large-rank residual does
    not improve on the base, there is nothing to compress and the ratio's sign
    and size carry no information.
    """
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps!r}")
    return jnp.asarray(delta_small, jnp.float32) / (
        jnp.asarray(delta_large, jnp.float32) + eps
    )


def compressibility_curve(
    deltas: Mapping[int, jax.Array | float], eps: float = 1e-8
) -> dict[int, jax.Array]:
    """``C_resid`` of every rank in a sweep, relative to the largest rank.

    ``deltas`` maps rank to ``Δ_D(r)`` for one domain, e.g. the sweep
    ``r ∈ {1, 2, 4, …, 64}``. The largest rank maps to ``≈ 1``.
    """
    if not deltas:
        raise ValueError("deltas is empty; expected at least one rank.")
    ranks = sorted(deltas)
    if ranks[0] < 1:
        raise ValueError(f"Ranks must be >= 1, got {ranks[0]}.")
    reference = deltas[ranks[-1]]
    return {r: residual_compressibility(deltas[r], reference, eps) for r in ranks}


__all__ = ["compressibility_curve", "loss_improvement", "residual_compressibility"]
