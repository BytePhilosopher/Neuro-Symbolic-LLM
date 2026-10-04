"""Representation metrics: loss improvement and low- vs high-rank residual compressibility."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp


def loss_improvement(
    base_loss: jax.Array | float, loss: jax.Array | float
) -> jax.Array:
    """Return ``L_base - L_adapted``; positive when the residual helps."""
    return jnp.asarray(base_loss, jnp.float32) - jnp.asarray(loss, jnp.float32)


def residual_compressibility(
    delta_small: jax.Array | float,
    delta_large: jax.Array | float,
    eps: float = 1e-8,
) -> jax.Array:
    """Return ``Δ(r_small) / (Δ(r_large) + eps)``; meaningful only if ``delta_large > 0``."""
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps!r}")
    return jnp.asarray(delta_small, jnp.float32) / (
        jnp.asarray(delta_large, jnp.float32) + eps
    )


def compressibility_curve(
    deltas: Mapping[int, jax.Array | float], eps: float = 1e-8
) -> dict[int, jax.Array]:
    """Return compressibility of each rank in ``{rank: Δ}`` relative to the largest rank."""
    if not deltas:
        raise ValueError("deltas is empty; expected at least one rank.")
    ranks = sorted(deltas)
    if ranks[0] < 1:
        raise ValueError(f"Ranks must be >= 1, got {ranks[0]}.")
    reference = deltas[ranks[-1]]
    return {r: residual_compressibility(deltas[r], reference, eps) for r in ranks}


__all__ = ["compressibility_curve", "loss_improvement", "residual_compressibility"]
