"""Task-performance and drift metrics (research plan §13-14).

All functions return ``jax.Array`` values so they can be used inside a
differentiated loss as well as for logging.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from frozenllm.substrate import FrozenSubstrate


def cross_entropy_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Mean next-token cross-entropy; labels equal to ``-100`` are ignored."""
    return FrozenSubstrate.compute_loss(logits, labels)


def kl_to_base(base_logits: jax.Array, adapted_logits: jax.Array) -> jax.Array:
    """Differentiable ``KL(p_base || p_adapted)``, averaged over positions.

    Same quantity as ``frozenllm.substrate.compute_kl_drift``, but returned as
    an unclamped ``jax.Array`` so gradients flow into ``adapted_logits``.
    """
    log_p = jax.nn.log_softmax(base_logits.astype(jnp.float32), axis=-1)
    log_q = jax.nn.log_softmax(adapted_logits.astype(jnp.float32), axis=-1)
    return jnp.mean(jnp.sum(jnp.exp(log_p) * (log_p - log_q), axis=-1))


def perplexity(loss: jax.Array | float) -> jax.Array:
    """Perplexity ``exp(loss)`` of a mean cross-entropy loss (in nats)."""
    return jnp.exp(jnp.asarray(loss, dtype=jnp.float32))


__all__ = ["cross_entropy_loss", "kl_to_base", "perplexity"]
