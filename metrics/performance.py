"""Task-performance, drift, continual-learning and compute metrics.

Loss and drift functions return ``jax.Array`` values so they can be used
inside a differentiated loss as well as for logging. ``training_step_time``
returns wall-clock seconds as a Python ``float``.

Continual-learning metrics  take a matrix ``A[t, j]``: the metric
on evaluation task ``j`` after training on task ``t``, with rows and columns
in training order and ``t`` 0-based.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

import jax
import jax.numpy as jnp

from frozenllm.substrate import FrozenSubstrate

T = TypeVar("T")


def cross_entropy_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Mean next-token cross-entropy; labels equal to ``-100`` are ignored."""
    return FrozenSubstrate.compute_loss(logits, labels)


def kl_to_base(base_logits: jax.Array, adapted_logits: jax.Array) -> jax.Array:
    """Differentiable ``KL(p_base || p_adapted)``, averaged over positions.

    Same quantity as ``frozenllm.substrate.compute_kl_drift``, but returned as
    an unclamped ``jax.Array`` so gradients flow into ``adapted_logits``.
    """
    # float32 log-space for stability over large vocabularies. Sum over V gives a
    # per-position KL [B, T]; the mean runs over batch and positions.
    log_p = jax.nn.log_softmax(base_logits.astype(jnp.float32), axis=-1)
    log_q = jax.nn.log_softmax(adapted_logits.astype(jnp.float32), axis=-1)
    return jnp.mean(jnp.sum(jnp.exp(log_p) * (log_p - log_q), axis=-1))


def perplexity(loss: jax.Array | float) -> jax.Array:
    """Perplexity ``exp(loss)`` of a mean cross-entropy loss (in nats)."""
    return jnp.exp(jnp.asarray(loss, dtype=jnp.float32))


def _task_matrix(acc: Any, t: int) -> jax.Array:
    """Validate ``acc`` as a 2-D matrix with a row and column for task ``t``."""
    a = jnp.asarray(acc, dtype=jnp.float32)
    if a.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix A[t, j], got shape {a.shape}.")
    if not isinstance(t, int) or t < 0:
        raise ValueError(f"t must be an integer >= 0, got {t!r}.")
    if t >= min(a.shape):
        raise ValueError(f"t={t} is out of range for A with shape {a.shape}.")
    return a


def mean_seen_accuracy(acc: Any, t: int) -> jax.Array:
    """Mean metric over the tasks seen so far, ``Ā_t = mean_{j<=t} A[t, j]``.

    ``t`` is 0-based, so the mean runs over ``t + 1`` tasks (the paper's
    1-based ``1/t · Σ_{j<=t}``). Works for any metric; for a loss matrix the
    result is the mean seen-task loss.
    """
    a = _task_matrix(acc, t)
    return jnp.mean(a[t, : t + 1])


def mean_forgetting(acc: Any, t: int, *, higher_is_better: bool = True) -> jax.Array:
    """Mean forgetting of earlier tasks after training task ``t``.

        F_t = mean_{j<t} ( best_{j<=s<t} A[s, j] - A[t, j] )

    ``best`` is ``max`` when ``higher_is_better`` (accuracy) and ``min`` with
    the sign flipped otherwise (loss), so forgetting is positive when
    performance on an old task got worse in both cases.

    The peak only ranges over ``s >= j``, i.e. after task ``j`` was trained
    (Chaudhry et al., 2018). The plan's ``max_{s<t}`` also admits rows from
    before task ``j`` was seen; those measure transfer, not retention, and
    would count negative transfer as forgetting.

    Undefined for ``t = 0`` (no earlier tasks), which raises ``ValueError``.
    """
    a = _task_matrix(acc, t)
    if t == 0:
        raise ValueError("Forgetting is undefined at t=0: there are no earlier tasks.")
    # Negate losses so "best" is always the max.
    x = a if higher_is_better else -a
    s = jnp.arange(t)[:, None]
    j = jnp.arange(t)[None, :]
    peak = jnp.max(jnp.where(s >= j, x[:t, :t], -jnp.inf), axis=0)
    return jnp.mean(peak - x[t, :t])


def training_step_time(
    step_fn: Callable[..., T], *args: Any, **kwargs: Any
) -> tuple[T, float]:
    """Run ``step_fn(*args, **kwargs)`` and return ``(output, seconds)``.

    JAX dispatches asynchronously, so the timer stops only after every array
    in the output is ready. The first call of a jitted function includes
    tracing and compilation; exclude it from throughput averages.
    """
    start = time.perf_counter()
    out = step_fn(*args, **kwargs)
    jax.block_until_ready(out)
    return out, time.perf_counter() - start


__all__ = [
    "cross_entropy_loss",
    "kl_to_base",
    "mean_forgetting",
    "mean_seen_accuracy",
    "perplexity",
    "training_step_time",
]
