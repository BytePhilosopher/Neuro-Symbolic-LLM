"""Stage A1: ordinary residual baseline.

    L_A1 = L_task + lambda_kl * KL(p_F0 || p_F0+R) + lambda_wd * ||phi||^2

No predictive coding and no symbolic head. Only the adapter parameters
``phi`` (the first positional argument) are differentiated; the substrate is
passed by keyword and never receives gradients.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import jax

from frozenllm.substrate import FrozenSubstrate, identity_modify
from metrics.performance import cross_entropy_loss, kl_to_base
from residual import AdapterParams, ResidualAdapter


@dataclass(frozen=True)
class A1Config:
    """Objective weights; mirrors the ``objective:`` YAML section."""

    lambda_kl: float = 0.0
    lambda_wd: float = 0.0

    def __post_init__(self) -> None:
        for name in ("lambda_kl", "lambda_wd"):
            value = getattr(self, name)
            if not isinstance(value, int | float) or value < 0:
                raise ValueError(f"{name} must be a number >= 0, got {value!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> A1Config:
        """Build from a mapping, rejecting unknown keys."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"Unknown A1Config keys: {unknown}. Allowed: {sorted(known)}."
            )
        return cls(**data)


def base_logits(substrate: FrozenSubstrate, input_ids: jax.Array) -> jax.Array:
    """Logits of the unmodified base model ``F0``, with gradients stopped."""
    # Empty intercept_layers: no hooks are attached, so this is the plain base forward.
    result = substrate.run_with_interception(
        input_ids, modify_fn=identity_modify, intercept_layers=()
    )
    return jax.lax.stop_gradient(result.logits)


def a1_objective(
    params: AdapterParams,
    *,
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    config: A1Config,
    input_ids: jax.Array,
    labels: jax.Array | None = None,
    base: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Return ``(total, terms)`` for one batch.

    ``labels`` default to ``input_ids`` (the next-token shift is done by the
    loss). ``base`` may carry precomputed base logits; either way they are
    passed through ``stop_gradient``.
    """
    adapted = substrate.run_with_interception(
        input_ids,
        modify_fn=adapter.modify_fn(params),
        intercept_layers=adapter.layers,
    ).logits
    base = (
        base_logits(substrate, input_ids)
        if base is None
        else jax.lax.stop_gradient(base)
    )
    task = cross_entropy_loss(adapted, input_ids if labels is None else labels)
    # Forward KL with the base as the target: penalises the adapter for dropping
    # probability mass the base assigns, keeping it near F0.
    kl = kl_to_base(base, adapted)
    wd = ResidualAdapter.l2_norm_sq(params)
    total = task + config.lambda_kl * kl + config.lambda_wd * wd
    return total, {"total": total, "task": task, "kl": kl, "wd": wd}


__all__ = ["A1Config", "a1_objective", "base_logits"]
