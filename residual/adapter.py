"""Hidden-state residual adapter .

At each adapted block ``l`` the frozen block output ``h`` (row vectors of
size ``d``) is replaced by

    h~ = h + s * sigma(h @ A_l) @ B_l,   A_l: [d, r],  B_l: [r, d]

with ``s = alpha / r`` (``1.0`` if ``alpha`` is unset). ``A ~ N(0, init_std^2)``
and ``B = 0``, so the adapted model equals the base model at initialisation.

The adapter parameters are a PyTree that is independent of the substrate
parameters; the only integration point is ``modify_fn(params)``, passed to
``FrozenSubstrate.run_with_interception``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp

from frozenllm.substrate.architecture import (
    Architecture,
    validate_interception_layers,
)

AdapterParams = dict[str, dict[str, jax.Array]]

ACTIVATIONS: Mapping[str, Callable[[jax.Array], jax.Array]] = MappingProxyType(
    {
        "identity": lambda x: x,
        "gelu": jax.nn.gelu,
        "relu": jax.nn.relu,
        "tanh": jnp.tanh,
    }
)

MIN_NUM_LAYERS = 4


@dataclass(frozen=True)
class ResidualConfig:
    """Adapter hyper-parameters; mirrors the ``adapter:`` YAML section."""

    rank: int = 16
    activation: str = "identity"
    alpha: float | None = None
    init_std: float | None = None
    late_start: int | None = None
    late_end: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError(f"rank must be an integer >= 1, got {self.rank!r}")
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {self.activation!r}. "
                f"Expected one of {sorted(ACTIVATIONS)}."
            )
        if self.alpha is not None and not self.alpha > 0:
            raise ValueError(f"alpha must be > 0 or null, got {self.alpha!r}")
        if self.init_std is not None and not self.init_std > 0:
            raise ValueError(f"init_std must be > 0 or null, got {self.init_std!r}")
        for name in ("late_start", "late_end"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, int):
                raise ValueError(f"{name} must be an integer or null, got {value!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResidualConfig:
        """Build from a mapping, rejecting unknown keys."""
        return cls(**_checked_kwargs(cls, data))

    @property
    def scale(self) -> float:
        """Residual scale ``s = alpha / r`` (``1.0`` if ``alpha`` is unset)."""
        return 1.0 if self.alpha is None else float(self.alpha) / self.rank


def _checked_kwargs(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            f"Unknown {cls.__name__} keys: {unknown}. Allowed: {sorted(known)}."
        )
    return dict(data)


def resolve_late_layers(
    num_layers: int,
    late_start: int | None = None,
    late_end: int | None = None,
) -> tuple[int, ...]:
    """Return the adapted blocks: the inclusive range ``(L // 2, L - 2]``.

    Defaults are ``late_start = L // 2 + 1`` and ``late_end = L - 2``. Bounds
    must satisfy ``L // 2 < late_start <= late_end <= L - 2``; they are never
    clamped.
    """
    l_mid = num_layers // 2
    start = l_mid + 1 if late_start is None else late_start
    end = num_layers - 2 if late_end is None else late_end
    valid = f"valid: {l_mid} < late_start <= late_end <= {num_layers - 2}"
    if num_layers < MIN_NUM_LAYERS:
        raise ValueError(
            f"Late-layer adapter needs a model with >= {MIN_NUM_LAYERS} blocks; "
            f"model has {num_layers}."
        )
    if not l_mid < start <= end <= num_layers - 2:
        raise ValueError(
            f"Invalid late-layer bounds late_start={start}, late_end={end} for a "
            f"model with {num_layers} blocks (0..{num_layers - 1}); {valid}."
        )
    return tuple(range(start, end + 1))


def layer_key(layer_idx: int) -> str:
    """PyTree key of the adapter at block ``layer_idx``."""
    return f"layer_{layer_idx}"


class ResidualAdapter:
    """Low-rank residual on the hidden states of the late frozen blocks."""

    def __init__(self, config: ResidualConfig, architecture: Architecture) -> None:
        self.config = config
        self.architecture = architecture
        self.hidden_size = architecture.hidden_size
        self.layers = validate_interception_layers(
            resolve_late_layers(
                architecture.num_layers, config.late_start, config.late_end
            ),
            architecture.num_layers,
        )
        self.scale = config.scale
        # 1/sqrt(d) keeps each component of h @ A at roughly the per-feature scale of h.
        self.init_std = (
            config.init_std
            if config.init_std is not None
            else 1.0 / math.sqrt(self.hidden_size)
        )
        self._activation = ACTIVATIONS[config.activation]

    def init_params(self) -> AdapterParams:
        """``A ~ N(0, init_std^2)`` and ``B = 0`` per layer, deterministic from seed."""
        base = jax.random.PRNGKey(self.config.seed)
        d, r = self.hidden_size, self.config.rank
        # Keys are folded by block index, not position in self.layers, so a
        # block's A does not change when late_start / late_end move.
        # B = 0 makes the residual exactly zero while dL/dB = sigma(hA)^T dL/dh
        # stays nonzero; A starts receiving gradient once B moves.
        return {
            layer_key(idx): {
                "A": self.init_std
                * jax.random.normal(
                    jax.random.fold_in(base, idx), (d, r), dtype=jnp.float32
                ),
                "B": jnp.zeros((r, d), dtype=jnp.float32),
            }
            for idx in self.layers
        }

    def residual(
        self, layer_params: Mapping[str, jax.Array], h: jax.Array
    ) -> jax.Array:
        """``s * sigma(h @ A) @ B`` in float32 (not yet cast back)."""
        # Upcast so the rank-r bottleneck does not lose precision on bf16/fp16
        # substrates. Shapes: [..., d] @ [d, r] -> [..., r] @ [r, d] -> [..., d].
        h32 = h.astype(jnp.float32)
        return self.scale * (
            self._activation(h32 @ layer_params["A"]) @ layer_params["B"]
        )

    def apply(self, params: AdapterParams, h: jax.Array, layer_idx: int) -> jax.Array:
        """Adapted hidden state; blocks without an entry pass through unchanged."""
        layer_params = params.get(layer_key(layer_idx))
        if layer_params is None:
            return h
        # Cast back so downstream frozen blocks see their native dtype.
        return h + self.residual(layer_params, h).astype(h.dtype)

    def modify_fn(self, params: AdapterParams) -> Callable[[jax.Array, int], jax.Array]:
        """A substrate ``ModifyFn`` ``(hidden, layer_idx) -> hidden`` over ``params``."""

        # Closing over params (possibly tracers) is what lets jax.grad reach the
        # adapter through the substrate's forward hooks.
        def modify(hidden: jax.Array, layer_idx: int) -> jax.Array:
            return self.apply(params, hidden, layer_idx)

        return modify

    @staticmethod
    def num_params(params: AdapterParams) -> int:
        """Total number of trainable scalars in ``params``."""
        return sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(params))

    @staticmethod
    def l2_norm_sq(params: AdapterParams) -> jax.Array:
        """``||phi||^2`` over all adapter parameters."""
        # Array-valued start so an empty PyTree still yields a float32 scalar.
        return sum(
            (jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(params)),
            start=jnp.zeros((), dtype=jnp.float32),
        )


__all__ = [
    "ACTIVATIONS",
    "AdapterParams",
    "ResidualAdapter",
    "ResidualConfig",
    "layer_key",
    "resolve_late_layers",
]
