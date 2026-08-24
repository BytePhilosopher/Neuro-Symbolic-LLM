"""Low-rank residual adapter applied to a transformer *hidden state*.

Research definition (Stage A1)::

    h_tilde_l = h0_l + R_phi,l(h0_l)
    R_phi,l(h)  = B_l  sigma(A_l h)

This is a LoRA-*style* low-rank factorisation, but it is emphatically **not**
`peft` LoRA: nothing here reparameterises a base weight matrix as ``W + BA``.
``R_phi`` is a standalone module that is added to a decoder layer's *output
activation*, structurally separate from every base weight. It is attached with
a forward hook (see :mod:`neurosymbolic.models.base`).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "identity": nn.Identity,
}


def build_activation(name: str) -> nn.Module:
    """Instantiate the activation ``sigma`` used inside the residual."""
    try:
        return _ACTIVATIONS[name]()
    except KeyError as exc:
        raise ValueError(
            f"unknown activation {name!r}; expected one of {sorted(_ACTIVATIONS)}"
        ) from exc


class LowRankResidualAdapter(nn.Module):
    """``R_phi(h) = scaling * B(dropout(sigma(A h)))`` with ``scaling = alpha / rank``.

    Parameters
    ----------
    hidden_size:
        Width of the hidden state this adapter corrects.
    rank:
        Bottleneck width ``r``. ``A: hidden -> r``, ``B: r -> hidden``.
    alpha:
        Scaling numerator. Using ``alpha / rank`` keeps the effective output
        magnitude comparable across a rank sweep (ENGINEERING CHOICE).
    dropout:
        Dropout applied to the bottleneck activation.
    activation:
        Name of ``sigma``.

    ``B`` is zero-initialised so that ``R_phi(h) == 0`` at step 0 and the
    adapted model is *exactly* the base model before any training
    (ENGINEERING CHOICE; standard low-rank-adapter practice, and it makes the
    KL term start at zero).
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        *,
        alpha: float = 16.0,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}")
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")

        self.hidden_size = hidden_size
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.activation_name = activation

        self.A = nn.Linear(hidden_size, rank, bias=False)
        self.B = nn.Linear(rank, hidden_size, bias=False)
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming-uniform ``A``, zero ``B`` (so the residual starts at exactly 0)."""
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Map ``h -> R_phi(h)``; the caller adds this to ``h``."""
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"adapter expects last dim {self.hidden_size}, got {tuple(hidden_states.shape)}"
            )
        projected: torch.Tensor = self.activation(self.A(hidden_states))
        residual: torch.Tensor = self.B(self.dropout(projected))
        return residual * self.scaling

    def num_parameters(self) -> int:
        """Parameter count of ``phi`` for this adapter."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, rank={self.rank}, alpha={self.alpha}, "
            f"scaling={self.scaling:.4f}, activation={self.activation_name}"
        )


class AdapterBank(nn.ModuleDict):
    """Container holding one :class:`LowRankResidualAdapter` per target layer.

    A plain ``ModuleDict`` keyed by layer index (as a string). This is *not* the
    research plan's "residual bank" construct from a later stage; it is only a
    registry so multiple target layers can be configured at A1.
    """

    def add(self, layer_index: int, adapter: LowRankResidualAdapter) -> None:
        self[str(layer_index)] = adapter

    def get_adapter(self, layer_index: int) -> LowRankResidualAdapter:
        adapter = self[str(layer_index)]
        assert isinstance(adapter, LowRankResidualAdapter)
        return adapter

    @property
    def layer_indices(self) -> list[int]:
        return sorted(int(key) for key in self)

    def adapters(self) -> Iterator[tuple[int, LowRankResidualAdapter]]:
        for index in self.layer_indices:
            yield index, self.get_adapter(index)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


@contextmanager
def eval_dropout(module: nn.Module) -> Iterator[None]:
    """Temporarily put ``module`` in eval mode (disables adapter dropout)."""
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


__all__ = [
    "AdapterBank",
    "LowRankResidualAdapter",
    "build_activation",
    "eval_dropout",
]
