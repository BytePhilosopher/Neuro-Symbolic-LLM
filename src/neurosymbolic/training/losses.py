r"""Stage A1 objective.

.. math::

    L_{A1} = L_{task}
             + \lambda_{KL} \, KL(p_{base} \| p_{adapted})
             + \lambda_{wd} \, \|\phi\|^2

The weight-decay term is **not** added to this loss: it is delegated to
``AdamW(weight_decay=...)``. Adding both would double-count the penalty
(ENGINEERING CHOICE, per the A1 spec).

KL reduction (ENGINEERING CHOICE, fixed here so training-time KL and the
post-hoc KL-drift evaluator are directly comparable):

.. code-block:: text

    KL_batch = sum(per_token_KL over all VALID tokens) / (count of VALID tokens)

i.e. a *token-weighted mean*, not a mean of per-sequence means. Validity comes
from the shifted label mask, so exactly the tokens that contribute to the task
loss contribute to the KL.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F  # noqa: N812

from neurosymbolic.constants import IGNORE_INDEX


@dataclass(frozen=True)
class A1LossOutput:
    """Components of the A1 objective for one batch."""

    total: torch.Tensor
    task: torch.Tensor
    kl: torch.Tensor
    weighted_kl: torch.Tensor
    num_valid_tokens: int
    extras: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "task_loss": float(self.task.detach()),
            "kl": float(self.kl.detach()),
            "weighted_kl": float(self.weighted_kl.detach()),
            "num_valid_tokens": self.num_valid_tokens,
            **self.extras,
        }


def shift_for_causal_lm(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop the last logit and the first label so position ``t`` predicts ``t+1``."""
    if logits.dim() != 3:
        raise ValueError(
            f"expected logits [batch, seq, vocab], got {tuple(logits.shape)}"
        )
    if labels.dim() != 2:
        raise ValueError(f"expected labels [batch, seq], got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"logits {tuple(logits.shape[:2])} and labels {tuple(labels.shape)} disagree"
        )
    return logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous()


def causal_lm_task_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Token-weighted-mean cross entropy over next-token predictions."""
    shifted_logits, shifted_labels = shift_for_causal_lm(logits, labels)
    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)).float(),
        shifted_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="mean",
    )


def per_token_kl(
    base_logits: torch.Tensor, adapted_logits: torch.Tensor
) -> torch.Tensor:
    r"""``KL(p_base || p_adapted)`` per position, shape ``[...]`` (vocab reduced).

    Computed in float32 from log-softmax outputs for numerical stability:
    :math:`\sum_v p_{base}(v) (\log p_{base}(v) - \log p_{adapted}(v))`.
    """
    base_log_probs = F.log_softmax(base_logits.float(), dim=-1)
    adapted_log_probs = F.log_softmax(adapted_logits.float(), dim=-1)
    base_probs = base_log_probs.exp()
    return (base_probs * (base_log_probs - adapted_log_probs)).sum(dim=-1)


def token_weighted_kl(
    base_logits: torch.Tensor,
    adapted_logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Token-weighted mean of ``KL(base || adapted)`` over valid positions.

    Returns ``(kl, num_valid_tokens)``. ``valid_mask`` is broadcast-compatible
    with the leading dims of the logits and is truthy for tokens that count.
    """
    kl_map = per_token_kl(base_logits, adapted_logits)
    if valid_mask.shape != kl_map.shape:
        raise ValueError(
            f"valid_mask {tuple(valid_mask.shape)} must match per-token KL "
            f"{tuple(kl_map.shape)}"
        )
    mask = valid_mask.to(dtype=kl_map.dtype)
    num_valid = int(mask.sum().item())
    if num_valid == 0:
        return kl_map.sum() * 0.0, 0
    return (kl_map * mask).sum() / mask.sum(), num_valid


def causal_lm_kl(
    base_logits: torch.Tensor,
    adapted_logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Token-weighted KL over the shifted next-token positions of a causal LM."""
    base_shifted, shifted_labels = shift_for_causal_lm(base_logits, labels)
    adapted_shifted, _ = shift_for_causal_lm(adapted_logits, labels)
    valid_mask = shifted_labels != IGNORE_INDEX
    return token_weighted_kl(base_shifted, adapted_shifted, valid_mask)


def classification_kl(
    base_logits: torch.Tensor, adapted_logits: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Per-example KL over class probabilities, averaged over the batch."""
    kl_per_example = per_token_kl(base_logits, adapted_logits)
    return kl_per_example.mean(), int(kl_per_example.numel())


def a1_loss(
    base_logits: torch.Tensor,
    adapted_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    lambda_kl: float,
    task: str = "causal_lm",
) -> A1LossOutput:
    """Assemble ``L_task + lambda_kl * KL(base || adapted)``.

    ``base_logits`` must already be detached (produced under ``no_grad`` with
    adapters disabled); the base branch never receives gradients.
    """
    if lambda_kl < 0:
        raise ValueError(f"lambda_kl must be >= 0, got {lambda_kl}")
    base_logits = base_logits.detach()

    if task == "causal_lm":
        task_loss = causal_lm_task_loss(adapted_logits, labels)
        kl, num_valid = causal_lm_kl(base_logits, adapted_logits, labels)
    elif task == "classification":
        task_loss = F.cross_entropy(
            adapted_logits.float(), labels, ignore_index=IGNORE_INDEX
        )
        kl, num_valid = classification_kl(base_logits, adapted_logits)
    else:
        raise ValueError(f"unsupported task {task!r}")

    weighted_kl = lambda_kl * kl
    return A1LossOutput(
        total=task_loss + weighted_kl,
        task=task_loss,
        kl=kl,
        weighted_kl=weighted_kl,
        num_valid_tokens=num_valid,
    )


def adapter_l2_squared(parameters: Sequence[torch.Tensor]) -> torch.Tensor:
    """``||phi||^2``, exposed for diagnostics only.

    NOT added to :func:`a1_loss` — weight decay is applied by AdamW. Reporting
    this separately keeps the ``lambda_wd ||phi||^2`` term of the written
    objective observable without double-counting it.
    """
    if not parameters:
        return torch.zeros(())
    return torch.stack([p.detach().float().pow(2).sum() for p in parameters]).sum()


__all__ = [
    "IGNORE_INDEX",
    "A1LossOutput",
    "a1_loss",
    "adapter_l2_squared",
    "causal_lm_kl",
    "causal_lm_task_loss",
    "classification_kl",
    "per_token_kl",
    "shift_for_causal_lm",
    "token_weighted_kl",
]
