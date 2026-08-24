"""KL drift evaluator: how far A1 moved the base model's predictive distribution.

Uses a **fixed anchor subset** of validation (the first ``max_batches`` batches
of the unshuffled validation loader) and the *same* token-weighted-mean
reduction as the training-time KL term, so the two numbers are directly
comparable::

    KL_drift = sum(per_token KL(p_base || p_A1) over valid tokens) / (valid tokens)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from neurosymbolic.models.base import ResidualAdaptedModel
from neurosymbolic.training.losses import (
    IGNORE_INDEX,
    per_token_kl,
    shift_for_causal_lm,
)


@dataclass(frozen=True)
class KLDriftResult:
    """``KL(p_base || p_A1)`` over a fixed anchor subset."""

    mean_kl: float
    max_token_kl: float
    num_tokens: int
    num_examples: int
    num_batches: int
    reduction: str = "token_weighted_mean"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@torch.no_grad()
def evaluate_kl_drift(
    model: ResidualAdaptedModel,
    dataloader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None = 8,
) -> KLDriftResult:
    """Compute KL drift between the frozen base and the adapted model."""
    was_training = model.training
    model.eval()
    total_kl = 0.0
    total_tokens = 0
    total_examples = 0
    batches = 0
    max_token_kl = 0.0
    try:
        for index, raw_batch in enumerate(dataloader):
            if max_batches is not None and index >= max_batches:
                break
            batch = {k: v.to(device) for k, v in raw_batch.items()}
            inputs = {
                k: v for k, v in batch.items() if k in ("input_ids", "attention_mask")
            }
            with model.adapters_disabled():
                base_logits = model(**inputs).logits
            adapted_logits = model(**inputs).logits

            base_shifted, shifted_labels = shift_for_causal_lm(
                base_logits, batch["labels"]
            )
            adapted_shifted, _ = shift_for_causal_lm(adapted_logits, batch["labels"])
            valid = shifted_labels != IGNORE_INDEX
            num_valid = int(valid.sum().item())
            if num_valid == 0:
                continue
            kl_map = per_token_kl(base_shifted, adapted_shifted)
            total_kl += float((kl_map * valid).sum().item())
            max_token_kl = max(max_token_kl, float(kl_map[valid].max().item()))
            total_tokens += num_valid
            total_examples += int(batch["labels"].size(0))
            batches += 1
    finally:
        model.train(was_training)

    mean_kl = total_kl / total_tokens if total_tokens else float("nan")
    return KLDriftResult(
        mean_kl=mean_kl,
        max_token_kl=max_token_kl,
        num_tokens=total_tokens,
        num_examples=total_examples,
        num_batches=batches,
    )


__all__ = ["KLDriftResult", "evaluate_kl_drift"]
