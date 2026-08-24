"""The A1 objective: KL reduction correctness, finiteness, no double-counted wd."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from neurosymbolic.constants import IGNORE_INDEX
from neurosymbolic.training.losses import (
    a1_loss,
    adapter_l2_squared,
    causal_lm_kl,
    causal_lm_task_loss,
    classification_kl,
    per_token_kl,
    token_weighted_kl,
)


def _reference_token_weighted_kl(
    base_logits: torch.Tensor, adapted_logits: torch.Tensor, mask: torch.Tensor
) -> float:
    """Independent reference: explicit Python loop over valid tokens.

    Deliberately written without any of the library's helpers so it cannot share
    a bug with the implementation under test.
    """
    total = 0.0
    count = 0
    batch, seq, vocab = base_logits.shape
    for b in range(batch):
        for t in range(seq):
            if not bool(mask[b, t]):
                continue
            p = F.softmax(base_logits[b, t].double(), dim=-1)
            q = F.softmax(adapted_logits[b, t].double(), dim=-1)
            total += sum(
                float(p[v]) * (math.log(float(p[v])) - math.log(float(q[v])))
                for v in range(vocab)
            )
            count += 1
    return total / count


def test_token_weighted_kl_matches_manual_reference_with_variable_lengths() -> None:
    """Variable valid-token counts per sequence — the case the reduction is for."""
    generator = torch.Generator().manual_seed(11)
    base_logits = torch.randn(3, 5, 7, generator=generator)
    adapted_logits = torch.randn(3, 5, 7, generator=generator)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    kl, num_valid = token_weighted_kl(base_logits, adapted_logits, mask)
    assert num_valid == 9
    assert float(kl) == pytest.approx(
        _reference_token_weighted_kl(base_logits, adapted_logits, mask), abs=1e-5
    )


def test_token_weighted_mean_differs_from_mean_of_sequence_means() -> None:
    """The two reductions genuinely disagree, so specifying one matters."""
    generator = torch.Generator().manual_seed(3)
    base_logits = torch.randn(2, 4, 5, generator=generator)
    adapted_logits = torch.randn(2, 4, 5, generator=generator) * 3.0
    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)

    token_weighted, _ = token_weighted_kl(base_logits, adapted_logits, mask)

    kl_map = per_token_kl(base_logits, adapted_logits)
    per_sequence = [
        float((kl_map[i] * mask[i]).sum() / mask[i].sum()) for i in range(mask.size(0))
    ]
    mean_of_means = sum(per_sequence) / len(per_sequence)

    assert float(token_weighted) != pytest.approx(mean_of_means, abs=1e-6)


def test_kl_of_identical_distributions_is_zero() -> None:
    logits = torch.randn(2, 3, 5)
    kl, num_valid = token_weighted_kl(
        logits, logits.clone(), torch.ones(2, 3, dtype=torch.bool)
    )
    assert float(kl) == pytest.approx(0.0, abs=1e-6)
    assert num_valid == 6


def test_kl_is_non_negative_and_finite_for_extreme_logits() -> None:
    base_logits = torch.tensor([[[100.0, -100.0, 0.0]]])
    adapted_logits = torch.tensor([[[-100.0, 100.0, 0.0]]])
    kl, _ = token_weighted_kl(
        base_logits, adapted_logits, torch.ones(1, 1, dtype=torch.bool)
    )
    assert math.isfinite(float(kl))
    assert float(kl) >= 0.0


def test_kl_with_no_valid_tokens_is_zero_not_nan() -> None:
    logits = torch.randn(2, 3, 5)
    kl, num_valid = token_weighted_kl(
        logits, logits * 2, torch.zeros(2, 3, dtype=torch.bool)
    )
    assert num_valid == 0
    assert float(kl) == 0.0
    assert math.isfinite(float(kl))


def test_mismatched_mask_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="must match per-token KL"):
        token_weighted_kl(torch.randn(2, 3, 5), torch.randn(2, 3, 5), torch.ones(2, 4))


def test_causal_lm_kl_uses_shifted_label_mask() -> None:
    """KL counts exactly the positions the task loss counts."""
    generator = torch.Generator().manual_seed(5)
    base_logits = torch.randn(2, 6, 4, generator=generator)
    adapted_logits = torch.randn(2, 6, 4, generator=generator)
    labels = torch.tensor(
        [[1, 2, 3, 1, 2, 3], [1, 2, 3, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]]
    )

    kl, num_valid = causal_lm_kl(base_logits, adapted_logits, labels)
    # Row 0 contributes 5 shifted positions, row 1 contributes 2.
    assert num_valid == 7
    expected = _reference_token_weighted_kl(
        base_logits[:, :-1, :], adapted_logits[:, :-1, :], labels[:, 1:] != IGNORE_INDEX
    )
    assert float(kl) == pytest.approx(expected, abs=1e-5)


def test_causal_lm_task_loss_ignores_padding() -> None:
    generator = torch.Generator().manual_seed(9)
    logits = torch.randn(2, 5, 6, generator=generator)
    padded_labels = torch.tensor(
        [[1, 2, 3, 4, 5], [1, 2, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]]
    )

    loss = causal_lm_task_loss(logits, padded_labels)
    manual = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, 6),
        padded_labels[:, 1:].reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    assert float(loss) == pytest.approx(float(manual), abs=1e-6)


def test_a1_total_loss_is_finite_and_composed_of_its_parts() -> None:
    generator = torch.Generator().manual_seed(13)
    base_logits = torch.randn(2, 5, 6, generator=generator)
    adapted_logits = torch.randn(2, 5, 6, generator=generator)
    labels = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, IGNORE_INDEX, IGNORE_INDEX]])

    out = a1_loss(base_logits, adapted_logits, labels, lambda_kl=0.25)
    assert math.isfinite(float(out.total))
    assert math.isfinite(float(out.task))
    assert math.isfinite(float(out.kl))
    assert float(out.weighted_kl) == pytest.approx(0.25 * float(out.kl), abs=1e-6)
    assert float(out.total) == pytest.approx(
        float(out.task) + float(out.weighted_kl), abs=1e-6
    )


def test_lambda_kl_zero_reduces_a1_loss_to_the_task_loss() -> None:
    generator = torch.Generator().manual_seed(17)
    base_logits = torch.randn(2, 4, 5, generator=generator)
    adapted_logits = torch.randn(2, 4, 5, generator=generator)
    labels = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])

    out = a1_loss(base_logits, adapted_logits, labels, lambda_kl=0.0)
    assert float(out.total) == pytest.approx(float(out.task), abs=1e-7)


def test_negative_lambda_kl_is_rejected() -> None:
    with pytest.raises(ValueError, match="lambda_kl must be >= 0"):
        a1_loss(
            torch.randn(1, 2, 3),
            torch.randn(1, 2, 3),
            torch.zeros(1, 2, dtype=torch.long),
            lambda_kl=-1.0,
        )


def test_base_branch_never_carries_gradient() -> None:
    """a1_loss detaches base logits, so no gradient can flow into the base path."""
    base_logits = torch.randn(1, 3, 4, requires_grad=True)
    adapted_logits = torch.randn(1, 3, 4, requires_grad=True)
    labels = torch.tensor([[1, 2, 3]])

    a1_loss(base_logits, adapted_logits, labels, lambda_kl=1.0).total.backward()
    assert base_logits.grad is None
    assert adapted_logits.grad is not None


def test_classification_kl_averages_over_the_batch() -> None:
    generator = torch.Generator().manual_seed(19)
    base_logits = torch.randn(4, 3, generator=generator)
    adapted_logits = torch.randn(4, 3, generator=generator)
    kl, count = classification_kl(base_logits, adapted_logits)
    assert count == 4
    assert float(kl) == pytest.approx(
        float(per_token_kl(base_logits, adapted_logits).mean()), abs=1e-6
    )


def test_adapter_l2_squared_is_a_diagnostic_only() -> None:
    params = [torch.ones(2, 2), torch.full((3,), 2.0)]
    assert float(adapter_l2_squared(params)) == pytest.approx(4 * 1.0 + 3 * 4.0)
    assert float(adapter_l2_squared([])) == 0.0


def test_a1_loss_does_not_include_a_weight_decay_term() -> None:
    """L_A1 as implemented == task + lambda_kl*KL exactly; wd lives in AdamW."""
    labels = torch.tensor([[1, 2, 3, 4]])
    base_logits = torch.randn(1, 4, 5)
    adapted_logits = torch.randn(1, 4, 5)
    out = a1_loss(base_logits, adapted_logits, labels, lambda_kl=0.5)
    assert float(out.total - out.task - out.weighted_kl) == pytest.approx(0.0, abs=1e-7)
