"""Freezing is a hard A1 requirement: only phi trains, the base never moves."""

from __future__ import annotations

import pytest
import torch

from neurosymbolic.models.base import FreezeViolationError
from neurosymbolic.training.losses import a1_loss
from neurosymbolic.training.trainer import build_optimizer


def _step(model, config, batch) -> None:
    """One real A1 optimisation step (two forward passes, backward, step)."""
    optimizer = build_optimizer(model, config)
    inputs = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask")}
    with torch.no_grad(), model.adapters_disabled():
        base_logits = model(**inputs).logits.detach()
    adapted_logits = model(**inputs).logits
    loss = a1_loss(
        base_logits,
        adapted_logits,
        batch["labels"],
        lambda_kl=config.training.lambda_kl,
    )
    loss.total.backward()
    optimizer.step()


def test_all_base_parameters_are_frozen(model) -> None:
    assert not any(p.requires_grad for p in model.base_model.parameters())
    model.verify_frozen()


def test_all_adapter_parameters_are_trainable(model) -> None:
    adapter_params = list(model.adapters.parameters())
    assert adapter_params
    assert all(p.requires_grad for p in adapter_params)


def test_verify_frozen_detects_an_unfrozen_base_parameter(model) -> None:
    next(iter(model.base_model.parameters())).requires_grad_(True)
    with pytest.raises(FreezeViolationError, match="trainable"):
        model.verify_frozen()


def test_optimizer_contains_only_adapter_parameters(model, config) -> None:
    optimizer = build_optimizer(model, config)
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    adapter_ids = {id(p) for p in model.adapters.parameters()}
    base_ids = {id(p) for p in model.base_model.parameters()}

    assert optimizer_ids == adapter_ids
    assert optimizer_ids.isdisjoint(base_ids)


def test_optimizer_weight_decay_carries_the_lambda_wd_term(model, config) -> None:
    """||phi||^2 is applied by AdamW, not added to the loss (no double counting)."""
    optimizer = build_optimizer(model, config)
    assert all(
        group["weight_decay"] == config.training.weight_decay
        for group in optimizer.param_groups
    )


def test_base_parameters_receive_no_gradients(model, config, batch) -> None:
    _step(model, config, batch)
    assert all(
        p.grad is None or torch.all(p.grad == 0) for p in model.base_model.parameters()
    )
    model.assert_no_base_gradients()


def test_adapter_parameters_receive_gradients(model, config, batch) -> None:
    optimizer = build_optimizer(model, config)
    inputs = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask")}
    with torch.no_grad(), model.adapters_disabled():
        base_logits = model(**inputs).logits.detach()
    adapted_logits = model(**inputs).logits
    a1_loss(
        base_logits,
        adapted_logits,
        batch["labels"],
        lambda_kl=config.training.lambda_kl,
    ).total.backward()
    del optimizer

    grads = [p.grad for p in model.adapters.parameters()]
    assert all(g is not None for g in grads)
    assert any(torch.any(g != 0) for g in grads if g is not None)


def test_base_parameters_unchanged_after_optimizer_step(model, config, batch) -> None:
    snapshot = {n: p.detach().clone() for n, p in model.base_model.named_parameters()}
    _step(model, config, batch)
    for name, original in snapshot.items():
        current = dict(model.base_model.named_parameters())[name]
        assert torch.equal(current.detach(), original), f"base parameter {name} changed"
    model.assert_base_unchanged(snapshot)


def test_adapter_parameters_do_change_after_optimizer_step(
    model, config, batch
) -> None:
    snapshot = {n: p.detach().clone() for n, p in model.named_adapter_parameters()}
    _step(model, config, batch)
    current = dict(model.named_adapter_parameters())
    changed = [
        n for n, original in snapshot.items() if not torch.equal(current[n], original)
    ]
    assert changed, "no adapter parameter moved after optimizer.step()"


def test_assert_base_unchanged_detects_tampering(model) -> None:
    snapshot = model.snapshot_base_parameters()
    with torch.no_grad():
        next(iter(model.base_model.parameters())).add_(1.0)
    with pytest.raises(FreezeViolationError, match="changed during training"):
        model.assert_base_unchanged(snapshot)


def test_snapshot_limit_bounds_the_number_of_tensors(model) -> None:
    total = len(list(model.base_model.named_parameters()))
    limited = model.snapshot_base_parameters(limit=3)
    assert 0 < len(limited) <= 3 < total


def test_trainable_percentage_is_tiny(model) -> None:
    counts = model.parameter_counts()
    assert counts.adapter > 0
    assert counts.base > counts.adapter
    assert 0.0 < counts.trainable_percentage < 100.0
