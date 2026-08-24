"""Adapter mechanics: shape, additivity, the disable context manager."""

from __future__ import annotations

import pytest
import torch

from neurosymbolic.models.base import find_decoder_layers
from neurosymbolic.models.residual_adapter import LowRankResidualAdapter


def test_residual_output_shape_equals_hidden_state_shape() -> None:
    adapter = LowRankResidualAdapter(32, 4, alpha=8.0)
    hidden = torch.randn(2, 5, 32)
    assert adapter(hidden).shape == hidden.shape


def test_adapter_rejects_wrong_hidden_size() -> None:
    adapter = LowRankResidualAdapter(32, 4)
    with pytest.raises(ValueError, match="expects last dim 32"):
        adapter(torch.randn(2, 5, 16))


def test_zero_initialised_b_makes_residual_exactly_zero_at_init() -> None:
    adapter = LowRankResidualAdapter(32, 4, alpha=8.0)
    hidden = torch.randn(2, 5, 32)
    assert torch.equal(adapter(hidden), torch.zeros_like(hidden))


def test_scaling_is_alpha_over_rank() -> None:
    adapter = LowRankResidualAdapter(16, 4, alpha=8.0)
    assert adapter.scaling == pytest.approx(2.0)


def test_factorisation_is_b_of_sigma_of_a() -> None:
    """R(h) == scaling * B(sigma(A h)) — the exact research-plan factorisation."""
    adapter = LowRankResidualAdapter(16, 4, alpha=4.0, activation="gelu")
    torch.nn.init.normal_(adapter.B.weight, std=0.1)
    hidden = torch.randn(2, 3, 16)
    expected = adapter.B(adapter.activation(adapter.A(hidden))) * adapter.scaling
    assert torch.allclose(adapter(hidden), expected, atol=1e-6)


def test_adapter_parameter_count_is_two_low_rank_matrices() -> None:
    adapter = LowRankResidualAdapter(32, 4)
    assert adapter.num_parameters() == 32 * 4 + 4 * 32


def test_hook_adds_residual_to_layer_output(model) -> None:
    """h_adapted == h_base + R_phi(h_base) at the hooked layer."""
    layers = find_decoder_layers(model.base_model)
    target = model.target_layers[0]
    adapter = model.adapters.get_adapter(target)
    torch.nn.init.normal_(adapter.B.weight, std=0.1)

    captured: dict[str, torch.Tensor] = {}

    def capture(_module, _inputs, output):
        captured["hidden"] = output[0].detach().clone()

    handle = layers[target].register_forward_hook(capture)
    input_ids = torch.randint(1, 64, (2, 6))
    try:
        with torch.no_grad(), model.adapters_disabled():
            model(input_ids=input_ids)
        base_hidden = captured["hidden"]

        with torch.no_grad():
            model(input_ids=input_ids)
        adapted_hidden = captured["hidden"]
    finally:
        handle.remove()

    # Our capture hook runs after the adapter hook, so `adapted_hidden` is the
    # post-residual value.
    expected = base_hidden + adapter(base_hidden)
    assert torch.allclose(adapted_hidden, expected, atol=1e-6)
    assert not torch.equal(adapted_hidden, base_hidden)


def test_disabled_context_manager_exactly_removes_the_residual(model) -> None:
    """BASE logits and adapter-disabled logits must match bit for bit."""
    adapter = model.adapters.get_adapter(model.target_layers[0])
    torch.nn.init.normal_(adapter.B.weight, std=0.1)
    input_ids = torch.randint(1, 64, (2, 6))

    with torch.no_grad():
        with model.adapters_disabled():
            disabled = model(input_ids=input_ids).logits.clone()
        enabled = model(input_ids=input_ids).logits.clone()

    # Detaching the hooks entirely gives the ground-truth base forward pass.
    model.remove_hooks()
    with torch.no_grad():
        pristine = model(input_ids=input_ids).logits.clone()

    assert torch.equal(disabled, pristine)
    assert not torch.equal(enabled, pristine)


def test_disabled_then_enabled_restores_previous_state(model) -> None:
    assert model.adapters_enabled
    with model.adapters_disabled():
        assert not model.adapters_enabled
        with model.adapters_disabled():
            assert not model.adapters_enabled
        assert not model.adapters_enabled
    assert model.adapters_enabled


def test_multiple_target_layers_get_independent_adapters(
    config_factory, device
) -> None:
    from neurosymbolic.models.base import build_a1_model
    from neurosymbolic.utils.config import AdapterConfig

    config = config_factory(adapter=AdapterConfig(rank=2, target_layers=[0, 1]))
    multi = build_a1_model(config, device)
    assert multi.target_layers == [0, 1]
    assert len(multi.adapters) == 2
    assert multi.adapters.get_adapter(0) is not multi.adapters.get_adapter(1)


def test_out_of_range_target_layer_is_rejected(config_factory, device) -> None:
    from neurosymbolic.models.base import build_a1_model
    from neurosymbolic.utils.config import AdapterConfig

    config = config_factory(adapter=AdapterConfig(rank=2, target_layers=[99]))
    with pytest.raises(ValueError, match="out of range"):
        build_a1_model(config, device)
