"""Adapter checkpoints: save, reload, and refuse mismatched geometry."""

from __future__ import annotations

import pytest
import torch

from neurosymbolic.models.base import build_a1_model
from neurosymbolic.utils.config import AdapterConfig


def test_checkpoint_round_trip_restores_adapter_weights(model, tmp_path) -> None:
    for adapter in (model.adapters.get_adapter(i) for i in model.target_layers):
        torch.nn.init.normal_(adapter.B.weight, std=0.2)
    original = {n: p.detach().clone() for n, p in model.named_adapter_parameters()}

    path = model.save_adapters(tmp_path / "adapter.pt", metadata={"note": "unit test"})
    assert path.is_file()

    # Zero the weights, then reload and confirm they come back exactly.
    with torch.no_grad():
        for _, parameter in model.named_adapter_parameters():
            parameter.zero_()
    metadata = model.load_adapters(path)

    assert metadata["note"] == "unit test"
    restored = dict(model.named_adapter_parameters())
    for name, weights in original.items():
        assert torch.equal(restored[name].detach(), weights)


def test_checkpoint_contains_only_adapter_weights(model, tmp_path) -> None:
    path = model.save_adapters(tmp_path / "adapter.pt")
    payload = torch.load(path, map_location="cpu")
    state = payload["adapter_state_dict"]
    assert state, "checkpoint has no adapter weights"
    assert all(key.endswith(("A.weight", "B.weight")) for key in state), sorted(state)
    base_names = {name for name, _ in model.base_model.named_parameters()}
    assert not (set(state) & base_names)


def test_checkpoint_records_reproduction_metadata(model, tmp_path) -> None:
    path = model.save_adapters(tmp_path / "adapter.pt")
    payload = torch.load(path, map_location="cpu")
    assert payload["stage"] == "A1"
    assert payload["adapter_config"]["type"] == "lora_residual"
    assert payload["resolved_target_layers"] == model.target_layers
    assert payload["hidden_size"] == model.hidden_size


def test_loading_into_a_different_rank_is_rejected(
    model, config_factory, device, tmp_path
) -> None:
    path = model.save_adapters(tmp_path / "adapter.pt")
    other = build_a1_model(config_factory(adapter=AdapterConfig(rank=2)), device)
    with pytest.raises(RuntimeError):
        other.load_adapters(path)


def test_loading_a_different_target_layer_is_rejected(
    model, config_factory, device, tmp_path
) -> None:
    path = model.save_adapters(tmp_path / "adapter.pt")
    other = build_a1_model(
        config_factory(
            adapter=AdapterConfig(rank=model.adapter_config.rank, target_layers=[0])
        ),
        device,
    )
    with pytest.raises(ValueError, match="target layers"):
        other.load_adapters(path)


def test_loading_a_missing_checkpoint_raises(model, tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="adapter checkpoint not found"):
        model.load_adapters(tmp_path / "nope.pt")


def test_reloaded_adapter_reproduces_identical_logits(
    model, config, device, tmp_path
) -> None:
    adapter = model.adapters.get_adapter(model.target_layers[0])
    torch.nn.init.normal_(adapter.B.weight, std=0.2)
    input_ids = torch.randint(1, 64, (2, 8))
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits.clone()

    path = model.save_adapters(tmp_path / "adapter.pt")
    from neurosymbolic.utils.seed import set_seed

    set_seed(config.training.seed)
    reloaded = build_a1_model(config, device)
    reloaded.load_adapters(path)
    with torch.no_grad():
        actual = reloaded(input_ids=input_ids).logits

    assert torch.allclose(expected, actual, atol=1e-6)
