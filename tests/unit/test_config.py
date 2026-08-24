"""Configuration loading, validation and CLI overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from neurosymbolic.utils.config import (
    ConfigError,
    apply_overrides,
    config_from_dict,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_shipped_a1_config_loads() -> None:
    config = load_config(REPO_ROOT / "configs" / "stage_A" / "a1_lora_baseline.yaml")
    assert config.model.name_or_path == "EleutherAI/pythia-160m"
    assert config.model.task == "causal_lm"
    assert config.adapter.type == "lora_residual"
    assert config.data.dataset == "wikitext"
    assert config.data.dataset_config == "wikitext-2-raw-v1"
    assert config.training.lambda_kl > 0
    assert (
        config.model.synthetic is None
    ), "the real config must not use a synthetic model"


def test_shipped_smoke_config_loads_with_synthetic_sections() -> None:
    config = load_config(REPO_ROOT / "configs" / "stage_A" / "a1_smoke.yaml")
    assert config.model.synthetic is not None
    assert config.data.synthetic is not None
    assert config.hardware.device == "cpu"


def test_a1_config_has_no_predictive_coding_or_symbolic_sections() -> None:
    """A1 must be structurally incapable of carrying PC/symbolic settings."""
    raw = yaml.safe_load(
        (REPO_ROOT / "configs" / "stage_A" / "a1_lora_baseline.yaml").read_text()
    )
    forbidden = {
        "pc",
        "predictive_coding",
        "latent",
        "inference_steps",
        "symbolic",
        "symbolic_head",
        "mork",
        "atomese",
        "metta",
        "causal_gate",
        "residual_bank",
        "columns",
        "routing",
    }
    keys = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                keys.add(str(key).lower())
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(raw)
    assert not (
        keys & forbidden
    ), f"A1 config carries later-stage keys: {keys & forbidden}"


def test_unknown_top_level_section_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown top-level config section"):
        config_from_dict({"predictive_coding": {"steps": 4}})


def test_unknown_key_within_a_section_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        config_from_dict({"adapter": {"rnak": 8}})


def test_missing_config_file_raises() -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(REPO_ROOT / "configs" / "stage_A" / "does_not_exist.yaml")


@pytest.mark.parametrize(
    ("section", "payload", "match"),
    [
        ("adapter", {"rank": 0}, "rank must be >= 1"),
        ("adapter", {"type": "peft_lora"}, "not implemented at A1"),
        ("adapter", {"activation": "mystery"}, "activation must be one of"),
        ("adapter", {"target_layers": []}, "must not be empty"),
        ("adapter", {"target_layers": [1, 1]}, "duplicates"),
        ("model", {"task": "seq2seq"}, "task must be one of"),
        ("model", {"dtype": "int8"}, "dtype must be one of"),
        ("training", {"lambda_kl": -0.5}, "lambda_kl must be >= 0"),
        ("training", {"warmup_ratio": 1.5}, "warmup_ratio must be in"),
        ("training", {"gradient_accumulation_steps": 0}, "must be >= 1"),
        ("data", {"max_length": 1}, "max_length must be >= 2"),
    ],
)
def test_invalid_values_are_rejected(section: str, payload: dict, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        config_from_dict({section: payload})


def test_overrides_are_applied_and_yaml_typed() -> None:
    raw = {"adapter": {"rank": 8}, "training": {"seed": 42}}
    out = apply_overrides(
        raw, ["adapter.rank=16", "training.seed=7", "training.max_steps=null"]
    )
    config = config_from_dict(out)
    assert config.adapter.rank == 16
    assert config.training.seed == 7
    assert config.training.max_steps is None
    assert raw["adapter"]["rank"] == 8, "apply_overrides must not mutate its input"


def test_override_without_equals_is_rejected() -> None:
    with pytest.raises(ConfigError, match="must look like"):
        apply_overrides({}, ["adapter.rank"])


def test_override_must_be_dotted() -> None:
    with pytest.raises(ConfigError, match="must be dotted"):
        apply_overrides({}, ["rank=4"])


def test_to_dict_round_trips_through_yaml() -> None:
    config = load_config(REPO_ROOT / "configs" / "stage_A" / "a1_smoke.yaml")
    restored = config_from_dict(yaml.safe_load(yaml.safe_dump(config.to_dict())))
    assert restored == config
