"""Typed configuration objects for Stage A experiments.

Configs are plain YAML mapped onto frozen dataclasses. Unknown keys raise
instead of being silently ignored, so a typo in a research config fails loudly
rather than silently changing the experiment.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

_VALID_TASKS = ("causal_lm", "classification")
_VALID_ACTIVATIONS = ("gelu", "relu", "silu", "tanh", "sigmoid", "identity")
_VALID_DTYPES = ("float32", "float16", "bfloat16")


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or internally inconsistent."""


@dataclass(frozen=True)
class SyntheticModelConfig:
    """Spec for a randomly initialised tiny transformer.

    TEMPORARY SMOKE-TEST SETTING. When present, no pretrained checkpoint is
    downloaded; a tiny GPTNeoX is constructed locally instead. Real experiments
    must leave this unset.
    """

    num_hidden_layers: int = 2
    hidden_size: int = 32
    num_attention_heads: int = 4
    intermediate_size: int = 64
    vocab_size: int = 128
    max_position_embeddings: int = 64


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str = "EleutherAI/pythia-160m"
    task: str = "causal_lm"
    dtype: str = "float32"
    synthetic: SyntheticModelConfig | None = None

    def __post_init__(self) -> None:
        if self.task not in _VALID_TASKS:
            raise ConfigError(
                f"model.task must be one of {_VALID_TASKS}, got {self.task!r}"
            )
        if self.dtype not in _VALID_DTYPES:
            raise ConfigError(
                f"model.dtype must be one of {_VALID_DTYPES}, got {self.dtype!r}"
            )


@dataclass(frozen=True)
class AdapterConfig:
    """Hidden-state low-rank residual adapter: R(h) = B * act(A h) * (alpha / rank).

    NOTE: `type: lora_residual` denotes a LoRA-*style* low-rank factorisation
    applied to a layer's output hidden state. It is NOT `peft` LoRA and does not
    reparameterise any base weight matrix.
    """

    type: str = "lora_residual"
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    activation: str = "gelu"
    target_layers: list[int] = field(default_factory=lambda: [-1])

    def __post_init__(self) -> None:
        if self.type != "lora_residual":
            raise ConfigError(
                f"adapter.type {self.type!r} is not implemented at A1; only 'lora_residual'."
            )
        if self.rank < 1:
            raise ConfigError(f"adapter.rank must be >= 1, got {self.rank}")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"adapter.dropout must be in [0, 1), got {self.dropout}")
        if self.activation not in _VALID_ACTIVATIONS:
            raise ConfigError(
                f"adapter.activation must be one of {_VALID_ACTIVATIONS}, got {self.activation!r}"
            )
        if not self.target_layers:
            raise ConfigError("adapter.target_layers must not be empty")
        if len(set(self.target_layers)) != len(self.target_layers):
            raise ConfigError(
                f"adapter.target_layers contains duplicates: {self.target_layers}"
            )


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    learning_rate: float = 1e-3
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    epochs: int = 1
    max_steps: int | None = None
    weight_decay: float = 0.01
    lambda_kl: float = 0.1
    warmup_ratio: float = 0.06
    max_grad_norm: float | None = 1.0
    log_every: int = 10
    freeze_check_every: int = 25

    def __post_init__(self) -> None:
        if self.gradient_accumulation_steps < 1:
            raise ConfigError("training.gradient_accumulation_steps must be >= 1")
        if self.batch_size < 1:
            raise ConfigError("training.batch_size must be >= 1")
        if self.lambda_kl < 0:
            raise ConfigError("training.lambda_kl must be >= 0")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ConfigError("training.warmup_ratio must be in [0, 1)")
        if self.epochs < 1 and self.max_steps is None:
            raise ConfigError("set training.epochs >= 1 or training.max_steps")


@dataclass(frozen=True)
class SyntheticDataConfig:
    """Random-token dataset spec. TEMPORARY SMOKE-TEST SETTING."""

    num_train: int = 16
    num_validation: int = 8
    num_test: int = 0
    min_length: int = 8
    max_length: int = 24


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "wikitext"
    dataset_config: str | None = "wikitext-2-raw-v1"
    text_column: str = "text"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str | None = None
    max_length: int = 256
    min_characters: int = 32
    max_train_examples: int | None = None
    max_eval_examples: int | None = None
    num_workers: int = 0
    synthetic: SyntheticDataConfig | None = None

    def __post_init__(self) -> None:
        if self.max_length < 2:
            raise ConfigError("data.max_length must be >= 2 for causal LM shifting")


@dataclass(frozen=True)
class EvaluationConfig:
    eval_every: int | None = 50
    save_every: int | None = None
    kl_anchor_batches: int = 8
    latency_batch_size: int = 1
    latency_sequence_length: int = 128
    latency_warmup_iters: int = 3
    latency_measure_iters: int = 10
    spectral_energy_ranks: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])


@dataclass(frozen=True)
class OutputConfig:
    directory: str = "runs/stage_A/a1"
    run_name: str = "a1_lora_baseline"
    save_checkpoint: bool = True


@dataclass(frozen=True)
class HardwareConfig:
    device: str = "auto"


@dataclass(frozen=True)
class A1Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_NESTED = {
    ("model", "synthetic"): SyntheticModelConfig,
    ("data", "synthetic"): SyntheticDataConfig,
}

_SECTIONS: dict[str, type] = {
    "model": ModelConfig,
    "adapter": AdapterConfig,
    "training": TrainingConfig,
    "data": DataConfig,
    "evaluation": EvaluationConfig,
    "output": OutputConfig,
    "hardware": HardwareConfig,
}


def _build_section(name: str, cls: type, raw: dict[str, Any]) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(
            f"config section {name!r} must be a mapping, got {type(raw).__name__}"
        )
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(
            f"unknown key(s) {unknown} in config section {name!r}; "
            f"valid keys are {sorted(known)}"
        )
    kwargs = dict(raw)
    for (section, key), nested_cls in _NESTED.items():
        if section == name and kwargs.get(key) is not None:
            kwargs[key] = _build_section(f"{name}.{key}", nested_cls, kwargs[key])
    return cls(**kwargs)


def config_from_dict(raw: dict[str, Any]) -> A1Config:
    """Build a validated :class:`A1Config` from a plain dict."""
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")
    unknown = sorted(set(raw) - set(_SECTIONS) - {"stage", "notes"})
    if unknown:
        raise ConfigError(
            f"unknown top-level config section(s) {unknown}; "
            f"valid sections are {sorted(_SECTIONS)}"
        )
    sections = {
        name: _build_section(name, cls, raw.get(name) or {})
        for name, cls in _SECTIONS.items()
    }
    return A1Config(**sections)


def load_config(path: str | Path) -> A1Config:
    """Load and validate a Stage A1 YAML config."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return config_from_dict(raw)


def _coerce(text: str) -> Any:
    """Parse a CLI override value using YAML scalar rules."""
    return yaml.safe_load(text)


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides to a raw config dict.

    Used for rank sweeps so a single pinned config file stays the source of
    truth and only the swept field varies.
    """
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in raw.items()}
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override {item!r} must look like 'section.key=value'")
        dotted, _, value = item.partition("=")
        keys = dotted.strip().split(".")
        if len(keys) < 2:
            raise ConfigError(
                f"override key {dotted!r} must be dotted, e.g. 'adapter.rank'"
            )
        cursor: dict[str, Any] = out
        for key in keys[:-1]:
            nxt = cursor.get(key)
            cursor[key] = dict(nxt) if isinstance(nxt, dict) else {}
            cursor = cursor[key]
        cursor[keys[-1]] = _coerce(value)
    return out


__all__ = [
    "A1Config",
    "AdapterConfig",
    "ConfigError",
    "DataConfig",
    "EvaluationConfig",
    "HardwareConfig",
    "ModelConfig",
    "OutputConfig",
    "SyntheticDataConfig",
    "SyntheticModelConfig",
    "TrainingConfig",
    "apply_overrides",
    "config_from_dict",
    "load_config",
]
