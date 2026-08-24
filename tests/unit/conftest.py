"""Shared fixtures for the Stage A1 unit tests.

Every fixture uses a *tiny, locally constructed* GPTNeoX. No test downloads a
checkpoint or loads the pinned pythia-160m model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurosymbolic.models.base import ResidualAdaptedModel  # noqa: E402
from neurosymbolic.utils.config import (  # noqa: E402
    A1Config,
    AdapterConfig,
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    OutputConfig,
    SyntheticDataConfig,
    SyntheticModelConfig,
    TrainingConfig,
)

TINY_MODEL = SyntheticModelConfig(
    num_hidden_layers=2,
    hidden_size=32,
    num_attention_heads=4,
    intermediate_size=64,
    vocab_size=64,
    max_position_embeddings=64,
)


def make_config(**overrides) -> A1Config:
    """Build a tiny synthetic A1 config, with keyword section overrides."""
    sections = {
        "model": ModelConfig(
            name_or_path="synthetic-tiny-gpt-neox",
            task="causal_lm",
            synthetic=TINY_MODEL,
        ),
        "adapter": AdapterConfig(rank=4, alpha=8.0, dropout=0.0, target_layers=[-1]),
        "training": TrainingConfig(
            seed=0,
            learning_rate=5e-3,
            batch_size=4,
            epochs=1,
            max_steps=3,
            lambda_kl=0.1,
            warmup_ratio=0.0,
            log_every=0,
            freeze_check_every=1,
        ),
        "data": DataConfig(
            dataset="synthetic",
            dataset_config=None,
            max_length=16,
            synthetic=SyntheticDataConfig(
                num_train=12, num_validation=8, min_length=6, max_length=16
            ),
        ),
        "evaluation": EvaluationConfig(
            eval_every=None,
            kl_anchor_batches=2,
            latency_sequence_length=16,
            latency_warmup_iters=1,
            latency_measure_iters=2,
        ),
        "output": OutputConfig(directory="unused", run_name="test"),
    }
    sections.update(overrides)
    return A1Config(**sections)


@pytest.fixture()
def config_factory():
    """Factory for tiny synthetic configs, with per-section overrides."""
    return make_config


@pytest.fixture()
def config() -> A1Config:
    return make_config()


@pytest.fixture()
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture()
def model(config: A1Config, device: torch.device) -> ResidualAdaptedModel:
    from neurosymbolic.models.base import build_a1_model
    from neurosymbolic.utils.seed import set_seed

    set_seed(config.training.seed)
    return build_a1_model(config, device)


@pytest.fixture()
def batch() -> dict[str, torch.Tensor]:
    """A batch with genuinely variable valid-token counts."""
    generator = torch.Generator().manual_seed(7)
    input_ids = torch.randint(1, 64, (3, 10), generator=generator)
    attention_mask = torch.tensor(
        [
            [1] * 10,
            [1] * 7 + [0] * 3,
            [1] * 4 + [0] * 6,
        ],
        dtype=torch.long,
    )
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    input_ids = input_ids * attention_mask
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
