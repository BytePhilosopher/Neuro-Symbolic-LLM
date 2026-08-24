"""Causal-LM data pipeline for Stage A1.

One clean task, deliberately not a generalised data framework. Pinned default
(ENGINEERING CHOICE): ``wikitext`` / ``wikitext-2-raw-v1`` — small, standard and
reproducible, and a natural pair for the pinned pythia-160m base model.

Examples are tokenised *per line* and truncated to ``data.max_length``, then
padded dynamically per batch by :class:`CausalLMCollator`. Padding positions get
label ``-100`` so they are excluded from both the task loss and the KL term.
Sequences therefore have genuinely different valid-token counts, which is
exactly the case the token-weighted KL reduction is specified for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch.utils.data import DataLoader, Dataset

from neurosymbolic.constants import IGNORE_INDEX
from neurosymbolic.utils.config import A1Config, DataConfig
from neurosymbolic.utils.logging import get_logger
from neurosymbolic.utils.seed import dataloader_generator

LOGGER = get_logger("data.datasets")


class TokenizerLike(Protocol):
    """The slice of the HF tokenizer API this module relies on."""

    pad_token_id: int | None

    def __call__(self, text: str, **kwargs: Any) -> Any: ...


class TokenizedTextDataset(Dataset):
    """In-memory dataset of variable-length ``input_ids`` lists."""

    def __init__(self, sequences: list[list[int]]) -> None:
        if not sequences:
            raise ValueError(
                "dataset is empty after filtering; loosen data.min_characters"
            )
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return {"input_ids": self.sequences[index]}


@dataclass
class CausalLMCollator:
    """Pad a batch to its longest sequence and build ``labels`` / ``attention_mask``.

    ``labels`` mirror ``input_ids`` on real tokens and are ``-100`` on padding.
    The mask (not the pad token id) determines validity, which stays correct when
    ``pad_token == eos_token`` — as it is for pythia.
    """

    pad_token_id: int
    label_pad_token_id: int = IGNORE_INDEX

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        lengths = [len(feature["input_ids"]) for feature in features]
        max_length = max(lengths)
        input_ids = torch.full(
            (len(features), max_length), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(features), max_length), dtype=torch.long)
        labels = torch.full(
            (len(features), max_length), self.label_pad_token_id, dtype=torch.long
        )
        for row, feature in enumerate(features):
            ids = torch.tensor(feature["input_ids"], dtype=torch.long)
            length = ids.numel()
            input_ids[row, :length] = ids
            attention_mask[row, :length] = 1
            labels[row, :length] = ids
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_tokenizer(config: A1Config) -> Any:
    """Load the tokenizer, guaranteeing a pad token exists.

    In synthetic (smoke-test) mode there is no pretrained tokenizer; a trivial
    stand-in carrying only ``pad_token_id`` and ``vocab_size`` is returned.
    """
    if config.model.synthetic is not None:
        return _SyntheticTokenizer(vocab_size=config.model.synthetic.vocab_size)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                f"tokenizer for {config.model.name_or_path} has neither a pad nor an eos token"
            )
        tokenizer.pad_token = tokenizer.eos_token
        LOGGER.info("Tokenizer had no pad token; using eos token as pad")
    return tokenizer


@dataclass
class _SyntheticTokenizer:
    """Stand-in tokenizer for synthetic smoke-test data."""

    vocab_size: int
    pad_token_id: int = 0

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        raise NotImplementedError("the synthetic tokenizer does not tokenise text")


def build_synthetic_split(
    *,
    num_examples: int,
    vocab_size: int,
    min_length: int,
    max_length: int,
    seed: int,
) -> TokenizedTextDataset:
    """Random-token dataset with variable sequence lengths (smoke test / tests)."""
    if min_length < 2:
        raise ValueError("min_length must be >= 2 so causal shifting leaves a target")
    generator = torch.Generator().manual_seed(seed)
    sequences: list[list[int]] = []
    for _ in range(num_examples):
        length = int(
            torch.randint(min_length, max_length + 1, (1,), generator=generator).item()
        )
        ids = torch.randint(1, max(2, vocab_size), (length,), generator=generator)
        sequences.append(ids.tolist())
    return TokenizedTextDataset(sequences)


def _tokenize_split(
    raw_texts: list[str],
    tokenizer: Any,
    data_config: DataConfig,
    limit: int | None,
) -> TokenizedTextDataset:
    sequences: list[list[int]] = []
    for text in raw_texts:
        stripped = text.strip()
        if len(stripped) < data_config.min_characters:
            continue
        encoded = tokenizer(
            stripped,
            truncation=True,
            max_length=data_config.max_length,
            add_special_tokens=False,
        )
        ids = encoded["input_ids"]
        if len(ids) < 2:
            continue
        sequences.append(list(ids))
        if limit is not None and len(sequences) >= limit:
            break
    return TokenizedTextDataset(sequences)


def build_datasets(config: A1Config, tokenizer: Any) -> dict[str, TokenizedTextDataset]:
    """Build train / validation (/ optional test) datasets."""
    data = config.data

    splits: dict[str, TokenizedTextDataset] = {}

    if data.synthetic is not None:
        vocab_size = getattr(tokenizer, "vocab_size", 128)
        splits = {
            "train": build_synthetic_split(
                num_examples=data.synthetic.num_train,
                vocab_size=vocab_size,
                min_length=data.synthetic.min_length,
                max_length=min(data.synthetic.max_length, data.max_length),
                seed=config.training.seed,
            ),
            "validation": build_synthetic_split(
                num_examples=data.synthetic.num_validation,
                vocab_size=vocab_size,
                min_length=data.synthetic.min_length,
                max_length=min(data.synthetic.max_length, data.max_length),
                seed=config.training.seed + 1,
            ),
        }
        if data.synthetic.num_test:
            splits["test"] = build_synthetic_split(
                num_examples=data.synthetic.num_test,
                vocab_size=vocab_size,
                min_length=data.synthetic.min_length,
                max_length=min(data.synthetic.max_length, data.max_length),
                seed=config.training.seed + 2,
            )
        return splits

    from datasets import load_dataset

    wanted = {"train": data.train_split, "validation": data.validation_split}
    if data.test_split:
        wanted["test"] = data.test_split

    limits = {
        "train": data.max_train_examples,
        "validation": data.max_eval_examples,
        "test": data.max_eval_examples,
    }

    for name, split in wanted.items():
        LOGGER.info("Loading %s/%s split=%s", data.dataset, data.dataset_config, split)
        raw = load_dataset(data.dataset, data.dataset_config, split=split)
        if data.text_column not in raw.column_names:
            raise ValueError(
                f"column {data.text_column!r} not in {raw.column_names} for split {split!r}"
            )
        splits[name] = _tokenize_split(
            list(raw[data.text_column]), tokenizer, data, limits[name]
        )
        LOGGER.info("Split %s: %d usable examples", name, len(splits[name]))
    return splits


def build_dataloaders(config: A1Config, tokenizer: Any) -> dict[str, DataLoader]:
    """Build seeded dataloaders. Only ``train`` shuffles; eval order is fixed.

    The train shuffle uses a generator seeded from ``training.seed`` so batch
    order is explicit and reproducible.
    """
    datasets = build_datasets(config, tokenizer)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        raise ValueError("tokenizer must expose a pad_token_id for collation")
    collator = CausalLMCollator(pad_token_id=int(pad_token_id))
    generator = dataloader_generator(config.training.seed)

    loaders: dict[str, DataLoader] = {}
    for name, dataset in datasets.items():
        is_train = name == "train"
        loaders[name] = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=is_train,
            generator=generator if is_train else None,
            collate_fn=collator,
            num_workers=config.data.num_workers,
            drop_last=False,
        )
    return loaders


__all__ = [
    "CausalLMCollator",
    "TokenizedTextDataset",
    "build_dataloaders",
    "build_datasets",
    "build_synthetic_split",
    "load_tokenizer",
]
