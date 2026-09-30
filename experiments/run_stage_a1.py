"""Run Stage A1: train a hidden-state residual adapter on a frozen LM.

Orchestration only: load the substrate, build the configs and the adapter,
train with ``optax.adam`` over the adapter PyTree, evaluate, and check that
the base parameters are unchanged.

    python experiments/run_stage_a1.py --config configs/stage_A/a1_lora_baseline.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run as a script, so the repo root is not on sys.path by default.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frozenllm.substrate import FrozenSubstrate  # noqa: E402
from metrics.performance import (  # noqa: E402
    cross_entropy_loss,
    kl_to_base,
    perplexity,
)
from residual import AdapterParams, ResidualAdapter, ResidualConfig  # noqa: E402
from stages.stage_A.a1 import A1Config, a1_objective, base_logits  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage_A" / "a1_lora_baseline.yaml"
SECTIONS = ("adapter", "objective", "training")

log = logging.getLogger("stage_a1")


@dataclass(frozen=True)
class TrainingConfig:
    """Run settings; mirrors the ``training:`` YAML section."""

    model: str = "gpt2"
    dataset: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    seq_len: int = 128
    batch_size: int = 8
    steps: int = 200
    learning_rate: float = 1.0e-3
    eval_batches: int = 8
    log_every: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("seq_len", "batch_size", "eval_batches", "log_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.steps < 0:
            raise ValueError(f"steps must be >= 0, got {self.steps}")
        if not self.learning_rate > 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainingConfig:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"Unknown TrainingConfig keys: {unknown}. Allowed: {sorted(known)}."
            )
        return cls(**data)


def load_configs(
    path: Path,
) -> tuple[ResidualConfig, A1Config, TrainingConfig]:
    """Parse the YAML file into the three config dataclasses."""
    raw = yaml.safe_load(path.read_text()) or {}
    unknown = sorted(set(raw) - set(SECTIONS))
    if unknown:
        raise ValueError(f"Unknown config sections in {path}: {unknown}.")
    return (
        ResidualConfig.from_dict(raw.get("adapter") or {}),
        A1Config.from_dict(raw.get("objective") or {}),
        TrainingConfig.from_dict(raw.get("training") or {}),
    )


def load_token_blocks(
    substrate: FrozenSubstrate, cfg: TrainingConfig, split: str
) -> np.ndarray:
    """Tokenize a dataset split and cut it into ``[n, seq_len]`` blocks."""
    from datasets import load_dataset

    if substrate.tokenizer is None:
        raise ValueError(f"No tokenizer available for {cfg.model!r}.")
    texts = load_dataset(cfg.dataset, cfg.dataset_config, split=split)["text"]
    # WikiText lines keep their trailing newlines, so a plain join gives one
    # contiguous token stream; the ragged tail is dropped.
    ids = substrate.tokenizer("".join(texts), return_tensors="np")["input_ids"][0]
    n = len(ids) // cfg.seq_len
    if n == 0:
        raise ValueError(f"Split {split!r} is shorter than seq_len={cfg.seq_len}.")
    return np.asarray(ids[: n * cfg.seq_len], dtype=np.int32).reshape(n, cfg.seq_len)


def batches(blocks: np.ndarray, batch_size: int, seed: int) -> Iterator[jax.Array]:
    """Endless, seeded stream of shuffled batches."""
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(len(blocks))
        # The last partial batch is dropped to keep batch shapes static.
        for i in range(0, len(order) - batch_size + 1, batch_size):
            yield jnp.asarray(blocks[order[i : i + batch_size]])


def evaluate(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    params: AdapterParams,
    eval_batches: Sequence[jax.Array],
) -> dict[str, float]:
    """Mean base/adapted loss, perplexity and KL over fixed eval batches."""
    base_losses, losses, kls = [], [], []
    for ids in eval_batches:
        base = base_logits(substrate, ids)
        adapted = substrate.run_with_interception(
            ids, modify_fn=adapter.modify_fn(params), intercept_layers=adapter.layers
        ).logits
        base_losses.append(float(cross_entropy_loss(base, ids)))
        losses.append(float(cross_entropy_loss(adapted, ids)))
        kls.append(float(kl_to_base(base, adapted)))
    base_loss, loss = float(np.mean(base_losses)), float(np.mean(losses))
    return {
        "base_loss": base_loss,
        "base_ppl": float(perplexity(base_loss)),
        "loss": loss,
        "ppl": float(perplexity(loss)),
        "kl": float(np.mean(kls)),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args(argv)

    adapter_cfg, objective_cfg, train_cfg = load_configs(args.config)
    if args.rank is not None:
        adapter_cfg = replace(adapter_cfg, rank=args.rank)
    if args.model is not None:
        train_cfg = replace(train_cfg, model=args.model)
    if args.steps is not None:
        train_cfg = replace(train_cfg, steps=args.steps)
    log.info("adapter=%s", adapter_cfg)
    log.info("objective=%s", objective_cfg)
    log.info("training=%s", train_cfg)

    substrate = FrozenSubstrate(train_cfg.model)
    arch = substrate.architecture
    max_pos = arch.max_position_embeddings
    if max_pos is not None and train_cfg.seq_len > max_pos:
        raise ValueError(f"seq_len={train_cfg.seq_len} exceeds {max_pos} positions.")

    adapter = ResidualAdapter(adapter_cfg, arch)
    params = adapter.init_params()
    n_trainable = ResidualAdapter.num_params(params)
    n_frozen = sum(int(p.numel()) for p in substrate.params.values())
    log.info(
        "model=%s L=%d d=%d layers=%s rank=%d",
        train_cfg.model,
        arch.num_layers,
        arch.hidden_size,
        list(adapter.layers),
        adapter_cfg.rank,
    )
    log.info(
        "trainable=%d frozen=%d ratio=%.3e",
        n_trainable,
        n_frozen,
        n_trainable / n_frozen,
    )

    frozen = substrate.verify_frozen()
    if not frozen["params_unchanged"]:
        log.error("Substrate not frozen before training: %s", frozen)
        return 1

    train_blocks = load_token_blocks(substrate, train_cfg, "train")
    eval_blocks = load_token_blocks(substrate, train_cfg, "validation")
    # Fixed, unshuffled eval batches so eval@init and eval@end are comparable.
    n_eval = min(len(eval_blocks), train_cfg.eval_batches * train_cfg.batch_size)
    eval_set = [
        jnp.asarray(eval_blocks[i : i + train_cfg.batch_size])
        for i in range(0, n_eval, train_cfg.batch_size)
    ]
    log.info("train_blocks=%d eval_batches=%d", len(train_blocks), len(eval_set))
    log.info("eval@init %s", evaluate(substrate, adapter, params, eval_set))

    # The optimizer only ever sees the adapter PyTree; base weights are torch
    # tensors outside it and cannot be updated.
    optimizer = optax.adam(train_cfg.learning_rate)
    opt_state = optimizer.init(params)
    grad_fn = jax.value_and_grad(a1_objective, has_aux=True)
    stream = batches(train_blocks, train_cfg.batch_size, train_cfg.seed)

    for step in range(1, train_cfg.steps + 1):
        ids = next(stream)
        start = time.perf_counter()
        # No precomputed base: a1_objective runs the base forward itself, so each
        # step costs two substrate forwards.
        (_, terms), grads = grad_fn(
            params,
            substrate=substrate,
            adapter=adapter,
            config=objective_cfg,
            input_ids=ids,
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        # JAX dispatches asynchronously; wait so step_time measures the real work.
        jax.block_until_ready(params)
        step_time = time.perf_counter() - start
        if step % train_cfg.log_every == 0 or step in (1, train_cfg.steps):
            log.info(
                "step=%d total=%.4f task=%.4f ppl=%.2f kl=%.3e wd=%.3e time=%.3fs",
                step,
                float(terms["total"]),
                float(terms["task"]),
                float(perplexity(terms["task"])),
                float(terms["kl"]),
                float(terms["wd"]),
                step_time,
            )

    log.info("eval@end %s", evaluate(substrate, adapter, params, eval_set))

    frozen = substrate.verify_frozen()
    log.info("verify_frozen=%s", frozen)
    if not frozen["params_unchanged"]:
        log.error("Base parameters changed during training.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
