"""Run Stage A1: train a hidden-state residual adapter on a frozen LM.

Orchestration only: load the substrate, build the configs and the adapter,
train with ``optax.adam`` over the adapter PyTree on a sequence of domains,
evaluate every domain after every task, write the run to disk, and check
that the base parameters are unchanged.

    python experiments/run_stage_a1.py --config configs/stage_A/a1_lora_baseline.yaml

A single-domain config is a sequence of length one. Artifacts are written to
``<output_dir>/<run_name>/``; see ``stages/results.py`` for the layout.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
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
    mean_forgetting,
    mean_seen_accuracy,
    perplexity,
    training_step_time,
)
from metrics.representation import loss_improvement  # noqa: E402
from residual import AdapterParams, ResidualAdapter, ResidualConfig  # noqa: E402
from stages.results import RunWriter, default_run_name, run_metadata  # noqa: E402
from stages.stage_A.a1 import A1Config, a1_objective, base_logits  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage_A" / "a1_lora_baseline.yaml"
SECTIONS = ("adapter", "objective", "training", "domains")
# Domain names become file names (params/task_<t>_<name>.npz).
DOMAIN_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")

log = logging.getLogger("stage_a1")


def _checked_kwargs(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            f"Unknown {cls.__name__} keys: {unknown}. Allowed: {sorted(known)}."
        )
    return dict(data)


@dataclass(frozen=True)
class TrainingConfig:
    """Run settings; mirrors the ``training:`` YAML section."""

    model: str = "gpt2"
    seq_len: int = 128
    batch_size: int = 8
    # Default steps per domain; a domain's own ``steps`` overrides it.
    steps: int = 200
    learning_rate: float = 1.0e-3
    eval_batches: int = 8
    log_every: int = 10
    seed: int = 0
    # Fresh Adam moments at each domain boundary, so stale moments from the
    # previous domain do not steer the first steps on the next one.
    reset_optimizer: bool = True
    output_dir: str = "runs"

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
        return cls(**_checked_kwargs(cls, data))


@dataclass(frozen=True)
class DomainConfig:
    """One domain of the sequence; an item of the ``domains:`` YAML list.

    Splits accept HF slicing, e.g. ``train[:20000]`` or ``test[:1000]+test[-1000:]``.
    """

    name: str
    dataset: str
    dataset_config: str | None = None
    text_field: str = "text"
    train_split: str = "train"
    eval_split: str = "validation"
    # Joins documents into one token stream. WikiText lines already end in
    # "\n", so "" keeps it contiguous; use "\n\n" for one-document-per-row sets.
    separator: str = ""
    steps: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not DOMAIN_NAME.match(self.name):
            raise ValueError(
                f"Domain name must match {DOMAIN_NAME.pattern}, got {self.name!r}"
            )
        if self.steps is not None and (
            not isinstance(self.steps, int) or self.steps < 0
        ):
            raise ValueError(f"Domain {self.name!r}: steps must be >= 0 or null.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DomainConfig:
        return cls(**_checked_kwargs(cls, data))


@dataclass(frozen=True)
class RunConfig:
    adapter: ResidualConfig
    objective: A1Config
    training: TrainingConfig
    domains: tuple[DomainConfig, ...]

    def to_dict(self) -> dict[str, Any]:
        """Same layout as the YAML file; ``load_configs`` reads it back."""
        return {
            "adapter": asdict(self.adapter),
            "objective": asdict(self.objective),
            "training": asdict(self.training),
            "domains": [asdict(d) for d in self.domains],
        }


def load_configs(path: Path) -> RunConfig:
    """Parse the YAML file into a ``RunConfig``."""
    raw = yaml.safe_load(path.read_text()) or {}
    unknown = sorted(set(raw) - set(SECTIONS))
    if unknown:
        raise ValueError(f"Unknown config sections in {path}: {unknown}.")
    domains_raw = raw.get("domains")
    if not isinstance(domains_raw, list) or not domains_raw:
        raise ValueError(f"{path}: 'domains' must be a non-empty list of domains.")
    domains = tuple(DomainConfig.from_dict(d) for d in domains_raw)
    names = [d.name for d in domains]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: domain names must be unique, got {names}.")
    return RunConfig(
        adapter=ResidualConfig.from_dict(raw.get("adapter") or {}),
        objective=A1Config.from_dict(raw.get("objective") or {}),
        training=TrainingConfig.from_dict(raw.get("training") or {}),
        domains=domains,
    )


@dataclass(frozen=True)
class Domain:
    """A domain with its data loaded and its step count resolved."""

    name: str
    train_blocks: np.ndarray
    eval_set: tuple[jax.Array, ...]
    steps: int


def load_token_blocks(
    substrate: FrozenSubstrate, domain: DomainConfig, split: str, seq_len: int
) -> np.ndarray:
    """Tokenize a dataset split and cut it into ``[n, seq_len]`` blocks."""
    from datasets import load_dataset

    if substrate.tokenizer is None:
        raise ValueError("The substrate has no tokenizer.")
    texts = load_dataset(domain.dataset, domain.dataset_config, split=split)[
        domain.text_field
    ]
    # One contiguous token stream; the ragged tail is dropped.
    ids = substrate.tokenizer(domain.separator.join(texts), return_tensors="np")[
        "input_ids"
    ][0]
    n = len(ids) // seq_len
    if n == 0:
        raise ValueError(
            f"Domain {domain.name!r} split {split!r} is shorter than seq_len={seq_len}."
        )
    # Annotated: numpy is opaque to mypy (follow_imports = "skip").
    blocks: np.ndarray = np.asarray(ids[: n * seq_len], dtype=np.int32)
    return blocks.reshape(n, seq_len)


def make_eval_set(
    blocks: np.ndarray, batch_size: int, eval_batches: int
) -> tuple[jax.Array, ...]:
    """Fixed, unshuffled eval batches, so every evaluation sees the same data."""
    n_eval = min(len(blocks), eval_batches * batch_size)
    return tuple(
        jnp.asarray(blocks[i : i + batch_size]) for i in range(0, n_eval, batch_size)
    )


def batches(blocks: np.ndarray, batch_size: int, seed: int) -> Iterator[jax.Array]:
    """Endless, seeded stream of shuffled batches."""
    if len(blocks) < batch_size:
        raise ValueError(f"{len(blocks)} blocks cannot fill batch_size={batch_size}.")
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


def summarize(
    names: Sequence[str],
    init: Mapping[str, Mapping[str, float]],
    rows: Sequence[Mapping[str, Mapping[str, float]]],
) -> dict[str, Any]:
    """Build the eval matrices and continual-learning summaries.

    Matrices are ``M[t][j]``: domain ``j`` evaluated after training task ``t``.
    Loss is the task metric, so forgetting uses ``higher_is_better=False``.
    """
    loss = np.array([[row[n]["loss"] for n in names] for row in rows])
    base_loss = [init[n]["base_loss"] for n in names]
    num_tasks = len(rows)
    return {
        "domains": list(names),
        "base_loss": base_loss,
        "loss": loss.tolist(),
        "ppl": [[row[n]["ppl"] for n in names] for row in rows],
        "kl": [[row[n]["kl"] for n in names] for row in rows],
        "improvement": [
            [
                float(loss_improvement(base_loss[j], loss[t, j]))
                for j in range(len(names))
            ]
            for t in range(num_tasks)
        ],
        "mean_seen_loss": [
            float(mean_seen_accuracy(loss, t)) for t in range(num_tasks)
        ],
        # Undefined at t=0 (no earlier task), stored as null.
        "mean_forgetting": [None]
        + [
            float(mean_forgetting(loss, t, higher_is_better=False))
            for t in range(1, num_tasks)
        ],
    }


def run_sequential(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    params: AdapterParams,
    domains: Sequence[Domain],
    objective: A1Config,
    training: TrainingConfig,
    writer: RunWriter,
) -> tuple[AdapterParams, dict[str, Any]]:
    """Train on ``domains`` in order, evaluating every domain after every task.

    Writes ``train``/``eval`` events to ``metrics.jsonl`` and the adapter after
    each task to ``params/task_<t>_<name>.npz``. Returns the final parameters
    and the summary from ``summarize`` plus step and token counts.
    """
    if not domains:
        raise ValueError("At least one domain is required.")
    names = [d.name for d in domains]

    def eval_all(after_task: int) -> dict[str, dict[str, float]]:
        row = {}
        for d in domains:
            row[d.name] = evaluate(substrate, adapter, params, d.eval_set)
            writer.log("eval", after_task=after_task, domain=d.name, **row[d.name])
        log.info("eval after_task=%d %s", after_task, _fmt_losses(row))
        return row

    # after_task=-1: before any training, where the adapted model equals the base.
    init = eval_all(-1)

    optimizer = optax.adam(training.learning_rate)
    grad_fn = jax.value_and_grad(a1_objective, has_aux=True)

    def train_step(
        params: AdapterParams, opt_state: optax.OptState, ids: jax.Array
    ) -> tuple[AdapterParams, optax.OptState, dict[str, jax.Array]]:
        # No precomputed base: a1_objective runs the base forward itself, so each
        # step costs two substrate forwards.
        (_, terms), grads = grad_fn(
            params,
            substrate=substrate,
            adapter=adapter,
            config=objective,
            input_ids=ids,
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, terms

    # The optimizer only ever sees the adapter PyTree; base weights are torch
    # tensors outside it and cannot be updated.
    opt_state = optimizer.init(params)
    rows: list[dict[str, dict[str, float]]] = []
    step_times: list[float] = []
    global_step = tokens = 0

    for t, domain in enumerate(domains):
        if t > 0 and training.reset_optimizer:
            opt_state = optimizer.init(params)
        log.info("task=%d domain=%s steps=%d", t, domain.name, domain.steps)
        # Per-task seed so domain t's batch order does not depend on earlier tasks.
        stream = batches(domain.train_blocks, training.batch_size, training.seed + t)
        for step in range(1, domain.steps + 1):
            ids = next(stream)
            (params, opt_state, terms), step_time = training_step_time(
                train_step, params, opt_state, ids
            )
            global_step += 1
            tokens += int(ids.size)
            step_times.append(step_time)
            scalars = {k: float(v) for k, v in terms.items()}
            writer.log(
                "train",
                task=t,
                domain=domain.name,
                step=step,
                global_step=global_step,
                tokens=tokens,
                step_time=step_time,
                ppl=float(perplexity(scalars["task"])),
                # Prefixed: the objective's "task" term would clash with the task index.
                **{f"loss_{k}": v for k, v in scalars.items()},
            )
            if step % training.log_every == 0 or step in (1, domain.steps):
                log.info(
                    "task=%d step=%d total=%.4f task_loss=%.4f kl=%.3e wd=%.3e "
                    "time=%.3fs",
                    t,
                    step,
                    scalars["total"],
                    scalars["task"],
                    scalars["kl"],
                    scalars["wd"],
                    step_time,
                )
        writer.save_params(f"task_{t}_{domain.name}", params)
        rows.append(eval_all(t))

    summary = summarize(names, init, rows)
    summary.update(
        steps=global_step,
        tokens=tokens,
        # The first step includes tracing; leave it out of the steady-state mean.
        mean_step_time=float(np.mean(step_times[1:])) if len(step_times) > 1 else None,
    )
    return params, summary


def _fmt_losses(row: Mapping[str, Mapping[str, float]]) -> str:
    return " ".join(f"{name}={r['loss']:.4f}" for name, r in row.items())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument(
        "--steps", type=int, default=None, help="Steps for every domain (smoke runs)."
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args(argv)


def apply_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    adapter, training, domains = cfg.adapter, cfg.training, cfg.domains
    if args.rank is not None:
        adapter = replace(adapter, rank=args.rank)
    if args.model is not None:
        training = replace(training, model=args.model)
    if args.output_dir is not None:
        training = replace(training, output_dir=args.output_dir)
    if args.steps is not None:
        training = replace(training, steps=args.steps)
        domains = tuple(replace(d, steps=args.steps) for d in domains)
    return replace(cfg, adapter=adapter, training=training, domains=domains)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args(argv)
    cfg = apply_overrides(load_configs(args.config), args)
    train_cfg = cfg.training
    log.info("adapter=%s", cfg.adapter)
    log.info("objective=%s", cfg.objective)
    log.info("training=%s", train_cfg)
    log.info("domains=%s", [d.name for d in cfg.domains])

    substrate = FrozenSubstrate(train_cfg.model)
    arch = substrate.architecture
    max_pos = arch.max_position_embeddings
    if max_pos is not None and train_cfg.seq_len > max_pos:
        raise ValueError(f"seq_len={train_cfg.seq_len} exceeds {max_pos} positions.")

    adapter = ResidualAdapter(cfg.adapter, arch)
    params = adapter.init_params()
    n_trainable = ResidualAdapter.num_params(params)
    n_frozen = sum(int(p.numel()) for p in substrate.params.values())
    log.info(
        "model=%s L=%d d=%d layers=%s rank=%d trainable=%d frozen=%d ratio=%.3e",
        train_cfg.model,
        arch.num_layers,
        arch.hidden_size,
        list(adapter.layers),
        cfg.adapter.rank,
        n_trainable,
        n_frozen,
        n_trainable / n_frozen,
    )

    frozen = substrate.verify_frozen()
    if not frozen["params_unchanged"]:
        log.error("Substrate not frozen before training: %s", frozen)
        return 1

    # Load all data before creating the run directory, so a bad dataset id
    # fails fast without leaving an empty run behind.
    domains = []
    for d in cfg.domains:
        train_blocks = load_token_blocks(substrate, d, d.train_split, train_cfg.seq_len)
        eval_blocks = load_token_blocks(substrate, d, d.eval_split, train_cfg.seq_len)
        domains.append(
            Domain(
                name=d.name,
                train_blocks=train_blocks,
                eval_set=make_eval_set(
                    eval_blocks, train_cfg.batch_size, train_cfg.eval_batches
                ),
                steps=train_cfg.steps if d.steps is None else d.steps,
            )
        )
        log.info(
            "domain=%s train_blocks=%d eval_batches=%d",
            d.name,
            len(train_blocks),
            len(domains[-1].eval_set),
        )

    run_name = args.run_name or default_run_name("a1")
    with RunWriter(train_cfg.output_dir, run_name) as writer:
        log.info("run_dir=%s", writer.dir)
        writer.write_config(cfg.to_dict())
        writer.write_json(
            "meta.json",
            {
                **run_metadata(),
                "stage": "A1",
                "model": train_cfg.model,
                "num_layers": arch.num_layers,
                "hidden_size": arch.hidden_size,
                "adapted_layers": list(adapter.layers),
                "trainable_params": n_trainable,
                "frozen_params": n_frozen,
            },
        )
        params, summary = run_sequential(
            substrate, adapter, params, domains, cfg.objective, train_cfg, writer
        )
        frozen = substrate.verify_frozen()
        summary["params_unchanged"] = frozen["params_unchanged"]
        writer.write_json("results.json", summary)

    log.info("mean_seen_loss=%s", summary["mean_seen_loss"])
    log.info("mean_forgetting=%s", summary["mean_forgetting"])
    log.info("verify_frozen=%s", frozen)
    if not frozen["params_unchanged"]:
        log.error("Base parameters changed during training.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
