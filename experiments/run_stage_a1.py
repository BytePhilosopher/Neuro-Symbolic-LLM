"""Run Stage A1: train a hidden-state residual adapter on a frozen LM.

Wiring only: parse arguments, load the substrate, build the configs and the
adapter, then hand the A1 objective to the shared sequential loop in
``stages/common.py`` (train with ``optax.adam`` over the adapter PyTree on a
sequence of domains, evaluating every domain after every task). Writes the
run to disk and checks that the base parameters are unchanged.

    python experiments/run_stage_a1.py --config configs/stage_A/a1_lora_baseline.yaml

A single-domain config is a sequence of length one. Artifacts are written to
``<output_dir>/<run_name>/``; see ``stages/results.py`` for the layout.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run as a script, so the repo root is not on sys.path by default.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frozenllm.substrate import FrozenSubstrate  # noqa: E402
from residual import ResidualAdapter  # noqa: E402
from stages.common import (  # noqa: E402
    Domain,
    RunConfig,
    load_configs,
    load_token_blocks,
    make_eval_set,
    run_sequential,
)
from stages.results import RunWriter, default_run_name, run_metadata  # noqa: E402
from stages.stage_A.a1 import A1Config, a1_objective  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage_A" / "a1_lora_baseline.yaml"

log = logging.getLogger("stage_a1")


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
    cfg = apply_overrides(load_configs(args.config, A1Config), args)
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
            substrate,
            adapter,
            params,
            domains,
            a1_objective,
            cfg.objective,
            train_cfg,
            writer,
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
