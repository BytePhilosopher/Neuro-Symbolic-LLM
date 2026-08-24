"""End-to-end Stage A1 pipeline.

Kept in the package (rather than inside ``experiments/run_stage_a1.py``) so the
CLI entry point and the smoke test exercise byte-identical code.

Order of operations mirrors the A1 specification: config -> seed -> tokenizer ->
frozen base -> adapters (forward hook) -> trainable-parameter verification ->
data -> optimiser (adapters only) -> train -> BASE/A1 evaluation -> KL drift ->
adapter spectral analysis -> latency -> checkpoint -> metrics.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from neurosymbolic import STAGE_A1, __version__
from neurosymbolic.data.datasets import build_dataloaders, load_tokenizer
from neurosymbolic.evaluation.kl_drift import KLDriftResult, evaluate_kl_drift
from neurosymbolic.evaluation.metrics import (
    EvalResult,
    LatencyResult,
    adapter_norm_metrics,
    benchmark_latency,
    evaluate_causal_lm,
    measure_residual_norm,
)
from neurosymbolic.evaluation.spectral_analysis import analyze_adapter_spectra
from neurosymbolic.models.base import ResidualAdaptedModel, build_a1_model
from neurosymbolic.training.trainer import A1Trainer, TrainingHistory
from neurosymbolic.utils.config import A1Config
from neurosymbolic.utils.logging import get_logger, write_csv, write_json
from neurosymbolic.utils.seed import resolve_device, set_seed

LOGGER = get_logger("training.pipeline")

_BASE_SNAPSHOT_TENSORS = 12


@dataclass
class A1RunResult:
    """Everything one A1 run produces."""

    config: A1Config
    metrics: dict[str, Any]
    history: TrainingHistory
    base_eval: EvalResult
    a1_eval: EvalResult
    kl_drift: KLDriftResult
    latency: dict[str, LatencyResult]
    spectral: dict[str, Any]
    output_dir: Path
    checkpoint_path: Path | None
    base_parameters_changed: bool
    adapter_parameters_changed: bool
    model: ResidualAdaptedModel | None = field(default=None, repr=False)


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "package_version": __version__,
    }


def _adapter_snapshot(model: ResidualAdaptedModel) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_adapter_parameters()}


def _any_changed(
    snapshot: dict[str, torch.Tensor], current: dict[str, torch.Tensor]
) -> bool:
    return any(
        not torch.equal(current[name].detach(), original)
        for name, original in snapshot.items()
        if name in current
    )


def run_a1(
    config: A1Config,
    *,
    keep_model: bool = False,
) -> A1RunResult:
    """Run the complete Stage A1 experiment described by ``config``."""
    device = resolve_device(config.hardware.device)
    set_seed(config.training.seed)
    LOGGER.info(
        "Stage %s | device=%s | seed=%d", STAGE_A1, device, config.training.seed
    )

    tokenizer = load_tokenizer(config)
    model = build_a1_model(config, device)
    counts = model.parameter_counts()
    LOGGER.info(
        "Parameters: base=%d adapter=%d (%.4f%% trainable)",
        counts.base,
        counts.adapter,
        counts.trainable_percentage,
    )

    loaders = build_dataloaders(config, tokenizer)
    if "train" not in loaders or "validation" not in loaders:
        raise ValueError("A1 requires both a train and a validation split")
    train_loader = loaders["train"]
    val_loader = loaders["validation"]

    base_snapshot = model.snapshot_base_parameters(limit=_BASE_SNAPSHOT_TENSORS)
    adapter_snapshot = _adapter_snapshot(model)

    # BASE condition: frozen pretrained model, adapters disabled.
    base_eval_result = evaluate_causal_lm(
        model, val_loader, device, condition="base", adapters_enabled=False
    )
    LOGGER.info(
        "BASE | val loss %.4f | ppl %.2f",
        base_eval_result.loss,
        base_eval_result.perplexity,
    )

    trainer = A1Trainer(model, config, device)
    history = trainer.train(train_loader, eval_loader=val_loader)

    a1_eval_result = trainer.evaluate(val_loader, condition="a1", adapters_enabled=True)
    LOGGER.info(
        "A1   | val loss %.4f | ppl %.2f",
        a1_eval_result.loss,
        a1_eval_result.perplexity,
    )

    kl_drift = evaluate_kl_drift(
        model, val_loader, device, max_batches=config.evaluation.kl_anchor_batches
    )
    LOGGER.info(
        "KL drift | mean %.6f over %d tokens", kl_drift.mean_kl, kl_drift.num_tokens
    )

    spectral = analyze_adapter_spectra(
        model, energy_ranks=config.evaluation.spectral_energy_ranks
    )

    vocab_size = int(getattr(model.base_model.config, "vocab_size", 128))
    latency = {
        condition: benchmark_latency(
            model,
            device,
            condition=condition,
            adapters_enabled=enabled,
            vocab_size=vocab_size,
            batch_size=config.evaluation.latency_batch_size,
            sequence_length=config.evaluation.latency_sequence_length,
            warmup_iters=config.evaluation.latency_warmup_iters,
            measure_iters=config.evaluation.latency_measure_iters,
            seed=config.training.seed,
        )
        for condition, enabled in (("base", False), ("a1", True))
    }

    residual_norms = measure_residual_norm(model, val_loader, device, max_batches=2)

    # Freezing verification: base untouched, adapters moved.
    # The flag is measured, not assumed; the assertion afterwards turns a
    # violation into a hard failure rather than a quietly-reported metric.
    base_changed = _any_changed(
        base_snapshot, dict(model.base_model.named_parameters())
    )
    adapter_changed = _any_changed(
        adapter_snapshot, dict(model.named_adapter_parameters())
    )
    model.assert_base_unchanged(base_snapshot)
    model.verify_frozen()

    output_dir = Path(config.output.directory) / config.output.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path: Path | None = None
    if config.output.save_checkpoint:
        checkpoint_path = model.save_adapters(
            output_dir / "adapter.pt",
            metadata={
                "stage": STAGE_A1,
                "run_name": config.output.run_name,
                "seed": config.training.seed,
                "model_name_or_path": config.model.name_or_path,
                "base_val_loss": base_eval_result.loss,
                "a1_val_loss": a1_eval_result.loss,
                "kl_drift": kl_drift.mean_kl,
            },
        )

    metrics: dict[str, Any] = {
        "stage": STAGE_A1,
        "run_name": config.output.run_name,
        "environment": _environment(),
        "device": str(device),
        "config": config.to_dict(),
        "parameters": counts.as_dict(),
        "training": history.summary(),
        "evaluation": {
            "base": base_eval_result.as_dict(),
            "a1": a1_eval_result.as_dict(),
            "improvement": {
                "validation_loss_delta": base_eval_result.loss - a1_eval_result.loss,
                "perplexity_delta": base_eval_result.perplexity
                - a1_eval_result.perplexity,
            },
            "interim": history.evaluations,
        },
        "kl_drift": kl_drift.as_dict(),
        "residual": {
            **adapter_norm_metrics(model),
            "mean_per_token_residual_norm": residual_norms,
        },
        "adapter_spectral_analysis": spectral,
        "latency": {name: result.as_dict() for name, result in latency.items()},
        "freezing": {
            "base_parameters_changed": base_changed,
            "adapter_parameters_changed": adapter_changed,
            "base_tensors_verified": len(base_snapshot),
        },
        "notes": {
            "c_resid": (
                "C_resid(D) is not computed per-run; it requires the rank sweep. "
                "See neurosymbolic.evaluation.compressibility."
            ),
        },
    }

    write_json(output_dir / "metrics.json", metrics)
    write_csv(output_dir / "training_steps.csv", history.steps)
    write_csv(
        output_dir / "summary.csv",
        [
            {
                "run_name": config.output.run_name,
                "rank": config.adapter.rank,
                "seed": config.training.seed,
                "base_val_loss": base_eval_result.loss,
                "a1_val_loss": a1_eval_result.loss,
                "base_perplexity": base_eval_result.perplexity,
                "a1_perplexity": a1_eval_result.perplexity,
                "validation_loss_delta": base_eval_result.loss - a1_eval_result.loss,
                "kl_drift": kl_drift.mean_kl,
                "adapter_parameters": counts.adapter,
                "trainable_percentage": counts.trainable_percentage,
                "base_latency_median_ms": latency["base"].median_ms,
                "a1_latency_median_ms": latency["a1"].median_ms,
                "training_time_seconds": history.wall_clock_seconds,
            }
        ],
    )
    LOGGER.info("Wrote metrics to %s", output_dir / "metrics.json")

    return A1RunResult(
        config=config,
        metrics=metrics,
        history=history,
        base_eval=base_eval_result,
        a1_eval=a1_eval_result,
        kl_drift=kl_drift,
        latency=latency,
        spectral=spectral,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        base_parameters_changed=base_changed,
        adapter_parameters_changed=adapter_changed,
        model=model if keep_model else None,
    )


def format_summary(result: A1RunResult) -> str:
    """Concise human-readable experiment summary."""
    counts = result.metrics["parameters"]
    training = result.metrics["training"]
    lines = [
        "",
        "=" * 62,
        f"STAGE A1 — {result.config.output.run_name}",
        "=" * 62,
        f"  model              : {result.config.model.name_or_path}",
        f"  dataset            : {result.config.data.dataset}"
        f"/{result.config.data.dataset_config}",
        f"  adapter            : rank={result.config.adapter.rank} "
        f"alpha={result.config.adapter.alpha} "
        f"layers={result.model.target_layers if result.model else result.config.adapter.target_layers}",
        f"  lambda_kl          : {result.config.training.lambda_kl}",
        "",
        f"  base parameters    : {counts['base_parameters']:,}",
        f"  adapter parameters : {counts['adapter_parameters']:,}",
        f"  trainable          : {counts['trainable_percentage']:.4f}%",
        "",
        f"  optimizer steps    : {training['total_optimizer_steps']}",
        f"  training time      : {training['training_time_seconds']:.1f}s",
        f"  throughput         : {training['throughput_tokens_per_second']:.0f} tok/s",
        f"  train loss         : {training['initial_train_loss']:.4f}"
        f" -> {training['final_train_loss']:.4f}",
        "",
        f"  BASE val loss      : {result.base_eval.loss:.4f} "
        f"(ppl {result.base_eval.perplexity:.2f})",
        f"  A1   val loss      : {result.a1_eval.loss:.4f} "
        f"(ppl {result.a1_eval.perplexity:.2f})",
        f"  improvement        : {result.base_eval.loss - result.a1_eval.loss:+.4f} nats/token",
        f"  KL drift           : {result.kl_drift.mean_kl:.6f} "
        f"({result.kl_drift.num_tokens} tokens)",
        "",
        f"  latency BASE       : {result.latency['base'].median_ms:.2f} ms (median)",
        f"  latency A1         : {result.latency['a1'].median_ms:.2f} ms (median)",
        "",
        f"  base params changed    : {'YES' if result.base_parameters_changed else 'NO'}",
        f"  adapter params changed : "
        f"{'YES' if result.adapter_parameters_changed else 'NO'}",
        "",
        f"  artifacts          : {result.output_dir}",
        "=" * 62,
        "",
    ]
    return "\n".join(lines)


def format_smoke_report(result: A1RunResult) -> str:
    """The A1 SMOKE TEST report block."""
    counts = result.metrics["parameters"]
    passed = (
        not result.base_parameters_changed
        and result.adapter_parameters_changed
        and _finite(result.history.initial_loss)
        and _finite(result.history.final_loss)
        and _finite(result.kl_drift.mean_kl)
        and result.checkpoint_path is not None
    )
    return "\n".join(
        [
            "",
            "A1 SMOKE TEST",
            "",
            f"    Base parameters: {counts['base_parameters']:,}",
            f"    Adapter parameters: {counts['adapter_parameters']:,}",
            f"    Trainable ratio: {counts['trainable_percentage']:.4f}%",
            "",
            f"    Base parameters changed: "
            f"{'YES' if result.base_parameters_changed else 'NO'}",
            f"    Adapter parameters changed: "
            f"{'YES' if result.adapter_parameters_changed else 'NO'}",
            "",
            f"    Initial loss: {result.history.initial_loss:.6f}",
            f"    Final loss: {result.history.final_loss:.6f}",
            f"    KL drift: {result.kl_drift.mean_kl:.6f}",
            "",
            f"    RESULT: {'PASS' if passed else 'FAIL'}",
            "",
        ]
    )


def _finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


def smoke_test_passed(result: A1RunResult) -> bool:
    """Whether the smoke run satisfied every pipeline invariant."""
    return "RESULT: PASS" in format_smoke_report(result)


__all__ = [
    "A1RunResult",
    "format_smoke_report",
    "format_summary",
    "run_a1",
    "smoke_test_passed",
]
