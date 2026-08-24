"""External metrics for Stage A1: loss, perplexity, latency, throughput, norms."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from neurosymbolic.models.base import ResidualAdaptedModel
from neurosymbolic.training.losses import (
    IGNORE_INDEX,
    causal_lm_task_loss,
    shift_for_causal_lm,
)
from neurosymbolic.utils.logging import get_logger

LOGGER = get_logger("evaluation.metrics")


@dataclass(frozen=True)
class EvalResult:
    """Validation metrics for one condition (BASE or A1)."""

    condition: str
    loss: float
    perplexity: float
    num_tokens: int
    num_examples: int
    num_batches: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _model_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Model kwargs only — ``labels`` are handled by our own loss functions."""
    return {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask")}


@torch.no_grad()
def evaluate_causal_lm(
    model: ResidualAdaptedModel,
    dataloader: DataLoader,
    device: torch.device,
    *,
    condition: str,
    adapters_enabled: bool = True,
    max_batches: int | None = None,
) -> EvalResult:
    """Token-weighted validation loss and perplexity under one condition.

    ``adapters_enabled=False`` gives the BASE condition (frozen model, no
    residual) via the adapter-disable context manager.
    """
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_examples = 0
    batches = 0
    try:
        for index, raw_batch in enumerate(dataloader):
            if max_batches is not None and index >= max_batches:
                break
            batch = _to_device(raw_batch, device)
            labels = batch["labels"]
            if adapters_enabled:
                outputs = model(**_model_inputs(batch))
            else:
                with model.adapters_disabled():
                    outputs = model(**_model_inputs(batch))
            _, shifted_labels = shift_for_causal_lm(outputs.logits, labels)
            num_valid = int((shifted_labels != IGNORE_INDEX).sum().item())
            if num_valid == 0:
                continue
            loss = causal_lm_task_loss(outputs.logits, labels)
            total_nll += float(loss.item()) * num_valid
            total_tokens += num_valid
            total_examples += int(labels.size(0))
            batches += 1
    finally:
        model.train(was_training)

    mean_loss = total_nll / total_tokens if total_tokens else float("nan")
    perplexity = (
        math.exp(mean_loss) if total_tokens and mean_loss < 700 else float("inf")
    )
    return EvalResult(
        condition=condition,
        loss=mean_loss,
        perplexity=perplexity,
        num_tokens=total_tokens,
        num_examples=total_examples,
        num_batches=batches,
    )


@dataclass(frozen=True)
class LatencyResult:
    """Inference latency for one condition, in milliseconds."""

    condition: str
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    batch_size: int
    sequence_length: int
    warmup_iters: int
    measure_iters: int
    device: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


@torch.no_grad()
def benchmark_latency(
    model: ResidualAdaptedModel,
    device: torch.device,
    *,
    condition: str,
    adapters_enabled: bool,
    vocab_size: int,
    batch_size: int = 1,
    sequence_length: int = 128,
    warmup_iters: int = 3,
    measure_iters: int = 10,
    seed: int = 0,
) -> LatencyResult:
    """Forward-pass latency under identical inputs for BASE and A1.

    The same synthetic input tensor is reused across conditions (fixed seed), so
    the only difference measured is the adapter's cost.
    """
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        0, max(2, vocab_size), (batch_size, sequence_length), generator=generator
    ).to(device)
    attention_mask = torch.ones_like(input_ids)

    was_training = model.training
    model.eval()

    def run_once() -> None:
        if adapters_enabled:
            model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            with model.adapters_disabled():
                model(input_ids=input_ids, attention_mask=attention_mask)

    try:
        for _ in range(warmup_iters):
            run_once()
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples: list[float] = []
        for _ in range(measure_iters):
            start = time.perf_counter()
            run_once()
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000.0)
    finally:
        model.train(was_training)

    return LatencyResult(
        condition=condition,
        mean_ms=statistics.fmean(samples),
        median_ms=statistics.median(samples),
        p95_ms=_percentile(samples, 0.95),
        min_ms=min(samples),
        max_ms=max(samples),
        batch_size=batch_size,
        sequence_length=sequence_length,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
        device=str(device),
    )


def adapter_norm_metrics(model: ResidualAdaptedModel) -> dict[str, Any]:
    """Residual-side metrics: adapter matrix norms, rank, parameter count."""
    per_layer: dict[str, Any] = {}
    for layer_index, adapter in model.adapters.adapters():
        per_layer[str(layer_index)] = {
            "a_frobenius_norm": float(adapter.A.weight.detach().float().norm().item()),
            "b_frobenius_norm": float(adapter.B.weight.detach().float().norm().item()),
            "a_shape": list(adapter.A.weight.shape),
            "b_shape": list(adapter.B.weight.shape),
            "rank": adapter.rank,
            "alpha": adapter.alpha,
            "scaling": adapter.scaling,
            "num_parameters": adapter.num_parameters(),
        }
    residual_norms = model.last_residual_norms()
    return {
        "adapter_rank": model.adapter_config.rank,
        "residual_parameter_count": model.adapters.num_parameters(),
        "per_layer": per_layer,
        "last_residual_norms": {str(k): v for k, v in residual_norms.items()},
        "last_residual_norm_total": (
            float(sum(residual_norms.values())) if residual_norms else 0.0
        ),
    }


@torch.no_grad()
def measure_residual_norm(
    model: ResidualAdaptedModel,
    dataloader: DataLoader,
    device: torch.device,
    *,
    max_batches: int = 4,
) -> dict[str, float]:
    """Mean per-token L2 norm of ``R_phi(h)`` over a fixed data subset."""
    was_training = model.training
    model.eval()
    sums: dict[int, float] = {}
    counts = 0
    try:
        for index, raw_batch in enumerate(dataloader):
            if index >= max_batches:
                break
            batch = _to_device(raw_batch, device)
            hidden_captures: dict[int, torch.Tensor] = {}
            handles = []
            for layer_index, adapter in model.adapters.adapters():
                handles.append(_capture_residual(adapter, layer_index, hidden_captures))
            try:
                model(**_model_inputs(batch))
            finally:
                for handle in handles:
                    handle.remove()
            mask = batch["attention_mask"].bool()
            for layer_index, residual in hidden_captures.items():
                token_norms = residual.float().norm(dim=-1)[mask]
                sums[layer_index] = sums.get(layer_index, 0.0) + float(
                    token_norms.sum().item()
                )
            counts += int(mask.sum().item())
    finally:
        model.train(was_training)
    if counts == 0:
        return {}
    return {f"layer_{k}": v / counts for k, v in sums.items()}


def _capture_residual(
    adapter: torch.nn.Module, layer_index: int, store: dict[int, torch.Tensor]
) -> Any:
    def hook(_module: torch.nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        store[layer_index] = output.detach()

    return adapter.register_forward_hook(hook)


__all__ = [
    "EvalResult",
    "LatencyResult",
    "adapter_norm_metrics",
    "benchmark_latency",
    "evaluate_causal_lm",
    "measure_residual_norm",
]
