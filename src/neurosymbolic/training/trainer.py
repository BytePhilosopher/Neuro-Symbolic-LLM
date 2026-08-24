"""Ordinary-backprop trainer for Stage A1.

Per optimisation micro-step:

1. base logits — ``with model.adapters_disabled(): under torch.no_grad()``
2. adapted logits — ordinary forward with the hook active
3. ``L = L_task + lambda_kl * KL(base || adapted)``
4. ``L.backward()``, then ``AdamW.step()`` over adapter parameters only

Weight decay lives in the optimiser, never in the loss. No predictive coding, no
latent inference, no symbolic components — plain backpropagation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from neurosymbolic.evaluation.metrics import EvalResult, evaluate_causal_lm
from neurosymbolic.models.base import ResidualAdaptedModel
from neurosymbolic.training.losses import a1_loss, adapter_l2_squared
from neurosymbolic.utils.config import A1Config
from neurosymbolic.utils.logging import get_logger

LOGGER = get_logger("training.trainer")

_MODEL_INPUT_KEYS = ("input_ids", "attention_mask")


@dataclass
class TrainingHistory:
    """Step-level training record plus interim evaluations."""

    steps: list[dict[str, Any]] = field(default_factory=list)
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    total_steps: int = 0
    total_micro_batches: int = 0
    total_tokens: int = 0
    wall_clock_seconds: float = 0.0

    @property
    def initial_loss(self) -> float:
        return self.steps[0]["loss"] if self.steps else float("nan")

    @property
    def final_loss(self) -> float:
        return self.steps[-1]["loss"] if self.steps else float("nan")

    def summary(self) -> dict[str, Any]:
        tokens_per_second = (
            self.total_tokens / self.wall_clock_seconds
            if self.wall_clock_seconds
            else 0.0
        )
        steps_per_second = (
            self.total_steps / self.wall_clock_seconds
            if self.wall_clock_seconds
            else 0.0
        )
        return {
            "total_optimizer_steps": self.total_steps,
            "total_micro_batches": self.total_micro_batches,
            "total_train_tokens": self.total_tokens,
            "training_time_seconds": self.wall_clock_seconds,
            "throughput_tokens_per_second": tokens_per_second,
            "throughput_steps_per_second": steps_per_second,
            "initial_train_loss": self.initial_loss,
            "final_train_loss": self.final_loss,
        }


def build_optimizer(model: ResidualAdaptedModel, config: A1Config) -> Optimizer:
    """AdamW over adapter parameters only.

    ``weight_decay`` here *is* the ``lambda_wd ||phi||^2`` term of the A1
    objective; it is applied decoupled by AdamW rather than added to the loss.
    """
    parameters = model.adapter_parameters()
    if not parameters:
        raise ValueError(
            "no trainable adapter parameters found; adapter attachment failed"
        )
    base_ids = {id(p) for p in model.base_model.parameters()}
    leaked = [p for p in parameters if id(p) in base_ids]
    if leaked:
        raise ValueError(
            f"{len(leaked)} base parameter(s) leaked into the optimizer parameter group"
        )
    return AdamW(
        parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )


def build_scheduler(
    optimizer: Optimizer, total_steps: int, warmup_ratio: float
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then linear decay (ENGINEERING CHOICE)."""
    warmup_steps = max(0, int(round(warmup_ratio * total_steps)))

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        remaining = total_steps - warmup_steps
        if remaining <= 0:
            return 1.0
        progress = (step - warmup_steps) / remaining
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _model_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in batch.items() if k in _MODEL_INPUT_KEYS}


def resolve_total_steps(config: A1Config, batches_per_epoch: int) -> int:
    """Number of optimiser steps implied by epochs / max_steps / accumulation."""
    accumulation = config.training.gradient_accumulation_steps
    steps_per_epoch = max(1, batches_per_epoch // accumulation)
    planned = steps_per_epoch * config.training.epochs
    if config.training.max_steps is not None:
        planned = min(planned, config.training.max_steps)
    return max(1, planned)


class A1Trainer:
    """Trains ``phi`` with ordinary backprop under the A1 objective."""

    def __init__(
        self,
        model: ResidualAdaptedModel,
        config: A1Config,
        device: torch.device,
        *,
        optimizer: Optimizer | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = optimizer or build_optimizer(model, config)
        self.history = TrainingHistory()

    def _forward_pair(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Two separate forward passes: base (no grad, hooks off) then adapted."""
        inputs = _model_inputs(batch)
        with torch.no_grad(), self.model.adapters_disabled():
            base_logits = self.model(**inputs).logits.detach()
        adapted_logits = self.model(**inputs).logits
        return base_logits, adapted_logits

    def train(
        self,
        train_loader: DataLoader,
        *,
        eval_loader: DataLoader | None = None,
    ) -> TrainingHistory:
        """Run the A1 training loop and return its history."""
        training_config = self.config.training
        accumulation = training_config.gradient_accumulation_steps
        total_steps = resolve_total_steps(self.config, len(train_loader))
        scheduler = build_scheduler(
            self.optimizer, total_steps, training_config.warmup_ratio
        )

        self.model.verify_frozen()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        step = 0
        micro_index = 0
        started = time.perf_counter()
        stop = False

        for epoch in range(training_config.epochs):
            if stop:
                break
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                base_logits, adapted_logits = self._forward_pair(batch)
                loss_out = a1_loss(
                    base_logits,
                    adapted_logits,
                    batch["labels"],
                    lambda_kl=training_config.lambda_kl,
                    task=self.config.model.task,
                )
                (loss_out.total / accumulation).backward()

                self.history.total_micro_batches += 1
                self.history.total_tokens += loss_out.num_valid_tokens
                micro_index += 1

                if micro_index % accumulation != 0:
                    continue

                if training_config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.adapter_parameters(), training_config.max_grad_norm
                    )
                if (
                    training_config.freeze_check_every
                    and step % training_config.freeze_check_every == 0
                ):
                    self.model.assert_no_base_gradients()

                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                step += 1

                record = {
                    "step": step,
                    "epoch": epoch,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "adapter_l2_squared": float(
                        adapter_l2_squared(self.model.adapter_parameters()).item()
                    ),
                    **loss_out.as_dict(),
                }
                self.history.steps.append(record)
                if training_config.log_every and step % training_config.log_every == 0:
                    LOGGER.info(
                        "step %d/%d | loss %.4f | task %.4f | kl %.5f | lr %.2e",
                        step,
                        total_steps,
                        record["loss"],
                        record["task_loss"],
                        record["kl"],
                        record["learning_rate"],
                    )

                eval_every = self.config.evaluation.eval_every
                if eval_loader is not None and eval_every and step % eval_every == 0:
                    interim = self.evaluate(eval_loader, condition=f"a1@step{step}")
                    self.history.evaluations.append({"step": step, **interim.as_dict()})
                    LOGGER.info(
                        "step %d | val loss %.4f | ppl %.2f",
                        step,
                        interim.loss,
                        interim.perplexity,
                    )
                    self.model.train()

                if step >= total_steps:
                    stop = True
                    break

        self.history.wall_clock_seconds = time.perf_counter() - started
        self.history.total_steps = step
        self.model.verify_frozen()
        return self.history

    def evaluate(
        self,
        dataloader: DataLoader,
        *,
        condition: str = "a1",
        adapters_enabled: bool = True,
        max_batches: int | None = None,
    ) -> EvalResult:
        """Validation pass under one condition (BASE or A1)."""
        return evaluate_causal_lm(
            self.model,
            dataloader,
            self.device,
            condition=condition,
            adapters_enabled=adapters_enabled,
            max_batches=max_batches,
        )


__all__ = [
    "A1Trainer",
    "TrainingHistory",
    "build_optimizer",
    "build_scheduler",
    "resolve_total_steps",
]
