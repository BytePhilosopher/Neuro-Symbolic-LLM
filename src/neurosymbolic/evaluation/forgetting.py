"""Sequential-task retention interface — a small extension point, not a system.

A1 trains on a single dataset, so nothing here runs in the A1 pipeline. What is
provided is the minimal reusable shape the research plan's forgetting analysis
needs:

1. evaluate on Task A,
2. adapt on Task B,
3. evaluate Task A again,
4. record retention / forgetting.

Extending to real sequential evaluation means passing more than one
:class:`SequentialTask` and calling :func:`evaluate_tasks` before and after the
adaptation phase. Deliberately kept thin so continual-learning machinery does
not dominate A1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader

from neurosymbolic.evaluation.metrics import EvalResult, evaluate_causal_lm
from neurosymbolic.models.base import ResidualAdaptedModel


@dataclass(frozen=True)
class SequentialTask:
    """A named evaluation task in a sequential protocol."""

    name: str
    dataloader: DataLoader
    max_batches: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def evaluate_tasks(
    model: ResidualAdaptedModel,
    tasks: list[SequentialTask],
    device: torch.device,
    *,
    adapters_enabled: bool = True,
    phase: str = "eval",
) -> dict[str, EvalResult]:
    """Evaluate the model on each task, returning ``{task_name: EvalResult}``."""
    return {
        task.name: evaluate_causal_lm(
            model,
            task.dataloader,
            device,
            condition=f"{phase}:{task.name}",
            adapters_enabled=adapters_enabled,
            max_batches=task.max_batches,
        )
        for task in tasks
    }


def compute_forgetting(
    before: dict[str, EvalResult],
    after: dict[str, EvalResult],
    *,
    lower_is_better: bool = True,
) -> dict[str, dict[str, float]]:
    """Per-task retention after adapting on a later task.

    ``forgetting`` is positive when the task got worse. ``retention`` is the
    ratio of after/before oriented so that 1.0 means "fully retained".
    """
    report: dict[str, dict[str, float]] = {}
    for name, before_result in before.items():
        after_result = after.get(name)
        if after_result is None:
            continue
        delta = after_result.loss - before_result.loss
        forgetting = delta if lower_is_better else -delta
        retention = (
            before_result.loss / after_result.loss
            if lower_is_better and after_result.loss
            else (
                after_result.loss / before_result.loss
                if before_result.loss
                else float("nan")
            )
        )
        report[name] = {
            "loss_before": before_result.loss,
            "loss_after": after_result.loss,
            "delta": delta,
            "forgetting": forgetting,
            "retention": retention,
        }
    return report


__all__ = ["SequentialTask", "compute_forgetting", "evaluate_tasks"]
