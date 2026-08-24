r"""The research plan's residual compressibility metric ``C_resid(D)``.

.. math::

    C_{resid}(D) = \frac{\Delta D(r_{small})}{\Delta D(r_{large})}

where :math:`\Delta D(r)` is the **external task-performance delta** achieved by
a residual of rank ``r`` relative to the frozen base on dataset ``D``. Computing
it therefore requires training at *multiple* ranks — it is only available after
the Section 21 rank sweep.

This is deliberately a separate module from
:mod:`neurosymbolic.evaluation.spectral_analysis`, which is a per-run
engineering diagnostic on a single adapter's matrices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RankRunSummary:
    """One point of the rank sweep: rank plus base/adapted validation loss."""

    rank: int
    base_loss: float
    adapted_loss: float

    @property
    def delta(self) -> float:
        """``Delta D(r)``: improvement over the frozen base (positive = better)."""
        return self.base_loss - self.adapted_loss


@dataclass(frozen=True)
class CResidResult:
    """``C_resid(D)`` for one (small, large) rank pair."""

    rank_small: int
    rank_large: int
    delta_small: float
    delta_large: float
    c_resid: float
    metric: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_c_resid(
    small: RankRunSummary,
    large: RankRunSummary,
    *,
    metric: str = "validation_loss_delta",
) -> CResidResult:
    """Ratio of the small-rank to the large-rank task-performance delta.

    ``C_resid`` near 1 means the large rank bought nothing the small rank did not
    already capture (the residual is compressible). Values near 0 mean capacity
    genuinely helped. If the large-rank delta is (near) zero the ratio is
    undefined and ``nan`` is returned rather than a misleading number.
    """
    if small.rank >= large.rank:
        raise ValueError(
            f"rank_small ({small.rank}) must be < rank_large ({large.rank})"
        )
    delta_large = large.delta
    c_resid = small.delta / delta_large if abs(delta_large) > 1e-12 else float("nan")
    return CResidResult(
        rank_small=small.rank,
        rank_large=large.rank,
        delta_small=small.delta,
        delta_large=delta_large,
        c_resid=c_resid,
        metric=metric,
    )


def compute_c_resid_sweep(
    runs: list[RankRunSummary], *, metric: str = "validation_loss_delta"
) -> dict[str, Any]:
    """Compute ``C_resid`` for every ascending rank pair in a sweep."""
    ordered = sorted(runs, key=lambda run: run.rank)
    if len(ordered) < 2:
        raise ValueError(
            "C_resid requires at least two ranks; run the rank sweep first"
        )
    pairs = [
        compute_c_resid(ordered[i], ordered[j], metric=metric).as_dict()
        for i in range(len(ordered))
        for j in range(i + 1, len(ordered))
    ]
    return {
        "metric": "C_resid(D)",
        "definition": "delta_D(r_small) / delta_D(r_large), delta_D(r) = base_loss - adapted_loss",
        "task_metric": metric,
        "runs": [
            {
                "rank": run.rank,
                "base_loss": run.base_loss,
                "adapted_loss": run.adapted_loss,
                "delta": run.delta,
            }
            for run in ordered
        ],
        "pairs": pairs,
    }


__all__ = ["CResidResult", "RankRunSummary", "compute_c_resid", "compute_c_resid_sweep"]
