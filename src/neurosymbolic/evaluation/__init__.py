"""Evaluation: task metrics, KL drift, adapter spectra, compressibility, forgetting."""

from neurosymbolic.evaluation.compressibility import (
    RankRunSummary,
    compute_c_resid,
    compute_c_resid_sweep,
)
from neurosymbolic.evaluation.forgetting import (
    SequentialTask,
    compute_forgetting,
    evaluate_tasks,
)
from neurosymbolic.evaluation.kl_drift import KLDriftResult, evaluate_kl_drift
from neurosymbolic.evaluation.metrics import (
    EvalResult,
    LatencyResult,
    adapter_norm_metrics,
    benchmark_latency,
    evaluate_causal_lm,
    measure_residual_norm,
)
from neurosymbolic.evaluation.spectral_analysis import (
    analyze_adapter_spectra,
    analyze_matrix,
    effective_rank,
)

__all__ = [
    "EvalResult",
    "KLDriftResult",
    "LatencyResult",
    "RankRunSummary",
    "SequentialTask",
    "adapter_norm_metrics",
    "analyze_adapter_spectra",
    "analyze_matrix",
    "benchmark_latency",
    "compute_c_resid",
    "compute_c_resid_sweep",
    "compute_forgetting",
    "effective_rank",
    "evaluate_causal_lm",
    "evaluate_kl_drift",
    "evaluate_tasks",
    "measure_residual_norm",
]
