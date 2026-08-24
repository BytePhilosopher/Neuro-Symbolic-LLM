"""End-to-end A1 smoke test: config -> ... -> evaluation -> checkpoint.

Runs the *same* :func:`run_a1` pipeline the CLI runs, on a tiny synthetic model
and dataset, and asserts every pipeline invariant the A1 spec lists.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from neurosymbolic.training.pipeline import (
    format_smoke_report,
    format_summary,
    run_a1,
    smoke_test_passed,
)
from neurosymbolic.utils.config import OutputConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    config = load_config(REPO_ROOT / "configs" / "stage_A" / "a1_smoke.yaml")
    output_dir = tmp_path_factory.mktemp("a1_smoke")
    config = type(config)(
        model=config.model,
        adapter=config.adapter,
        training=config.training,
        data=config.data,
        evaluation=config.evaluation,
        output=OutputConfig(
            directory=str(output_dir), run_name="a1_smoke", save_checkpoint=True
        ),
        hardware=config.hardware,
    )
    return run_a1(config, keep_model=True)


def test_pipeline_completes_and_trains(result) -> None:
    assert result.history.total_steps > 0
    assert result.history.total_micro_batches >= result.history.total_steps
    assert result.history.total_tokens > 0


def test_losses_are_finite(result) -> None:
    assert math.isfinite(result.history.initial_loss)
    assert math.isfinite(result.history.final_loss)
    for step in result.history.steps:
        assert math.isfinite(step["loss"])
        assert math.isfinite(step["task_loss"])
        assert math.isfinite(step["kl"])
        assert step["kl"] >= 0.0


def test_base_parameters_did_not_change(result) -> None:
    assert result.base_parameters_changed is False
    assert result.model is not None
    result.model.verify_frozen()


def test_adapter_parameters_did_change(result) -> None:
    assert result.adapter_parameters_changed is True


def test_both_evaluation_conditions_are_present(result) -> None:
    assert result.base_eval.condition == "base"
    assert result.a1_eval.condition == "a1"
    assert math.isfinite(result.base_eval.loss)
    assert math.isfinite(result.a1_eval.loss)
    assert result.base_eval.num_tokens > 0
    assert result.a1_eval.num_tokens == result.base_eval.num_tokens


def test_kl_drift_is_finite_and_non_negative(result) -> None:
    assert math.isfinite(result.kl_drift.mean_kl)
    assert result.kl_drift.mean_kl >= 0.0
    assert result.kl_drift.num_tokens > 0
    assert result.kl_drift.reduction == "token_weighted_mean"


def test_latency_measured_for_both_conditions(result) -> None:
    assert set(result.latency) == {"base", "a1"}
    for measurement in result.latency.values():
        assert measurement.median_ms > 0.0
        assert measurement.measure_iters > 0


def test_spectral_analysis_is_labelled_and_not_c_resid(result) -> None:
    assert result.spectral["analysis"] == "adapter_spectral_analysis"
    assert "C_resid" in result.spectral["note"]
    per_layer = result.spectral["per_layer"]
    assert per_layer
    for entry in per_layer.values():
        for matrix in ("A", "B"):
            assert entry[matrix]["shape"]
            assert math.isfinite(entry[matrix]["frobenius_norm"])
            assert entry[matrix][
                "cumulative_energy"
            ], "no energy ranks within matrix rank"
            # Ranks above the matrix's actual rank must be skipped.
            for key in entry[matrix]["cumulative_energy"]:
                assert (
                    int(key.removeprefix("top_"))
                    <= entry[matrix]["num_singular_values"]
                )


def test_checkpoint_written_and_loadable(result) -> None:
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.is_file()
    payload = torch.load(result.checkpoint_path, map_location="cpu")
    assert payload["stage"] == "A1"
    assert payload["adapter_state_dict"]
    assert result.model is not None
    result.model.load_adapters(result.checkpoint_path)


def test_metrics_json_written_with_expected_sections(result) -> None:
    metrics_path = result.output_dir / "metrics.json"
    assert metrics_path.is_file()
    payload = json.loads(metrics_path.read_text())
    for section in (
        "parameters",
        "training",
        "evaluation",
        "kl_drift",
        "residual",
        "adapter_spectral_analysis",
        "latency",
        "freezing",
    ):
        assert section in payload, f"metrics.json missing {section}"
    assert payload["parameters"]["adapter_parameters"] > 0
    assert payload["freezing"]["base_parameters_changed"] is False


def test_csv_artifacts_written(result) -> None:
    assert (result.output_dir / "training_steps.csv").is_file()
    assert (result.output_dir / "summary.csv").is_file()


def test_smoke_report_passes(result) -> None:
    report = format_smoke_report(result)
    assert "A1 SMOKE TEST" in report
    assert "Base parameters changed: NO" in report
    assert "Adapter parameters changed: YES" in report
    assert "RESULT: PASS" in report
    assert smoke_test_passed(result)


def test_summary_renders(result) -> None:
    summary = format_summary(result)
    assert "STAGE A1" in summary
    assert "BASE val loss" in summary
    assert "KL drift" in summary
