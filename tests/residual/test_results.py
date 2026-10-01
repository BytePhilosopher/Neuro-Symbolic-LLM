"""Unit tests for the run-artifact writer. No model needed."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from stages.results import RunWriter, default_run_name, load_params, run_metadata


def test_writes_all_artifacts(tmp_path: Path) -> None:
    params = {"layer_7": {"A": jnp.ones((4, 2)), "B": jnp.zeros((2, 4))}}
    with RunWriter(tmp_path, "run") as writer:
        writer.write_config({"training": {"steps": 3}, "domains": [{"name": "d"}]})
        writer.write_json("results.json", {"loss": jnp.asarray([[1.5]]), "x": None})
        writer.log("train", step=1, loss=jnp.float32(2.0), ids=np.arange(3))
        writer.log("eval", after_task=0)
        path = writer.save_params("task_0_d", params)

    run = tmp_path / "run"
    assert yaml.safe_load((run / "config.yaml").read_text()) == {
        "training": {"steps": 3},
        "domains": [{"name": "d"}],
    }
    assert json.loads((run / "results.json").read_text()) == {
        "loss": [[1.5]],
        "x": None,
    }
    lines = [
        json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()
    ]
    assert lines == [
        {"event": "train", "step": 1, "loss": 2.0, "ids": [0, 1, 2]},
        {"event": "eval", "after_task": 0},
    ]
    loaded = load_params(path)
    assert set(loaded) == {"layer_7"}
    np.testing.assert_array_equal(loaded["layer_7"]["A"], np.ones((4, 2)))
    np.testing.assert_array_equal(loaded["layer_7"]["B"], np.zeros((2, 4)))
    # Atomic writes leave no temp files behind.
    assert not [p for p in run.iterdir() if p.name.startswith(".")]


def test_refuses_existing_run_dir(tmp_path: Path) -> None:
    RunWriter(tmp_path, "run").close()
    with pytest.raises(FileExistsError):
        RunWriter(tmp_path, "run")


@pytest.mark.parametrize("name", ["", "a/b", "../x"])
def test_rejects_non_plain_run_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="plain directory name"):
        RunWriter(tmp_path, name)


def test_log_after_close_raises(tmp_path: Path) -> None:
    writer = RunWriter(tmp_path, "run")
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.log("train")


def test_metadata_and_run_name() -> None:
    meta = run_metadata()
    assert {"started_at", "git_commit", "git_dirty", "python", "versions"} <= set(meta)
    assert default_run_name("a1").startswith("a1-")
