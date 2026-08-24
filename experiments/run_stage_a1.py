#!/usr/bin/env python
"""Stage A1 entry point — ordinary residual baseline (no PC, no symbolic head).

Usage::

    python experiments/run_stage_a1.py --config configs/stage_A/a1_lora_baseline.yaml

    # fast synthetic pipeline check, no downloads
    python experiments/run_stage_a1.py --config configs/stage_A/a1_smoke.yaml --smoke

    # rank sweep point, holding everything else fixed
    python experiments/run_stage_a1.py \
        --config configs/stage_A/a1_lora_baseline.yaml \
        --set adapter.rank=4 --set output.run_name=a1_rank4

All orchestration lives in :func:`neurosymbolic.training.pipeline.run_a1`; this
file is only argument parsing and reporting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurosymbolic.training.pipeline import (  # noqa: E402
    format_smoke_report,
    format_summary,
    run_a1,
    smoke_test_passed,
)
from neurosymbolic.utils.config import (  # noqa: E402
    ConfigError,
    apply_overrides,
    config_from_dict,
)
from neurosymbolic.utils.logging import configure_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage A1: frozen transformer + low-rank hidden-state residual."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "stage_A" / "a1_lora_baseline.yaml",
        help="Path to the A1 YAML config.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override a config value (repeatable), e.g. --set adapter.rank=16",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Print the A1 SMOKE TEST report and exit non-zero unless it passes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"error: config file not found: {config_path}", file=sys.stderr)
        return 2
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    try:
        config = config_from_dict(apply_overrides(raw, args.overrides))
    except ConfigError as exc:
        print(f"error: invalid configuration: {exc}", file=sys.stderr)
        return 2

    result = run_a1(config)

    if args.smoke:
        print(format_smoke_report(result))
        return 0 if smoke_test_passed(result) else 1

    print(format_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
