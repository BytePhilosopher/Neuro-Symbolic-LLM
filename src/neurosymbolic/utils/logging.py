"""Minimal logging + artifact writing helpers.

Deliberately thin: a single stdlib logger factory and JSON/CSV writers. No
global mutable state beyond the stdlib logging registry.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Install a single stream handler on the package logger (idempotent)."""
    logger = logging.getLogger("neurosymbolic")
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the package logger."""
    configure_logging()
    return logging.getLogger(f"neurosymbolic.{name}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if hasattr(value, "item") and getattr(value, "ndim", 0) == 0:
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a mapping as pretty-printed JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path | None:
    """Write flat rows as CSV. Returns ``None`` when there is nothing to write."""
    rows = [dict(row) for row in rows]
    if not rows:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonable(row.get(k)) for k in columns})
    return path


__all__ = ["configure_logging", "get_logger", "write_csv", "write_json"]
