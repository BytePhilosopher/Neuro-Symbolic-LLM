"""Reproducible seeding for model init, adapter init and dataloader shuffling."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch (CPU + CUDA) RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def dataloader_generator(seed: int) -> torch.Generator:
    """Return a CPU generator dedicated to dataloader shuffling.

    Shuffle order is seeded explicitly from the run seed rather than left to
    torch's global RNG state, so batch order is stable across reruns even if
    the amount of RNG consumed during model construction changes.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a device spec, where ``auto`` prefers CUDA, then MPS, then CPU."""
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


__all__ = ["dataloader_generator", "resolve_device", "set_seed"]
