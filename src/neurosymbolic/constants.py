"""Constants shared across data, training and evaluation.

Kept in a leaf module with no intra-package imports so that data, training and
evaluation can all depend on it without creating an import cycle.
"""

from __future__ import annotations

#: Label value excluded from the task loss and from the KL term (torch default).
IGNORE_INDEX: int = -100

__all__ = ["IGNORE_INDEX"]
