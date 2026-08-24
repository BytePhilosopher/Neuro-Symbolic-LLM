"""Training: the A1 objective, the trainer, and the end-to-end pipeline.

Import submodules directly (``neurosymbolic.training.pipeline``,
``...trainer``, ``...losses``). This package intentionally re-exports nothing:
``pipeline`` depends on :mod:`neurosymbolic.data` and
:mod:`neurosymbolic.evaluation`, both of which depend on
:mod:`neurosymbolic.training.losses`, so eager re-exports here would form an
import cycle.
"""
