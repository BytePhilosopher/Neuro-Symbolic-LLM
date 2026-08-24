"""Frozen pretrained transformer + hidden-state residual adapters.

The A1 model is::

    F(x) = F0(x) + R_phi(x)

``F0`` is a Hugging Face pretrained transformer with **every** parameter frozen.
``R_phi`` is one :class:`~neurosymbolic.models.residual_adapter.LowRankResidualAdapter`
per selected decoder layer, attached with a ``forward`` hook that rewrites the
layer's output hidden state::

    h_adapted = h_base + R_phi(h_base)

Two design points that are ENGINEERING CHOICES, not research-plan requirements:

* **Forward hooks** (rather than wrapping/replacing layer modules) keep the base
  module graph byte-identical, which makes "the base was not modified" trivially
  checkable.
* **Adapter enable/disable via a context manager** plus *two separate forward
  passes* is how base vs. adapted logits are obtained. No monkeypatching of HF
  internals to share compute — at A1 scale the second pass is cheap and
  correctness outranks the micro-optimisation.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from neurosymbolic.models.residual_adapter import AdapterBank, LowRankResidualAdapter
from neurosymbolic.utils.config import A1Config, AdapterConfig, ModelConfig
from neurosymbolic.utils.logging import get_logger

LOGGER = get_logger("models.base")

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Attribute paths where HF decoder-only architectures keep their layer stack.
_LAYER_PATHS: tuple[tuple[str, ...], ...] = (
    ("gpt_neox", "layers"),
    ("transformer", "h"),
    ("model", "layers"),
    ("model", "decoder", "layers"),
    ("decoder", "layers"),
    ("transformer", "layers"),
)


class FreezeViolationError(RuntimeError):
    """Raised when a base parameter is trainable, gets a gradient, or changes."""


@dataclass(frozen=True)
class ParameterCounts:
    """Parameter accounting for the A1 report."""

    base: int
    adapter: int

    @property
    def total(self) -> int:
        return self.base + self.adapter

    @property
    def trainable_percentage(self) -> float:
        return 100.0 * self.adapter / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "total_parameters": self.total,
            "base_parameters": self.base,
            "adapter_parameters": self.adapter,
            "trainable_percentage": self.trainable_percentage,
        }


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPES[name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported dtype {name!r}; expected {sorted(_DTYPES)}"
        ) from exc


def find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the decoder layer stack of a HF decoder-only model.

    Tries the known attribute paths first, then falls back to the largest
    ``nn.ModuleList`` in the module tree. Raises with an actionable message if
    nothing plausible is found.
    """
    for path in _LAYER_PATHS:
        cursor: Any = model
        for attribute in path:
            cursor = getattr(cursor, attribute, None)
            if cursor is None:
                break
        if isinstance(cursor, nn.ModuleList) and len(cursor) > 0:
            return cursor

    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.ModuleList) and len(module) > 0
    ]
    if not candidates:
        raise ValueError(
            f"could not locate a decoder layer stack on {type(model).__name__}; "
            "pass a decoder-only HF model or extend _LAYER_PATHS"
        )
    name, layers = max(candidates, key=lambda item: len(item[1]))
    LOGGER.warning(
        "Using fallback layer-stack discovery: %s (%d layers)", name, len(layers)
    )
    if not isinstance(layers, nn.ModuleList):  # pragma: no cover - defensive
        raise TypeError(
            f"expected an nn.ModuleList at {name}, got {type(layers).__name__}"
        )
    return layers


def _normalise_layer_indices(indices: Sequence[int], num_layers: int) -> list[int]:
    """Map possibly-negative layer indices onto ``[0, num_layers)``."""
    resolved: list[int] = []
    for index in indices:
        actual = index + num_layers if index < 0 else index
        if not 0 <= actual < num_layers:
            raise ValueError(
                f"target layer {index} is out of range for a model with {num_layers} layers"
            )
        if actual in resolved:
            raise ValueError(
                f"target layer {index} resolves to duplicate index {actual}"
            )
        resolved.append(actual)
    return sorted(resolved)


def build_synthetic_causal_lm(config: A1Config) -> nn.Module:
    """Construct a randomly initialised tiny GPTNeoX (no network access).

    TEMPORARY SMOKE-TEST SETTING / test helper. Used by the smoke config and the
    unit tests so neither needs to download the pinned pythia-160m checkpoint.
    """
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    spec = config.model.synthetic
    if spec is None:
        raise ValueError("model.synthetic must be set to build a synthetic model")
    hf_config: Any = GPTNeoXConfig(
        vocab_size=spec.vocab_size,
        hidden_size=spec.hidden_size,
        num_hidden_layers=spec.num_hidden_layers,
        num_attention_heads=spec.num_attention_heads,
        intermediate_size=spec.intermediate_size,
        max_position_embeddings=spec.max_position_embeddings,
        use_cache=False,
    )
    model: nn.Module = GPTNeoXForCausalLM(hf_config)
    return model


def load_base_model(
    model_config: ModelConfig, config: A1Config | None = None
) -> nn.Module:
    """Load the pretrained base model ``F0`` (or a synthetic tiny stand-in)."""
    if model_config.synthetic is not None:
        if config is None:
            raise ValueError("config is required to build a synthetic model")
        LOGGER.info("Building synthetic tiny model (smoke-test mode)")
        return build_synthetic_causal_lm(config)

    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

    dtype = resolve_dtype(model_config.dtype)
    LOGGER.info(
        "Loading base model %s (task=%s)", model_config.name_or_path, model_config.task
    )
    loaded: nn.Module
    if model_config.task == "causal_lm":
        loaded = AutoModelForCausalLM.from_pretrained(
            model_config.name_or_path, torch_dtype=dtype
        )
    else:
        loaded = AutoModelForSequenceClassification.from_pretrained(
            model_config.name_or_path, torch_dtype=dtype
        )
    return loaded


class ResidualAdaptedModel(nn.Module):
    """A frozen base model with trainable hidden-state residual adapters.

    Every base parameter is frozen in ``__init__`` and verified; only adapter
    parameters (``phi``) are trainable. Call the instance exactly like the
    wrapped HF model.
    """

    def __init__(
        self,
        base_model: nn.Module,
        adapter_config: AdapterConfig,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.adapter_config = adapter_config

        self._freeze_base()

        layers = find_decoder_layers(base_model)
        self.num_layers = len(layers)
        self.target_layers = _normalise_layer_indices(
            adapter_config.target_layers, self.num_layers
        )
        hidden_size = self._infer_hidden_size(base_model)
        self.hidden_size = hidden_size

        self.adapters = AdapterBank()
        self._handles: list[RemovableHandle] = []
        self._adapters_enabled = True
        self._last_residual_norms: dict[int, float] = {}

        for index in self.target_layers:
            adapter = LowRankResidualAdapter(
                hidden_size,
                adapter_config.rank,
                alpha=adapter_config.alpha,
                dropout=adapter_config.dropout,
                activation=adapter_config.activation,
            )
            adapter.to(dtype=self._base_dtype(), device=self._base_device())
            self.adapters.add(index, adapter)
            self._handles.append(
                layers[index].register_forward_hook(
                    self._make_hook(index), with_kwargs=False
                )
            )

        self.verify_frozen()
        LOGGER.info(
            "Attached %d residual adapter(s) at layer(s) %s (hidden_size=%d, rank=%d)",
            len(self.target_layers),
            self.target_layers,
            hidden_size,
            adapter_config.rank,
        )

    # ---------------------------------------------------------------- freezing

    def _freeze_base(self) -> None:
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)

    def verify_frozen(self) -> None:
        """Assert no base parameter is trainable. Raises :class:`FreezeViolationError`."""
        offenders = [
            name
            for name, param in self.base_model.named_parameters()
            if param.requires_grad
        ]
        if offenders:
            raise FreezeViolationError(
                f"{len(offenders)} base parameter(s) are trainable, expected none: "
                f"{offenders[:5]}"
            )
        untrainable = [
            name
            for name, param in self.adapters.named_parameters()
            if not param.requires_grad
        ]
        if untrainable:
            raise FreezeViolationError(
                f"adapter parameter(s) are not trainable: {untrainable[:5]}"
            )

    def assert_no_base_gradients(self) -> None:
        """Assert no base parameter has accumulated a gradient."""
        offenders = [
            name
            for name, param in self.base_model.named_parameters()
            if param.grad is not None and torch.any(param.grad != 0)
        ]
        if offenders:
            raise FreezeViolationError(
                f"base parameter(s) received gradients: {offenders[:5]}"
            )

    def snapshot_base_parameters(
        self, limit: int | None = None
    ) -> dict[str, torch.Tensor]:
        """Clone base parameters (or a deterministic subset) for change detection.

        ``limit`` caps the number of tensors snapshotted, which keeps the memory
        cost bounded for real-sized models while still covering embeddings, an
        attention block and the LM head.
        """
        names = [name for name, _ in self.base_model.named_parameters()]
        if limit is not None and limit < len(names):
            # Always keep the first and last tensors — the input embeddings and
            # the output head are where accidental base updates would show up
            # first — and stride across the middle for the rest.
            middle = names[1:-1]
            stride = max(1, len(middle) // max(1, limit - 2))
            names = [names[0], *middle[::stride][: limit - 2], names[-1]]
        selected = set(names)
        return {
            name: param.detach().clone()
            for name, param in self.base_model.named_parameters()
            if name in selected
        }

    def assert_base_unchanged(
        self, snapshot: dict[str, torch.Tensor], *, atol: float = 0.0
    ) -> None:
        """Assert every snapshotted base parameter still holds its original value."""
        current = dict(self.base_model.named_parameters())
        drifted: list[str] = []
        for name, original in snapshot.items():
            param = current.get(name)
            if param is None:
                drifted.append(f"{name} (missing)")
            elif not torch.allclose(param.detach(), original, atol=atol, rtol=0.0):
                drifted.append(name)
        if drifted:
            raise FreezeViolationError(
                f"{len(drifted)} base parameter(s) changed during training: {drifted[:5]}"
            )

    # ------------------------------------------------------------------- hooks

    def _make_hook(self, layer_index: int):
        """Build the forward hook that adds ``R_phi(h)`` to a layer's output."""

        def hook(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
        ) -> Any:
            if not self._adapters_enabled:
                return output
            adapter = self.adapters.get_adapter(layer_index)
            if isinstance(output, tuple):
                hidden = output[0]
                adapted = self._apply_adapter(adapter, layer_index, hidden)
                return (adapted, *output[1:])
            if isinstance(output, torch.Tensor):
                return self._apply_adapter(adapter, layer_index, output)
            # Some HF versions return a dataclass-like ModelOutput per layer.
            hidden = getattr(output, "hidden_states", None)
            if isinstance(hidden, torch.Tensor):
                output.hidden_states = self._apply_adapter(adapter, layer_index, hidden)
                return output
            raise TypeError(
                f"layer {layer_index} returned unsupported output type {type(output).__name__}; "
                "cannot attach a hidden-state residual adapter"
            )

        return hook

    def _apply_adapter(
        self,
        adapter: LowRankResidualAdapter,
        layer_index: int,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        residual: torch.Tensor = adapter(hidden.to(dtype=adapter.A.weight.dtype))
        self._last_residual_norms[layer_index] = float(
            residual.detach().float().norm().item()
        )
        adapted: torch.Tensor = hidden + residual.to(dtype=hidden.dtype)
        return adapted

    @contextmanager
    def adapters_disabled(self) -> Iterator[None]:
        """Run the *base* model: hooks stay registered but return output untouched.

        Used to obtain base logits (the BASE evaluation condition and the KL
        reference distribution) without removing/re-adding hooks.
        """
        previous = self._adapters_enabled
        self._adapters_enabled = False
        try:
            yield
        finally:
            self._adapters_enabled = previous

    @property
    def adapters_enabled(self) -> bool:
        return self._adapters_enabled

    def remove_hooks(self) -> None:
        """Detach all forward hooks, restoring the pristine base model."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # ------------------------------------------------------------------ compute

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model(*args, **kwargs)

    def base_forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass with adapters disabled and gradients detached."""
        with torch.no_grad(), self.adapters_disabled():
            return self.base_model(*args, **kwargs)

    # ------------------------------------------------------------------ metadata

    def adapter_parameters(self) -> list[nn.Parameter]:
        """The optimiser's *only* parameter group: ``phi``."""
        return [p for p in self.adapters.parameters() if p.requires_grad]

    def named_adapter_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [
            (name, p) for name, p in self.adapters.named_parameters() if p.requires_grad
        ]

    def parameter_counts(self) -> ParameterCounts:
        return ParameterCounts(
            base=sum(p.numel() for p in self.base_model.parameters()),
            adapter=self.adapters.num_parameters(),
        )

    def last_residual_norms(self) -> dict[int, float]:
        """Frobenius norm of the most recent residual tensor, per target layer."""
        return dict(self._last_residual_norms)

    def _base_dtype(self) -> torch.dtype:
        for parameter in self.base_model.parameters():
            return parameter.dtype
        return torch.float32

    def _base_device(self) -> torch.device:
        for parameter in self.base_model.parameters():
            return parameter.device
        return torch.device("cpu")

    @staticmethod
    def _infer_hidden_size(base_model: nn.Module) -> int:
        config = getattr(base_model, "config", None)
        for attribute in ("hidden_size", "n_embd", "d_model"):
            value = getattr(config, attribute, None)
            if isinstance(value, int):
                return value
        raise ValueError(
            f"could not infer hidden size from {type(base_model).__name__}.config"
        )

    # --------------------------------------------------------------- checkpoints

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            k: v.detach().cpu().clone() for k, v in self.adapters.state_dict().items()
        }

    def save_adapters(
        self, path: str | Path, metadata: dict[str, Any] | None = None
    ) -> Path:
        """Save *only* the trainable adapter weights plus reproduction metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "stage": "A1",
            "adapter_config": {
                "type": self.adapter_config.type,
                "rank": self.adapter_config.rank,
                "alpha": self.adapter_config.alpha,
                "dropout": self.adapter_config.dropout,
                "activation": self.adapter_config.activation,
                "target_layers": self.adapter_config.target_layers,
            },
            "resolved_target_layers": self.target_layers,
            "hidden_size": self.hidden_size,
            "adapter_state_dict": self.adapter_state_dict(),
            "metadata": metadata or {},
        }
        torch.save(payload, path)
        LOGGER.info("Saved adapter checkpoint to %s", path)
        return path

    def load_adapters(self, path: str | Path, *, strict: bool = True) -> dict[str, Any]:
        """Load adapter weights saved by :meth:`save_adapters`."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"adapter checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu")
        if payload.get("hidden_size") != self.hidden_size:
            raise ValueError(
                f"checkpoint hidden_size {payload.get('hidden_size')} != model {self.hidden_size}"
            )
        if strict and payload.get("resolved_target_layers") != self.target_layers:
            raise ValueError(
                f"checkpoint target layers {payload.get('resolved_target_layers')} "
                f"!= model target layers {self.target_layers}"
            )
        self.adapters.load_state_dict(payload["adapter_state_dict"], strict=strict)
        self.adapters.to(device=self._base_device(), dtype=self._base_dtype())
        metadata: dict[str, Any] = payload.get("metadata", {})
        return metadata


def build_a1_model(config: A1Config, device: torch.device) -> ResidualAdaptedModel:
    """Load ``F0``, freeze it, and attach the A1 residual adapters."""
    base_model = load_base_model(config.model, config)
    base_model.to(device)
    model = ResidualAdaptedModel(base_model, config.adapter)
    model.to(device)
    model.verify_frozen()
    return model


__all__ = [
    "FreezeViolationError",
    "ParameterCounts",
    "ResidualAdaptedModel",
    "build_a1_model",
    "build_synthetic_causal_lm",
    "find_decoder_layers",
    "load_base_model",
    "resolve_dtype",
]
