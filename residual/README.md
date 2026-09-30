# Neuro-Symbolic-LLM — Hidden-State Residual Adapter

A trainable low-rank **hidden-state residual** inserted at the late blocks of a frozen `FrozenSubstrate` (see [`frozenllm/README.md`](../frozenllm/README.md)). It is the only trainable component in Stage A1. It is **not** weight-level LoRA: it changes block outputs, never the base weights ($\nabla \theta_0 = 0$).

Provides late-layer selection with strict bound validation, zero-residual initialisation (the adapted model equals the base at step 0), a parameter PyTree independent of `substrate.params`, and a `modify_fn` that plugs into `FrozenSubstrate.run_with_interception`.

---

## Table of Contents

1. [Architectural Overview](#architectural-overview)
2. [Implementation Design & Data Flow](#implementation-design--data-flow)
3. [File Breakdown](#file-breakdown)
4. [Test Suite Documentation](#test-suite-documentation)
5. [Installation & Requirements](#installation--requirements)
6. [How to Run](#how-to-run)
   - [Run the Stage A1 runner](#1-run-the-stage-a1-runner)
   - [Run the test suites](#2-run-tests)
   - [Run pre-commit / lint checks](#3-code-quality-checks)
   - [Quickstart: use ResidualAdapter in Python](#4-quickstart-python-example)
7. [Core Architectural Guarantees](#core-architectural-guarantees)

---

## Architectural Overview

1. **Residual Map:** At each adapted block $l$, the frozen block output $h_l$ (row vectors of size $d$) is replaced by

   $$\tilde h_l = h_l + s \cdot \sigma(h_l A_l)\, B_l, \qquad A_l \in \mathbb{R}^{d \times r},\ B_l \in \mathbb{R}^{r \times d}$$

   with $s = \alpha / r$, or $1.0$ when `alpha` is `None` (`ResidualConfig.scale`).
2. **Activations:** $\sigma \in$ {`identity` (default), `gelu`, `relu`, `tanh`} (`ACTIVATIONS`). With `identity`, the residual is a sum of $r$ rank-one read/write atoms.
3. **Precision:** The residual is computed in float32 and cast back to the dtype of $h_l$ before the addition.
4. **Late-Layer Selection:** With $L$ = `architecture.num_layers` (blocks $0 \dots L-1$), the adapted blocks are the late layers strictly after the midpoint, $(L/\!/2,\ L-2]$. Block $L-1$ is never adapted.

   | $L$ | $L/\!/2$ | adapted blocks |
   |---|---|---|
   | 12 | 6 | 7 … 10 |
   | 24 | 12 | 13 … 22 |
   | 32 | 16 | 17 … 30 |

   `late_start` / `late_end` replace the defaults $L/\!/2 + 1$ and $L-2$ (both inclusive). `resolve_late_layers` raises `ValueError` if $L < 4$ or if $L/\!/2 <$ `late_start` $\le$ `late_end` $\le L-2$ does not hold; the message gives the depth, the requested bounds and the valid range. Bounds are never clamped.
5. **Zero-Residual Initialisation:** $A_l \sim \mathcal{N}(0, 1/d)$ (std $1/\sqrt{d}$, overridable with `init_std`) and $B_l = 0$, so the adapted model equals the base model at step 0. `init_params()` is deterministic in `seed`.
6. **Independent Parameter PyTree:** `{"layer_7": {"A": f32[d, r], "B": f32[r, d]}, ..., "layer_10": {...}}`, keyed by `layer_key(idx)`. Size is $2 \cdot d \cdot r \cdot |\text{layers}|$; for GPT-2 small ($d = 768$, $r = 16$, blocks 7…10) that is 98,304.
7. **Single Integration Point:** The adapter is built from `substrate.architecture` only. `modify_fn(params)` returns a substrate `ModifyFn` `(hidden, layer_idx) -> hidden`; blocks without an entry pass through unchanged.

---

## Implementation Design & Data Flow

```text
ResidualConfig (adapter: section of a1_lora_baseline.yaml, via from_dict)
     │
     ▼
ResidualAdapter(config, substrate.architecture)
     │ resolve_late_layers(L, late_start, late_end) ─► validate_interception_layers
     │ -> adapter.layers, e.g. (7, 8, 9, 10)
     ▼
init_params() -> {"layer_l": {"A": N(0, init_std²), "B": 0}}   (independent of θ0)
     │
     ▼
substrate.run_with_interception(ids,
        modify_fn=adapter.modify_fn(params), intercept_layers=adapter.layers)
     │
     ├── Blocks 0 … l-1 (frozen)
     ▼
Block l output h_l ──[hook]──► intermediates[l] = h_l (pristine)
     │
     ▼ apply(params, h_l, l)
     ├── 1. params.get(layer_key(l)); no entry -> return h_l unchanged
     ├── 2. residual: s · σ(h_l.astype(float32) @ A) @ B
     └── 3. h̃_l = h_l + residual.astype(h_l.dtype)
     │
     ▼
Blocks l+1 … L-1 (frozen, consume h̃_l) -> ln_f -> LM head -> logits
     │
     ▼
loss (e.g. a1_objective) ─► jax.grad w.r.t. params only ─► optax.adam(params)
```

---

## File Breakdown

### Adapter (`residual/`)

- **`__init__.py`**: Re-exports the public API: `ResidualConfig`, `resolve_late_layers`, `ResidualAdapter`, `layer_key`, `AdapterParams`, `ACTIVATIONS`.
- **`adapter.py`**: The implementation.

| Name | Signature / description |
|---|---|
| `ResidualConfig` | `ResidualConfig(rank: int = 16, activation: str = "identity", alpha: float \| None = None, init_std: float \| None = None, late_start: int \| None = None, late_end: int \| None = None, seed: int = 0)`. Frozen dataclass, validated on construction; `from_dict(data)` rejects unknown keys; `scale` is $s$. |
| `resolve_late_layers` | `resolve_late_layers(num_layers: int, late_start: int \| None = None, late_end: int \| None = None) -> tuple[int, ...]` |
| `ResidualAdapter` | `ResidualAdapter(config: ResidualConfig, architecture: Architecture)`; attributes `layers`, `hidden_size`, `scale`, `init_std`. |
| `.init_params` | `init_params() -> AdapterParams` |
| `.residual` | `residual(layer_params: Mapping[str, jax.Array], h: jax.Array) -> jax.Array`: float32, not cast back. |
| `.apply` | `apply(params: AdapterParams, h: jax.Array, layer_idx: int) -> jax.Array` |
| `.modify_fn` | `modify_fn(params: AdapterParams) -> Callable[[jax.Array, int], jax.Array]` |
| `.num_params` | `num_params(params: AdapterParams) -> int` (static method) |
| `.l2_norm_sq` | `l2_norm_sq(params: AdapterParams) -> jax.Array` (static method): $\lVert\phi\rVert^2$. |
| `layer_key` | `layer_key(layer_idx: int) -> str`: `f"layer_{layer_idx}"`. |
| `AdapterParams` | `dict[str, dict[str, jax.Array]]` |
| `ACTIVATIONS` | Read-only mapping of activation name to function. |

### Related Stage A1 code

- **[`stages/stage_A/a1.py`](../stages/stage_A/a1.py)**: `A1Config` and `a1_objective` (task CE + KL to base + weight decay).
- **[`metrics/performance.py`](../metrics/performance.py)**: `cross_entropy_loss`, differentiable `kl_to_base`, `perplexity`.
- **[`experiments/run_stage_a1.py`](../experiments/run_stage_a1.py)**: training runner (`optax.adam` over the adapter PyTree only).
- **[`configs/stage_A/a1_lora_baseline.yaml`](../configs/stage_A/a1_lora_baseline.yaml)**: `adapter:`, `objective:`, `training:` sections.

---

## Test Suite Documentation

| Test File | Focus Areas |
|---|---|
| `tests/residual/test_adapter.py` | Layer resolution for $L$ = 12, 24, 32, explicit bounds and every invalid-bound case; parameter shapes and keys; $B = 0$ and std of $A$ at init; `init_std` override and seeding; `num_params` $= 2dr\,|\text{layers}|$; `l2_norm_sq`; zero residual at init; untouched non-adapted layers; residual formula, rank-one atoms and float32 cast-back; config validation and unknown-key rejection. No substrate needed. |
| `tests/stage_A/test_a1_training.py` | On the 12-block GPT-2 and Pythia fixtures: adapted logits equal base at init (`atol=1e-5`); pristine intermediates; $\partial L/\partial B \ne 0$ and $\partial L/\partial A = 0$ at init, $\partial L/\partial A \ne 0$ once $B \ne 0$; finite-difference check; no gradient on substrate params; objective terms; optimizer state holds only adapter leaves; after training, loss decreases, KL > 0, $B \ne 0$ and `params_unchanged()` is `True`; `A1Config` and shipped-YAML validation. |

---

## Installation & Requirements

The adapter has no dependencies beyond the project's (`jax`, plus `frozenllm` for `Architecture` and `validate_interception_layers`). Install the whole package from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

`torch>=2.6` has no wheels for Intel macOS; there, use the CPU image built from the repository `Dockerfile`:

```bash
docker build -t neuro-symbolic-llm .
docker run --rm -it -v "$PWD":/app neuro-symbolic-llm
```

The install is editable. After changing the package list in `pyproject.toml`, rerun `pip install -e ".[dev]"` (or rebuild the image) so that `residual` is importable outside the repository root.

---

## How to Run

### 1. Run the Stage A1 Runner

Trains the adapter on WikiText-2 over frozen GPT-2 (downloads the model and dataset on first run):

```bash
python experiments/run_stage_a1.py --config configs/stage_A/a1_lora_baseline.yaml

# Override the rank and number of steps (--model is also accepted)
python experiments/run_stage_a1.py --config configs/stage_A/a1_lora_baseline.yaml --rank 8 --steps 20
```

The runner logs the resolved layers, trainable/frozen parameter counts, loss terms, KL, perplexity and step time. It exits non-zero if `verify_frozen()` reports changed base parameters.

The step is not jitted, so CPU training is slow. With the shipped config (`batch_size: 8`, `seq_len: 128`), one step took about 3 minutes in a 3.8 GiB Docker VM, and the process was then killed, most likely out of memory. A copy of the config with `batch_size: 2` and `seq_len: 64` completed 20 steps in the same environment.

### 2. Run Tests

```bash
# Adapter unit tests and Stage A1 integration tests
pytest tests/residual/ tests/stage_A/ -v
```

### 3. Code Quality Checks

```bash
pre-commit run --all-files
```

### 4. Quickstart: Python Example

```python
import jax
import jax.numpy as jnp
from transformers import GPT2Config, GPT2LMHeadModel

from frozenllm.substrate import FrozenSubstrate
from residual import ResidualAdapter, ResidualConfig

# 1. Frozen substrate: tiny 12-block GPT-2 (the test fixture config).
#    FrozenSubstrate("gpt2") loads the real checkpoint instead.
cfg = GPT2Config(n_layer=12, n_head=4, n_embd=32, n_positions=32, vocab_size=64,
                 bos_token_id=1, eos_token_id=2)
substrate = FrozenSubstrate(GPT2LMHeadModel(cfg).eval(), config=cfg)

# 2. Adapter on the late layers, zero residual at init
adapter = ResidualAdapter(ResidualConfig(rank=4), substrate.architecture)
params = adapter.init_params()
print(adapter.layers)  # (7, 8, 9, 10)
ids = jnp.arange(9, dtype=jnp.int32)[None, :]  # [batch, seq_len]

# 3. Loss through the frozen substrate, as a function of the adapter only
def loss_fn(params):
    out = substrate.run_with_interception(
        ids, modify_fn=adapter.modify_fn(params), intercept_layers=adapter.layers
    )
    return substrate.compute_loss(out.logits, ids)

# 4. Gradients w.r.t. the adapter. At init dL/dB != 0 and dL/dA == 0 (B = 0).
loss, grads = jax.value_and_grad(loss_fn)(params)

# 5. Base parameters are untouched
assert substrate.params_unchanged()
```

---

## Core Architectural Guarantees

1. **Frozen Base ($\nabla \theta_0 = 0$):** The adapter holds no reference to `substrate.params`. They receive no gradient, the optimizer state contains only adapter leaves, and `params_unchanged()` is `True` after training (`test_substrate_params_receive_no_gradient`, `test_optimizer_state_holds_only_adapter_leaves`, `test_training_moves_only_the_adapter`).
2. **Pristine Intermediates:** `ForwardResult.hidden_state(l)` is $h_l$ before the residual (`test_intermediates_are_pristine`).
3. **Zero-Residual Start:** At init the adapted logits equal the base logits (`atol=1e-5`) and blocks outside `adapter.layers` are untouched (`test_initial_logits_equal_base`, `test_zero_residual_at_init`, `test_non_adapted_layers_untouched`).
4. **No Silent Layer Changes:** Invalid depths or bounds raise `ValueError`; they are never clamped or skipped (`test_invalid_bounds_raise`, `test_too_shallow_model_raises`).
5. **Strict Configuration:** `ResidualConfig.from_dict` rejects unknown keys, so the YAML `adapter:` section must mirror the dataclass (`test_config_rejects_unknown_keys`, `test_shipped_yaml_loads_and_rejects_unknown_keys`).

Out of scope for this package: predictive coding (Stages A2/A3), FabricPC, symbolic heads (Stage A4) and weight-level LoRA. Design rationale: [`stages/stage_A/architecture.md`](../stages/stage_A/architecture.md) and research plan `pc_residual_research_plan_v8` §7.2–7.3 (not stored in this repository).
