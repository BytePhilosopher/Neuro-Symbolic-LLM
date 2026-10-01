# Stage A1 — Hidden-State Residual Adapter

Stage A1 is the baseline for adapting a frozen language model: a small, trainable, low-rank residual is added to the hidden states of the model's late blocks, and only that residual is trained. The base model is wrapped in a [`FrozenSubstrate`](../frozenllm/README.md) and never changes.

The adapter is **not** weight-level LoRA. It modifies block *outputs* at inference time; the base weights receive no gradient ($\nabla \theta_0 = 0$).

---

## Contents

- [Overview](#overview)
- [The Adapter](#the-adapter)
- [The A1 Objective](#the-a1-objective)
- [Training Loop](#training-loop)
- [Code Layout](#code-layout)
- [API](#api)
- [Configuration](#configuration)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Guarantees](#guarantees)

---

## Overview

| | |
|---|---|
| **Base model** | Frozen (GPT-2 by default), run through `FrozenSubstrate` |
| **Trainable part** | Low-rank residual on late blocks, $2 \cdot d \cdot r$ parameters per block |
| **Objective** | Next-token cross-entropy, optional KL to the base model, optional weight decay |
| **Data** | WikiText-2 (`Salesforce/wikitext`, `wikitext-2-raw-v1`) |
| **Optimizer** | `optax.adam`, over the adapter parameters only |
| **Starting point** | Identical to the base model (zero residual at step 0) |

---

## The Adapter

### Residual

At each adapted block $l$, the frozen output $h_l \in \mathbb{R}^{d}$ is replaced by:

$$\tilde h_l = h_l + s \cdot \sigma(h_l A_l)\, B_l, \qquad A_l \in \mathbb{R}^{d \times r},\ B_l \in \mathbb{R}^{r \times d}$$

| Symbol | Meaning |
|---|---|
| $r$ | Rank (`rank`) |
| $s$ | Scale: $\alpha / r$, or $1.0$ when `alpha` is `null` |
| $\sigma$ | Activation: `identity` (default), `gelu`, `relu` or `tanh` |

The residual is computed in float32, so the rank-$r$ bottleneck keeps its precision on bf16/fp16 models, and is cast back to the dtype of $h_l$ before the addition.

### Layer Selection

For a model with $L$ blocks ($0 \dots L-1$), the adapter targets the late blocks $(L/\!/2,\ L-2]$. The final block is never adapted.

| $L$ | Adapted blocks |
|---|---|
| 12 (GPT-2 small) | 7 – 10 |
| 24 | 13 – 22 |
| 32 | 17 – 30 |

`late_start` and `late_end` (both inclusive) override the defaults. Invalid bounds, or $L < 4$, raise `ValueError` with the valid range. Bounds are never clamped.

### Initialisation

- $A_l \sim \mathcal{N}(0, 1/d)$ (override with `init_std`) and $B_l = 0$.
- Because $B_l = 0$, the residual is exactly zero and the adapted model equals the base model at step 0. The gradient with respect to $B_l$ is still non-zero, so training starts immediately; $A_l$ begins to receive gradient once $B_l$ moves.
- Initialisation is deterministic in `seed`. Random keys are derived from the block index, so a block's $A_l$ does not change when the layer range is moved.

### Parameters

The parameters form a PyTree that is fully separate from the substrate's weights:

```python
{"layer_7": {"A": f32[d, r], "B": f32[r, d]}, ..., "layer_10": {...}}
```

For GPT-2 small ($d = 768$, $r = 16$, blocks 7–10) this is **98,304** trainable parameters.

---

## The A1 Objective

Implemented in [`stages/stage_A/a1.py`](../stages/stage_A/a1.py):

$$\mathcal{L}_{A1} = \mathcal{L}_{\text{task}} + \lambda_{\text{kl}} \cdot \mathrm{KL}\big(p_{F_0} \,\|\, p_{F_0 + R}\big) + \lambda_{\text{wd}} \cdot \lVert\phi\rVert^2$$

| Term | Description |
|---|---|
| $\mathcal{L}_{\text{task}}$ | Next-token cross-entropy of the adapted model. Labels default to the input ids; the loss applies the shift. |
| $\mathrm{KL}$ | Forward KL from the base model $F_0$ to the adapted model $F_0 + R$, averaged over batch and positions. Penalises the adapter for moving away from the base distribution. Base logits are wrapped in `stop_gradient`. |
| $\lVert\phi\rVert^2$ | Squared L2 norm of all adapter parameters. |

`a1_objective` returns `(total, terms)`, where `terms` holds `total`, `task`, `kl` and `wd`. KL is always computed and logged, even when $\lambda_{\text{kl}} = 0$ (the default).

---

## Training Loop

Implemented in [`experiments/run_stage_a1.py`](../experiments/run_stage_a1.py):

1. Load the config into a `RunConfig`: `ResidualConfig`, `A1Config`, `TrainingConfig` and one `DomainConfig` per domain.
2. Load the frozen model into `FrozenSubstrate` and confirm its weights are frozen.
3. Build the adapter and initialise its parameters.
4. Tokenise each domain's train and eval splits into contiguous `[n, seq_len]` blocks. Train batches are shuffled and seeded per task; eval batches are fixed, so every evaluation sees the same data.
5. Create the run directory and write `config.yaml` and `meta.json` (see [Run Artifacts](#run-artifacts)).
6. Evaluate every domain at init: base and adapted loss, perplexity and KL.
7. Train on the domains in order. Each step computes `a1_objective` and its gradient with respect to the adapter, then applies an `optax.adam` update. Adam's moments are reset at each domain boundary unless `reset_optimizer: false`.
8. After each task, save the adapter and evaluate every domain. This fills the loss matrix `L[t][j]` (domain `j` after task `t`).
9. Write `results.json` and confirm the base weights are unchanged. The script exits with a non-zero code if they are not.

A single-domain config is a sequence of length one, which is a plain A1 run.

```text
input ids ──► FrozenSubstrate ──────────────────────────────► base logits (stop_gradient)
          │                                                         │
          └─► FrozenSubstrate + adapter hook on late blocks ──► adapted logits
                                                                    │
                                   task CE + λ_kl·KL + λ_wd·‖φ‖² ◄──┘
                                                │
                                   jax.grad w.r.t. adapter only ──► optax.adam
```

Each step runs two forward passes (base and adapted). The step is not JIT-compiled, so training on CPU is slow.

---

## Code Layout

| File | Purpose |
|---|---|
| [`residual/adapter.py`](adapter.py) | `ResidualConfig`, `ResidualAdapter`, layer selection |
| [`stages/stage_A/a1.py`](../stages/stage_A/a1.py) | `A1Config`, `a1_objective`, `base_logits` |
| [`metrics/performance.py`](../metrics/performance.py) | `cross_entropy_loss`, differentiable `kl_to_base`, `perplexity`, `mean_seen_accuracy`, `mean_forgetting`, `training_step_time` |
| [`metrics/representation.py`](../metrics/representation.py) | `loss_improvement`, `residual_compressibility`, `compressibility_curve` |
| [`stages/results.py`](../stages/results.py) | `RunWriter`: run directory, JSONL metrics, atomic JSON, adapter checkpoints |
| [`experiments/run_stage_a1.py`](../experiments/run_stage_a1.py) | Multi-domain sequential training and evaluation runner |
| [`configs/stage_A/a1_lora_baseline.yaml`](../configs/stage_A/a1_lora_baseline.yaml) | Run configuration: WikiText-2 → AG News → IMDB |
| [`tests/residual/test_adapter.py`](../tests/residual/test_adapter.py) | Adapter unit tests |
| [`tests/stage_A/test_a1_training.py`](../tests/stage_A/test_a1_training.py) | End-to-end A1 tests |
| [`tests/stage_A/test_a1_sequential.py`](../tests/stage_A/test_a1_sequential.py) | Sequential runner and domain config tests |
| [`tests/metrics/`](../tests/metrics/), [`tests/stages/`](../tests/stages/) | Metric and run-writer unit tests |

---

## API

### `residual`

| Name | Description |
|---|---|
| `ResidualConfig` | Frozen, validated dataclass: `rank=16`, `activation="identity"`, `alpha=None`, `init_std=None`, `late_start=None`, `late_end=None`, `seed=0`. `from_dict()` rejects unknown keys; `.scale` returns $s$. |
| `ResidualAdapter(config, architecture)` | Builds the adapter from the substrate's architecture. Attributes: `layers`, `hidden_size`, `scale`, `init_std`. |
| `.init_params()` | Returns the initial `AdapterParams`. |
| `.apply(params, h, layer_idx)` | Adds the residual; blocks without parameters pass through unchanged. |
| `.residual(layer_params, h)` | Returns the raw residual in float32. |
| `.modify_fn(params)` | Returns a `(hidden, layer_idx) -> hidden` hook for `FrozenSubstrate.run_with_interception`. |
| `ResidualAdapter.num_params(params)` | Total trainable parameter count. |
| `ResidualAdapter.l2_norm_sq(params)` | $\lVert\phi\rVert^2$ as a float32 scalar. |
| `resolve_late_layers(num_layers, late_start=None, late_end=None)` | Returns the adapted block indices. |
| `layer_key(layer_idx)` | Returns `f"layer_{layer_idx}"`. |
| `AdapterParams` | Type alias: `dict[str, dict[str, jax.Array]]`. |
| `ACTIVATIONS` | Read-only mapping of activation name to function. |

### `stages.stage_A.a1`

| Name | Description |
|---|---|
| `A1Config` | Frozen dataclass: `lambda_kl=0.0`, `lambda_wd=0.0` (both must be `>= 0`). `from_dict()` rejects unknown keys. |
| `a1_objective(params, *, substrate, adapter, config, input_ids, labels=None, base=None)` | Returns `(total, terms)`. `base` accepts precomputed base logits. |
| `base_logits(substrate, input_ids)` | Base-model logits with gradients stopped. |

---

## Configuration

[`configs/stage_A/a1_lora_baseline.yaml`](../configs/stage_A/a1_lora_baseline.yaml) has four sections, each mapping one-to-one to a dataclass. Unknown sections or keys are rejected.

```yaml
adapter:                # → ResidualConfig
  rank: 16
  activation: identity  # identity | gelu | relu | tanh
  alpha: null           # null → scale 1.0, else alpha / rank
  init_std: null        # null → 1/sqrt(d)
  late_start: null      # null → L//2 + 1
  late_end: null        # null → L - 2
  seed: 0

objective:              # → A1Config
  lambda_kl: 0.0
  lambda_wd: 0.0

training:               # → TrainingConfig
  model: gpt2
  seq_len: 128
  batch_size: 8
  steps: 200            # per domain, unless the domain sets its own
  learning_rate: 1.0e-3
  eval_batches: 8
  log_every: 10
  seed: 0
  reset_optimizer: true # fresh Adam moments at each domain boundary
  output_dir: runs

domains:                # → DomainConfig each; trained in order, names unique
  - name: wikitext2     # [A-Za-z0-9_.-]+, used in checkpoint file names
    dataset: Salesforce/wikitext
    dataset_config: wikitext-2-raw-v1
    text_field: text
    train_split: train  # HF slicing works: "train[:20000]"
    eval_split: validation
    separator: ""       # joins rows into one stream; "\n\n" for one-doc-per-row sets
    steps: null         # null → training.steps
```

`domains` is required and must be non-empty.

---

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

On Intel macOS, where `torch>=2.6` has no wheels, use the Docker image:

```bash
docker build -t neuro-symbolic-llm .
docker run --rm -it -v "$PWD":/app neuro-symbolic-llm
```

---

## Usage

### Train

The model and dataset are downloaded on the first run.

```bash
# Sequential: WikiText-2 → AG News → IMDB
python experiments/run_stage_a1.py --config configs/stage_A/a1_lora_baseline.yaml

# Override rank, model, steps (applied to every domain) or where the run is written
python experiments/run_stage_a1.py --config configs/stage_A/a1_lora_baseline.yaml \
    --rank 8 --steps 20 --output-dir runs --run-name smoke
```

The runner logs the adapted layers, trainable and frozen parameter counts, loss terms, perplexity, KL and step time, and writes everything to the run directory.

### Run Artifacts

Each run writes a fresh directory, `<output_dir>/<run_name>/` (default name `a1-<UTC timestamp>`). An existing directory is never overwritten.

| File | Contents |
|---|---|
| `config.yaml` | The resolved config after CLI overrides. It can be passed back with `--config`. |
| `meta.json` | Git commit and dirty flag, library versions, argv, start time, model, adapted layers, parameter counts |
| `metrics.jsonl` | One event per line. `train`: `task`, `domain`, `step`, `global_step`, `tokens`, `step_time`, `ppl`, `loss_total`, `loss_task`, `loss_kl`, `loss_wd`. `eval`: `after_task` (`-1` = init), `domain`, `base_loss`, `loss`, `ppl`, `kl`. |
| `results.json` | `domains`, `base_loss[j]`, matrices `loss`, `ppl`, `kl` and `improvement` indexed `[t][j]`, `mean_seen_loss[t]`, `mean_forgetting[t]` (`null` at `t = 0`), `steps`, `tokens`, `mean_step_time`, `params_unchanged` |
| `params/task_<t>_<name>.npz` | Adapter parameters after task `t`; read with `stages.results.load_params` |

Forgetting is computed on loss, so a positive value means an earlier domain's loss rose above its best value since that domain was trained.

> **Note:** On CPU or with limited memory, lower `batch_size` and `seq_len` (for example `2` and `64`), or use a GPU.

### Python

```python
import jax
import jax.numpy as jnp
from transformers import GPT2Config, GPT2LMHeadModel

from frozenllm.substrate import FrozenSubstrate
from residual import ResidualAdapter, ResidualConfig
from stages.stage_A.a1 import A1Config, a1_objective

# Tiny 12-block GPT-2. Use FrozenSubstrate("gpt2") for the real checkpoint.
cfg = GPT2Config(n_layer=12, n_head=4, n_embd=32, n_positions=32, vocab_size=64,
                 bos_token_id=1, eos_token_id=2)
substrate = FrozenSubstrate(GPT2LMHeadModel(cfg).eval(), config=cfg)

adapter = ResidualAdapter(ResidualConfig(rank=4), substrate.architecture)
params = adapter.init_params()
print(adapter.layers)  # (7, 8, 9, 10)

ids = jnp.arange(9, dtype=jnp.int32)[None, :]

(total, terms), grads = jax.value_and_grad(a1_objective, has_aux=True)(
    params,
    substrate=substrate,
    adapter=adapter,
    config=A1Config(lambda_kl=0.1),
    input_ids=ids,
)

assert substrate.params_unchanged()
```

---

## Testing

```bash
pytest tests/residual/ tests/stage_A/ tests/stages/ tests/metrics/ -v
pre-commit run --all-files
```

| Suite | Covers |
|---|---|
| `tests/residual/test_adapter.py` | Layer resolution and bound validation, parameter shapes and initialisation, seeding, parameter count, residual math, float32 cast-back, config validation. Runs without a model. |
| `tests/stage_A/test_a1_training.py` | On 12-block GPT-2 and Pythia fixtures: logits equal the base at init, unmodified intermediates, gradients reach the adapter only, finite-difference check, objective terms, and a short training run that leaves the base unchanged. |
| `tests/stage_A/test_a1_sequential.py` | Two-domain run on the GPT-2 fixture with synthetic data: loss matrix and summaries, event order in `metrics.jsonl`, saved checkpoints, config round-trip, `--steps` override, domain config validation. |
| `tests/metrics/` | `mean_seen_accuracy`, `mean_forgetting` (accuracy and loss), `training_step_time`, `residual_compressibility`, `compressibility_curve`. Runs without a model. |
| `tests/stages/test_results.py` | `RunWriter` artifacts, atomic writes, refusal to overwrite, checkpoint round-trip. Runs without a model. |

---

## Guarantees

Each guarantee is enforced by tests.

| Guarantee | Test(s) |
|---|---|
| **Frozen base:** base weights get no gradient, are absent from the optimizer state, and are unchanged after training. | `test_substrate_params_receive_no_gradient`, `test_optimizer_state_holds_only_adapter_leaves`, `test_training_moves_only_the_adapter` |
| **Unmodified intermediates:** `ForwardResult.hidden_state(l)` returns $h_l$ before the residual is added. | `test_intermediates_are_pristine` |
| **Zero-residual start:** adapted logits equal base logits at init (`atol=1e-5`); non-adapted blocks are untouched. | `test_initial_logits_equal_base`, `test_zero_residual_at_init`, `test_non_adapted_layers_untouched` |
| **No silent layer changes:** invalid depths or bounds raise `ValueError`. | `test_invalid_bounds_raise`, `test_too_shallow_model_raises` |
| **Strict configuration:** unknown config keys are rejected. | `test_config_rejects_unknown_keys`, `test_shipped_yaml_loads_and_rejects_unknown_keys` |
