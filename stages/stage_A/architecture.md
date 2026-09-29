# Stage A1 — Architecture

Ordinary residual baseline: **no predictive coding, no symbolic head**
(research plan §7.3). A1 is the reference that A2–A4 are measured against, so it
is kept as plain as possible.

> **Status:** design only. As of commit `e6c1005` on `Base`, `residual/`,
> `stages/stage_A/`, `metrics/*.py`, `experiments/run_stage_a1.py` and
> `configs/stage_A/a1_lora_baseline.yaml` exist but are empty.

**Sources.** Each design choice below is tagged with where it comes from:

- **[plan §x]**: stated in the research plan (`pc_residual_research_plan_v8`).
- **[repo]**: dictated by existing code on `Base`.
- **[team]**: a team decision. The plan does not specify it.

---

## 1. What A1 is

A1 consists of a frozen base `F0` (from `frozenllm.substrate`) plus a small
trainable residual `R_φ`, trained by ordinary backprop **[plan §7.1, §7.3]**:

```
F(x) = F0(x; θ0) + R_φ(x)          θ0 frozen, only φ trained
```

### 1.1 Residual adapter

At each adapted layer `l`, the frozen block output `h_l` is replaced by the
following **[plan §7.2]**:

```
h̃_l = h_l + s · σ(h_l A_l) B_l

A_l ∈ R^{d×r}    read directions  a_i  (columns of A_l)
B_l ∈ R^{r×d}    write directions b_i  (rows of B_l)
s   = alpha / r  if alpha is set, else 1.0                    [team]
```

- This document uses the **row-vector convention** (`h` has shape `[..., d]`).
  The plan writes the same map as `B_l σ(A_l h)` with column vectors, which is
  the transpose.
- `σ = identity` (the default) gives the **rank-one atomized** form
  `Σ_i b_i (a_iᵀ h)` from §7.2. It exposes per-atom read/write directions for
  later Stage A metrics (redundancy, A6 audits) **[plan §7.2, §7.8]**.
- `σ ∈ {gelu, relu, tanh}` gives the nonlinear low-rank form **[plan §7.2]**.
  The specific activation set is a **[team]** choice.
- The residual is computed in float32 and cast back to the hidden-state dtype
  before the addition **[team]**.

This is a **hidden-state** adapter. It is not weight-level LoRA.

### 1.2 Initialisation **[team]**

- `A_l ~ N(0, σ_A²)` with `σ_A = 1/√d` by default (variance `1/d`).
  `init_std` overrides `σ_A`, and it is a standard deviation.
- `B_l = 0`. At step 0 the adapted model therefore **is** the base model
  (logits equal, KL = 0). This is a tested invariant.

A consequence for gradient tests: at `B = 0`, `∂L/∂A = 0` exactly and
`∂L/∂B ≠ 0` in general.

### 1.3 Which layers: late layers strictly after the midpoint

The plan places adapters at "one layer or a small set of layers,
`l ∈ {l_mid, l_late}`" **[plan §7.2]**. A1 keeps the plan's `l_late` and uses
`l_mid` as an **exclusive lower bound**. The adapted layers form a contiguous
late block that starts *after* the midpoint, not at it **[team]**.

With `L = architecture.num_layers` and zero-based block indices `0 … L−1`
**[repo]**:

```
l_mid      = L // 2                         exclusive lower bound
l_late     = L − 2                          last adapted block
late_start = config.late_start or l_mid + 1
late_end   = config.late_end   or l_late
layers     = late_start … late_end          inclusive
           = range(late_start, late_end + 1)
```

The final block (`L−1`) is skipped, so every adapter output still passes
through at least one frozen block before `ln_f` and the LM head.

| `L` (model) | `l_mid` | default layers | `|layers|` |
|---|---|---|---|
| 12 (GPT-2 small, Pythia-160m, test fixtures) | 6 | 7 … 10 | 4 |
| 24 (GPT-2 medium, Pythia-410m / 1.4b) | 12 | 13 … 22 | 10 |
| 32 (Pythia-2.8b / 6.9b) | 16 | 17 … 30 | 14 |

**Validation** (fail with a clear message):

- `L ≥ 4`
- `l_mid < late_start ≤ late_end ≤ L − 2`

Setting `late_start = late_end = L − 2` recovers the plan's single-late-layer
option. The resolved tuple is also passed through the substrate's
`validate_interception_layers` **[repo]**. No layer indices are hard-coded
anywhere.

### 1.4 Why a hidden-state adapter and not weight-level LoRA **[repo]**

The substrate exposes exactly one intervention point:
`modify_fn(hidden, layer_idx)` on block outputs
(`frozenllm/substrate/interception.py`). The base parameter dict passed to
`functional_call` is internal to `FrozenSubstrate`, so weight-level LoRA
(`W + BA` on q/k/v/o) would require changing `frozenllm/`. The hidden-state
adapter of §7.2 fits the existing contract exactly.

The config filename `a1_lora_baseline.yaml` is kept for continuity. It refers to
the LoRA-*shaped* hidden-state adapter.

### 1.5 Objective **[plan §7.3]**

```
L_A1 = L_task + λ_KL · KL(p_F0 ‖ p_F0+R) + λ_wd · ‖φ‖²
```

- `L_task` is next-token cross-entropy (`FrozenSubstrate.compute_loss`
  **[repo]**).
- `p_F0` comes from a base-only forward pass with the identity `modify_fn`.
  Its logits are wrapped in `jax.lax.stop_gradient`.
- `‖φ‖²` is the sum of squares over all `A_l` and `B_l`.
- Weight decay lives **only** in `λ_wd`. The optimizer is plain `optax.adam`,
  not `adamw`, so the penalty is not applied twice.
- Defaults `λ_KL = 0.0` and `λ_wd = 0.0` are **[team]** starting values to be
  swept. KL to base is **always logged** as the drift metric, even when
  `λ_KL = 0`.

---

## 2. Data flow

```
                experiments/run_stage_a1.py  (orchestrator)
                             │
   ┌───────────────┬─────────┴─────────┬──────────────────────┐
   ▼               ▼                   ▼                      ▼
frozenllm.     residual/           stages/stage_A/        metrics/
substrate      adapter.py          a1.py                  performance.py
FrozenSubstrate ResidualAdapter    a1_objective           cross_entropy_loss
(base, frozen)  init / modify_fn   (term weighting)       kl_to_base, perplexity
   │               │
   │ run_with_interception(ids, modify_fn(φ), intercept_layers=adapter.layers)
   │◄──────────────┘
   ▼
 block l output h_l ──hook──► modify_fn(h_l, l) = h_l + R_l(h_l)
   │                                  │
   │ intermediates[l] = h_l (pristine)│ h̃_l fed to block l+1
   ▼                                  ▼
 ForwardResult(logits, intermediates) ──► metrics ──► L_A1 ──► ∂L/∂φ ──► optax
```

---

## 3. Code layout

| Path | Owns | Must not contain |
|---|---|---|
| `residual/adapter.py` | `ResidualConfig`, `resolve_late_layers`, `ResidualAdapter`. This is the reusable component; A2/A3 will reuse it. | losses, data, training loops, FabricPC |
| `stages/stage_A/a1.py` | `A1Config` (objective weights) and `a1_objective(...)`: what makes A1 *A1* | adapter math, metric formulas |
| `metrics/performance.py` | `cross_entropy_loss`, `kl_to_base` (differentiable), `perplexity` | stage-specific logic |
| `experiments/run_stage_a1.py` | loading the substrate, data, optimizer, train/eval loop, logging, invariant checks | adapter or loss math |
| `configs/stage_A/a1_lora_baseline.yaml` | `adapter:`, `objective:`, `training:` sections | layer index lists |

### 3.1 `residual/adapter.py`

```python
@dataclass(frozen=True)
class ResidualConfig:
    rank: int = 16                 # [plan §12.2 uses r=16]
    activation: str = "identity"   # identity | gelu | relu | tanh
    alpha: float | None = None
    init_std: float | None = None  # None → 1/sqrt(d)
    late_start: int | None = None  # None → L//2 + 1
    late_end: int | None = None    # None → L - 2
    seed: int = 0

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ResidualConfig: ...
    # from_dict rejects unknown keys.

def resolve_late_layers(num_layers, late_start=None, late_end=None) -> tuple[int, ...]: ...

class ResidualAdapter:
    def __init__(self, config: ResidualConfig, architecture: Architecture): ...
    layers: tuple[int, ...]
    hidden_size: int
    def init_params(self) -> Params: ...                  # deterministic from seed
    def residual(self, layer_params, h) -> jax.Array: ...
    def modify_fn(self, params) -> ModifyFn: ...          # (hidden, layer_idx) -> hidden
    def num_params(self, params) -> int: ...
    def l2_norm_sq(self, params) -> jax.Array: ...
```

The parameter PyTree is independent of `substrate.params`. Its keys are
generated from the resolved layers:

```python
{"layer_7": {"A": f32[d, r], "B": f32[r, d]}, ..., "layer_10": {...}}
```

Layers without an entry pass through unchanged. Adapter size is
`2·d·r·|layers|` (for example, GPT-2 small, `d=768`, `r=16`, 4 layers:
98,304 parameters, about 8×10⁻⁴ of the base). The runner logs this ratio; the
plan targets 10⁻⁴–10⁻² **[plan §5]**.

### 3.2 Contract with the substrate **[repo]**

- `run_with_interception` always calls hooks with `to_jax=True`. `hidden`
  arrives as a `jax.Array`, and the returned `jax.Array` is converted back via
  `torch_view`.
- `ForwardResult.intermediates[l]` holds the **pristine** `h_l`, captured
  before `modify_fn` runs.
- The adapter is built from `substrate.architecture` (`hidden_size`,
  `num_layers`) and never inspects model internals.
- Nothing in A1 touches `substrate.params`. `substrate.params_unchanged()` must
  remain `True` after training.
- `compute_kl_drift` in `frozenllm/substrate/drift.py` calls `float(...)`, so it
  is usable for logging but not inside a differentiated loss.
  `metrics.performance.kl_to_base` is the differentiable version and returns a
  `jax.Array`.

---

## 4. Gradient path: verify before building the trainer

Training needs `∂L/∂φ` through the frozen blocks above each adapted layer. The
current tests only check that the frozen params receive **no** torch gradient.
No test takes a JAX gradient through the substrate yet.

1. **Pure JAX (preferred).** Call `jax.value_and_grad` on a loss that invokes
   `substrate.run_with_interception` with `modify_fn` closing over `φ`. The
   tracer enters at the hook and flows through torchax ops on concrete frozen
   weights into the logits. This path handles multiple adapted layers, and θ0
   gets no gradient by construction.
2. **Fallback: torch autograd via `call_jax_differentiable`**
   (`torchax_backend.py`). **Risk:** the hard-coded `to_jax=True` means each
   hooked `h` passes through `jax_view`, which cuts the torch tape. A lower
   adapter would then receive no gradient through an upper one.

Checks to run (Linux, CI, or the repo's Docker image, because torch ≥ 2.6 has
no Intel-macOS wheels):

- `∂L/∂B` is finite and non-zero at init.
- After `B` is set non-zero, `∂L/∂A` is finite and non-zero at `late_start` and
  `late_end`.
- A finite-difference check on one entry of `A` at `late_start` matches the
  analytic gradient.
- Loss decreases over a few steps, and `params_unchanged()` still holds.
- *(Nice to have)* `jax.jit` over the step. Eager `grad` is sufficient for A1.

If path 1 fails, the options are path 2 or a minimal `frozenllm/` change (for
example, allowing `to_jax=False`). That choice belongs to the team, not to the
A1 code, and must never involve unfreezing θ0.

---

## 5. Invariants and tests

1. `B = 0` at init ⇒ adapted logits equal base logits (`atol 1e-5`), for both
   the GPT-2 and Pythia fixtures.
2. `intermediates[l]` is pristine at the adapted layers.
3. Layer resolution: `L=12 → 7…10`, `L=24 → 13…22`, `L=32 → 17…30`. Explicit
   `late_start`/`late_end` are honoured, and invalid bounds raise.
4. Shapes: `A: (d, r)`, `B: (r, d)`. `num_params == 2·d·r·|layers|`. Layers
   outside the block have no params and are unmodified.
5. Gradients follow the checks in §4.
6. The optimizer state contains only adapter leaves: its tree structure matches
   the adapter PyTree.
7. After a few steps: `L_task` decreases, KL > 0, `B ≠ 0`, and
   `substrate.params_unchanged()` is `True`.
8. Config validation rejects `rank < 1`, unknown activation, non-positive
   `alpha`/`init_std`, negative λ, and unknown YAML keys.

The existing 12-layer fixtures in `tests/frozenllm/conftest.py` resolve to
layers `7…10`, so no special deep test model is required.

---

## 6. Out of scope for this A1 PR

The plan also asks A1 to provide forgetting baselines and residual-
compressibility curves over `r ∈ {1,2,4,8,16,32,64}` **[plan §7.3, §14.2]**.
This PR makes those possible (`--rank` override, logged step time and parameter
counts) but does not implement `mean_forgetting` or `residual_compressibility`.
They belong in `metrics/` in a follow-up.

Also out of scope: predictive coding, FabricPC, symbolic heads, and weight-level
LoRA.

---

## 7. Housekeeping **[repo]**

- In `pyproject.toml`, add `residual*`, `stages*`, `metrics*` to
  `[tool.setuptools.packages.find] include`, and `residual`, `stages`, `metrics`
  to `[tool.ruff.lint.isort] known-first-party`.
- `.github/workflows/ci.yml` only runs `tests/frozenllm/unit/`. Extend it to run
  the new A1 tests.
- Pre-commit runs black, ruff, ruff-format and mypy, and all of them must pass.
