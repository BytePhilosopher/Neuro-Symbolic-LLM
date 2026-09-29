"""Unit tests for the hidden-state residual adapter (no substrate needed)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from frozenllm.substrate.architecture import Architecture
from residual import (
    ACTIVATIONS,
    ResidualAdapter,
    ResidualConfig,
    layer_key,
    resolve_late_layers,
)


def _arch(num_layers: int = 12, hidden_size: int = 32) -> Architecture:
    return Architecture(
        model_family="gpt2",
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=4,
        head_dim=hidden_size // 4,
        vocab_size=64,
    )


# ── layer resolution ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("num_layers", "expected"),
    [
        (12, (7, 8, 9, 10)),
        (24, tuple(range(13, 23))),
        (32, tuple(range(17, 31))),
    ],
)
def test_default_late_layers(num_layers: int, expected: tuple[int, ...]) -> None:
    layers = resolve_late_layers(num_layers)
    assert layers == expected
    assert min(layers) > num_layers // 2
    assert num_layers - 1 not in layers


def test_shallowest_valid_model_gives_single_layer() -> None:
    # L=4 passes the depth check but (2, 2] is empty, so bounds validation fails.
    with pytest.raises(ValueError, match="2 < late_start <= late_end <= 2"):
        resolve_late_layers(4)
    assert resolve_late_layers(5) == (3,)


@pytest.mark.parametrize(
    ("num_layers", "start", "end", "expected"),
    [
        (12, 7, 7, (7,)),
        (12, 8, 10, (8, 9, 10)),
        (12, None, 9, (7, 8, 9)),
        (12, 9, None, (9, 10)),
        (24, 20, 22, (20, 21, 22)),
    ],
)
def test_explicit_bounds(
    num_layers: int, start: int | None, end: int | None, expected: tuple[int, ...]
) -> None:
    assert resolve_late_layers(num_layers, start, end) == expected


@pytest.mark.parametrize(
    ("num_layers", "start", "end"),
    [
        (12, 6, None),  # starts at the midpoint
        (12, 3, 10),  # starts before the midpoint
        (12, None, 11),  # adapts the final block
        (12, None, 12),  # beyond the model
        (12, 10, 8),  # start > end
        (12, 0, None),  # zero is a bound, not "unset"
        (12, -1, None),
    ],
)
def test_invalid_bounds_raise(
    num_layers: int, start: int | None, end: int | None
) -> None:
    with pytest.raises(ValueError) as err:
        resolve_late_layers(num_layers, start, end)
    msg = str(err.value)
    assert f"{num_layers} blocks" in msg
    assert f"late_start={start if start is not None else num_layers // 2 + 1}" in msg
    assert f"{num_layers // 2} < late_start <= late_end <= {num_layers - 2}" in msg


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_too_shallow_model_raises(num_layers: int) -> None:
    with pytest.raises(ValueError, match=f"model has {num_layers}"):
        resolve_late_layers(num_layers)


def test_adapter_uses_resolved_layers() -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=2), _arch(12))
    assert adapter.layers == resolve_late_layers(12)
    with pytest.raises(ValueError):
        ResidualAdapter(ResidualConfig(rank=2, late_end=11), _arch(12))


# ── init, shapes, size ───────────────────────────────────────────────────────


def test_param_shapes_and_keys() -> None:
    d, r = 32, 4
    adapter = ResidualAdapter(ResidualConfig(rank=r), _arch(12, d))
    params = adapter.init_params()
    assert set(params) == {layer_key(i) for i in adapter.layers}
    for layer_params in params.values():
        assert set(layer_params) == {"A", "B"}
        assert layer_params["A"].shape == (d, r)
        assert layer_params["B"].shape == (r, d)
        assert layer_params["A"].dtype == jnp.float32
        assert layer_params["B"].dtype == jnp.float32


def test_init_b_zero_and_a_std() -> None:
    d, r = 256, 64
    adapter = ResidualAdapter(ResidualConfig(rank=r), _arch(12, d))
    params = adapter.init_params()
    a_all = np.concatenate([np.asarray(p["A"]).ravel() for p in params.values()])
    assert all(np.all(np.asarray(p["B"]) == 0.0) for p in params.values())
    assert abs(a_all.std() - 1.0 / math.sqrt(d)) / (1.0 / math.sqrt(d)) < 0.02
    assert abs(a_all.mean()) < 5 * a_all.std() / math.sqrt(a_all.size)


def test_init_std_override() -> None:
    d = 256
    adapter = ResidualAdapter(ResidualConfig(rank=64, init_std=0.5), _arch(12, d))
    a = np.asarray(adapter.init_params()[layer_key(adapter.layers[0])]["A"])
    assert abs(a.std() - 0.5) / 0.5 < 0.05


def test_init_is_deterministic_and_seeded() -> None:
    arch = _arch(12)
    adapter = ResidualAdapter(ResidualConfig(rank=4, seed=0), arch)
    p0 = adapter.init_params()
    p0b = ResidualAdapter(ResidualConfig(rank=4, seed=0), arch).init_params()
    p1 = ResidualAdapter(ResidualConfig(rank=4, seed=1), arch).init_params()
    first, second = (layer_key(i) for i in adapter.layers[:2])
    np.testing.assert_array_equal(p0[first]["A"], p0b[first]["A"])
    assert not np.array_equal(p0[first]["A"], p1[first]["A"])
    assert not np.array_equal(p0[first]["A"], p0[second]["A"])


@pytest.mark.parametrize(("num_layers", "d", "r"), [(12, 32, 4), (24, 64, 16)])
def test_num_params(num_layers: int, d: int, r: int) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=r), _arch(num_layers, d))
    params = adapter.init_params()
    assert ResidualAdapter.num_params(params) == 2 * d * r * len(adapter.layers)


def test_l2_norm_sq() -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4), _arch(12))
    params = adapter.init_params()
    expected = sum(float(np.sum(np.asarray(p["A"]) ** 2)) for p in params.values())
    np.testing.assert_allclose(
        float(ResidualAdapter.l2_norm_sq(params)), expected, rtol=1e-5
    )


# ── residual math ────────────────────────────────────────────────────────────


def _hidden(d: int, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(42), (2, 5, d)).astype(dtype)


@pytest.mark.parametrize("activation", sorted(ACTIVATIONS))
def test_zero_residual_at_init(activation: str) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4, activation=activation), _arch(12))
    params = adapter.init_params()
    h = _hidden(32)
    fn = adapter.modify_fn(params)
    for idx in adapter.layers:
        np.testing.assert_array_equal(fn(h, idx), h)


def test_non_adapted_layers_untouched() -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4), _arch(12))
    params = jax.tree_util.tree_map(jnp.ones_like, adapter.init_params())
    h = _hidden(32)
    fn = adapter.modify_fn(params)
    for idx in range(adapter.architecture.num_layers):
        out = fn(h, idx)
        if idx in adapter.layers:
            assert not np.array_equal(out, h)
        else:
            assert out is h


@pytest.mark.parametrize(
    ("activation", "alpha"), [("identity", None), ("gelu", 8.0), ("tanh", 2.0)]
)
def test_residual_formula(activation: str, alpha: float | None) -> None:
    d, r = 32, 4
    adapter = ResidualAdapter(
        ResidualConfig(rank=r, activation=activation, alpha=alpha), _arch(12, d)
    )
    rng = np.random.default_rng(0)
    A = jnp.asarray(rng.normal(size=(d, r)), dtype=jnp.float32)
    B = jnp.asarray(rng.normal(size=(r, d)), dtype=jnp.float32)
    idx = adapter.layers[0]
    params = {layer_key(idx): {"A": A, "B": B}}
    h = _hidden(d)
    scale = 1.0 if alpha is None else alpha / r
    expected = h + scale * ACTIVATIONS[activation](h @ A) @ B
    np.testing.assert_allclose(
        adapter.apply(params, h, idx), expected, rtol=1e-5, atol=1e-5
    )


def test_identity_is_sum_of_rank_one_atoms() -> None:
    d, r = 32, 3
    adapter = ResidualAdapter(ResidualConfig(rank=r), _arch(12, d))
    rng = np.random.default_rng(1)
    A = jnp.asarray(rng.normal(size=(d, r)), dtype=jnp.float32)
    B = jnp.asarray(rng.normal(size=(r, d)), dtype=jnp.float32)
    h = _hidden(d)
    atoms = sum((h @ A[:, i])[..., None] * B[i] for i in range(r))
    np.testing.assert_allclose(
        adapter.residual({"A": A, "B": B}, h), atoms, rtol=1e-4, atol=1e-4
    )


def test_residual_cast_back_to_hidden_dtype() -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4), _arch(12))
    params = jax.tree_util.tree_map(jnp.ones_like, adapter.init_params())
    idx = adapter.layers[0]
    h = _hidden(32, jnp.bfloat16)
    assert adapter.residual(params[layer_key(idx)], h).dtype == jnp.float32
    assert adapter.apply(params, h, idx).dtype == jnp.bfloat16


# ── config validation ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank": 0},
        {"rank": -2},
        {"rank": 1.5},
        {"activation": "swish"},
        {"alpha": 0.0},
        {"alpha": -1.0},
        {"init_std": 0.0},
        {"late_start": 7.5},
    ],
)
def test_config_rejects_invalid(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ResidualConfig(**kwargs)  # type: ignore[arg-type]


def test_config_from_dict() -> None:
    cfg = ResidualConfig.from_dict({"rank": 8, "activation": "gelu", "alpha": 16.0})
    assert cfg.rank == 8 and cfg.activation == "gelu"
    assert cfg.scale == 2.0
    assert ResidualConfig().scale == 1.0


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown ResidualConfig keys: \\['layers'\\]"):
        ResidualConfig.from_dict({"rank": 8, "layers": [6, 10]})
