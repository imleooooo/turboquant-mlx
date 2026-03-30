import numpy as np
import pytest

from turboquant_mlx.rotation import apply_srht, apply_inverse_srht, generate_signs, fwht


def test_fwht_self_inverse():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(64).astype(np.float32)
    y = fwht(fwht(x)) / 64
    np.testing.assert_allclose(x, y, atol=1e-4)


def test_srht_norm_preservation():
    """SRHT preserves the L2 norm of each block (up to float32 tolerance)."""
    rng = np.random.default_rng(2)
    in_dim = 128
    x = rng.standard_normal(in_dim).astype(np.float32)
    signs = generate_signs(in_dim, seed=0, block_size=64)
    y = apply_srht(x, signs, block_size=64)
    # Each block should preserve norm
    norm_x = float(np.linalg.norm(x[:64]))
    norm_y0 = float(np.linalg.norm(y[:64]))
    np.testing.assert_allclose(norm_x, norm_y0, rtol=1e-4)


def test_srht_inverse_roundtrip():
    rng = np.random.default_rng(3)
    in_dim = 100
    block_size = 64
    x = rng.standard_normal(in_dim).astype(np.float32)
    signs = generate_signs(in_dim, seed=7, block_size=block_size)
    y = apply_srht(x, signs, block_size)
    x_hat = apply_inverse_srht(y, signs, in_dim, block_size)
    np.testing.assert_allclose(x, x_hat, atol=1e-4)


def test_srht_inverse_roundtrip_power2():
    rng = np.random.default_rng(4)
    in_dim = 256
    block_size = 256
    x = rng.standard_normal(in_dim).astype(np.float32)
    signs = generate_signs(in_dim, seed=99, block_size=block_size)
    y = apply_srht(x, signs, block_size)
    x_hat = apply_inverse_srht(y, signs, in_dim, block_size)
    np.testing.assert_allclose(x, x_hat, atol=1e-4)


def test_generate_signs_shape():
    signs = generate_signs(300, seed=0, block_size=128)
    assert signs.shape == (3, 128)
    assert signs.dtype == np.int8
    assert set(np.unique(signs)).issubset({-1, 1})


def test_srht_output_approx_gaussian():
    """After SRHT, coordinates should be approximately Gaussian (test kurtosis ≈ 3)."""
    rng = np.random.default_rng(5)
    in_dim = 512
    x = rng.standard_normal(in_dim).astype(np.float32) * 3.0  # non-unit variance
    signs = generate_signs(in_dim, seed=0, block_size=512)
    y = apply_srht(x, signs, block_size=512)[:in_dim]
    from scipy.stats import kurtosis
    k = kurtosis(y, fisher=False)  # normal kurtosis = 3
    assert 2.0 < k < 5.0, f"Kurtosis {k:.2f} is far from Gaussian"
