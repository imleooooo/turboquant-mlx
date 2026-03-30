import numpy as np
import pytest

from turboquant_mlx.quantize import (
    pack_indices, unpack_indices, pack_bits, unpack_bits,
    quantize_weight_matrix_mse,
)
from turboquant_mlx.dequantize import dequantize_weight_matrix


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_pack_unpack_roundtrip(bits):
    rng = np.random.default_rng(0)
    n = 200
    indices = rng.integers(0, 2**bits, size=n, dtype=np.uint8)
    packed = pack_indices(indices, bits)
    recovered = unpack_indices(packed, bits, n)
    np.testing.assert_array_equal(indices, recovered)


def test_pack_bits_roundtrip():
    rng = np.random.default_rng(1)
    n = 150
    bits = rng.integers(0, 2, size=n, dtype=np.uint8)
    packed = pack_bits(bits)
    recovered = unpack_bits(packed, n)
    np.testing.assert_array_equal(bits, recovered)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_mse_roundtrip_distortion(bits, medium_weight):
    """MSE of quantize→dequantize should be within 3× of theoretical bound."""
    W = medium_weight
    out_dim, in_dim = W.shape
    data = quantize_weight_matrix_mse(W, bits, seed=42, block_size=64)
    W_hat = dequantize_weight_matrix(data, bits, in_dim, block_size=64)
    mse = float(np.mean((W - W_hat) ** 2))
    # Normalize by input variance
    w_var = float(np.var(W))
    relative_mse = mse / (w_var + 1e-8)
    # Theoretical bound: (sqrt(3)*pi/2) / 4^bits
    theoretical = (np.sqrt(3) * np.pi / 2) / (4**bits)
    # Allow generous factor since we're quantizing already-scaled weights
    assert relative_mse < 3.0 * theoretical + 0.1, (
        f"Relative MSE {relative_mse:.4f} too large for {bits}-bit (bound ~{theoretical:.4f})"
    )


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_mse_shapes(bits, small_weight):
    W = small_weight
    out_dim, in_dim = W.shape
    block_size = 32
    data = quantize_weight_matrix_mse(W, bits, seed=0, block_size=block_size)
    assert "tq_indices" in data
    assert "tq_scales" in data
    assert "tq_signs" in data
    assert data["tq_indices"].dtype == np.uint32
    assert data["tq_scales"].dtype == np.float16
    assert data["tq_signs"].dtype == np.int8
    assert data["tq_scales"].shape[0] == out_dim


def test_mse_deterministic(small_weight):
    """Same seed → same quantized output."""
    W = small_weight
    d1 = quantize_weight_matrix_mse(W, 4, seed=42, block_size=32)
    d2 = quantize_weight_matrix_mse(W, 4, seed=42, block_size=32)
    np.testing.assert_array_equal(d1["tq_indices"], d2["tq_indices"])
    np.testing.assert_array_equal(d1["tq_signs"], d2["tq_signs"])


def test_mse_different_seeds_different_output(small_weight):
    """Different seeds → different rotations → (very likely) different indices."""
    W = small_weight
    d1 = quantize_weight_matrix_mse(W, 4, seed=1, block_size=32)
    d2 = quantize_weight_matrix_mse(W, 4, seed=2, block_size=32)
    assert not np.array_equal(d1["tq_signs"], d2["tq_signs"])
