import numpy as np
import pytest

from turboquant_mlx.codebooks import get_codebook, SUPPORTED_BITS


@pytest.mark.parametrize("bits", sorted(SUPPORTED_BITS))
def test_codebook_length(bits):
    cb, bd = get_codebook(bits)
    assert len(cb) == 2**bits
    assert len(bd) == 2**bits - 1


@pytest.mark.parametrize("bits", sorted(SUPPORTED_BITS))
def test_codebook_symmetry(bits):
    cb, _ = get_codebook(bits)
    np.testing.assert_allclose(cb, -cb[::-1], atol=1e-5)


@pytest.mark.parametrize("bits", sorted(SUPPORTED_BITS))
def test_codebook_monotone(bits):
    cb, _ = get_codebook(bits)
    assert np.all(np.diff(cb) > 0), "Codebook must be strictly increasing"


@pytest.mark.parametrize("bits", sorted(SUPPORTED_BITS))
def test_boundaries_are_midpoints(bits):
    cb, bd = get_codebook(bits)
    expected = (cb[:-1] + cb[1:]) / 2
    np.testing.assert_allclose(bd, expected, atol=1e-6)


@pytest.mark.parametrize("bits", sorted(SUPPORTED_BITS))
def test_mse_within_theoretical_bound(bits):
    """Empirical MSE on N(0,1) samples should be near the TurboQuant bound.

    The formula (√3π/2)/4^b is a theoretical upper bound for the ALGORITHM
    (including SRHT rotation).  For 2–4 bit it's tight; at 8-bit the bound
    is an asymptotic approximation so we allow a wider margin.
    """
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200_000).astype(np.float32)
    cb, bd = get_codebook(bits)
    idx = np.searchsorted(bd, x)
    x_hat = cb[idx]
    mse = float(np.mean((x - x_hat) ** 2))
    theoretical = (np.sqrt(3) * np.pi / 2) / (4**bits)
    # 8-bit: bound is approximate; allow 3× margin.  Other bits: 1.1×.
    factor = 3.0 if bits == 8 else 1.1
    assert mse <= factor * theoretical, (
        f"MSE={mse:.6f} exceeds {factor}×theoretical={factor*theoretical:.6f} for {bits}-bit"
    )


def test_unsupported_bits():
    with pytest.raises(ValueError, match="Unsupported bit width"):
        get_codebook(5)
