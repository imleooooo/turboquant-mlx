"""
Block-wise Subsampled Randomized Hadamard Transform (SRHT).

Given a vector x ∈ R^d, the rotation y = SRHT(x) maps coordinates to an
approximately Gaussian distribution, enabling near-optimal scalar quantization
via Lloyd-Max codebooks.

The transform is structured as:
    y_block = FWHT(x_block ⊙ D) / sqrt(B)
where D ∈ {±1}^B is a random sign diagonal, B is the block size (power of 2),
and FWHT is the Fast Walsh-Hadamard Transform.

FWHT is self-inverse (up to scaling):  FWHT(FWHT(x)/sqrt(B)) * sqrt(B) = x
So the inverse is simply:              x_block = FWHT(y_block) ⊙ D / sqrt(B) * sqrt(B)
                                                = FWHT(y_block) ⊙ D

Reference: arXiv:2504.19874, Section 3 and Lemma 2.1
"""

from __future__ import annotations

import numpy as np

from .utils import n_blocks, next_power_of_2


def fwht(a: np.ndarray) -> np.ndarray:
    """In-place Fast Walsh-Hadamard Transform. len(a) must be a power of 2.

    Normalised so that FWHT(FWHT(x)) == len(x) * x, i.e. the unnormalised
    transform is applied. The caller is responsible for dividing by sqrt(B).
    """
    a = a.copy()
    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            a[i : i + h], a[i + h : i + 2 * h] = (
                a[i : i + h] + a[i + h : i + 2 * h],
                a[i : i + h] - a[i + h : i + 2 * h],
            )
        h *= 2
    return a


def generate_signs(in_dim: int, seed: int, block_size: int = 4096) -> np.ndarray:
    """Generate random ±1 sign matrix for block-wise SRHT.

    Returns:
        signs: int8 array of shape (n_blocks, block_size)
    """
    nb = n_blocks(in_dim, block_size)
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 2, size=(nb, block_size), dtype=np.int8)
    signs = raw * 2 - 1  # map {0,1} → {-1,+1}
    return signs.astype(np.int8)


def apply_srht(
    x: np.ndarray,
    signs: np.ndarray,
    block_size: int = 4096,
) -> np.ndarray:
    """Apply block-wise SRHT to a 1-D float32 vector.

    Args:
        x:          float32 array of shape (in_dim,)
        signs:      int8 array of shape (n_blocks, block_size) — from generate_signs
        block_size: must be a power of 2

    Returns:
        y: float32 array of shape (n_blocks * block_size,)
           (last block may contain padding zeros beyond in_dim)
    """
    in_dim = len(x)
    nb = signs.shape[0]
    scale = 1.0 / np.sqrt(block_size)
    out = np.empty(nb * block_size, dtype=np.float32)
    for b in range(nb):
        start = b * block_size
        end = min(start + block_size, in_dim)
        chunk = np.zeros(block_size, dtype=np.float32)
        chunk[: end - start] = x[start:end]
        chunk *= signs[b].astype(np.float32)
        out[b * block_size : (b + 1) * block_size] = fwht(chunk) * scale
    return out


def apply_inverse_srht(
    y: np.ndarray,
    signs: np.ndarray,
    in_dim: int,
    block_size: int = 4096,
) -> np.ndarray:
    """Invert block-wise SRHT.

    Args:
        y:       float32 array of shape (n_blocks * block_size,)
        signs:   int8 array of shape (n_blocks, block_size)
        in_dim:  original input dimension (to trim padding)

    Returns:
        x: float32 array of shape (in_dim,)
    """
    nb = signs.shape[0]
    scale = 1.0 / np.sqrt(block_size)
    out = np.empty(nb * block_size, dtype=np.float32)
    for b in range(nb):
        chunk = y[b * block_size : (b + 1) * block_size].copy()
        # FWHT is self-inverse: FWHT(FWHT(x)) = B*x, so FWHT(y)*scale recovers x*signs
        out[b * block_size : (b + 1) * block_size] = (
            fwht(chunk) * scale * signs[b].astype(np.float32)
        )
    return out[:in_dim]
