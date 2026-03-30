"""
TurboQuant dequantization.

Implements:
  - dequantize_row_mse         — reconstruct a single row from packed indices
  - dequantize_weight_matrix   — reconstruct full float32 weight matrix (offline)
  - estimate_inner_product_qjl — QJL inner product correction (prod variant, inference)

Reference: arXiv:2504.19874, Section 3
"""

from __future__ import annotations

import numpy as np

from .codebooks import get_codebook
from .quantize import unpack_indices, unpack_bits
from .rotation import apply_inverse_srht, apply_srht


def dequantize_row_mse(
    packed_indices: np.ndarray,
    scales: np.ndarray,
    signs: np.ndarray,
    codebook: np.ndarray,
    bits: int,
    in_dim: int,
    block_size: int = 4096,
) -> np.ndarray:
    """Reconstruct a single weight row from TurboQuant_mse compressed data.

    Args:
        packed_indices: uint32 array (packed row indices)
        scales:         float16 array of shape (n_blk,)
        signs:          int8 array of shape (n_blk, block_size)
        codebook:       float32 array of shape (2**bits,)
        bits:           quantization bit width
        in_dim:         original input dimension
        block_size:     SRHT block size

    Returns:
        row: float32 array of shape (in_dim,)
    """
    nb = signs.shape[0]
    n_padded = nb * block_size
    indices = unpack_indices(packed_indices, bits, n_padded)  # uint8 (n_padded,)

    y_hat = np.empty(n_padded, dtype=np.float32)
    for b in range(nb):
        chunk_idx = indices[b * block_size : (b + 1) * block_size]
        y_hat[b * block_size : (b + 1) * block_size] = (
            codebook[chunk_idx].astype(np.float32) * float(scales[b])
        )

    return apply_inverse_srht(y_hat, signs, in_dim, block_size)


def dequantize_weight_matrix(
    data: dict[str, np.ndarray],
    bits: int,
    in_dim: int,
    block_size: int = 4096,
) -> np.ndarray:
    """Reconstruct a full float32 weight matrix from TurboQuant_mse data dict.

    Args:
        data:       dict with keys tq_indices, tq_scales, tq_signs
        bits:       quantization bit width
        in_dim:     original in_features (columns)
        block_size: SRHT block size

    Returns:
        W: float32 array of shape (out_dim, in_dim)
    """
    codebook, _ = get_codebook(bits)
    indices = data["tq_indices"]   # (out_dim, n_packed)
    scales = data["tq_scales"]     # (out_dim, n_blk)
    signs = data["tq_signs"]       # (n_blk, block_size)

    out_dim = indices.shape[0]
    W = np.empty((out_dim, in_dim), dtype=np.float32)
    for i in range(out_dim):
        W[i] = dequantize_row_mse(
            indices[i], scales[i], signs, codebook, bits, in_dim, block_size
        )
    return W


def estimate_inner_product_qjl(
    x: np.ndarray,
    Hx: np.ndarray,
    qjl_sign_bits: np.ndarray,
    res_norm: float,
    in_dim: int,
) -> float:
    """Estimate residual · x using QJL (1-bit inner product estimator).

    The QJL estimator from the paper:
        E[<sign(Hr_i), sign(Hx)>] = (2/π) arcsin(<r_i, x> / (‖r_i‖ ‖x‖))

    Inverted:  r_i · x ≈ (π/2) ‖r_i‖ ‖x‖ corr / in_dim

    Args:
        x:              float32 query vector of shape (in_dim,)
        Hx:             float32 rotated query SRHT(x)[:in_dim]
        qjl_sign_bits:  uint32 packed sign bits for this output neuron
        res_norm:       float16 residual norm ‖r_i‖₂
        in_dim:         dimension

    Returns:
        scalar estimate of r_i · x
    """
    sign_r = unpack_bits(qjl_sign_bits, in_dim).astype(np.float32) * 2 - 1  # {-1,+1}
    sign_x = np.sign(Hx).astype(np.float32)
    sign_x[sign_x == 0] = 1.0
    corr = float(np.dot(sign_r, sign_x)) / in_dim
    x_norm = float(np.linalg.norm(x))
    # Invert: corr = (2/π) arcsin(cos_sim)  =>  cos_sim = sin((π/2) * corr)
    # Then r_i · x = cos_sim * ||r_i|| * ||x||
    cos_sim = float(np.sin((np.pi / 2.0) * corr))
    return cos_sim * float(res_norm) * x_norm
