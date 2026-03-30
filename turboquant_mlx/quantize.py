"""
TurboQuant quantization algorithms.

Implements:
  - pack_indices / unpack_indices     — bit-packing for uint8 indices → uint32
  - quantize_row_mse                  — TurboQuant_mse for a single row vector
  - quantize_weight_matrix_mse        — apply TurboQuant_mse to a 2-D weight matrix
  - quantize_weight_matrix_prod       — apply TurboQuant_prod (MSE + QJL residual)

Reference: arXiv:2504.19874, Algorithms 1 & 2
"""

from __future__ import annotations

import numpy as np

from .codebooks import get_codebook
from .rotation import apply_srht, generate_signs
from .utils import n_blocks, seed_for_layer


# ---------------------------------------------------------------------------
# Index packing
# ---------------------------------------------------------------------------

def pack_indices(indices: np.ndarray, bits: int) -> np.ndarray:
    """Pack uint8 indices (values in [0, 2**bits)) into uint32 words.

    32 // bits indices per word, LSB-first.  Input is zero-padded to a
    multiple of (32 // bits).

    Args:
        indices: uint8 array of shape (n,)
        bits:    quantization bit width (2, 3, 4, or 8)

    Returns:
        packed: uint32 array of shape (ceil(n / (32//bits)),)
    """
    k = 32 // bits
    n = len(indices)
    pad = (-n) % k
    if pad:
        indices = np.concatenate([indices, np.zeros(pad, dtype=np.uint8)])
    indices = indices.astype(np.uint32)
    result = np.zeros(len(indices) // k, dtype=np.uint32)
    mask = np.uint32((1 << bits) - 1)
    for i in range(k):
        result |= (indices[i::k] & mask) << np.uint32(i * bits)
    return result


def unpack_indices(packed: np.ndarray, bits: int, n_original: int) -> np.ndarray:
    """Unpack uint32 words back to uint8 indices.

    Args:
        packed:     uint32 array
        bits:       quantization bit width
        n_original: number of valid indices to return (trims padding)

    Returns:
        indices: uint8 array of shape (n_original,)
    """
    k = 32 // bits
    n_padded = len(packed) * k
    result = np.zeros(n_padded, dtype=np.uint8)
    mask = np.uint32((1 << bits) - 1)
    for i in range(k):
        result[i::k] = ((packed >> np.uint32(i * bits)) & mask).astype(np.uint8)
    return result[:n_original]


# ---------------------------------------------------------------------------
# Single-row quantization (TurboQuant_mse)
# ---------------------------------------------------------------------------

def quantize_row_mse(
    row: np.ndarray,
    codebook: np.ndarray,
    boundaries: np.ndarray,
    signs: np.ndarray,
    bits: int,
    block_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a single weight row using TurboQuant_mse.

    Steps:
      1. Apply block-wise SRHT: y = SRHT(row, signs)
      2. Per-block scale = std(y_block)
      3. Normalize and apply Lloyd-Max codebook lookup
      4. Return packed indices + scales

    Args:
        row:       float32 array of shape (in_dim,)
        codebook:  float32 centroids of shape (2**bits,)
        boundaries: float32 boundaries of shape (2**bits - 1,)
        signs:     int8 array of shape (n_blk, block_size) — SRHT signs
        bits:      quantization bit width
        block_size: SRHT block size

    Returns:
        packed_indices: uint32 array
        scales:         float16 array of shape (n_blk,)
    """
    in_dim = len(row)
    nb = signs.shape[0]
    y = apply_srht(row.astype(np.float32), signs, block_size)  # (nb*block_size,)

    scales = np.empty(nb, dtype=np.float16)
    raw_indices = np.empty(nb * block_size, dtype=np.uint8)

    for b in range(nb):
        chunk = y[b * block_size : (b + 1) * block_size]
        std = float(np.std(chunk))
        if std < 1e-8:
            std = 1e-8
        scales[b] = np.float16(std)
        chunk_norm = chunk / std
        # np.digitize returns indices in [0, 2**bits - 1] when boundaries has 2**bits - 1 elements
        idx = np.searchsorted(boundaries, chunk_norm)
        raw_indices[b * block_size : (b + 1) * block_size] = idx.astype(np.uint8)

    # Only pack indices up to the padded dim (nb * block_size), trimming happens at unpack
    packed = pack_indices(raw_indices, bits)
    return packed, scales


# ---------------------------------------------------------------------------
# Full weight matrix quantization
# ---------------------------------------------------------------------------

def quantize_weight_matrix_mse(
    W: np.ndarray,
    bits: int,
    seed: int,
    block_size: int = 4096,
) -> dict[str, np.ndarray]:
    """Quantize a 2-D weight matrix using TurboQuant_mse.

    All rows share the same SRHT sign matrix (one signs matrix per layer).

    Args:
        W:          float32 array of shape (out_dim, in_dim)
        bits:       quantization bit width
        seed:       random seed for generating SRHT signs
        block_size: SRHT block size (must be power of 2)

    Returns dict with keys:
        tq_indices:  uint32 (out_dim, n_packed_per_row)
        tq_scales:   float16 (out_dim, n_blk)
        tq_signs:    int8 (n_blk, block_size)
    """
    W = W.astype(np.float32)
    out_dim, in_dim = W.shape
    codebook, boundaries = get_codebook(bits)
    signs = generate_signs(in_dim, seed, block_size)  # (n_blk, block_size)
    nb = signs.shape[0]

    # Determine packed width for a single row
    n_padded = nb * block_size
    k = 32 // bits
    n_packed = (n_padded + k - 1) // k

    all_indices = np.empty((out_dim, n_packed), dtype=np.uint32)
    all_scales = np.empty((out_dim, nb), dtype=np.float16)

    for i in range(out_dim):
        packed, sc = quantize_row_mse(W[i], codebook, boundaries, signs, bits, block_size)
        all_indices[i] = packed
        all_scales[i] = sc

    return {
        "tq_indices": all_indices,
        "tq_scales": all_scales,
        "tq_signs": signs,
    }


# ---------------------------------------------------------------------------
# TurboQuant_prod (MSE + QJL residual)
# ---------------------------------------------------------------------------

def pack_bits(bits_array: np.ndarray) -> np.ndarray:
    """Pack a binary (0/1) array into uint32 words, 32 bits per word, LSB-first."""
    n = len(bits_array)
    pad = (-n) % 32
    if pad:
        bits_array = np.concatenate([bits_array, np.zeros(pad, dtype=np.uint8)])
    bits_array = bits_array.astype(np.uint32)
    result = np.zeros(len(bits_array) // 32, dtype=np.uint32)
    for i in range(32):
        result |= bits_array[i::32] << np.uint32(i)
    return result


def unpack_bits(packed: np.ndarray, n_original: int) -> np.ndarray:
    """Unpack uint32 words to binary (0/1) uint8 array."""
    n_padded = len(packed) * 32
    result = np.zeros(n_padded, dtype=np.uint8)
    for i in range(32):
        result[i::32] = ((packed >> np.uint32(i)) & np.uint32(1)).astype(np.uint8)
    return result[:n_original]


def quantize_weight_matrix_prod(
    W: np.ndarray,
    bits: int,
    seed_mse: int,
    seed_qjl: int,
    block_size: int = 4096,
) -> dict[str, np.ndarray]:
    """Quantize a weight matrix using TurboQuant_prod (Algorithm 2).

    Uses (bits-1)-bit TurboQuant_mse + 1-bit QJL on the residual.

    Returns dict with all keys from _mse plus:
        tq_qjl_bits:   uint32 (out_dim, ceil(in_dim/32)) — QJL sign bits
        tq_qjl_signs:  int8 (n_blk_qjl, block_size)
        tq_res_norms:  float16 (out_dim,)
    """
    from .dequantize import dequantize_weight_matrix

    W = W.astype(np.float32)
    out_dim, in_dim = W.shape

    if bits < 2:
        raise ValueError("TurboQuant_prod requires bits >= 2 (uses bits-1 for MSE stage)")

    # Stage 1: (bits-1)-bit MSE quantization
    mse_data = quantize_weight_matrix_mse(W, bits - 1, seed_mse, block_size)

    # Dequantize to compute residual
    W_mse = dequantize_weight_matrix(mse_data, bits - 1, in_dim, block_size)
    R = W - W_mse  # residual: shape (out_dim, in_dim)

    # Stage 2: QJL — 1-bit sign of SRHT(residual)
    signs_qjl = generate_signs(in_dim, seed_qjl, block_size)
    n_packed_qjl = (in_dim + 31) // 32

    qjl_bits = np.empty((out_dim, n_packed_qjl), dtype=np.uint32)
    res_norms = np.empty(out_dim, dtype=np.float16)

    for i in range(out_dim):
        Hr_i = apply_srht(R[i], signs_qjl, block_size)[:in_dim]
        sign_bits = (Hr_i > 0).astype(np.uint8)
        qjl_bits[i] = pack_bits(sign_bits)[:n_packed_qjl]
        res_norms[i] = np.float16(float(np.linalg.norm(R[i])))

    return {
        **mse_data,
        "tq_qjl_bits": qjl_bits,
        "tq_qjl_signs": signs_qjl,
        "tq_res_norms": res_norms,
    }
