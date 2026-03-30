"""
Model conversion orchestrator.

Walks through weight keys, identifies linear layers to quantize, applies
TurboQuant, and assembles the output weights dict.
"""

from __future__ import annotations

import numpy as np

from .quantize import quantize_weight_matrix_mse, quantize_weight_matrix_prod
from .utils import matches_quantizable_pattern, seed_for_layer


def convert_model(
    weights: dict[str, np.ndarray],
    config: dict,
    bits: int,
    variant: str = "mse",
    block_size: int = 4096,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Quantize all eligible linear weight matrices in a model weights dict.

    Eligible keys: 2-D weight tensors matching quantizable patterns
    (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, etc.)

    Non-eligible keys (embeddings, norms, biases) are passed through as-is.

    Args:
        weights:    flat {key: np.ndarray} from load_model_weights()
        config:     model config dict (unused, reserved for future use)
        bits:       quantization bit width (2, 3, 4, or 8)
        variant:    'mse' or 'prod'
        block_size: SRHT block size (power of 2, default 4096)
        seed:       global random seed; per-layer seeds are derived from this
        verbose:    if True, print per-layer progress

    Returns:
        output_weights: dict with quantized layers expanded to tq_* keys,
                        and non-quantized layers passed through unchanged
    """
    output: dict[str, np.ndarray] = {}
    n_quantized = 0
    n_skipped = 0

    for key, tensor in weights.items():
        if (
            matches_quantizable_pattern(key)
            and tensor.ndim == 2
            and tensor.shape[0] > 1
            and tensor.shape[1] > 1
        ):
            W = tensor.astype(np.float32)
            layer_prefix = key[: -len(".weight")]
            layer_seed = seed_for_layer(key, seed)

            if verbose:
                print(f"  Quantizing {key} {list(W.shape)} → {bits}-bit {variant} ...")

            if variant == "mse":
                compressed = quantize_weight_matrix_mse(W, bits, layer_seed, block_size)
            elif variant == "prod":
                compressed = quantize_weight_matrix_prod(
                    W, bits, layer_seed, layer_seed ^ 0xDEADBEEF, block_size
                )
            else:
                raise ValueError(f"Unknown variant '{variant}'")

            for suffix, arr in compressed.items():
                output[f"{layer_prefix}.{suffix}"] = arr

            # Preserve bias as-is if present
            bias_key = key.replace(".weight", ".bias")
            if bias_key in weights:
                output[bias_key] = weights[bias_key]

            n_quantized += 1
        else:
            output[key] = tensor
            n_skipped += 1

    if verbose:
        print(f"\nQuantized {n_quantized} layers, passed through {n_skipped} tensors.")

    return output


def get_quantizable_keys(weights: dict[str, np.ndarray]) -> list[str]:
    """Return the list of weight keys that would be quantized."""
    return [
        k for k, v in weights.items()
        if matches_quantizable_pattern(k) and v.ndim == 2
    ]
