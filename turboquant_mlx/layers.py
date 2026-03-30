"""
TurboQuantLinear — MLX nn.Module that stores TurboQuant-compressed weights.

For both 'mse' and 'prod' variants, dequantization is performed in NumPy on
each forward pass and the result is converted back to an mx.array before the
matmul.  An optional eager_dequant=True flag pre-dequantizes on load (faster
inference, higher memory).

Usage:
    # Quantize from an existing nn.Linear
    tq_layer = TurboQuantLinear.from_linear(linear, bits=4, variant='mse')

    # Or reconstruct from a saved weights dict
    tq_layer = TurboQuantLinear.from_weights_dict(
        in_features=4096, out_features=4096,
        weights=weights_dict, bits=4, variant='mse'
    )

    # Forward pass (drop-in for nn.Linear)
    output = tq_layer(x)
"""

from __future__ import annotations

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False

from .codebooks import get_codebook
from .dequantize import dequantize_weight_matrix, estimate_inner_product_qjl
from .quantize import quantize_weight_matrix_mse, quantize_weight_matrix_prod
from .rotation import apply_srht
from .utils import seed_for_layer


if _HAS_MLX:
    class TurboQuantLinear(nn.Module):
        """Drop-in replacement for nn.Linear with TurboQuant-compressed weights."""

        def __init__(
            self,
            in_features: int,
            out_features: int,
            bits: int,
            variant: str = "mse",
            block_size: int = 4096,
            eager_dequant: bool = False,
        ):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.bits = bits
            self.variant = variant
            self.block_size = block_size
            self._eager_dequant = eager_dequant
            self._weight_cache: np.ndarray | None = None

            # These will be set via _set_compressed_weights
            # Declared here so MLX's parameter tracking sees them
            self.tq_indices: mx.array | None = None
            self.tq_scales: mx.array | None = None
            self.tq_signs: mx.array | None = None
            self.bias: mx.array | None = None

            # prod-only
            self.tq_qjl_bits: mx.array | None = None
            self.tq_qjl_signs: mx.array | None = None
            self.tq_res_norms: mx.array | None = None

        def _set_compressed_weights(
            self, data: dict[str, np.ndarray], bias: np.ndarray | None = None
        ) -> None:
            """Load compressed weight arrays as mx.array buffers."""
            self.tq_indices = mx.array(data["tq_indices"])
            self.tq_scales = mx.array(data["tq_scales"])
            self.tq_signs = mx.array(data["tq_signs"])
            if "tq_qjl_bits" in data:
                self.tq_qjl_bits = mx.array(data["tq_qjl_bits"])
                self.tq_qjl_signs = mx.array(data["tq_qjl_signs"])
                self.tq_res_norms = mx.array(data["tq_res_norms"])
            if bias is not None:
                self.bias = mx.array(bias)
            if self._eager_dequant:
                self._weight_cache = self._dequantize_numpy()

        def _get_numpy_data(self) -> dict[str, np.ndarray]:
            """Convert stored mx.arrays to numpy for dequantization."""
            data = {
                "tq_indices": np.array(self.tq_indices),
                "tq_scales": np.array(self.tq_scales),
                "tq_signs": np.array(self.tq_signs),
            }
            if self.tq_qjl_bits is not None:
                data["tq_qjl_bits"] = np.array(self.tq_qjl_bits)
                data["tq_qjl_signs"] = np.array(self.tq_qjl_signs)
                data["tq_res_norms"] = np.array(self.tq_res_norms)
            return data

        def _dequantize_numpy(self) -> np.ndarray:
            """Dequantize weights to float32 numpy array."""
            data = self._get_numpy_data()
            return dequantize_weight_matrix(
                data, self.bits if self.variant == "mse" else self.bits - 1,
                self.in_features, self.block_size
            )

        def __call__(self, x: mx.array) -> mx.array:
            if self.variant == "prod" and self.tq_qjl_bits is not None:
                # prod path: MSE dequant + QJL correction.
                # Must not short-circuit through _weight_cache as that only
                # contains the (bits-1)-bit MSE reconstruction without QJL.
                W_np = (
                    self._weight_cache
                    if self._eager_dequant and self._weight_cache is not None
                    else self._dequantize_numpy()
                )
                data = self._get_numpy_data()
                x_np = np.array(x.astype(mx.float32))   # (..., in_features)
                if x_np.shape[-1] != self.in_features:
                    raise ValueError(
                        f"TurboQuantLinear expected last dim {self.in_features}, "
                        f"got {x_np.shape}"
                    )
                batch_shape = x_np.shape[:-1]
                flat = x_np.reshape(-1, self.in_features)  # (B, in_features)

                signs_qjl = data["tq_qjl_signs"]
                qjl_bits = data["tq_qjl_bits"]
                res_norms = data["tq_res_norms"]

                out_flat = np.empty((flat.shape[0], self.out_features), dtype=np.float32)
                for b in range(flat.shape[0]):
                    vec = flat[b]
                    Hx = apply_srht(vec, signs_qjl, self.block_size)[:self.in_features]
                    corrections = np.array([
                        estimate_inner_product_qjl(
                            vec, Hx, qjl_bits[i], float(res_norms[i]), self.in_features
                        )
                        for i in range(self.out_features)
                    ], dtype=np.float32)
                    out_flat[b] = W_np @ vec + corrections

                result = mx.array(out_flat.reshape(batch_shape + (self.out_features,)))
                if self.bias is not None:
                    result = result + self.bias
                return result

            # mse path (or prod without QJL data — fallback)
            if self._eager_dequant and self._weight_cache is not None:
                W = mx.array(self._weight_cache)
            else:
                W = mx.array(self._dequantize_numpy())

            result = x @ W.T
            if self.bias is not None:
                result = result + self.bias
            return result

        @classmethod
        def from_linear(
            cls,
            linear: "nn.Linear",
            bits: int,
            variant: str = "mse",
            block_size: int = 4096,
            seed: int = 42,
            eager_dequant: bool = False,
        ) -> "TurboQuantLinear":
            """Quantize an nn.Linear layer and return a TurboQuantLinear."""
            W = np.array(linear.weight).astype(np.float32)
            out_dim, in_dim = W.shape
            bias = np.array(linear.bias) if linear.bias is not None else None

            if variant == "mse":
                data = quantize_weight_matrix_mse(W, bits, seed, block_size)
            elif variant == "prod":
                data = quantize_weight_matrix_prod(W, bits, seed, seed + 1, block_size)
            else:
                raise ValueError(f"Unknown variant '{variant}'. Choose 'mse' or 'prod'.")

            layer = cls(in_dim, out_dim, bits, variant, block_size, eager_dequant)
            layer._set_compressed_weights(data, bias)
            return layer

        @classmethod
        def from_weights_dict(
            cls,
            in_features: int,
            out_features: int,
            weights: dict[str, np.ndarray],
            bits: int,
            variant: str = "mse",
            block_size: int = 4096,
            bias: np.ndarray | None = None,
            eager_dequant: bool = False,
        ) -> "TurboQuantLinear":
            """Reconstruct TurboQuantLinear from a loaded weights dictionary."""
            layer = cls(in_features, out_features, bits, variant, block_size, eager_dequant)
            layer._set_compressed_weights(weights, bias)
            return layer

else:
    # Stub for environments without MLX (useful for unit tests of pure numpy code)
    class TurboQuantLinear:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("mlx is required for TurboQuantLinear")
