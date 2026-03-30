# turboquant-mlx

TurboQuant weight quantization for MLX language models on Apple Silicon.

Implements the TurboQuant algorithm from [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026) — a data-oblivious vector quantization method that achieves near-optimal distortion rates within 2.7× of the information-theoretic lower bound, with no calibration data or fine-tuning required.

## How it works

TurboQuant quantizes each weight matrix in two steps:

1. **Random rotation (SRHT)** — applies a block-wise Subsampled Randomized Hadamard Transform to each weight row, which maps coordinates to an approximately Gaussian distribution regardless of the original weight distribution.
2. **Lloyd-Max scalar quantization** — applies an optimal scalar quantizer (precomputed for N(0,1)) independently to each coordinate. Because the distribution is now known, no per-tensor calibration is needed.

Two variants are supported:

- **`mse`** (Algorithm 1) — minimises mean-squared error. Good general-purpose weight compression.
- **`prod`** (Algorithm 2) — adds a 1-bit QJL (Quantized Johnson-Lindenstrauss) correction on the quantization residual, producing an unbiased inner-product estimator. Better for attention and dot-product-heavy layers.

## Installation

Requires Python 3.11+ and an Apple Silicon Mac.

```bash
git clone https://github.com/imleooooo/turboquant-mlx
cd turboquant-mlx
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Usage

### CLI

```bash
# Quantize a model to 4-bit (mse variant)
python -m turboquant_mlx \
  --model ./path/to/mlx-model \
  --output ./path/to/output \
  --bits 4

# Quantize with the prod variant (unbiased inner product)
python -m turboquant_mlx \
  --model ./path/to/mlx-model \
  --output ./path/to/output \
  --bits 4 \
  --variant prod

# Dry run — list quantizable layers without modifying anything
python -m turboquant_mlx \
  --model ./path/to/mlx-model \
  --output /tmp/unused \
  --list-layers
```

**All options:**

| Flag | Default | Description |
|---|---|---|
| `--model` / `-m` | required | Input MLX model directory |
| `--output` / `-o` | required | Output directory for quantized model |
| `--bits` | `4` | Bit width: `2`, `3`, `4`, or `8` |
| `--variant` | `mse` | `mse` or `prod` |
| `--block-size` | `4096` | SRHT block size (must be power of 2) |
| `--seed` | `42` | Global random seed for rotation matrices |
| `--max-shard-gb` | `4.0` | Maximum shard size in GB |
| `--verbose` / `-v` | off | Print per-layer progress |
| `--list-layers` | off | Dry run: list quantizable layers and exit |

### Python API

```python
from turboquant_mlx import load_model_weights, convert_model, save_quantized_model
from pathlib import Path

weights, config = load_model_weights(Path("./llama-3-8b-mlx"))

quantized = convert_model(
    weights, config,
    bits=4,
    variant="mse",   # or "prod"
    seed=42,
    verbose=True,
)

save_quantized_model(
    Path("./llama-3-8b-tq4"),
    quantized, config,
    bits=4, variant="mse", block_size=4096,
)
```

### Using quantized layers directly

```python
import mlx.nn as nn
from turboquant_mlx.layers import TurboQuantLinear

# Quantize an existing nn.Linear layer
tq_layer = TurboQuantLinear.from_linear(
    linear,          # nn.Linear instance
    bits=4,
    variant="mse",   # or "prod"
    seed=42,
)

# Drop-in forward pass — accepts any (..., in_features) input
output = tq_layer(x)
```

## Output format

> **Note:** The output directory is **not** a standard MLX model directory.
> Quantized linear layers no longer have `.weight` tensors — they are replaced
> by `.tq_*` arrays that existing MLX/mlx-lm loaders cannot consume.
> To run inference you must reconstruct each layer using
> `TurboQuantLinear.from_weights_dict()` (see [Loading for inference](#loading-for-inference)).

The output contains safetensors files and an updated `config.json`.
Each quantized linear layer's `.weight` tensor is replaced by:

| Tensor | dtype | Description |
|---|---|---|
| `{layer}.tq_indices` | uint32 | Packed Lloyd-Max codebook indices |
| `{layer}.tq_scales` | float16 | Per-block SRHT scale factors |
| `{layer}.tq_signs` | int8 | SRHT rotation signs |
| `{layer}.tq_qjl_bits` | uint32 | QJL sign bits (`prod` only) |
| `{layer}.tq_qjl_signs` | int8 | QJL rotation signs (`prod` only) |
| `{layer}.tq_res_norms` | float16 | QJL residual norms (`prod` only) |

`config.json` gains a `"quantization"` section:

```json
{
  "quantization": {
    "quant_method": "turboquant",
    "bits": 4,
    "variant": "mse",
    "block_size": 4096
  }
}
```

## Loading for inference

Because the output uses a custom serialization, you must load the saved weights
and reconstruct `TurboQuantLinear` layers manually before running inference.
A minimal example:

```python
import numpy as np
from pathlib import Path
from turboquant_mlx.model_io import load_quantized_weights
from turboquant_mlx.layers import TurboQuantLinear

weights, config = load_quantized_weights(Path("./llama-3-8b-tq4"))
q = config["quantization"]
bits, variant, block_size = q["bits"], q["variant"], q["block_size"]

# Reconstruct a single layer (e.g. model.layers.0.self_attn.q_proj)
prefix = "model.layers.0.self_attn.q_proj"
layer_weights = {
    k.removeprefix(prefix + "."): v
    for k, v in weights.items()
    if k.startswith(prefix + ".tq_")
}
bias = weights.get(prefix + ".bias")

tq_layer = TurboQuantLinear.from_weights_dict(
    in_features=4096,
    out_features=4096,
    weights=layer_weights,
    bits=bits,
    variant=variant,
    block_size=block_size,
    bias=bias,
)

output = tq_layer(x)  # x: mx.array of shape (..., in_features)
```

A full model loader that wires all layers into an existing MLX model graph is
not yet included in this repository.

## Layers quantized

All linear projection layers matching standard LLM naming conventions are quantized: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. This covers Llama, Mistral, Gemma, Phi, and Qwen architectures. Embeddings and norm layers are passed through unchanged.

## Compression estimates (Llama-7B)

| Bit width | Approx. size | vs fp16 |
|---|---|---|
| 2-bit | ~1.5 GB | 7.7× |
| 3-bit | ~2.3 GB | 4.9× |
| 4-bit | ~3.0 GB | 3.8× |
| 8-bit | ~5.9 GB | 1.9× |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Reference

Zandieh, Daliri, Hadian, Mirrokni.
**TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.**
ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
