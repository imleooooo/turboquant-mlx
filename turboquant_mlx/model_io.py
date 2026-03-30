"""
Load and save MLX models in safetensors format.

Handles:
  - Loading multi-shard safetensors models
  - Saving quantized models with metadata + updated config.json
  - Reading/writing model.safetensors.index.json for multi-shard models
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Load a single safetensors file into a numpy dict."""
    from safetensors.numpy import load_file
    return load_file(str(path))


def load_model_weights(model_path: Path | str) -> tuple[dict[str, np.ndarray], dict]:
    """Load all weights from an MLX model directory.

    Supports single-shard (model.safetensors) and multi-shard
    (model.safetensors.index.json) models.

    Returns:
        weights: flat dict {key: np.ndarray}
        config:  parsed config.json dict
    """
    model_path = Path(model_path)
    weights: dict[str, np.ndarray] = {}

    index_file = model_path / "model.safetensors.index.json"
    single_file = model_path / "model.safetensors"

    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        for shard_name in shard_files:
            shard_path = model_path / shard_name
            weights.update(_load_safetensors(shard_path))
    elif single_file.exists():
        weights.update(_load_safetensors(single_file))
    else:
        # Try any .safetensors file
        st_files = sorted(model_path.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")
        for f in st_files:
            weights.update(_load_safetensors(f))

    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    return weights, config


def save_quantized_model(
    output_path: Path | str,
    quantized_weights: dict[str, np.ndarray],
    config: dict,
    bits: int,
    variant: str,
    block_size: int,
    original_model: str = "",
    max_shard_gb: float = 4.0,
) -> None:
    """Save TurboQuant-quantized weights to a model directory.

    Writes:
      - One or more .safetensors shards
      - model.safetensors.index.json (if multi-shard)
      - config.json with quantization metadata
    """
    from safetensors.numpy import save_file

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "format": "turboquant_mlx",
        "tq_version": "1.0",
        "tq_bits": str(bits),
        "tq_variant": variant,
        "tq_block_size": str(block_size),
        "tq_original_model": str(original_model),
    }

    # Split into shards
    from .utils import make_shards
    shards = make_shards(quantized_weights, max_shard_gb)

    if len(shards) == 1:
        out_file = output_path / "model.safetensors"
        save_file(shards[0], str(out_file), metadata=metadata)
    else:
        weight_map: dict[str, str] = {}
        n_digits = len(str(len(shards)))
        for i, shard in enumerate(shards):
            shard_name = f"model-{i+1:0{n_digits}d}-of-{len(shards):0{n_digits}d}.safetensors"
            out_file = output_path / shard_name
            save_file(shard, str(out_file), metadata=metadata)
            for key in shard:
                weight_map[key] = shard_name

        index = {
            "metadata": {"total_size": sum(a.nbytes for a in quantized_weights.values())},
            "weight_map": weight_map,
        }
        with open(output_path / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)

    # Write updated config.json
    updated_config = dict(config)
    updated_config["quantization"] = {
        "quant_method": "turboquant",
        "bits": bits,
        "variant": variant,
        "block_size": block_size,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(updated_config, f, indent=2)

    print(f"Saved quantized model to {output_path}")


def load_quantized_weights(model_path: Path | str) -> tuple[dict[str, np.ndarray], dict]:
    """Load TurboQuant-quantized model weights (same as load_model_weights)."""
    return load_model_weights(model_path)
