"""End-to-end roundtrip test: quantize → save → load → verify."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from turboquant_mlx.convert import convert_model, get_quantizable_keys
from turboquant_mlx.model_io import load_model_weights, save_quantized_model


def make_fake_model(tmp_path: Path, n_layers: int = 2) -> Path:
    """Create a minimal fake MLX model directory with safetensors + config.json."""
    from safetensors.numpy import save_file

    rng = np.random.default_rng(42)
    weights = {}
    for i in range(n_layers):
        # Attention projections (quantizable)
        weights[f"model.layers.{i}.self_attn.q_proj.weight"] = rng.standard_normal((32, 64)).astype(np.float32)
        weights[f"model.layers.{i}.self_attn.k_proj.weight"] = rng.standard_normal((32, 64)).astype(np.float32)
        weights[f"model.layers.{i}.self_attn.v_proj.weight"] = rng.standard_normal((32, 64)).astype(np.float32)
        weights[f"model.layers.{i}.self_attn.o_proj.weight"] = rng.standard_normal((64, 32)).astype(np.float32)
        # MLP (quantizable)
        weights[f"model.layers.{i}.mlp.gate_proj.weight"] = rng.standard_normal((128, 64)).astype(np.float32)
        weights[f"model.layers.{i}.mlp.down_proj.weight"] = rng.standard_normal((64, 128)).astype(np.float32)
        # Norm (not quantizable)
        weights[f"model.layers.{i}.input_layernorm.weight"] = rng.standard_normal((64,)).astype(np.float32)
    # Embeddings (not quantizable)
    weights["model.embed_tokens.weight"] = rng.standard_normal((1000, 64)).astype(np.float32)

    save_file(weights, str(tmp_path / "model.safetensors"))
    config = {"hidden_size": 64, "num_layers": n_layers, "model_type": "fake"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def test_load_model_weights():
    with tempfile.TemporaryDirectory() as tmp:
        model_path = make_fake_model(Path(tmp))
        weights, config = load_model_weights(model_path)
        assert len(weights) > 0
        assert "hidden_size" in config


def test_get_quantizable_keys():
    with tempfile.TemporaryDirectory() as tmp:
        model_path = make_fake_model(Path(tmp))
        weights, _ = load_model_weights(model_path)
        keys = get_quantizable_keys(weights)
        assert len(keys) > 0
        for k in keys:
            assert "proj" in k or "gate" in k or "up" in k or "down" in k


@pytest.mark.parametrize("bits", [4, 3])
def test_convert_and_save_mse(bits):
    with tempfile.TemporaryDirectory() as src_tmp, tempfile.TemporaryDirectory() as dst_tmp:
        src = make_fake_model(Path(src_tmp))
        dst = Path(dst_tmp)

        weights, config = load_model_weights(src)
        quantized = convert_model(weights, config, bits=bits, variant="mse", block_size=32, seed=42, verbose=False)

        # Check quantized keys are present for each layer
        assert any("tq_indices" in k for k in quantized)
        assert any("tq_scales" in k for k in quantized)
        assert any("tq_signs" in k for k in quantized)
        # Check norm/embedding keys pass through
        assert any("layernorm" in k for k in quantized)

        save_quantized_model(dst, quantized, config, bits=bits, variant="mse",
                             block_size=32, original_model=str(src), max_shard_gb=1.0)

        # Reload and check config
        _, saved_config = load_model_weights(dst)
        assert saved_config["quantization"]["quant_method"] == "turboquant"
        assert saved_config["quantization"]["bits"] == bits


def test_no_weight_key_after_quantization():
    """Original .weight keys should be replaced by .tq_* keys."""
    with tempfile.TemporaryDirectory() as tmp:
        src = make_fake_model(Path(tmp))
        weights, config = load_model_weights(src)
        quantized = convert_model(weights, config, bits=4, variant="mse",
                                  block_size=32, seed=0, verbose=False)
        for key in quantized:
            if ".tq_" not in key:
                # Remaining keys should be non-quantizable (norms, embeddings, biases)
                assert not (key.endswith(".weight") and weights.get(key, np.zeros(1)).ndim == 2
                            and "proj" in key), f"Quantizable key {key!r} was not quantized"
