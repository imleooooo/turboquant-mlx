"""Command-line interface for turboquant-mlx."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="turboquant-mlx",
        description="Quantize MLX language models using the TurboQuant algorithm (arXiv:2504.19874)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m", required=True,
        help="Path to input MLX model directory (must contain safetensors + config.json)",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to output directory for the quantized model",
    )
    parser.add_argument(
        "--bits", type=int, default=4, choices=[2, 3, 4, 8],
        help="Quantization bit width",
    )
    parser.add_argument(
        "--variant", choices=["mse", "prod"], default="mse",
        help=(
            "TurboQuant variant: "
            "'mse' (Algorithm 1, MSE-optimal) or "
            "'prod' (Algorithm 2, unbiased inner product via QJL residual)"
        ),
    )
    parser.add_argument(
        "--block-size", type=int, default=4096,
        help="SRHT block size (must be a power of 2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed for rotation matrices",
    )
    parser.add_argument(
        "--max-shard-gb", type=float, default=4.0,
        help="Maximum shard size in GB",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-layer quantization progress",
    )
    parser.add_argument(
        "--list-layers", action="store_true",
        help="List quantizable layers and exit (dry run)",
    )

    args = parser.parse_args()

    # Validate block_size is power of 2
    if args.block_size & (args.block_size - 1) != 0:
        parser.error("--block-size must be a power of 2 (e.g. 64, 128, 256, 512, 1024, 4096)")

    from .model_io import load_model_weights, save_quantized_model
    from .convert import convert_model, get_quantizable_keys

    model_path = Path(args.model)
    output_path = Path(args.output)

    print(f"Loading model from {model_path} ...")
    weights, config = load_model_weights(model_path)
    print(f"Loaded {len(weights)} tensors.")

    if args.list_layers:
        keys = get_quantizable_keys(weights)
        print(f"\nQuantizable layers ({len(keys)}):")
        for k in keys:
            shape = list(weights[k].shape)
            print(f"  {k}  {shape}")
        sys.exit(0)

    print(f"\nQuantizing with TurboQuant_{args.variant}, {args.bits}-bit, block_size={args.block_size} ...")
    quantized = convert_model(
        weights, config,
        bits=args.bits,
        variant=args.variant,
        block_size=args.block_size,
        seed=args.seed,
        verbose=args.verbose,
    )

    print(f"\nSaving to {output_path} ...")
    save_quantized_model(
        output_path, quantized, config,
        bits=args.bits,
        variant=args.variant,
        block_size=args.block_size,
        original_model=str(model_path),
        max_shard_gb=args.max_shard_gb,
    )

    print("Done.")
