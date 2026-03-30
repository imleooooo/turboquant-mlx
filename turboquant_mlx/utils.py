"""Shared utilities."""

from __future__ import annotations

import hashlib
import re
from typing import Iterator

import numpy as np

# Weight key patterns for layers to quantize (covers Llama, Mistral, Gemma, Phi, Qwen)
_QUANTIZABLE_RE = re.compile(
    r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
    r"|query_key_value|dense|dense_h_to_4h|dense_4h_to_h"
    r"|fc1|fc2|out_proj)\.weight$"
)


def matches_quantizable_pattern(key: str) -> bool:
    return bool(_QUANTIZABLE_RE.search(key))


def seed_for_layer(layer_key: str, global_seed: int) -> int:
    """Deterministic per-layer seed derived from layer key + global seed."""
    h = hashlib.md5(f"{global_seed}:{layer_key}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def n_blocks(in_dim: int, block_size: int) -> int:
    return (in_dim + block_size - 1) // block_size


def make_shards(
    weights: dict[str, np.ndarray], max_gb: float = 4.0
) -> list[dict[str, np.ndarray]]:
    """Split a weights dict into shards each ≤ max_gb bytes."""
    max_bytes = int(max_gb * 1024**3)
    shards: list[dict[str, np.ndarray]] = [{}]
    current_bytes = 0
    for key, arr in weights.items():
        arr_bytes = arr.nbytes
        if current_bytes + arr_bytes > max_bytes and current_bytes > 0:
            shards.append({})
            current_bytes = 0
        shards[-1][key] = arr
        current_bytes += arr_bytes
    return shards


def iter_layer_groups(
    weights: dict[str, np.ndarray],
) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
    """Yield (prefix, sub_dict) grouping all keys sharing the same layer prefix."""
    from collections import defaultdict

    groups: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for key, arr in weights.items():
        # Find the longest matching quantized prefix (ending in .tq_*)
        if ".tq_" in key:
            prefix, suffix = key.split(".tq_", 1)
            groups[prefix][f"tq_{suffix}"] = arr
        else:
            # Pass-through key; not grouped
            yield key, {key: arr}
            continue
    for prefix, sub in groups.items():
        yield prefix, sub
