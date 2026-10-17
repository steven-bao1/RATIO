"""Resolve auxiliary tensors without tying checkpoints to their export path."""
from pathlib import Path


def partition_kv_parameter(value, num_kv_heads, head_dim, tp_rank, tp_size):
    """Shard or replicate KV channels in the same order as QKVParallelLinear."""
    if (any(type(n) is not int or n <= 0 for n in (num_kv_heads, head_dim, tp_size))
            or type(tp_rank) is not int or not 0 <= tp_rank < tp_size):
        raise ValueError("Invalid tensor-parallel KV geometry")
    if value.ndim != 2 or value.shape != (num_kv_heads * head_dim, 1):
        raise ValueError("KV scale/zero must have shape (num_kv_heads * head_dim, 1)")
    if num_kv_heads >= tp_size:
        if num_kv_heads % tp_size:
            raise ValueError("KV head count must be divisible by tensor-parallel size")
        width = (num_kv_heads // tp_size) * head_dim
        start = tp_rank * width
    else:
        if tp_size % num_kv_heads:
            raise ValueError("Tensor-parallel size must be divisible by KV head count")
        width = head_dim
        start = (tp_rank // (tp_size // num_kv_heads)) * head_dim
    return value.narrow(0, start, width)


def resolve_kv_scale_path(model_path, configured_path):
    """Prefer the bundled KV scales; support legacy external paths as fallback.

    Earlier exports stored the original output directory in k_scale_path. A
    checkpoint-local copy must win even if that old directory still exists.
    """
    root = Path(model_path)
    bundled = root / "pre_rope_k_scale_zero.pth"
    if bundled.is_file():
        return str(bundled)
    path = Path(configured_path)
    candidates = [path] if path.is_absolute() else [root / path, path]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"Missing KVQuant scales in {root}; configured path: {configured_path}")
