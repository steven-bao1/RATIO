"""I/O helpers and stable schemas for shared-prefix analysis artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa


SCORE_SCHEMA = pa.schema(
    [
        ("trajectory_index", pa.int32()),
        ("trajectory_id", pa.string()),
        ("source_task", pa.string()),
        ("rollout_metric_name", pa.string()),
        ("rollout_metric_value", pa.float32()),
        ("rollout_correct", pa.bool_()),
        ("prompt_token_count", pa.int32()),
        ("response_position", pa.int32()),
        ("sequence_position", pa.int32()),
        ("token_id", pa.int32()),
        ("raw_logprob", pa.float32()),
        ("raw_probability", pa.float32()),
        ("sampling_logprob", pa.float32()),
        ("sampling_probability", pa.float32()),
        ("truncated_logprob", pa.float32()),
        ("truncated_probability", pa.float32()),
        ("rank", pa.int32()),
        ("cumulative_mass_before", pa.float32()),
        ("in_nucleus", pa.bool_()),
        ("nucleus_size", pa.int32()),
        ("nucleus_token_ids", pa.list_(pa.int32())),
        ("nucleus_logprobs", pa.list_(pa.float32())),
        ("nucleus_truncated_logprobs", pa.list_(pa.float32())),
    ]
)


COMPARISON_SCHEMA = pa.schema(
    [
        ("trajectory_index", pa.int32()),
        ("trajectory_id", pa.string()),
        ("source_task", pa.string()),
        ("prompt_token_count", pa.int32()),
        ("response_position", pa.int32()),
        ("sequence_position", pa.int32()),
        ("token_id", pa.int32()),
        ("quant_raw_logprob", pa.float32()),
        ("bf16_raw_logprob", pa.float32()),
        ("quant_minus_bf16_raw_logprob", pa.float32()),
        ("quant_sampling_logprob", pa.float32()),
        ("bf16_sampling_logprob", pa.float32()),
        ("quant_truncated_logprob", pa.float32()),
        ("bf16_truncated_logprob", pa.float32()),
        ("quant_rank", pa.int32()),
        ("bf16_rank", pa.int32()),
        ("quant_cumulative_mass_before", pa.float32()),
        ("bf16_cumulative_mass_before", pa.float32()),
        ("actual_in_quant_nucleus", pa.bool_()),
        ("actual_in_bf16_nucleus", pa.bool_()),
        ("actual_quant_only", pa.bool_()),
        ("quant_nucleus_size", pa.int32()),
        ("bf16_nucleus_size", pa.int32()),
        ("intersection_size", pa.int32()),
        ("union_size", pa.int32()),
        ("nucleus_jaccard", pa.float32()),
        ("nucleus_overlap_coefficient", pa.float32()),
        ("quant_only_token_ids", pa.list_(pa.int32())),
        ("quant_only_logprobs", pa.list_(pa.float32())),
        ("bf16_only_token_ids", pa.list_(pa.int32())),
        ("bf16_only_logprobs", pa.list_(pa.float32())),
    ]
)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        while block := reader.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def trajectory_id(prompt_token_ids: list[int], generated_token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for label, values in ((b"prompt", prompt_token_ids), (b"response", generated_token_ids)):
        digest.update(label)
        digest.update(len(values).to_bytes(8, "little"))
        for token_id in values:
            digest.update(int(token_id).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def prompt_token_id_sha256(prompt_token_ids: list[int]) -> str:
    """Hash exact prompt IDs using the experiment's stable binary encoding."""
    digest = hashlib.sha256()
    for token_id in prompt_token_ids:
        digest.update(int(token_id).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def rollout_correctness(
    row: dict,
) -> tuple[str | None, float | None, bool | None]:
    """Extract the single LightEval correctness metric saved with a rollout."""
    metrics = row.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        # Keep the scorer backward compatible with synthetic/legacy exact-ID
        # trajectories. Formal experiment rollouts always contain this field.
        return None, None, None
    if len(metrics) != 1:
        raise ValueError("Expected exactly one correctness metric")
    metric_name, metric_value = next(iter(metrics.items()))
    value = float(metric_value)
    if not 0 <= value <= 1:
        raise ValueError("Correctness metric must be in [0, 1]")
    return str(metric_name), value, bool(value > 0.5)


def load_exact_trajectories(path: Path, max_samples: int | None = None) -> list[dict]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise TypeError(f"Expected a JSON list in {path}")
    if max_samples is not None:
        rows = rows[:max_samples]
    for index, row in enumerate(rows):
        # Public inference always saves exact input IDs; never re-tokenize text.
        if "prompt_token_ids" not in row and "input_token_ids" in row:
            row["prompt_token_ids"] = row["input_token_ids"]
        if "prompt_token_ids" not in row or "generated_token_ids" not in row:
            raise ValueError(
                f"Row {index} has no exact token IDs. Regenerate the no-penalty "
                "rollout with the collection entry point; decoded text must "
                "not be re-tokenized for a shared-prefix experiment."
            )
        prompt_ids = row["prompt_token_ids"]
        response_ids = row["generated_token_ids"]
        for values in (prompt_ids, response_ids):
            if not isinstance(values, list) or any(type(value) is not int or value < 0 for value in values):
                raise ValueError(f"Trajectory {index} contains invalid exact token IDs")
        if not prompt_ids or not response_ids:
            raise ValueError(f"Row {index} has an empty prompt or response")
        if int(row.get("num_generated_tokens", len(response_ids))) != len(response_ids):
            raise ValueError(f"Row {index} generated-token count does not match its IDs")
        if float(row.get("overthinking_penalty", 0.0)) != 0.0 or float(row.get("fixed_penalty", 0.0)) != 0.0 or row.get("token_biases"):
            raise ValueError(
                f"Row {index} was generated with a nonzero penalty; only normal "
                "quantized rollouts are valid for this experiment"
            )
        if row.get("rollout_temperature") is not None and float(
            row["rollout_temperature"]
        ) <= 0.0:
            raise ValueError(f"Row {index} has invalid rollout temperature")
    return rows


def rows_to_table(rows: list[dict], schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=schema)
