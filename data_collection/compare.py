#!/usr/bin/env python3
"""Align quantized/BF16 score shards and compare their top-p nuclei."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

from data_collection.io import (
    COMPARISON_SCHEMA,
    atomic_write_json,
    rows_to_table,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant-scores", type=Path, required=True)
    parser.add_argument("--bf16-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def iter_parquet_rows(path: Path, batch_size: int):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def compare_row(quant: dict, bf16: dict) -> dict:
    alignment_fields = (
        "trajectory_index",
        "trajectory_id",
        "source_task",
        "prompt_token_count",
        "response_position",
        "sequence_position",
        "token_id",
    )
    for field in alignment_fields:
        if quant[field] != bf16[field]:
            raise ValueError(
                f"Shared-prefix alignment mismatch for {field}: "
                f"{quant[field]!r} != {bf16[field]!r}"
            )

    quant_ids = quant["nucleus_token_ids"]
    bf16_ids = bf16["nucleus_token_ids"]
    quant_set = set(quant_ids)
    bf16_set = set(bf16_ids)
    intersection = quant_set & bf16_set
    union = quant_set | bf16_set
    quant_only = [token_id for token_id in quant_ids if token_id not in bf16_set]
    bf16_only = [token_id for token_id in bf16_ids if token_id not in quant_set]
    quant_lp = dict(zip(quant_ids, quant["nucleus_logprobs"]))
    bf16_lp = dict(zip(bf16_ids, bf16["nucleus_logprobs"]))
    minimum_size = min(len(quant_set), len(bf16_set))
    token_id = quant["token_id"]

    return {
        "trajectory_index": quant["trajectory_index"],
        "trajectory_id": quant["trajectory_id"],
        "source_task": quant["source_task"],
        "prompt_token_count": quant["prompt_token_count"],
        "response_position": quant["response_position"],
        "sequence_position": quant["sequence_position"],
        "token_id": token_id,
        "quant_raw_logprob": quant["raw_logprob"],
        "bf16_raw_logprob": bf16["raw_logprob"],
        "quant_minus_bf16_raw_logprob": (
            quant["raw_logprob"] - bf16["raw_logprob"]
        ),
        "quant_sampling_logprob": quant["sampling_logprob"],
        "bf16_sampling_logprob": bf16["sampling_logprob"],
        "quant_truncated_logprob": quant["truncated_logprob"],
        "bf16_truncated_logprob": bf16["truncated_logprob"],
        "quant_rank": quant["rank"],
        "bf16_rank": bf16["rank"],
        "quant_cumulative_mass_before": quant["cumulative_mass_before"],
        "bf16_cumulative_mass_before": bf16["cumulative_mass_before"],
        "actual_in_quant_nucleus": quant["in_nucleus"],
        "actual_in_bf16_nucleus": bf16["in_nucleus"],
        "actual_quant_only": token_id in quant_set and token_id not in bf16_set,
        "quant_nucleus_size": len(quant_set),
        "bf16_nucleus_size": len(bf16_set),
        "intersection_size": len(intersection),
        "union_size": len(union),
        "nucleus_jaccard": len(intersection) / len(union) if union else 1.0,
        "nucleus_overlap_coefficient": (
            len(intersection) / minimum_size if minimum_size else 1.0
        ),
        "quant_only_token_ids": quant_only,
        "quant_only_logprobs": [quant_lp[token_id] for token_id in quant_only],
        "bf16_only_token_ids": bf16_only,
        "bf16_only_logprobs": [bf16_lp[token_id] for token_id in bf16_only],
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    quant_manifest = json.loads((args.quant_scores / "manifest.json").read_text())
    bf16_manifest = json.loads((args.bf16_scores / "manifest.json").read_text())
    if quant_manifest.get("role") != "quant" or bf16_manifest.get("role") != "bf16":
        raise ValueError("Score directories must have quant and bf16 roles respectively")
    quant_backend = quant_manifest.get("execution_backend")
    if quant_backend is None and quant_manifest.get("score_source") == "online-vllm-generation":
        quant_backend = "vllm"
    bf16_backend = bf16_manifest.get("execution_backend")
    for manifest in (quant_manifest, bf16_manifest):
        versions = manifest.get("online_record_schema_versions", [])
        if manifest.get("nucleus_method") == "vllm" and versions and min(versions) < 5:
            raise ValueError("vLLM recording schema version must be at least 5")
    if quant_manifest.get("sampler_parameter_policy") != bf16_manifest.get("sampler_parameter_policy"):
        raise ValueError("Quant and BF16 used different sampler parameter precision policies")
    if bf16_manifest.get("score_source") == "forced-vllm-replay":
        if quant_backend != "vllm" or bf16_backend != "vllm":
            raise ValueError(
                "Forced BF16 replay comparisons require vLLM on both sides"
            )
        if not bf16_manifest.get("forced_replay_verified_exact"):
            raise ValueError("BF16 manifest does not verify exact fixed-prefix replay")
        if quant_manifest.get("vllm_version") != bf16_manifest.get("vllm_version"):
            raise ValueError("Quant and BF16 replay used different vLLM versions")
        if quant_manifest.get("vllm_sampler_sha256") != bf16_manifest.get(
            "vllm_sampler_sha256"
        ):
            raise ValueError("Quant and BF16 replay used different vLLM samplers")
    for field in (
        "rollouts_sha256",
        "tokenizer_vocab_sha256",
        "temperature",
        "top_p",
        "nucleus_method",
    ):
        if quant_manifest.get(field) != bf16_manifest.get(field):
            raise ValueError(f"Score manifest mismatch for {field}")

    quant_trajectories = quant_manifest["trajectories"]
    bf16_trajectories = bf16_manifest["trajectories"]
    if not quant_trajectories or not bf16_trajectories:
        raise ValueError("Cannot compare empty score collections")
    if quant_manifest.get("fixed_prefix_replay") and not quant_manifest.get("forced_replay_verified_exact"):
        raise ValueError("Quantized fixed-prefix replay has not been verified")
    if len({row["trajectory_id"] for row in bf16_trajectories}) != len(bf16_trajectories):
        raise ValueError("BF16 manifest contains duplicate trajectories")
    quant_by_id = {
        record["trajectory_id"]: record for record in quant_trajectories
    }
    if len(quant_by_id) != len(quant_trajectories):
        raise ValueError("Quantized manifest contains duplicate trajectory IDs")
    trajectory_pairs = []
    for bf16_record in bf16_trajectories:
        quant_record = quant_by_id.get(bf16_record["trajectory_id"])
        if quant_record is None:
            raise ValueError(
                "BF16 selection contains a trajectory absent from quant scores: "
                f"{bf16_record['trajectory_id']}"
            )
        trajectory_pairs.append((quant_record, bf16_record))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    input_config = {
        "schema_version": 1,
        "quant_manifest": str((args.quant_scores / "manifest.json").resolve()),
        "quant_manifest_sha256": sha256_file(args.quant_scores / "manifest.json"),
        "bf16_manifest": str((args.bf16_scores / "manifest.json").resolve()),
        "bf16_manifest_sha256": sha256_file(args.bf16_scores / "manifest.json"),
        "batch_size": args.batch_size,
    }
    input_config_path = args.output_dir / "comparison-inputs.json"
    atomic_write_json(input_config_path, input_config)
    output_records = []
    for ordinal, (quant_record, bf16_record) in enumerate(trajectory_pairs):
        index = int(quant_record["trajectory_index"])
        if index != int(bf16_record["trajectory_index"]):
            raise ValueError(
                f"Trajectory index differs for {quant_record['trajectory_id']}: "
                f"{index} != {bf16_record['trajectory_index']}"
            )
        digest = quant_record["trajectory_id"]
        output_path = args.output_dir / f"trajectory-{index:05d}-{digest[:12]}.parquet"
        quant_path = args.quant_scores / quant_record["file"]
        bf16_path = args.bf16_scores / bf16_record["file"]
        temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
        writer = pq.ParquetWriter(
            temporary,
            COMPARISON_SCHEMA,
            compression="zstd",
            use_dictionary=True,
        )
        compared = 0
        try:
            quant_rows = iter_parquet_rows(quant_path, args.batch_size)
            bf16_rows = iter_parquet_rows(bf16_path, args.batch_size)
            pending = []
            sentinel = object()
            for quant_row, bf16_row in itertools.zip_longest(
                quant_rows, bf16_rows, fillvalue=sentinel
            ):
                if quant_row is sentinel or bf16_row is sentinel:
                    raise ValueError(f"Shard row count mismatch for trajectory {index}")
                pending.append(compare_row(quant_row, bf16_row))
                if len(pending) >= args.batch_size:
                    writer.write_table(rows_to_table(pending, COMPARISON_SCHEMA))
                    compared += len(pending)
                    pending.clear()
            if pending:
                writer.write_table(rows_to_table(pending, COMPARISON_SCHEMA))
                compared += len(pending)
        except Exception:
            writer.close()
            temporary.unlink(missing_ok=True)
            raise
        writer.close()
        if compared != quant_record["response_token_count"]:
            temporary.unlink(missing_ok=True)
            raise AssertionError(
                f"Compared {compared} rows, expected {quant_record['response_token_count']}"
            )
        os.replace(temporary, output_path)
        print(
            f"[{ordinal + 1}/{len(trajectory_pairs)}] WROTE "
            f"{output_path.name} ({compared} positions)"
        )
        output_records.append({**quant_record, "file": output_path.name})

    manifest = {
        "schema_version": 2,
        "kind": "shared-prefix-quant-bf16-comparison",
        "quant_model": quant_manifest["model"],
        "bf16_model": bf16_manifest["model"],
        "rollouts": quant_manifest["rollouts"],
        "rollouts_sha256": quant_manifest["rollouts_sha256"],
        "tokenizer": quant_manifest["tokenizer"],
        "tokenizer_vocab_sha256": quant_manifest["tokenizer_vocab_sha256"],
        "temperature": quant_manifest["temperature"],
        "top_p": quant_manifest["top_p"],
        "nucleus_method": quant_manifest.get("nucleus_method", "minimal"),
        "quant_execution_backend": quant_backend,
        "bf16_execution_backend": bf16_backend,
        "backend_matched": bool(
            quant_backend is not None and quant_backend == bf16_backend
        ),
        "quant_manifest_sha256": input_config["quant_manifest_sha256"],
        "bf16_manifest_sha256": input_config["bf16_manifest_sha256"],
        "quant_trajectory_count": len(quant_trajectories),
        "bf16_selected_trajectory_count": len(bf16_trajectories),
        "bf16_selected_original_indices": [
            record["trajectory_index"] for record in bf16_trajectories
        ],
        "trajectories": output_records,
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    print(f"Comparison complete: {args.output_dir}")


if __name__ == "__main__":
    main()
