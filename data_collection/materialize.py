#!/usr/bin/env python3
"""Join online vLLM nucleus records with sampled rollout token IDs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path

import msgpack
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from data_collection.io import (
    SCORE_SCHEMA,
    atomic_write_json,
    load_exact_trajectories,
    prompt_token_id_sha256,
    rollout_correctness,
    rows_to_table,
    sha256_file,
    trajectory_id,
)
from data_collection.online import ONLINE_SCHEMA_VERSION, ONLINE_POST_SAMPLER_SCHEMA_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-indices", type=Path, default=None)
    parser.add_argument("--role", choices=("quant", "bf16"), default="quant")
    parser.add_argument(
        "--score-source",
        choices=("online-vllm-generation", "forced-vllm-replay"),
        default="online-vllm-generation",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--initial-top-k", type=int, default=128)
    parser.add_argument(
        "--nucleus-method", choices=("minimal", "vllm"), default="vllm"
    )
    parser.add_argument("--buffer-rows", type=int, default=256)
    return parser.parse_args()


def tokenizer_fingerprint(tokenizer) -> str:
    digest = hashlib.sha256()
    for token, token_id in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
        digest.update(int(token_id).to_bytes(4, "little", signed=False))
        digest.update(token.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def vllm_sampler_fingerprint() -> tuple[str, str]:
    spec = importlib.util.find_spec("vllm.model_executor.layers.sampler")
    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot locate the active vLLM sampler implementation")
    path = Path(spec.origin).resolve()
    return str(path), sha256_file(path)


def materialize(args: argparse.Namespace) -> None:
    if args.buffer_rows <= 0:
        raise ValueError("buffer_rows must be positive")
    if args.score_source == "forced-vllm-replay":
        if args.sample_indices is None:
            raise ValueError("Forced vLLM replay requires --sample-indices")
    elif args.role != "quant":
        raise ValueError("Online vLLM generation records must use role=quant")
    trajectories = load_exact_trajectories(args.rollouts)
    selected_trajectories = list(enumerate(trajectories))
    if args.sample_indices is not None:
        selection = json.loads(args.sample_indices.read_text())
        indices = selection.get("indices")
        if not isinstance(indices, list) or not indices:
            raise ValueError("Selection file must contain a non-empty indices list")
        indices = [int(index) for index in indices]
        if len(indices) != len(set(indices)):
            raise ValueError("Selection file contains duplicate indices")
        if min(indices) < 0 or max(indices) >= len(trajectories):
            raise ValueError("Selection index is outside the rollout population")
        selected_trajectories = [(index, trajectories[index]) for index in indices]
    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "manifest.json").exists() or any(args.output_dir.glob("trajectory-*.parquet")):
        raise FileExistsError("Score artifacts already exist; use a new output directory")

    contexts: dict[bytes, dict] = {}
    shard_records = []
    record_versions: set[int] = set()
    missing_sampled_tokens: list[dict] = []
    for index, row in selected_trajectories:
        recorded_method = row.get("rollout_online_nucleus_method")
        # For online-generation scores, the rollout itself owns the recorded
        # nucleus distribution and therefore must declare the requested method.
        # In forced replay, however, the source rollout supplies only the fixed
        # token trajectory; the replay records own the distribution.
        if args.score_source == "online-vllm-generation":
            if recorded_method is not None and recorded_method != args.nucleus_method:
                raise ValueError(
                    f"Trajectory {index} recorded nucleus_method={recorded_method}, "
                    f"but materialization requested {args.nucleus_method}"
                )
            if args.nucleus_method == "vllm" and recorded_method != "vllm":
                raise ValueError(
                    f"Trajectory {index} does not declare the required exact vLLM "
                    "online nucleus method"
                )
        prompt_ids = [int(value) for value in row["prompt_token_ids"]]
        response_ids = [int(value) for value in row["generated_token_ids"]]
        prompt_digest = prompt_token_id_sha256(prompt_ids)
        key = bytes.fromhex(prompt_digest)[:16]
        if key in contexts:
            raise ValueError("Duplicate prompts or truncated prompt-hash collision")
        digest = trajectory_id(prompt_ids, response_ids)
        metric_name, metric_value, correct = rollout_correctness(row)
        output_path = args.output_dir / f"trajectory-{index:05d}-{digest[:12]}.parquet"
        temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
        metadata = {
            b"trajectory_id": digest.encode(),
            b"source_task": str(row.get("source_task", "")).encode(),
            b"prompt_token_count": str(len(prompt_ids)).encode(),
            b"response_token_count": str(len(response_ids)).encode(),
            b"score_source": args.score_source.encode(),
        }
        contexts[key] = {
            "index": index,
            "row": row,
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "digest": digest,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "correct": correct,
            "output_path": output_path,
            "temporary": temporary,
            "writer": None,
            "metadata": metadata,
            "buffer": [],
            "written": 0,
            "missing_sampled_tokens": 0,
        }

    def flush(context: dict) -> None:
        if context["buffer"]:
            if context["writer"] is None:
                context["writer"] = pq.ParquetWriter(
                    context["temporary"], SCORE_SCHEMA.with_metadata(context["metadata"]),
                    compression="zstd", use_dictionary=True)
            context["writer"].write_table(
                rows_to_table(context["buffer"], SCORE_SCHEMA)
            )
            context["written"] += len(context["buffer"])
            context["buffer"].clear()
            if context["written"] == len(context["response_ids"]):
                context["writer"].close()
                context["writer"] = None

    try:
        for records_path in args.records:
            with records_path.open("rb") as stream:
                unpacker = msgpack.Unpacker(stream, raw=False, max_buffer_size=2**31)
                for packed in unpacker:
                    if not isinstance(packed, list) or len(packed) not in (6, 7, 12):
                        raise ValueError("Malformed online nucleus record")
                    version = int(packed[0])
                    if args.nucleus_method == "vllm" and version < ONLINE_POST_SAMPLER_SCHEMA_VERSION:
                        raise ValueError("Pre-fix vLLM records lack sampler-dtype temperature alignment; recollect, do not relabel")
                    replay_target = None
                    if version == 1 and len(packed) == 6:
                        key, position, ids_raw, raw_lp_raw, sampling_lp_raw = packed[1:]
                        truncated_lp_raw = None
                    elif version in (2, 3, ONLINE_POST_SAMPLER_SCHEMA_VERSION) and len(packed) == 7:
                        (
                            _, key, position, ids_raw, raw_lp_raw,
                            sampling_lp_raw, truncated_lp_raw,
                        ) = packed
                    elif version == ONLINE_SCHEMA_VERSION and len(packed) == 12:
                        (
                            _, key, position, ids_raw, raw_lp_raw,
                            sampling_lp_raw, truncated_lp_raw, target_id,
                            target_raw_logprob, target_sampling_logprob,
                            target_rank, target_cumulative_mass_before,
                        ) = packed
                        replay_target = (
                            int(target_id),
                            float(target_raw_logprob),
                            float(target_sampling_logprob),
                            int(target_rank),
                            float(target_cumulative_mass_before),
                        )
                    else:
                        raise ValueError(f"Unsupported online record schema {version}")
                    if version == ONLINE_SCHEMA_VERSION and args.score_source != "forced-vllm-replay":
                        raise ValueError("Replay record used with online-generation score source")
                    if version != ONLINE_SCHEMA_VERSION and args.score_source == "forced-vllm-replay":
                        raise ValueError("Forced replay requires schema-6 target records")
                    record_versions.add(version)
                    context = contexts.get(key)
                    if context is None:
                        raise ValueError(f"Online record has unknown prompt key {key.hex()}")
                    position = int(position)
                    expected_position = context["written"] + len(context["buffer"])
                    if position != expected_position:
                        raise ValueError(
                            f"Trajectory {context['index']} expected position "
                            f"{expected_position}, found {position}"
                        )
                    nucleus_ids = np.frombuffer(ids_raw, dtype="<i4")
                    raw_logprobs = np.frombuffer(raw_lp_raw, dtype="<f4")
                    sampling_logprobs = np.frombuffer(sampling_lp_raw, dtype="<f4")
                    if truncated_lp_raw is None:
                        maximum = float(sampling_logprobs.max())
                        log_normalizer = maximum + math.log(
                            float(
                                np.exp(sampling_logprobs - maximum).sum(
                                    dtype=np.float64
                                )
                            )
                        )
                        truncated_logprobs = sampling_logprobs - log_normalizer
                    else:
                        truncated_logprobs = np.frombuffer(
                            truncated_lp_raw, dtype="<f4"
                        )
                    if not (
                        len(nucleus_ids)
                        == len(raw_logprobs)
                        == len(sampling_logprobs)
                        == len(truncated_logprobs)
                    ):
                        raise ValueError("Online nucleus arrays have different lengths")
                    if position >= len(context["response_ids"]):
                        raise ValueError("Online record exceeds generated response length")
                    target = context["response_ids"][position]
                    matches = np.flatnonzero(nucleus_ids == target)
                    if replay_target is not None:
                        (
                            replay_id,
                            target_raw_lp,
                            target_sampling_lp,
                            rank,
                            cumulative_mass_before,
                        ) = replay_target
                        if replay_id != target:
                            raise ValueError(
                                f"Replay target mismatch at trajectory "
                                f"{context['index']} position {position}"
                            )
                        in_nucleus = len(matches) == 1
                        target_truncated_lp = (
                            float(truncated_logprobs[int(matches[0])])
                            if in_nucleus
                            else float("-inf")
                        )
                    elif len(matches) == 0:
                        if version >= 3:
                            raise ValueError(
                                f"Post-sampler schema-{version} record is missing "
                                f"sampled token {target} at trajectory "
                                f"{context['index']} position {position}"
                            )
                        target_raw_lp = float("nan")
                        target_sampling_lp = float("nan")
                        target_truncated_lp = float("nan")
                        greater = np.zeros(len(raw_logprobs), dtype=bool)
                        rank = 0
                        cumulative_mass_before = float("nan")
                        in_nucleus = False
                        context["missing_sampled_tokens"] += 1
                        missing_sampled_tokens.append(
                            {
                                "trajectory_index": context["index"],
                                "response_position": position,
                                "token_id": target,
                                "record_schema_version": version,
                            }
                        )
                    else:
                        target_index = int(matches[0])
                        target_raw_lp = float(raw_logprobs[target_index])
                        target_sampling_lp = float(sampling_logprobs[target_index])
                        target_truncated_lp = float(truncated_logprobs[target_index])
                        greater = raw_logprobs > target_raw_lp
                        rank = int(greater.sum()) + 1
                        cumulative_mass_before = float(
                            np.exp(sampling_logprobs[greater]).sum(dtype=np.float64)
                        )
                        in_nucleus = True
                    source_task = str(context["row"].get("source_task", ""))
                    context["buffer"].append(
                        {
                            "trajectory_index": context["index"],
                            "trajectory_id": context["digest"],
                            "source_task": source_task,
                            "rollout_metric_name": context["metric_name"],
                            "rollout_metric_value": context["metric_value"],
                            "rollout_correct": context["correct"],
                            "prompt_token_count": len(context["prompt_ids"]),
                            "response_position": position,
                            "sequence_position": len(context["prompt_ids"]) + position,
                            "token_id": target,
                            "raw_logprob": target_raw_lp,
                            "raw_probability": math.exp(target_raw_lp),
                            "sampling_logprob": target_sampling_lp,
                            "sampling_probability": math.exp(target_sampling_lp),
                            "truncated_logprob": target_truncated_lp,
                            "truncated_probability": math.exp(target_truncated_lp),
                            "rank": rank,
                            "cumulative_mass_before": cumulative_mass_before,
                            "in_nucleus": in_nucleus,
                            "nucleus_size": len(nucleus_ids),
                            "nucleus_token_ids": nucleus_ids.tolist(),
                            "nucleus_logprobs": sampling_logprobs.tolist(),
                            "nucleus_truncated_logprobs": (
                                truncated_logprobs.tolist()
                            ),
                        }
                    )
                    if len(context["buffer"]) >= args.buffer_rows or position + 1 == len(context["response_ids"]):
                        flush(context)
    except Exception:
        for context in contexts.values():
            if context["writer"] is not None:
                context["writer"].close()
            context["temporary"].unlink(missing_ok=True)
        raise

    for context in contexts.values():
        flush(context)
        if context["writer"] is not None:
            context["writer"].close()
        expected = len(context["response_ids"])
        if context["written"] != expected:
            context["temporary"].unlink(missing_ok=True)
            raise ValueError(
                f"Trajectory {context['index']} has {context['written']} online "
                f"records, expected {expected}"
            )
        os.replace(context["temporary"], context["output_path"])
        shard_records.append(
            {
                "trajectory_index": context["index"],
                "trajectory_id": context["digest"],
                "source_task": str(context["row"].get("source_task", "")),
                "prompt_token_count": len(context["prompt_ids"]),
                "response_token_count": expected,
                "rollout_metric_name": context["metric_name"],
                "rollout_metric_value": context["metric_value"],
                "rollout_correct": context["correct"],
                "missing_sampled_token_positions": (
                    context["missing_sampled_tokens"]
                ),
                "file": context["output_path"].name,
            }
        )

    shard_records.sort(key=lambda item: item["trajectory_index"])
    sampler_path, sampler_sha256 = vllm_sampler_fingerprint()
    manifest = {
        "schema_version": 3,
        "kind": "shared-prefix-model-scores",
        "role": args.role,
        "score_source": args.score_source,
        "execution_backend": "vllm",
        "vllm_version": importlib.metadata.version("vllm"),
        "vllm_sampler_path": sampler_path,
        "vllm_sampler_sha256": sampler_sha256,
        "model": str(args.model.resolve()),
        "tokenizer": str(tokenizer_path.resolve()),
        "tokenizer_vocab_sha256": tokenizer_fingerprint(tokenizer),
        "rollouts": str(args.rollouts.resolve()),
        "rollouts_sha256": sha256_file(args.rollouts),
        "online_records": (
            str(args.records[0].resolve())
            if len(args.records) == 1
            else [str(path.resolve()) for path in args.records]
        ),
        "online_records_sha256": (
            sha256_file(args.records[0])
            if len(args.records) == 1
            else [sha256_file(path) for path in args.records]
        ),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "initial_top_k": args.initial_top_k,
        "nucleus_method": args.nucleus_method,
        "online_record_schema_version": (
            max(record_versions) if record_versions else None
        ),
        "online_record_schema_versions": sorted(record_versions),
        "sampler_parameter_policy": "incoming-logits-dtype" if args.nucleus_method == "vllm" else "requested-float",
        "exact_post_sampler_support": args.nucleus_method == "vllm" and bool(record_versions) and min(record_versions) >= 5,
        "fixed_prefix_replay": args.score_source == "forced-vllm-replay",
        "missing_sampled_token_count": len(missing_sampled_tokens),
        "missing_sampled_token_positions": missing_sampled_tokens,
        "total_rollout_trajectories": len(trajectories),
        "selected_trajectory_count": len(selected_trajectories),
        "selected_original_indices": [index for index, _ in selected_trajectories],
        "selection_file": (
            str(args.sample_indices.resolve()) if args.sample_indices is not None else None
        ),
        "selection_file_sha256": (
            sha256_file(args.sample_indices) if args.sample_indices is not None else None
        ),
        "trajectories": shard_records,
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    print(
        f"Materialized {len(shard_records)} {args.role} vLLM score shards; "
        f"missing sampled-token scores={len(missing_sampled_tokens)}"
    )


if __name__ == "__main__":
    materialize(parse_args())
