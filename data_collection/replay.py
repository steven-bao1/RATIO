#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_USE_V1"] = "0"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


from data_collection.io import (
    atomic_write_json,
    load_exact_trajectories,
    sha256_file,
)
from data_collection.online import OnlineNucleusRecorderLogitsProcessor
from data_collection.replay_processor import ForcedTrajectoryLogitsProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--sample-indices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("quant", "bf16"), default="bf16")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-length", type=int, default=65536)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser.parse_args()


def selected_trajectories(
    rollouts: list[dict], selection_path: Path
) -> list[tuple[int, dict]]:
    selection = json.loads(selection_path.read_text())
    indices = selection.get("indices")
    if not isinstance(indices, list) or not indices:
        raise ValueError("Selection file must contain a non-empty indices list")
    indices = [int(index) for index in indices]
    if len(indices) != len(set(indices)):
        raise ValueError("Selection file contains duplicate indices")
    if min(indices) < 0 or max(indices) >= len(rollouts):
        raise ValueError("Selection index is outside the rollout population")
    return [(index, rollouts[index]) for index in indices]


def run_replay(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from data_collection.backend import require_recording_backend
    require_recording_backend()
    from vllm import LLM, SamplingParams
    from vllm_custom.model_executor.fake_quantized_models.registry import register_fake_quantized_models
    register_fake_quantized_models()
    vllm_version = importlib.metadata.version("vllm")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    if not math.isfinite(args.temperature) or args.temperature <= 0.0:
        raise ValueError("--temperature must be positive")
    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be positive")

    rollouts = load_exact_trajectories(args.rollouts)
    selected = selected_trajectories(rollouts, args.sample_indices)
    requested_max_model_length = args.max_model_length
    required_length = max(
        len(row["prompt_token_ids"]) + len(row["generated_token_ids"])
        for _, row in selected
    )
    # A decoder can emit one last token from a prefix at the context limit.
    # Keep the complete fixed trajectory rather than dropping its last token.
    # Only this one-slot boundary case is allowed; larger overruns still fail.
    if required_length == requested_max_model_length + 1:
        args.max_model_length = required_length
        print(
            f"Replay reserves one final-token slot: "
            f"{requested_max_model_length} -> {args.max_model_length}",
            flush=True,
        )
    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    from data_collection.materialize import tokenizer_fingerprint
    model_tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer_fingerprint(model_tokenizer) != tokenizer_fingerprint(tokenizer):
        raise ValueError("Replay model and trajectory tokenizer have different token-ID mappings")
    vocab_limit = max(tokenizer.get_vocab().values()) + 1

    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output_dir / "manifest.json"
    records_dir = args.output_dir / "online-records"
    records_dir.mkdir(parents=True, exist_ok=True)
    record_paths: list[Path] = []
    prompts = []
    sampling_parameters = []
    expected_responses: list[tuple[int, ...]] = []

    for index, row in selected:
        prompt_ids = tuple(int(value) for value in row["prompt_token_ids"])
        response_ids = tuple(int(value) for value in row["generated_token_ids"])
        if not prompt_ids or not response_ids:
            raise ValueError(f"Trajectory {index} has an empty prompt or response")
        if len(prompt_ids) + len(response_ids) > args.max_model_length:
            raise ValueError(
                f"Trajectory {index} requires {len(prompt_ids) + len(response_ids)} "
                f"tokens, exceeding --max-model-length={args.max_model_length}"
            )
        highest_id = max(max(prompt_ids), max(response_ids))
        if highest_id >= vocab_limit:
            raise ValueError(
                f"Trajectory {index} token {highest_id} is outside tokenizer vocabulary"
            )

        record_path = records_dir / f"trajectory-{index:05d}.msgpack"
        recorder = OnlineNucleusRecorderLogitsProcessor(
            str(record_path),
            temperature=args.temperature,
            top_p=args.top_p,
            nucleus_method="vllm",
            replay_target_token_ids=response_ids,
            expected_prompt_token_ids=prompt_ids,
        )
        forcer = ForcedTrajectoryLogitsProcessor(
            prompt_token_ids=prompt_ids,
            response_token_ids=response_ids,
        )
        prompts.append({"prompt_token_ids": list(prompt_ids)})
        sampling_parameters.append(
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=-1,
                seed=args.seed,
                max_tokens=len(response_ids),
                ignore_eos=True,
                detokenize=False,
                logits_processors=[recorder, forcer],
            )
        )
        expected_responses.append(response_ids)
        record_paths.append(record_path)

    config = {
        "schema_version": 1,
        "kind": "shared-prefix-vllm-fixed-trajectory-replay",
        "role": args.role,
        "score_source": "forced-vllm-replay",
        "execution_backend": "vllm",
        "vllm_version": vllm_version,
        "model": str(args.model.resolve()),
        "tokenizer": str(tokenizer_path.resolve()),
        "rollouts": str(args.rollouts.resolve()),
        "rollouts_sha256": sha256_file(args.rollouts),
        "selection_file": str(args.sample_indices.resolve()),
        "selection_file_sha256": sha256_file(args.sample_indices),
        "selected_original_indices": [index for index, _ in selected],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "dtype": args.dtype,
        "max_model_length": args.max_model_length,
        "requested_max_model_length": requested_max_model_length,
        "final_token_extra_slot": args.max_model_length > requested_max_model_length,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "flashinfer_sampler_forced_disabled": True,
        "vllm_v1_forced_disabled": True,
        "logits_stage": "raw-model-logits-before-forced-token-processor",
    }
    atomic_write_json(args.output_dir / "replay-config.json", config)

    engine = LLM(
        model=str(args.model),
        tokenizer=str(tokenizer_path),
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        trust_remote_code=False,
    )
    outputs = engine.generate(
        prompts,
        sampling_parameters,
        use_tqdm=True,
    )
    if len(outputs) != len(selected):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} responses for {len(selected)} trajectories"
        )
    for (index, _), output, expected, record_path in zip(
        selected, outputs, expected_responses, record_paths
    ):
        if len(output.outputs) != 1:
            raise RuntimeError(f"Trajectory {index} returned multiple generations")
        actual = tuple(int(value) for value in output.outputs[0].token_ids)
        if actual != expected:
            mismatch = next(
                (
                    position
                    for position, (left, right) in enumerate(zip(actual, expected))
                    if left != right
                ),
                min(len(actual), len(expected)),
            )
            raise RuntimeError(
                f"Forced replay diverged at trajectory {index}, position {mismatch}"
            )
        if not record_path.is_file() or record_path.stat().st_size == 0:
            raise RuntimeError(f"Trajectory {index} produced no online records")

    command = [
        sys.executable,
        "-m",
        "data_collection.materialize",
        "--records",
        *(str(path) for path in record_paths),
        "--rollouts",
        str(args.rollouts),
        "--model",
        str(args.model),
        "--tokenizer",
        str(tokenizer_path),
        "--output-dir",
        str(args.output_dir),
        "--sample-indices",
        str(args.sample_indices),
        "--role",
        args.role,
        "--score-source",
        "forced-vllm-replay",
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--nucleus-method",
        "vllm",
    ]
    subprocess.run(command, check=True)

    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "vllm_version": vllm_version,
            "replay_config": str((args.output_dir / "replay-config.json").resolve()),
            "replay_config_sha256": sha256_file(args.output_dir / "replay-config.json"),
            "forced_replay_verified_exact": True,
            "dtype": args.dtype,
        }
    )
    atomic_write_json(manifest_path, manifest)
    print(
        f"Completed {args.role} vLLM replay scoring for {len(selected)} trajectories: "
        f"{args.output_dir}"
    )


def main() -> None:
    run_replay(parse_args())


if __name__ == "__main__":
    main()
