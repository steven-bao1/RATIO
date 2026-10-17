"""Evaluate an upstream-compatible model, optionally applying logit penalties."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from evaluation import atomic_json, responses_from_details, summarize
from penalty import FIXED_PENALTY_CANDIDATES, load_bias_config

TASKS = {
    "AIME-2024": ("aime24", "reasoning.py"),
    "AIME-2025": ("aime25", "reasoning.py"),
    "AIME-90": ("aime90", "reasoning.py"),
    "AIME-120": ("aime90,aime25", "reasoning.py"),
    "GPQA-Diamond": ("gpqa:diamond", "reasoning.py"),
    "GSM8K": ("gsm8k", "reasoning.py"),
    "MATH-500": ("math_500", "reasoning.py"),
    "HumanEval": ("humaneval", "humaneval.py"),
    "NuminaMath-1.5": ("numina_math", "reasoning.py"),
    "LiveCodeBench": ("lcb:codegeneration", "livecodebench.py"),
}


def parser_gen(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="Your checkpoint path (absolute or relative); no prescribed parent directory")
    p.add_argument("--dataset", choices=TASKS, required=True)
    p.add_argument("--output_dir", required=True, help="Unique run directory; never overwritten")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "auto"], default="bfloat16")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=-1,
                   help="-1 disables top-k (default); otherwise a positive integer, independent of model path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--engine_seed", type=int, default=1234,
                   help="Historical vLLM engine seed; sampling seed remains --seed (42 by default)")
    p.add_argument("--max_new_tokens", type=int, default=32768)
    p.add_argument("--max_model_length", type=int, default=32768)
    p.add_argument("--max_samples", type=int)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--fixed_penalty", type=float, default=0.0,
                       help="Uniform raw-logit penalty on the Meta paper's 50 markers; "
                            f"paper candidates: {', '.join(map(str, FIXED_PENALTY_CANDIDATES))}. "
                            "0 disables it; other nonnegative strengths are allowed.")
    group.add_argument("--token_bias_map", help='JSON: {"token_biases": {"123": -1.0}}')
    p.add_argument("--allow-code-execution", action="store_true",
                   help="Code benchmarks execute generated code; use an external sandbox")
    p.add_argument("--dry-run", action="store_true", help="Print config without loading model/GPU")
    args = p.parse_args(argv)
    if args.top_k != -1 and args.top_k <= 0:
        p.error("top_k must be -1 (disabled) or a positive integer")
    for name in ("tensor_parallel_size", "max_new_tokens", "max_model_length"):
        if getattr(args, name) <= 0:
            p.error(f"{name} must be positive")
    # Match historical inference: a requested generation cap may equal/exceed
    # the context size. vLLM stops at the remaining context after the prompt.
    if args.max_samples is not None and args.max_samples <= 0:
        p.error("max_samples must be positive")
    if not 0 < args.gpu_memory_utilization <= 1 or not 0 < args.top_p <= 1:
        p.error("GPU utilization and top_p must be in (0,1]")
    for name in ("temperature", "fixed_penalty"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            p.error(f"{name} must be finite and non-negative")
    if args.dataset in ("HumanEval", "LiveCodeBench") and not args.allow_code_execution:
        p.error("Code benchmarks require --allow-code-execution and an external sandbox")
    return args


def run_config(args):
    config = vars(args).copy()
    config.pop("dry_run")
    config["schema_version"] = 1
    biases, metadata = load_bias_config(args.token_bias_map) if args.token_bias_map else (None, None)
    config["token_biases"], config["token_bias_metadata"] = biases, metadata
    if args.token_bias_map:
        config["token_bias_map_sha256"] = hashlib.sha256(Path(args.token_bias_map).read_bytes()).hexdigest()
    return config


def evaluate(config):
    # --help/--dry-run and result aggregation work without CUDA packages.
    import transformers
    from lighteval.models.model_input import GenerationParameters
    from lighteval_custom.models.vllm.vllm_model import VLLMModelConfig
    from lighteval_custom.main_vllm import vllm
    from vllm_custom.model_executor.fake_quantized_models.registry import register_fake_quantized_models
    register_fake_quantized_models()
    transformers.set_seed(config["seed"])
    generation = GenerationParameters(
        temperature=config["temperature"], top_p=config["top_p"], top_k=config["top_k"],
        max_new_tokens=config["max_new_tokens"], seed=config["seed"])
    model_config = VLLMModelConfig(
        pretrained=config["model"], dtype=config["dtype"], seed=config["engine_seed"],
        max_model_length=config["max_model_length"], tensor_parallel_size=config["tensor_parallel_size"],
        gpu_memory_utilization=config["gpu_memory_utilization"], enforce_eager=True,
        enable_prefix_caching=False, enable_chunked_prefill=False, generation_parameters=generation,
        init_model=True, fixed_penalty=config["fixed_penalty"], token_biases=config["token_biases"],
        token_bias_metadata=config.get("token_bias_metadata"))
    task_names, task_file = TASKS[config["dataset"]]
    _, details = vllm(
        model_config=model_config, use_chat_template=True,
        tasks=",".join(f"custom|{name}|0|0" for name in task_names.split(",")),
        custom_tasks=str(Path(__file__).parent / "lighteval_custom" / "tasks" / task_file),
        max_samples=config["max_samples"], output_dir=config["output_dir"])
    return responses_from_details(details)


def main(args):
    config = run_config(args)
    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return
    if Path.cwd().resolve() != Path(__file__).parent.resolve():
        raise ValueError("Run from the project root containing inference.py (see README)")
    run_dir = Path(args.output_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=False)  # Prevent stale result reuse and duplicate writers.
    atomic_json(run_dir / "config.json", config)
    atomic_json(run_dir / "status.json", {"state": "running"})
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if args.allow_code_execution:
        os.environ["RATIO_ALLOW_CODE_EXECUTION"] = "1"
    try:
        rows = evaluate(config)
        atomic_json(run_dir / "responses.json", rows)
        summary = summarize(rows)
        atomic_json(run_dir / "summary.json", summary)
        atomic_json(run_dir / "status.json", {"state": "complete"})
        print(json.dumps(summary, indent=2))
    except BaseException as exc:
        atomic_json(run_dir / "status.json", {"state": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main(parser_gen())
