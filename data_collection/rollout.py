"""No-penalty rollout and independent correctness on a frozen question subset."""
import argparse
import json
import os
from pathlib import Path

from evaluation import atomic_json, responses_from_details, summarize
from data_collection.io import sha256_file


def annotate_responses(rows, questions, benchmark, record_nucleus):
    by_id = {}
    for question in questions:
        metadata = question["_collection"]
        question_id = metadata.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("Prepared question must have a non-empty question_id")
        if question_id in by_id:
            raise ValueError("Duplicate prepared question_id")
        by_id[question_id] = metadata
    used = set()
    for row in rows:
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or question_id not in by_id:
            raise ValueError("Missing or unknown question_id in collection response")
        metadata = by_id[question_id]
        if metadata["question_id"] in used:
            raise ValueError("Duplicate response for a prepared question")
        used.add(metadata["question_id"])
        row.update(metadata)
        row.update({"source_task": benchmark, "prompt_token_ids": row["input_token_ids"],
                    "num_generated_tokens": row["generated_tokens"], "overthinking_penalty": 0.0,
                    "rollout_online_nucleus_method": "vllm" if record_nucleus else None})
    if len(rows) != len(questions):
        raise ValueError("Not all prepared questions produced an output")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--benchmark", choices=["math", "gsm8k", "aime", "gpqa"], required=True)
    p.add_argument("--role", choices=["bf16", "awq", "gptq"], required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    config = json.loads((args.workspace / "collection.json").read_text())
    generation = config["generation"]
    questions_path = args.workspace / "inputs" / args.benchmark / "questions/train.json"
    questions = json.loads(questions_path.read_text())
    os.environ["RATIO_COLLECTION_QUESTIONS"] = str(questions_path.resolve())
    os.environ["RATIO_COLLECTION_BENCHMARK"] = args.benchmark
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from data_collection.backend import require_recording_backend
    require_recording_backend()
    import transformers
    from lighteval.models.model_input import GenerationParameters
    from lighteval_custom.models.vllm.vllm_model import VLLMModelConfig
    from lighteval_custom.main_vllm import vllm
    from vllm_custom.model_executor.fake_quantized_models.registry import register_fake_quantized_models
    register_fake_quantized_models()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record_nucleus = args.role != "bf16"
    model_config = VLLMModelConfig(
        pretrained=config["models"][args.role], dtype=generation["dtype"],
        seed=generation["seed"], tensor_parallel_size=generation["tensor_parallel_size"],
        gpu_memory_utilization=generation["gpu_memory_utilization"], max_model_length=generation["max_model_length"],
        enforce_eager=True, enable_prefix_caching=False, enable_chunked_prefill=False, init_model=True,
        generation_parameters=GenerationParameters(temperature=generation["temperature"], top_p=generation["top_p"],
                                                  top_k=-1, max_new_tokens=generation["max_new_tokens"], seed=generation["seed"]),
        nucleus_record_path=str((args.output_dir / "online.msgpack").resolve()) if record_nucleus else None)
    transformers.set_seed(generation["seed"])
    atomic_json(args.output_dir / "config.json", {"role": args.role, "benchmark": args.benchmark,
                "generation": generation, "model": config["models"][args.role], "penalty": 0,
                "question_file": str(questions_path.resolve()), "question_file_sha256": sha256_file(questions_path)})
    atomic_json(args.output_dir / "status.json", {"state": "running"})
    try:
        _, details = vllm(model_config=model_config, tasks="custom|collection|0|0", use_chat_template=True,
                          custom_tasks=str(Path(__file__).resolve().parents[1] / "lighteval_custom/tasks/collection.py"),
                          max_samples=None, output_dir=str(args.output_dir))
        rows = annotate_responses(responses_from_details(details), questions, args.benchmark, record_nucleus)
        atomic_json(args.output_dir / "responses.json", rows)
        atomic_json(args.output_dir / "summary.json", summarize(rows))
        atomic_json(args.output_dir / "status.json", {"state": "complete"})
    except BaseException as exc:
        atomic_json(args.output_dir / "status.json", {"state": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
