"""Freeze data selection and reference tokens without launching GPU work."""
import hashlib
import json
import os
from pathlib import Path
import random

from evaluation import atomic_json

GENERATION_DEFAULTS = {
    "seed": 42, "temperature": 0.6, "top_p": 0.95,
    "max_new_tokens": 65536, "max_model_length": 65536,
    "tensor_parallel_size": 1, "gpu_memory_utilization": 0.9, "dtype": "bfloat16",
}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def checkpoint_identity(path):
    """Hash configuration/tokenizers; record weight sizes and mtimes (not content hashes)."""
    path = Path(path)
    if not (path / "config.json").is_file():
        raise FileNotFoundError(path / "config.json")
    metadata = {}
    for item in sorted(path.iterdir()):
        if item.is_file() and item.suffix in (".json", ".safetensors", ".bin", ".pt", ".model"):
            stat = item.stat()
            metadata[item.name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if item.suffix in (".json", ".model"):
                metadata[item.name]["sha256"] = hashlib.sha256(item.read_bytes()).hexdigest()
    return metadata


def shuffled_indices(size, count, seed):
    if type(count) is not int or not 0 < count <= size:
        raise ValueError(f"Cannot select {count} questions from {size}")
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    return indices[:count]


def replay_selection(size, count, seed, benchmark):
    # Preserve the historical independent, benchmark-specific 50-of-100 rule.
    name = "math500" if benchmark == "math" else benchmark
    derived = int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big")
    if type(count) is not int or not 0 < count <= size:
        raise ValueError("replay_count must be positive and no larger than question_count")
    return {"indices": sorted(random.Random(derived).sample(range(size), count)),
            "population_size": size, "sample_count": count, "selection_seed": seed,
            "derived_seed": derived, "seed_namespace": name,
            "index_space": "saved rollout order (LightEval applies its fixed seed-42 shuffle)"}


def source_rows(source, base):
    """Local JSON/JSONL/Parquet/Arrow or HF snapshot directory, with explicit split."""
    from datasets import Dataset, load_dataset
    cache = os.environ.get("HF_DATASETS_CACHE", str(Path.cwd() / ".cache/huggingface/datasets"))
    if not source.get("split"):
        raise ValueError("Every source must explicitly declare its split")
    path = (base / source["path"]).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        if path.suffix == ".arrow":
            data = Dataset.from_file(str(path))
        elif path.suffix == ".parquet":
            data = Dataset.from_parquet(str(path), cache_dir=cache)
        elif path.suffix in (".json", ".jsonl"):
            data = Dataset.from_json(str(path), cache_dir=cache)
        else:
            raise ValueError(f"Unsupported source file: {path}")
    else:
        data = load_dataset(str(path), name=source.get("subset"), split=source["split"], cache_dir=cache)
    rows = list(data)
    if not rows:
        raise ValueError(f"Empty source: {path}")
    provenance = {**source, "path": str(path), "rows": len(rows), "content_sha256": digest_json(rows)}
    return rows, provenance


def question_text(row, benchmark):
    return str(row[{"math": "problem", "gsm8k": "question", "aime": "problem", "gpqa": "Question"}[benchmark]])


def prepare_questions(spec, benchmark, base, count, seed):
    sources = spec.get("sources") or [spec["source"]]
    pool, provenance = [], []
    for ordinal, source in enumerate(sources):
        if benchmark in ("math", "gsm8k") and source.get("split") != "train":
            raise ValueError(f"Direction-1 {benchmark} requires a training split, not evaluation data")
        rows, record = source_rows(source, base)
        provenance.append(record)
        for index, row in enumerate(rows):
            text = question_text(row, benchmark)
            pool.append({**row, "_collection": {
                "benchmark": benchmark, "source_ordinal": ordinal, "source_index": index,
                "source_id": str(row.get("id", row.get("unique_id", row.get("Record ID", index)))),
                "question_id": digest_json([benchmark, ordinal, index, text]),
                "question_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }})
    chosen = shuffled_indices(len(pool), count, seed)
    selected = [pool[index] for index in chosen]
    questions = [question_text(row, benchmark) for row in selected]
    if len(set(questions)) != len(questions):
        raise ValueError(f"Selected {benchmark} samples contain duplicate questions")
    evaluation_record = None
    if benchmark in ("math", "gsm8k") and "evaluation" not in spec:
        raise ValueError(f"Provide {benchmark}.evaluation for exact-question overlap checking")
    if "evaluation" in spec:
        heldout, evaluation_record = source_rows(spec["evaluation"], base)
        if set(questions) & {question_text(row, benchmark) for row in heldout}:
            raise ValueError(f"Selected {benchmark} questions overlap evaluation data")
    return selected, {"sources": provenance, "selected_pool_indices": chosen, "selection_seed": seed,
                      "count": count, "evaluation_source": evaluation_record,
                      "exact_question_overlap_checked": evaluation_record is not None,
                      "disclosure": "selection/evidence data, not an independent test" if benchmark in ("aime", "gpqa") else "training questions"}


def reference_contexts(rows, tokenizer, seed, limit=None):
    count = len(rows) if limit is None else limit
    indices = shuffled_indices(len(rows), count, seed)
    contexts = []
    for ordinal, index in enumerate(indices):
        row = rows[index]
        query = str(row["problem"]) + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                               tokenize=False, add_generation_prompt=True)
        continuation = " " + str(row["solution"])
        # Preserve the historical render-then-tokenize policy (including BOS).
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        response_ids = tokenizer.encode(continuation, add_special_tokens=False)
        if not prompt_ids or not response_ids:
            raise ValueError(f"Empty reference token sequence: {index}")
        contexts.append({"trajectory_kind": "math-cot-reference-solution", "trajectory_index": ordinal,
                         "source_index": index, "source_split": "train", "source_task": str(row.get("type", "math")),
                         "math_category": str(row.get("type", "")), "math_level": str(row.get("level", "")),
                         "problem": row["problem"], "solution": row["solution"], "rendered_prompt": prompt,
                         "reference_continuation": continuation, "prompt_token_ids": prompt_ids,
                         "generated_token_ids": response_ids, "num_generated_tokens": len(response_ids),
                         "overthinking_penalty": 0.0})
    return contexts


def prepare(config_path, output_dir):
    from transformers import AutoTokenizer
    from data_collection.materialize import tokenizer_fingerprint
    config_path, output_dir = Path(config_path).resolve(), Path(output_dir).resolve()
    config = json.loads(config_path.read_text())
    base = config_path.parent
    if set(config["models"]) != {"bf16", "awq", "gptq"}:
        raise ValueError("Provide exactly bf16, awq and gptq model paths")
    models = {role: str((base / path).resolve()) for role, path in config["models"].items()}
    generation = {**GENERATION_DEFAULTS, **config.get("generation", {})}
    if set(generation) != set(GENERATION_DEFAULTS):
        raise ValueError("Unknown generation option; collection does not accept penalty or model-specific flags")
    from data_collection.workflow import validate_generation
    validate_generation(generation)
    tokenizers = {role: AutoTokenizer.from_pretrained(path) for role, path in models.items()}
    fingerprints = {role: tokenizer_fingerprint(t) for role, t in tokenizers.items()}
    if len(set(fingerprints.values())) != 1:
        raise ValueError("BF16/AWQ/GPTQ must share the same tokenizer token-ID mapping")
    prompt_signatures = {role: digest_json({"chat_template": t.chat_template,
                         "bos_token_id": t.bos_token_id, "eos_token_id": t.eos_token_id})
                         for role, t in tokenizers.items()}
    if len(set(prompt_signatures.values())) != 1:
        raise ValueError("BF16/AWQ/GPTQ chat templates or special-token policies differ")
    if output_dir.exists():
        raise FileExistsError("Preparation output already exists; use a new directory")
    count, replay_count = config.get("question_count", 100), config.get("replay_count", 50)
    datasets = config.get("datasets", {})
    if set(datasets) - {"math", "gsm8k", "gpqa", "aime"}:
        raise ValueError("Unknown direction-1 benchmark")
    prepared, selections, manifests = {}, {}, {}
    for benchmark, spec in datasets.items():
        prepared[benchmark], manifests[benchmark] = prepare_questions(spec, benchmark, base, count, generation["seed"])
        selections[benchmark] = replay_selection(count, replay_count, generation["seed"], benchmark)
    reference = config.get("reference")
    contexts, reference_manifest = None, None
    if reference:
        if reference["source"].get("split") != "train":
            raise ValueError("Direction-2 reference solutions must come from a training split")
        rows, reference_manifest = source_rows(reference["source"], base)
        contexts = reference_contexts(rows, tokenizers["bf16"], generation["seed"], reference.get("limit"))
    if not datasets and contexts is None:
        raise ValueError("No collection inputs configured")
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in prepared.items():
        atomic_json(output_dir / "inputs" / name / "questions/train.json", rows)
        atomic_json(output_dir / "inputs" / name / "selection.json", selections[name])
    if contexts is not None:
        atomic_json(output_dir / "inputs/reference/contexts.json", contexts)
        atomic_json(output_dir / "inputs/reference/selection.json", {"indices": list(range(len(contexts)))})
    manifest = {"schema_version": 1, "models": models, "generation": generation,
                "checkpoint_identities": {role: checkpoint_identity(path) for role, path in models.items()},
                "tokenizer_fingerprints": fingerprints, "datasets": manifests,
                "prompt_signatures": prompt_signatures,
                "reference": reference_manifest, "question_count": count, "replay_count": replay_count,
                "reference_count": len(contexts) if contexts is not None else 0,
                "configuration_sha256": digest_json(config),
                "prompt_policy": "chat-template rendering followed by add_special_tokens=True",
                "top_k": -1, "nucleus_method": "vllm", "penalty": 0}
    from data_collection.io import sha256_file
    manifest["input_hashes"] = {str(p.relative_to(output_dir)): sha256_file(p) for p in sorted((output_dir / "inputs").rglob("*.json"))}
    atomic_json(output_dir / "collection.json", manifest)
    return manifest
