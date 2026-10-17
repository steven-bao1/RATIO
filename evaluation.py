"""Auditable response storage and macro benchmark summaries (stdlib only)."""
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def token_list(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)) or any(type(x) is not int or x < 0 for x in value):
        raise ValueError("Missing/invalid generated token IDs; refusing to estimate length")
    return list(value)


def responses_from_details(details):
    rows = []
    for task, samples in details.items():
        for index, sample in enumerate(samples):
            if len(sample["predictions"]) != 1:
                raise ValueError("This entry point expects one generation per problem")
            tokens = token_list(sample["cont_tokens"])
            prompt = sample["full_prompt"]
            rows.append({
                "task": task, "sample_index": index,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "full_prompt": prompt, "generated_text": sample["predictions"][0],
                "gold": sample["gold"], "metrics": sample["metrics"],
                "input_token_ids": token_list(sample["input_tokens"]),
                "generated_token_ids": tokens, "generated_tokens": len(tokens),
            })
            specifics = sample.get("specifics")
            if isinstance(specifics, dict) and "collection_question_id" in specifics:
                question_id = specifics["collection_question_id"]
                if not isinstance(question_id, str) or not question_id:
                    raise ValueError("Invalid collection question_id in evaluation details")
                rows[-1]["question_id"] = question_id
    return rows


def summarize(rows):
    if not rows:
        raise ValueError("No responses")
    scores, lengths = [], []
    for row in rows:
        # Each supported task has exactly one correctness metric. Do not silently
        # pick a latency/auxiliary metric when a future task introduces one.
        metrics = row["metrics"]
        if len(metrics) != 1:
            raise ValueError(f"Expected one accuracy metric, got {list(metrics)}")
        score = float(next(iter(metrics.values())))
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Accuracy metric must be in [0,1]")
        length = len(token_list(row["generated_token_ids"]))
        if row["generated_tokens"] != length:
            raise ValueError("Token count does not match stored IDs")
        scores.append(score)
        lengths.append(length)
    return {"n": len(rows), "accuracy_percent": statistics.mean(scores) * 100,
            "mean_generated_tokens": statistics.mean(lengths)}
