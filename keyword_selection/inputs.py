"""Explicit, read-only raw-data contracts and provenance validation."""
import json
from pathlib import Path

from data_collection.io import sha256_file

QUANTIZERS = ("awq", "gptq")
BENCHMARKS = ("aime", "gpqa", "gsm8k", "math")


def read_json(path):
    return json.loads(Path(path).read_text())


def shard_path(directory, record):
    base = Path(directory).resolve()
    path = (base / record["file"]).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError(f"Invalid or missing score shard: {path}")
    return path


def pair_manifests(quant, full, require_equal=False):
    if quant.get("role") != "quant" or full.get("role") != "bf16":
        raise ValueError("Expected quantized and BF16 score roles")
    for field in ("rollouts_sha256", "tokenizer_vocab_sha256", "temperature", "top_p", "nucleus_method"):
        if quant.get(field) is None or quant[field] != full.get(field):
            raise ValueError(f"Missing or mismatched score provenance: {field}")
    def backend(m):
        return m.get("execution_backend") or ("vllm" if m.get("score_source") == "online-vllm-generation" else None)
    if backend(quant) is None or backend(quant) != backend(full):
        raise ValueError("Mixed or missing scoring backend; use backend-matched raw data")
    for m in (quant, full):
        if m.get("score_source") == "forced-vllm-replay" and not m.get("forced_replay_verified_exact"):
            raise ValueError("Unverified forced replay")
    if backend(quant) == "vllm":
        for m in (quant, full):
            versions = m.get("online_record_schema_versions", [])
            if versions and min(versions) < 5:
                raise ValueError("Pre-fix vLLM recording precision is not valid for public selection; recollect raw scores")
        if quant.get("sampler_parameter_policy") != full.get("sampler_parameter_policy"):
            raise ValueError("Mismatched sampler parameter precision policies")
        for field in ("vllm_version", "vllm_sampler_sha256"):
            if quant.get(field) is None or quant[field] != full.get(field):
                raise ValueError(f"Missing or mismatched backend provenance: {field}")
    def index(m):
        rows = m["trajectories"]
        by_id = {r["trajectory_id"]: r for r in rows}
        if not rows or len(by_id) != len(rows):
            raise ValueError("Empty or duplicate trajectory records")
        if len({r["trajectory_index"] for r in rows}) != len(rows):
            raise ValueError("Duplicate trajectory indices")
        return by_id
    q, f = index(quant), index(full)
    if not set(f).issubset(q) or (require_equal and set(q) != set(f)):
        raise ValueError("Mismatched trajectory sets")
    pairs = []
    for right in full["trajectories"]:
        left = q[right["trajectory_id"]]
        for field in ("trajectory_index", "source_task", "prompt_token_count", "response_token_count"):
            if field not in left or left[field] != right.get(field):
                raise ValueError(f"Trajectory mismatch: {field}")
        pairs.append((left, right))
    return pairs


class Provenance:
    def __init__(self):
        self.files = {}
        self.stats = {}

    def add(self, filename):
        path = Path(filename).resolve()
        if path not in self.stats:
            before = path.stat()
            self.files[str(path)] = sha256_file(path)
            after = path.stat()
            stamp = (after.st_size, after.st_mtime_ns)
            if stamp != (before.st_size, before.st_mtime_ns):
                raise ValueError(f"Input changed while hashing: {path}")
            self.stats[path] = stamp
        return path

    def verify_unchanged(self):
        for path, stamp in self.stats.items():
            now = path.stat()
            if (now.st_size, now.st_mtime_ns) != stamp:
                raise ValueError(f"Input changed during analysis: {path}")


def workspace_inputs(workspace):
    """Resolve the public collector layout, without loading model weights."""
    base = Path(workspace).resolve()
    config = read_json(base / "collection.json")
    return {
        "tokenizer": config["models"]["bf16"],
        "collection_manifest": str(base / "collection.json"),
        "direction2": {q: str(base / "direction2" / q / "scores") for q in ("bf16", *QUANTIZERS)},
        "direction1": [
            {"quantizer": q, "benchmark": task,
             "quant_scores": str(base / "direction1" / task / q / "quant-scores"),
             "bf16_scores": str(base / "direction1" / task / q / "bf16-scores"),
             "rollouts": str(base / "direction1" / task / q / "rollout/responses.json"),
             "bf16_rollouts": str(base / "direction1" / task / "bf16/rollout/responses.json")}
            for q in QUANTIZERS for task in config["datasets"]
        ],
    }


def config_inputs(filename):
    """Alternative explicit paths for previously collected research artifacts."""
    path = Path(filename).resolve()
    value = read_json(path)
    for key in value:
        if key not in ("tokenizer", "direction1", "direction2"):
            raise ValueError(f"Unknown raw-input config key: {key}")
    def resolve(p):
        return str((path.parent / p).resolve())
    value["tokenizer"] = resolve(value["tokenizer"])
    value["direction2"] = {q: resolve(p) for q, p in value["direction2"].items()}
    for row in value["direction1"]:
        if set(row) != {"quantizer", "benchmark", "quant_scores", "bf16_scores", "rollouts", "bf16_rollouts"}:
            raise ValueError("Each Direction-1 cell needs explicit scores and both rollout paths")
        for key in ("quant_scores", "bf16_scores", "rollouts", "bf16_rollouts"):
            row[key] = resolve(row[key])
    return value


def validate_inputs(inputs, audit):
    if set(inputs["direction2"]) != {"bf16", *QUANTIZERS}:
        raise ValueError("Direction 2 needs BF16, AWQ and GPTQ scores")
    cells = [(c["quantizer"], c["benchmark"]) for c in inputs["direction1"]]
    expected = {(q, d) for q in QUANTIZERS for d in BENCHMARKS}
    if set(cells) != expected or len(cells) != len(expected):
        raise ValueError("Direction 1 requires exactly AWQ/GPTQ × aime/gpqa/gsm8k/math; no silent missing cells")
    full = read_json(audit.add(Path(inputs["direction2"]["bf16"]) / "manifest.json"))
    identity = {key: full[key] for key in ("tokenizer_vocab_sha256", "temperature", "top_p", "nucleus_method")}
    if full.get("execution_backend") == "vllm":
        identity.update({key: full[key] for key in ("vllm_version", "vllm_sampler_sha256")})
    models = {"bf16": full["model"]}
    collection = None
    if inputs.get("collection_manifest"):
        collection_path = audit.add(inputs["collection_manifest"])
        collection = read_json(collection_path)
        for relative, digest in collection["input_hashes"].items():
            path = (collection_path.parent / relative).resolve()
            if not path.is_relative_to(collection_path.parent):
                raise ValueError("Prepared input path escapes collection workspace")
            if sha256_file(audit.add(path)) != digest:
                raise ValueError(f"Prepared collection input changed: {relative}")
        if len(full["trajectories"]) != collection["reference_count"]:
            raise ValueError("Direction-2 reference collection is incomplete")
    for q in QUANTIZERS:
        quant = read_json(audit.add(Path(inputs["direction2"][q]) / "manifest.json"))
        pair_manifests(quant, full, require_equal=True)
        models[q] = quant["model"]
    directories = set(inputs["direction2"].values())
    selections = {}
    for cell in inputs["direction1"]:
        q = read_json(audit.add(Path(cell["quant_scores"]) / "manifest.json"))
        f = read_json(audit.add(Path(cell["bf16_scores"]) / "manifest.json"))
        pair_manifests(q, f)
        selected = sorted(r["trajectory_index"] for r in f["trajectories"])
        task = cell["benchmark"]
        if task in selections and selected != selections[task]:
            raise ValueError("AWQ/GPTQ must use the same selected rollout indices per benchmark")
        selections[task] = selected
        if collection:
            index_path = collection_path.parent / "inputs" / task / "selection.json"
            expected = read_json(index_path)["indices"]
            if selected != sorted(expected) or len(q["trajectories"]) != collection["question_count"]:
                raise ValueError(f"Incomplete or incompatible Direction-1 selection: {task}")
        for m, role in ((q, cell["quantizer"]), (f, "bf16")):
            if m["model"] != models[role] or any(m.get(k) != v for k, v in identity.items()):
                raise ValueError("Direction-1/2 model, tokenizer or sampling configuration mismatch")
        actual = sha256_file(audit.add(cell["rollouts"]))
        if q["rollouts_sha256"] != actual:
            raise ValueError("Rollout source differs from scoring source")
        audit.add(cell["bf16_rollouts"])
        directories.update([cell["quant_scores"], cell["bf16_scores"]])
    for directory in sorted(directories):
        m = read_json(audit.add(Path(directory) / "manifest.json"))
        for record in m["trajectories"]:
            audit.add(shard_path(directory, record))
    return identity
