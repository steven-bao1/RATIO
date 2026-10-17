"""Explicit stage plans, provenance checks, and sequential execution."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from evaluation import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def validate_generation(g):
    for key in ("max_new_tokens", "max_model_length", "tensor_parallel_size"):
        if type(g[key]) is not int or g[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(g["seed"]) is not int:
        raise ValueError("seed must be an integer")
    if not math.isfinite(g["temperature"]) or g["temperature"] <= 0:
        raise ValueError("Collection requires positive finite temperature")
    if not 0 < g["top_p"] <= 1 or not 0 < g["gpu_memory_utilization"] <= 1:
        raise ValueError("Invalid top_p or GPU memory utilization")
    if g["dtype"] not in ("bfloat16", "float16"):
        raise ValueError("Collection dtype must be explicitly bfloat16 or float16")
    if g["max_new_tokens"] > g["max_model_length"]:
        raise ValueError("Generation budget cannot exceed the context window")


def build_plan(workspace, config):
    workspace = Path(workspace).resolve()
    g, models = config["generation"], config["models"]
    validate_generation(g)
    stages = []

    def add(name, module, args, dependencies=()):
        stages.append({"name": name, "command": [sys.executable, "-m", module, *map(str, args)],
                       "dependencies": list(dependencies)})

    def score(name, trajectories, selection, model, role, output, dependencies=()):
        args = ["--rollouts", trajectories, "--sample-indices", selection, "--model", model,
                "--tokenizer", models["bf16"], "--role", role, "--output-dir", output]
        for key in ("seed", "temperature", "top_p", "dtype", "max_model_length", "tensor_parallel_size", "gpu_memory_utilization"):
            args.extend(["--" + key.replace("_", "-"), g[key]])
        add(name, "data_collection.replay", args, dependencies)

    def compare(name, quant, full, output, dependencies):
        add(name, "data_collection.compare", ["--quant-scores", quant, "--bf16-scores", full,
                                               "--output-dir", output], dependencies)

    if config.get("reference_count"):
        inputs = workspace / "inputs/reference"
        out = workspace / "direction2"
        score("direction2.bf16.score", inputs / "contexts.json", inputs / "selection.json",
              models["bf16"], "bf16", out / "bf16/scores")
        for role in ("awq", "gptq"):
            score(f"direction2.{role}.score", inputs / "contexts.json", inputs / "selection.json",
                  models[role], "quant", out / role / "scores")
            compare(f"direction2.{role}.compare", out / role / "scores", out / "bf16/scores",
                    out / role / "comparison", ["direction2.bf16.score", f"direction2.{role}.score"])

    for benchmark in config["datasets"]:
        inputs = workspace / "inputs" / benchmark
        out = workspace / "direction1" / benchmark
        full_stage = f"direction1.bf16.{benchmark}.rollout"
        add(full_stage, "data_collection.rollout", ["--workspace", workspace, "--benchmark", benchmark,
             "--role", "bf16", "--output-dir", out / "bf16/rollout"])
        for role in ("awq", "gptq"):
            prefix, run = f"direction1.{role}.{benchmark}", out / role
            add(prefix + ".rollout", "data_collection.rollout", ["--workspace", workspace,
                "--benchmark", benchmark, "--role", role, "--output-dir", run / "rollout"])
            add(prefix + ".materialize", "data_collection.materialize", [
                "--rollouts", run / "rollout/responses.json", "--records", run / "rollout/online.msgpack",
                "--model", models[role], "--tokenizer", models["bf16"], "--role", "quant",
                "--output-dir", run / "quant-scores", "--nucleus-method", "vllm",
                "--temperature", g["temperature"], "--top-p", g["top_p"]], [prefix + ".rollout"])
            score(prefix + ".replay", run / "rollout/responses.json", inputs / "selection.json",
                  models["bf16"], "bf16", run / "bf16-scores", [prefix + ".rollout"])
            compare(prefix + ".compare", run / "quant-scores", run / "bf16-scores", run / "comparison",
                    [prefix + ".materialize", prefix + ".replay"])
            add(prefix + ".correctness", "data_collection.correctness", ["--quant", run / "rollout/responses.json",
                 "--bf16", out / "bf16/rollout/responses.json", "--output-dir", run / "correctness"],
                [prefix + ".rollout", full_stage])
    gpu_modules = {"data_collection.rollout", "data_collection.replay"}
    return sorted(stages, key=lambda stage: stage["command"][2] not in gpu_modules)


def validate_workspace(workspace):
    from data_collection.io import sha256_file
    from data_collection.prepare import checkpoint_identity
    workspace = Path(workspace).resolve()
    config = json.loads((workspace / "collection.json").read_text())
    for role, model in config["models"].items():
        if checkpoint_identity(model) != config["checkpoint_identities"][role]:
            raise ValueError(f"Checkpoint changed after preparation: {role}")
    for relative, digest in config["input_hashes"].items():
        if sha256_file(workspace / relative) != digest:
            raise ValueError(f"Prepared input changed: {relative}; create a new workspace")
    return config


def validate_completed_stage(workspace, stage, saved, manifest_hash):
    """Validate a finished stage before reuse; directory existence is insufficient."""
    from data_collection.io import sha256_file
    name = stage["name"]
    if not isinstance(saved, dict):
        raise ValueError(f"Malformed stage record: {name}")
    if saved.get("state") != "complete":
        raise ValueError(f"Stage {name} is {saved.get('state')!r}, not complete. "
                         "It may still be running or have been interrupted. "
                         "--skip-completed never retries or overwrites partial stages; "
                         "inspect its log and preserve existing outputs.")
    if saved.get("collection_sha256") != manifest_hash or saved.get("command") != stage["command"]:
        raise ValueError(f"Completed stage configuration changed: {name}")
    artifacts = saved.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"Completed stage has no artifact manifest: {name}")
    output = Path(stage["command"][stage["command"].index("--output-dir") + 1]).resolve()
    if not output.is_relative_to(workspace):
        raise ValueError(f"Stage output is outside the workspace: {name}")
    actual = {str(p.relative_to(workspace)) for p in output.rglob("*") if p.is_file()}
    if actual != set(artifacts):
        raise ValueError(f"Completed stage artifact file set changed: {name}")
    for filename, digest in artifacts.items():
        path = workspace / filename
        if not path.resolve().is_relative_to(output) or sha256_file(path) != digest:
            raise ValueError(f"Completed stage artifact changed: {filename}")


def run_stage(workspace, name, *, skip_completed=False):
    from data_collection.io import sha256_file
    workspace = Path(workspace).resolve()
    config = validate_workspace(workspace)
    plan = {stage["name"]: stage for stage in build_plan(workspace, config)}
    if name not in plan:
        raise ValueError(f"Unknown stage {name}; use plan to list names")
    stage = plan[name]
    states = workspace / "stages"
    states.mkdir(exist_ok=True)
    manifest_hash = sha256_file(workspace / "collection.json")
    verified_dependencies = set()

    def validate_dependency(dependency):
        if dependency in verified_dependencies:
            return
        prerequisite = plan[dependency]
        for ancestor in prerequisite["dependencies"]:
            validate_dependency(ancestor)
        path = states / (dependency + ".json")
        if not path.exists():
            raise ValueError(f"Missing data prerequisite: {dependency}")
        saved = json.loads(path.read_text())
        validate_completed_stage(workspace, prerequisite, saved, manifest_hash)
        verified_dependencies.add(dependency)

    for dependency in stage["dependencies"]:
        validate_dependency(dependency)
    path = states / (name + ".json")
    if skip_completed and path.exists():
        print(f"Validating completed stage: {name}", flush=True)
        saved = json.loads(path.read_text())
        validate_completed_stage(workspace, stage, saved, manifest_hash)
        print(f"Skipped completed stage: {name}", flush=True)
        return
    output_index = stage["command"].index("--output-dir") + 1
    output = Path(stage["command"][output_index])
    log_path = states / (name + ".log")
    if path.exists() or output.exists() or log_path.exists():
        raise FileExistsError(f"Existing stage state/output/log: {name}. "
                              "Use --skip-completed for verified complete stages; "
                              "partial or untracked output is never reused or overwritten.")
    with path.open("x") as stream:
        json.dump({"state": "running", "command": stage["command"], "collection_sha256": manifest_hash}, stream)
    environment = {**os.environ, "VLLM_USE_V1": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0",
                   "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
    try:
        print(f"Starting {name}; log: {log_path}", flush=True)
        with log_path.open("x") as log:
            result = subprocess.run(stage["command"], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Stage failed (exit {result.returncode}); inspect {states / (name + '.log')}")
        print(f"Validating output files: {name}", flush=True)
        artifacts = {str(p.relative_to(workspace)): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}
        if not artifacts:
            raise RuntimeError("Stage produced no artifacts")
        atomic_json(path, {"state": "complete", "command": stage["command"], "artifacts": artifacts,
                           "collection_sha256": manifest_hash})
    except BaseException as exc:
        atomic_json(path, {"state": "failed", "error": str(exc), "command": stage["command"],
                           "collection_sha256": manifest_hash})
        raise
    print(f"Completed {name}", flush=True)
