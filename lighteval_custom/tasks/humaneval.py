# HumanEval executes generated code. This checker is not a security sandbox;
# run evaluations inside an appropriately isolated environment.

"""Local OpenAI HumanEval task for LightEval."""

from __future__ import annotations

import ast
import builtins
import contextlib
import faulthandler
import io
import multiprocessing
import os
import re
import shutil
import signal
import subprocess
import tempfile
from typing import Any

import numpy as np
from aenum import extend_enum

from lighteval.metrics.metrics import MetricCategory, Metrics, MetricUseCase, SampleLevelMetric
from lighteval.tasks.lighteval_task import Doc, LightevalTaskConfig


DATASET_PATH = "./datasets/humaneval"
DATASET_SUBSET = "openai_humaneval"
TIMEOUT_SECONDS = 5.0


def humaneval_prompt_fn(line: dict[str, Any], task_name: str = "humaneval") -> Doc:
    """Use the canonical HumanEval prompt without exposing solutions or tests."""
    return Doc(
        task_name=task_name,
        query=line["prompt"],
        choices=[line["canonical_solution"]],
        gold_index=0,
        specific={
            "prompt": line["prompt"],
            "test": line["test"],
            "entry_point": line["entry_point"],
            "task_id": line["task_id"],
        },
    )


def extract_candidate(model_output: str, prompt: str, entry_point: str) -> str:
    """Accept a fenced full function or a traditional HumanEval completion."""
    fenced = re.findall(r"```(?:python|py)?\s*\n(.*?)```", model_output, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced[-1] if fenced else model_output
    candidate = candidate.strip("\n")
    if re.search(rf"^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", candidate, flags=re.MULTILINE):
        imports = [node for node in ast.parse(prompt).body if isinstance(node, (ast.Import, ast.ImportFrom))]
        if imports:
            insertion_line = 0
            try:
                statements = ast.parse(candidate).body
            except SyntaxError:
                statements = []
            for index, node in enumerate(statements):
                is_docstring = (
                    index == 0 and isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
                )
                is_future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
                if not (is_docstring or is_future):
                    break
                insertion_line = node.end_lineno
            lines = candidate.splitlines(keepends=True)
            prefix = "".join(lines[:insertion_line])
            if prefix and not prefix.endswith("\n"):
                prefix += "\n"
            candidate = prefix + "\n".join(ast.unparse(node) for node in imports) + "\n" + "".join(lines[insertion_line:])
        return candidate + "\n"
    # The official format is an indented function body continuing `prompt`.
    return prompt + candidate + "\n"


class _Timeout(Exception):
    pass


def _time_limit(seconds: float):
    @contextlib.contextmanager
    def manager():
        def handler(_signum, _frame):
            raise _Timeout("timed out")

        previous = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    return manager()


def _apply_reliability_guard() -> None:
    """Disable common destructive operations inside the disposable child."""
    faulthandler.disable()
    os.environ["OMP_NUM_THREADS"] = "1"
    builtins.exit = None
    builtins.quit = None
    builtins.help = None
    for name in (
        "kill", "killpg", "system", "putenv", "remove", "removedirs", "rmdir",
        "fchdir", "setuid", "fork", "forkpty", "rename", "renames", "truncate",
        "replace", "unlink", "chmod", "chown", "chroot", "lchown", "getcwd", "chdir",
    ):
        if hasattr(os, name):
            setattr(os, name, None)
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None
    subprocess.Popen = None


def _unsafe_execute(program: str, passed: Any) -> None:
    # Save every patched function before applying the child-only guard so the
    # temporary directory can always be cleaned without noisy tracebacks.
    guarded_os_names = (
        "kill", "killpg", "system", "putenv", "remove", "removedirs", "rmdir",
        "fchdir", "setuid", "fork", "forkpty", "rename", "renames", "truncate",
        "replace", "unlink", "chmod", "chown", "chroot", "lchown", "getcwd", "chdir",
    )
    saved_os = {name: getattr(os, name) for name in guarded_os_names if hasattr(os, name)}
    saved_builtins = {name: getattr(builtins, name) for name in ("exit", "quit", "help")}
    saved_shutil = {name: getattr(shutil, name) for name in ("rmtree", "move", "chown")}
    saved_popen = subprocess.Popen
    original_directory = os.getcwd()
    directory = tempfile.mkdtemp(prefix="qrm-humaneval-")
    os.chdir(directory)
    try:
        _apply_reliability_guard()
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            with _time_limit(TIMEOUT_SECONDS):
                exec(program, {})
        passed.value = 1
    except BaseException:
        passed.value = 0
    finally:
        for name, value in saved_os.items():
            setattr(os, name, value)
        for name, value in saved_builtins.items():
            setattr(builtins, name, value)
        for name, value in saved_shutil.items():
            setattr(shutil, name, value)
        subprocess.Popen = saved_popen
        os.chdir(original_directory)
        shutil.rmtree(directory, ignore_errors=True)


def check_program(program: str) -> float:
    """Execute one candidate in a killable process with a wall-clock timeout."""
    if os.environ.get("RATIO_ALLOW_CODE_EXECUTION") != "1":
        raise RuntimeError("HumanEval requires explicit code-execution consent and an external sandbox")
    context = multiprocessing.get_context("fork")
    passed = context.Value("b", 0)
    process = context.Process(target=_unsafe_execute, args=(program, passed))
    process.start()
    process.join(TIMEOUT_SECONDS + 1.0)
    if process.is_alive():
        process.kill()
        process.join()
    return float(bool(passed.value))


def humaneval_metric(predictions: list[str], formatted_doc: Doc, **_kwargs) -> float:
    """Compute single-sample pass@1 against HumanEval's functional tests."""
    prompt = formatted_doc.specific["prompt"]
    test = formatted_doc.specific["test"]
    entry_point = formatted_doc.specific["entry_point"]
    scores = []
    for prediction in predictions:
        candidate = extract_candidate(prediction, prompt, entry_point)
        program = candidate + "\n" + test + "\n" + f"check({entry_point})\n"
        scores.append(check_program(program))
    # Our experiments generate one response per problem; multi-response aggregation is unused.
    return max(scores, default=0.0)


humaneval_pass_at_1 = SampleLevelMetric(
    metric_name="humaneval_pass@1",
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.REASONING,
    higher_is_better=True,
    sample_level_fn=humaneval_metric,
    corpus_level_fn=np.mean,
)

extend_enum(Metrics, "humaneval_pass_at_1", humaneval_pass_at_1)

humaneval = LightevalTaskConfig(
    name="humaneval",
    suite=["custom"],
    prompt_function=humaneval_prompt_fn,
    hf_repo=DATASET_PATH,
    hf_subset=DATASET_SUBSET,
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[Metrics.humaneval_pass_at_1],
    stop_sequence=[],
    trust_dataset=True,
    version=1,
)

TASKS_TABLE = [humaneval]
