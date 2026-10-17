"""Prepared selection-data tasks; separate from the public evaluation datasets."""
import os
from pathlib import Path
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval_custom.tasks.reasoning import (
    prompt_fn, aime25_prompt_fn, gsm8k_prompt_fn, gpqa_prompt_fn,
    latex_gold_metric, expr_gold_metric, expr_gsm8k_metric, gpqa_metric,
)

benchmark = os.environ["RATIO_COLLECTION_BENCHMARK"]
path = Path(os.environ["RATIO_COLLECTION_QUESTIONS"]).resolve()
functions = {"math": (prompt_fn, latex_gold_metric), "aime": (aime25_prompt_fn, expr_gold_metric),
             "gsm8k": (gsm8k_prompt_fn, expr_gsm8k_metric), "gpqa": (gpqa_prompt_fn, gpqa_metric)}
prompt, metric = functions[benchmark]


def collection_prompt(line, task_name: str = None):
    question_id = line["_collection"]["question_id"]
    if not isinstance(question_id, str) or not question_id:
        raise ValueError("Prepared question must have a non-empty question_id")
    doc = prompt(line, task_name)
    doc.specific = {**(doc.specific or {}), "collection_question_id": question_id}
    return doc


TASKS_TABLE = [LightevalTaskConfig(
    name="collection", suite=["custom"], prompt_function=collection_prompt,
    hf_repo=str(path.parent), hf_subset="default", hf_avail_splits=["train"], evaluation_splits=["train"],
    few_shots_split=None, few_shots_select=None,
    generation_size=65536, metric=[metric], version=1,
)]
