"""Join independent BF16/quantized correctness by frozen question identity."""
import argparse
import json
from pathlib import Path
from evaluation import atomic_json
from data_collection.io import rollout_correctness, sha256_file


def join_correctness(quant, bf16):
    lookup = {row["question_id"]: row for row in bf16}
    if len(lookup) != len(bf16) or len({r["question_id"] for r in quant}) != len(quant):
        raise ValueError("Duplicate question IDs")
    if set(lookup) != {row["question_id"] for row in quant}:
        raise ValueError("Independent correctness rollouts use different question sets")
    output = []
    for index, row in enumerate(quant):
        full = lookup[row["question_id"]]
        if row["prompt_token_ids"] != full["prompt_token_ids"]:
            raise ValueError("BF16/quantized prompts do not match for a correctness pair")
        qm, qv, qc = rollout_correctness(row)
        fm, fv, fc = rollout_correctness(full)
        if qc is None or fc is None or qm != fm:
            raise ValueError("Missing or mismatched correctness metrics")
        output.append({"trajectory_index": index, "question_id": row["question_id"],
                       "quant_correct": qc, "bf16_correct": fc,
                       "quant_metric": qv, "bf16_metric": fv, "metric_name": qm})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quant", type=Path, required=True)
    p.add_argument("--bf16", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    rows = join_correctness(json.loads(args.quant.read_text()), json.loads(args.bf16.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output_dir / "pairs.json", rows)
    atomic_json(args.output_dir / "manifest.json", {"quant_sha256": sha256_file(args.quant),
                 "bf16_sha256": sha256_file(args.bf16), "count": len(rows)})


if __name__ == "__main__":
    main()
