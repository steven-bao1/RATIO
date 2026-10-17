"""Recompute accuracy and length from completed runs; do not trust cached averages."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    args = p.parse_args()
    print("| Run | N | Accuracy (%) | Mean generated tokens |")
    print("| --- | ---: | ---: | ---: |")
    for path in sorted(args.root.rglob("responses.json")):
        status = path.parent / "status.json"
        if not status.exists() or json.loads(status.read_text())["state"] != "complete":
            continue
        result = summarize(json.loads(path.read_text()))
        label = str(path.parent.relative_to(args.root)) or "."
        print(f"| {label} | {result['n']} | {result['accuracy_percent']:.2f} | {result['mean_generated_tokens']:.2f} |")


if __name__ == "__main__":
    main()
