"""Run the five benchmarks sequentially on explicitly selected visible GPUs."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", required=True)
    p.add_argument("--dry-run", action="store_true")
    args, forwarded = p.parse_known_args()
    if any(x.split("=")[0] in ("--output_dir", "--dataset") for x in forwarded):
        p.error("Use --output-root; dataset is managed by this script")
    commands = []
    for task in ("GPQA-Diamond", "AIME-120", "GSM8K", "MATH-500", "HumanEval"):
        if not args.dry_run and (ROOT / args.output_root / task).exists():
            p.error(f"Run directory already exists: {Path(args.output_root) / task}")
        command = [sys.executable, str(ROOT / "inference.py"), *forwarded,
                   "--dataset", task, "--output_dir", str(Path(args.output_root) / task)]
        # Validate every configuration BEFORE the first expensive model load.
        subprocess.run([*command, "--dry-run"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        commands.append(command)
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
