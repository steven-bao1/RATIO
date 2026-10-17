"""Explicit AWQ/GPTQ weight-only fake-quantization launch commands."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("method", choices=["awq", "gptq"])
    p.add_argument("--model", required=True, help="Your source checkpoint path (absolute or relative)")
    p.add_argument("--output", required=True, help="Your new quantized checkpoint directory (must not exist)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bits", type=int, choices=range(2, 9), default=3,
                   help="Weight bit width (2-8); default 3 preserves the historical W3 recipe")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--samples", type=int, default=128,
                   help="Pile-val AWQ: eligible documents; GPTQ/explicit fixed-block AWQ: blocks")
    p.add_argument("--calibration", required=True, help="Explicit calibration dataset; see README")
    p.add_argument("--awq-local-data", help="Local calibration data for AWQ")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.samples <= 0 or args.group_size <= 0:
        p.error("samples and group-size must be positive")
    if args.method == "gptq" and args.awq_local_data:
        p.error("--awq-local-data only applies to AWQ")
    supported = {
        "awq": {"pile-uncopyrighted", "pileval"},
        "gptq": {"wikitext2", "ptb", "c4", "pile", "reasoning-numina-math-1.5",
                 "reasoning-math-500", "reasoning-livecodebench", "reasoning-gpqa"},
    }
    if args.calibration not in supported[args.method]:
        p.error(f"Unsupported calibration dataset for {args.method}: {args.calibration}")
    if Path.cwd().resolve() != ROOT:
        p.error("Run from the project root")
    common = ["--model", args.model, "--w_bits", str(args.bits), "--w_groupsize", str(args.group_size),
              "--w_asym", "--save_qmodel_path", args.output]
    if args.method == "awq":
        command = [sys.executable, "-m", "quantization.awq.run_awq", *common,
                   "--n_samples", str(args.samples), "--seqlen", "512",
                   "--calib_seed", str(args.seed), "--calib_data", args.calibration]
        if args.awq_local_data:
            command += ["--calib_dataset_path", args.awq_local_data]
    else:
        command = [sys.executable, "-m", "quantization.quarot_gptq.save_fake_quant", *common,
                   "--nsamples", str(args.samples), "--seqlen", "2048",
                   "--seed", str(args.seed), "--cal_dataset", args.calibration,
                   "--w_clip", "--act_order", "--tp", "1"]
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.mkdir(exist_ok=False)
        atomic_json(out / "quantization_run.json", {"arguments": vars(args), "command": command, "state": "running"})
        result = subprocess.run(command)
        atomic_json(out / "quantization_run.json", {"arguments": vars(args), "command": command,
                    "state": "complete" if result.returncode == 0 else "failed", "returncode": result.returncode})
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
