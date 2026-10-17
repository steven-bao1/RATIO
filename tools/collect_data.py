"""Prepare, inspect, and run the two raw-data collection directions."""
import argparse
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare", help="Freeze data and token IDs; no GPU execution")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path,
                         help="Exact output path; default: outputs/<BF16 model directory name>/collection")
    plan = commands.add_parser("plan", help="Print all explicit stage commands; no GPU execution")
    plan.add_argument("--workspace", type=Path, required=True)
    run = commands.add_parser("run", help="Execute one stage or the complete sequential plan")
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--stage", required=True, help="Stage name from plan, or all")
    run.add_argument("--skip-completed", action="store_true",
                     help="Verify and skip complete stages; refuse running/failed/changed stages")
    args = p.parse_args()
    if args.action == "prepare":
        from data_collection.paths import collection_output_dir
        from data_collection.prepare import prepare as prepare_data
        output = args.output_dir if args.output_dir is not None else collection_output_dir(args.config)
        print(f"Collection workspace: {output.resolve()}", flush=True)
        manifest = prepare_data(args.config, output)
        print(json.dumps({"workspace": str(output.resolve()), "datasets": list(manifest["datasets"]),
                          "reference_count": manifest["reference_count"]}, indent=2))
    else:
        from data_collection.workflow import build_plan, run_stage, validate_workspace
        config = validate_workspace(args.workspace)
        stages = build_plan(args.workspace, config)
        if args.action == "plan":
            for stage in stages:
                print(stage["name"] + "\n  " + shlex.join(stage["command"]))
        elif args.stage == "all":
            for stage in stages:
                run_stage(args.workspace, stage["name"], skip_completed=args.skip_completed)
        else:
            run_stage(args.workspace, args.stage, skip_completed=args.skip_completed)


if __name__ == "__main__":
    main()
