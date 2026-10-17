"""Analyze existing raw data and export preliminary keywords plus review candidates."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--workspace", type=Path, help="Completed public collection workspace")
    source.add_argument("--inputs", type=Path, help="Explicit local raw-score and rollout paths in JSON")
    parser.add_argument("--output-dir", type=Path,
                        help="Exact output path; default: selection next to workspace, or outputs/<model>/selection with --inputs")
    parser.add_argument("--policy", type=Path, help="Explicit numerical overrides, recorded as a protocol variant")
    parser.add_argument("--table10-lexicon", type=Path, help="Optional replacement lexical-family policy")
    parser.add_argument("--d1-exclusions", type=Path, help="Optional replacement explicit exclusion policy")
    parser.add_argument("--post-qsti-exclusions", type=Path, help="Optional replacement post-QSTI semantic exclusion policy")
    args = parser.parse_args()
    from keyword_selection.inputs import workspace_inputs, config_inputs
    from keyword_selection.workflow import run
    from data_collection.paths import model_output_dir
    inputs = workspace_inputs(args.workspace) if args.workspace else config_inputs(args.inputs)
    output = args.output_dir
    if output is None:
        output = (args.workspace.resolve().parent / "selection" if args.workspace
                  else model_output_dir(inputs["tokenizer"], "selection"))
    print(f"Selection directory: {output.resolve()}", flush=True)
    run(inputs, output, policy_file=args.policy, lexicon_file=args.table10_lexicon,
        exclusions_file=args.d1_exclusions, source_config=args.inputs,
        post_qsti_exclusions_file=args.post_qsti_exclusions)


if __name__ == "__main__":
    main()
