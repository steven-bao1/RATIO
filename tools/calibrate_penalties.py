"""Confirm reviewed token IDs, calibrate V8 penalties and export an inference bias map."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True, help="Completed selection output directory")
    parser.add_argument("--decisions", type=Path, help="Reviewed CSV; defaults to selection/review/user_decisions.csv")
    parser.add_argument("--output-dir", type=Path,
                        help="Exact output path; default: calibration next to selection; never overwrite")
    args = parser.parse_args()
    from keyword_selection.calibration import calibrate
    output = args.output_dir if args.output_dir is not None else args.selection.resolve().parent / "calibration"
    print(f"Calibration directory: {output.resolve()}", flush=True)
    calibrate(args.selection, output, args.decisions)


if __name__ == "__main__":
    main()
