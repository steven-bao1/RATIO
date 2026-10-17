"""Explicitly apply the minimal nucleus-recording patch to pinned vLLM source."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vllm-root", type=Path, default=root / "third-party/vllm")
    p.add_argument("--check", action="store_true", help="Only check applicability; do not edit dependencies")
    args = p.parse_args()
    expected = json.loads((root / "dependencies.json").read_text())["vllm"]["revision"]
    actual = subprocess.check_output(["git", "-C", str(args.vllm_root), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise RuntimeError(f"Unsupported vLLM revision: {actual}; expected {expected}")
    patch = str(root / "patches/vllm-nucleus-recording.patch")
    git = ["git", "-C", str(args.vllm_root), "apply"]
    if subprocess.run([*git, "--reverse", "--check", patch], capture_output=True).returncode == 0:
        print("Recording patch already present")
        return
    subprocess.run([*git, "--check", patch], check=True)
    if not args.check:
        subprocess.run([*git, patch], check=True)
    print("Patch applicable" if args.check else "Patch applied; reinstall this checkout if your vLLM installation is not editable")


if __name__ == "__main__":
    main()
