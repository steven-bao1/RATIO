"""Fetch pinned upstream sources; never reset or modify an existing checkout."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    dependencies = json.loads((ROOT / "dependencies.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=sorted(dependencies),
                        help="Fetch only these dependencies (default: all inherited sources)")
    args = parser.parse_args(argv)
    # Upstream LightEval LFS objects are historical reference-test outputs, not
    # runtime assets. Leave pointers in place rather than fetching them implicitly.
    git_env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    for name, entry in dependencies.items():
        if args.only and name not in args.only:
            continue
        path = ROOT / "third-party" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
            dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()
            if revision != entry["revision"] or dirty:
                raise RuntimeError(f"Existing {path} is dirty or at a different revision; inspect manually")
            continue
        # Download the frozen snapshot, not the repository's entire history.
        # Keep a real checkout so the recording-hook installer can verify HEAD.
        subprocess.run(["git", "init", str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "remote", "add", "origin", entry["url"]], check=True)
        subprocess.run(["git", "-C", str(path), "fetch", "--depth", "1", "origin", entry["revision"]], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", "FETCH_HEAD"], check=True, env=git_env)
        subprocess.run(["git", "-C", str(path), "submodule", "update", "--init", "--recursive"], check=True, env=git_env)


if __name__ == "__main__":
    main()
