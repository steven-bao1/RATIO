"""Download pinned evaluation datasets into the directories used by RATIO."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs/datasets.lock.json"
DATA_SUFFIXES = {".parquet", ".arrow", ".json", ".jsonl", ".csv", ".tsv", ".gz", ".zip"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or str(path) != value:
        raise ValueError(f"Invalid relative dataset path: {value!r}")
    return path


def read_lock():
    manifest = json.loads(LOCK.read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("datasets"):
        raise ValueError("Unsupported or empty dataset lock file")
    directories = set()
    for spec in manifest["datasets"].values():
        directory = safe_relative(spec["directory"])
        if len(directory.parts) != 1 or str(directory) in directories:
            raise ValueError("Dataset directories must be distinct single directory names")
        directories.add(str(directory))
        if not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]):
            raise ValueError("Dataset revisions must be full commit hashes")
        if not spec["files"]:
            raise ValueError("Each dataset must list its files and checksums")
        for filename, digest in spec["files"].items():
            safe_relative(filename)
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"Invalid SHA-256 for {filename}")
    return manifest["datasets"]


def inspect_existing(destination, spec):
    if destination.is_symlink():
        raise ValueError(f"Refusing a symlink dataset directory: {destination}")
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Not a directory: {destination}")
    missing = []
    for filename, expected in spec["files"].items():
        path = destination / filename
        for parent in (path, *path.parents):
            if parent == destination:
                break
            if parent.is_symlink():
                raise ValueError(f"Refusing a symlink in dataset files: {parent}")
        if not path.exists():
            missing.append(filename)
        elif not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Existing file differs from the pinned version: {path}. Use a clean --output-dir; nothing was overwritten.")
    if destination.exists():
        for path in destination.rglob("*"):
            relative = path.relative_to(destination)
            if relative.parts[0] == ".cache":
                continue
            if path.is_symlink():
                raise ValueError(f"Refusing a symlink in dataset directory: {path}")
            if path.is_file() and path.suffix.lower() in DATA_SUFFIXES and relative.as_posix() not in spec["files"]:
                raise ValueError(f"Unlisted data file may change dataset loading: {path}. Use a clean --output-dir.")
    return missing


def obtain(spec, destination, verify_only=False):
    missing = inspect_existing(destination, spec)
    if verify_only and missing:
        raise ValueError(f"Missing files in {destination}: {', '.join(missing)}")
    if missing:
        from huggingface_hub import hf_hub_download
        destination.mkdir(parents=True, exist_ok=True)
        for filename in missing:
            print(f"  Downloading {filename}", flush=True)
            hf_hub_download(repo_id=spec["repo_id"], repo_type="dataset", revision=spec["revision"],
                            filename=filename, local_dir=str(destination))
            if sha256(destination / filename) != spec["files"][filename]:
                raise ValueError(f"Downloaded checksum mismatch: {destination / filename}")
    if inspect_existing(destination, spec):
        raise ValueError(f"Dataset download is incomplete: {destination}")


def main(argv=None):
    specs = read_lock()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["all", *specs], default=["all"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets",
                        help="Dataset root (default: repository datasets/)")
    parser.add_argument("--list", action="store_true", help="Show pinned sources without downloading")
    parser.add_argument("--verify-only", action="store_true", help="Check existing files without network access")
    args = parser.parse_args(argv)
    if "all" in args.datasets and len(args.datasets) != 1:
        parser.error("Use all alone, or list individual dataset names")
    selected = list(specs) if args.datasets == ["all"] else list(dict.fromkeys(args.datasets))
    root = args.output_dir.resolve()
    failed = []
    for name in selected:
        spec = specs[name]
        destination = root / spec["directory"]
        print(f"{name}: {spec['repo_id']}@{spec['revision']} -> {destination}", flush=True)
        if args.list:
            continue
        try:
            obtain(spec, destination, verify_only=args.verify_only)
        except Exception as error:
            print(f"FAILED {name}: {error}", file=sys.stderr, flush=True)
            if name == "gpqa":
                print("GPQA requires access approval at https://huggingface.co/datasets/Idavidrein/gpqa "
                      "and an authorized Hugging Face login or HF_TOKEN.", file=sys.stderr, flush=True)
            failed.append(name)
        else:
            print(f"Verified {name}: {len(spec['files'])} files", flush=True)
    if failed:
        print("Incomplete datasets: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
