import os
from pathlib import Path

import torch
from datasets import DatasetDict, IterableDataset, load_dataset, load_from_disk


PILE_UNCOPYRIGHTED_REPO = "monology/pile-uncopyrighted"
PILE_UNCOPYRIGHTED_PARQUET_REPO = "monology/pile-uncopyrighted-parquet"
LAST_CALIBRATION_AUDIT = {}


def _pack_legacy_documents(dataset, tokenizer, n_samples, block_size, seed):
    """Original Pile-val recipe: sample documents, not a fixed block count."""
    if isinstance(dataset, IterableDataset):
        raise ValueError("Legacy Pile-val requires exact full-dataset shuffling, not streaming")
    samples = []
    lengths = []
    for row in dataset.shuffle(seed=seed):
        # Preserve encode defaults, including tokenizer special-token behavior.
        encoded = tokenizer.encode(row["text"].strip())
        if len(encoded) > 512:
            continue
        sample = torch.tensor([encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        lengths.append(sample.numel())
        if len(samples) == n_samples:
            break
    if len(samples) != n_samples:
        raise ValueError(f"Only {len(samples)}/{n_samples} eligible calibration documents")
    concatenated = torch.cat(samples, dim=1)
    count = concatenated.shape[1] // block_size
    if not count:
        raise ValueError("Calibration documents do not fill one block")
    LAST_CALIBRATION_AUDIT.clear()
    LAST_CALIBRATION_AUDIT.update(
        recipe="legacy-pileval-documents", seed=seed, requested_documents=n_samples,
        accepted_documents=len(samples), accepted_tokens=int(concatenated.shape[1]),
        maximum_document_tokens=512, block_size=block_size, calibration_blocks=count,
        discarded_tail_tokens=int(concatenated.shape[1] % block_size),
        min_document_tokens=min(lengths), max_document_tokens=max(lengths))
    print(f" * Legacy Pile-val: {len(samples)} documents -> {count} blocks")
    return [concatenated[:, i * block_size:(i + 1) * block_size] for i in range(count)]


def _select_train_split(dataset):
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError("The local calibration dataset has no train split.")
        return dataset["train"]
    return dataset


def _load_local_dataset(dataset_path):
    path = Path(dataset_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Calibration dataset path does not exist: {path}")

    if path.is_dir() and (
        (path / "dataset_info.json").is_file()
        or (path / "dataset_dict.json").is_file()
    ):
        return _select_train_split(load_from_disk(str(path))), str(path)

    if path.is_file():
        files = [str(path)]
    else:
        parquet_files = sorted(str(item) for item in path.rglob("*.parquet"))
        json_files = sorted(
            str(item)
            for pattern in ("*.json", "*.jsonl", "*.json.gz", "*.jsonl.gz", "*.jsonl.zst")
            for item in path.rglob(pattern)
        )
        files = parquet_files or json_files

    if not files:
        raise ValueError(
            f"No saved Hugging Face dataset, Parquet, or JSON files found under {path}."
        )

    builder = "parquet" if files[0].endswith(".parquet") else "json"
    dataset = load_dataset(
        builder,
        data_files={"train": files},
        split="train",
        streaming=True,
    )
    return dataset, str(path)


def _shuffle_dataset(dataset, seed):
    if isinstance(dataset, IterableDataset):
        return dataset.shuffle(seed=seed, buffer_size=10_000)
    return dataset.shuffle(seed=seed)


def _pack_fixed_length_samples(dataset, tokenizer, n_samples, block_size, seed):
    dataset = _shuffle_dataset(dataset, seed)
    separator = tokenizer.encode("\n\n", add_special_tokens=False)
    token_buffer = []
    samples = []

    for row in dataset:
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue

        # A few Pile documents are extremely large. One block per document is
        # enough for calibration and avoids tokenizing multi-megabyte examples.
        encoded = tokenizer.encode(
            text.strip(),
            add_special_tokens=False,
            truncation=True,
            max_length=block_size,
        )
        if not encoded:
            continue

        token_buffer.extend(encoded)
        token_buffer.extend(separator)
        while len(token_buffer) >= block_size:
            block = token_buffer[:block_size]
            del token_buffer[:block_size]
            samples.append(torch.tensor([block], dtype=torch.long))
            if len(samples) == n_samples:
                return samples

    raise RuntimeError(
        f"Calibration dataset ended after producing {len(samples)}/{n_samples} "
        f"blocks of {block_size} tokens."
    )


def _load_cached_samples(cache_path, n_samples, block_size):
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    input_ids = payload["input_ids"] if isinstance(payload, dict) else payload
    expected_shape = (n_samples, block_size)
    if not isinstance(input_ids, torch.Tensor) or tuple(input_ids.shape) != expected_shape:
        raise ValueError(
            f"Calibration cache {cache_path} has shape "
            f"{getattr(input_ids, 'shape', None)}, expected {expected_shape}."
        )
    print(f" * Loaded {n_samples}x{block_size} calibration tokens from {cache_path}")
    return [row.unsqueeze(0) for row in input_ids]


def _save_cached_samples(cache_path, samples, source, seed):
    path = Path(cache_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    input_ids = torch.cat(samples, dim=0)
    torch.save(
        {
            "input_ids": input_ids,
            "source": source,
            "seed": seed,
        },
        path,
    )
    print(f" * Saved calibration token cache to {path}")


def get_calib_dataset(
    data="pileval",
    tokenizer=None,
    n_samples=128,
    block_size=512,
    dataset_path=None,
    cache_path=None,
    seed=42,
):
    """Pile-val uses the historical document sampler (variable block count).

    The explicitly selected pile-uncopyrighted alternative retains its newer
    fixed-block recipe; it is not equivalent to historical Pile-val experiments.
    """
    if tokenizer is None:
        raise ValueError("A tokenizer is required for AWQ calibration.")
    if n_samples <= 0 or block_size <= 0:
        raise ValueError("n_samples and block_size must be positive.")

    LAST_CALIBRATION_AUDIT.clear()
    if data == "pileval":
        # Old caches identify blocks, not the selected documents; never silently
        # reuse a fixed-block cache as a historical document-sampler result.
        if cache_path:
            raise ValueError("Legacy Pile-val does not accept fixed-block token caches")
        source = str(Path(dataset_path or "./datasets/pile-val-backup").resolve())
        dataset = load_dataset(source, split="validation")
        samples = _pack_legacy_documents(dataset, tokenizer, n_samples, block_size, seed)
        LAST_CALIBRATION_AUDIT["source"] = source
        return samples

    if cache_path and Path(cache_path).expanduser().is_file():
        return _load_cached_samples(cache_path, n_samples, block_size)

    if data == "pile-uncopyrighted":
        local_path = dataset_path or os.environ.get("QRM_PILE_UNCOPYRIGHTED_DIR")
        if local_path:
            dataset, source = _load_local_dataset(local_path)
            sources = [(dataset, source)]
        else:
            # Prefer the canonical repository. The Parquet mirror is an
            # auto-converted copy used only if streaming the JSON/Zstandard
            # source fails in the installed datasets version.
            sources = []
            for repo in (PILE_UNCOPYRIGHTED_REPO, PILE_UNCOPYRIGHTED_PARQUET_REPO):
                try:
                    sources.append(
                        (load_dataset(repo, split="train", streaming=True), repo)
                    )
                except Exception as error:
                    print(f" * Could not initialize streaming from {repo}: {error}")
    else:
        raise NotImplementedError(f"Unsupported AWQ calibration dataset: {data}")

    errors = []
    for dataset, source in sources:
        try:
            print(f" * Building AWQ calibration blocks from {source}")
            samples = _pack_fixed_length_samples(
                dataset,
                tokenizer=tokenizer,
                n_samples=n_samples,
                block_size=block_size,
                seed=seed,
            )
            print(f" * Prepared exactly {len(samples)} blocks of {block_size} tokens")
            if cache_path:
                _save_cached_samples(cache_path, samples, source, seed)
            return samples
        except Exception as error:
            errors.append(f"{source}: {error}")
            print(f" * Streaming calibration from {source} failed: {error}")

    details = "\n".join(errors) if errors else "No dataset source could be initialized."
    raise RuntimeError(f"Unable to build Pile calibration data:\n{details}")
