"""User-confirmed Direction-2 V8 norm-noclip calibration; CPU-only."""
from array import array
from collections import Counter
import itertools
import math
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from data_collection.io import sha256_file
from data_collection.materialize import tokenizer_fingerprint
from evaluation import atomic_json
from keyword_selection.decisions import load_selection, unique_json
from keyword_selection.inputs import pair_manifests, shard_path

ROOT = Path(__file__).resolve().parents[1]
SATURATION_LOGPROB = -1e-7
SCORE_COLUMNS = ("trajectory_index", "trajectory_id", "source_task", "response_position",
                 "token_id", "nucleus_token_ids", "nucleus_logprobs")
EVENT_SCHEMA = pa.schema([
    ("quantizer", pa.string()), ("trajectory_index", pa.int32()), ("trajectory_id", pa.string()),
    ("source_task", pa.string()), ("response_position", pa.int32()), ("token_id", pa.int32()),
    ("q_in_nucleus", pa.bool_()), ("bf16_in_nucleus", pa.bool_()),
    ("q_logprob_used", pa.float64()), ("bf16_logprob_used", pa.float64()), ("delta_logp", pa.float64()),
    ("exact_logit_correction_magnitude", pa.float64()), ("used_for_calibration", pa.bool_()),
    ("exclusion_reason", pa.string()),
])


def exact_correction(q_logp, f_logp):
    """Use saved temperature-scaled logps; do NOT multiply by temperature."""
    if not math.isfinite(q_logp) or not math.isfinite(f_logp) or q_logp > 0 or f_logp > 0:
        raise ValueError("Invalid saved probability for exact-logit correction")
    if q_logp <= f_logp:
        return None, "nonpositive_shift"
    if q_logp >= SATURATION_LOGPROB or f_logp >= SATURATION_LOGPROB:
        return None, "saturated_log_probability"
    q_logit = q_logp - math.log(-math.expm1(q_logp))
    f_logit = f_logp - math.log(-math.expm1(f_logp))
    value = q_logit - f_logit
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Invalid positive exact-logit correction")
    return value, ""


class ExactMedians:
    """Spill doubles to temporary files; exact event medians, not sketches/means."""
    def __init__(self, directory, token_ids):
        self.directory = Path(directory)
        self.buffers = {k: array("d") for k in token_ids}
        self.counts = Counter()

    def add(self, key, value):
        self.buffers[key].append(value)
        self.counts[key] += 1
        if len(self.buffers[key]) >= 4096:
            self.flush(key)

    def flush(self, key):
        buffer = self.buffers[key]
        if buffer:
            with (self.directory / f"{key}.bin").open("ab") as stream:
                buffer.tofile(stream)
            self.buffers[key] = array("d")

    def median(self, key):
        if not self.counts[key]:
            raise ValueError(f"Token {key} has no valid positive calibration events in this quantizer")
        self.flush(key)
        path = self.directory / f"{key}.bin"
        if path.stat().st_size != self.counts[key] * np.dtype(np.float64).itemsize:
            raise ValueError("Temporary exact-median value count mismatch")
        values = np.memmap(path, dtype=np.float64, mode="r+")
        try:
            return float(np.median(values, overwrite_input=True))
        finally:
            del values


def aligned_rows(qpath, fpath, record):
    qfile, ffile = pq.ParquetFile(qpath), pq.ParquetFile(fpath)
    length = record["response_token_count"]
    if not length or qfile.metadata.num_rows != length or ffile.metadata.num_rows != length:
        raise ValueError(f"Incomplete score shard: {qpath}")
    def rows(file):
        for batch in file.iter_batches(batch_size=512, columns=list(SCORE_COLUMNS)):
            yield from batch.to_pylist()
    sentinel = object()
    for position, (q, f) in enumerate(itertools.zip_longest(rows(qfile), rows(ffile), fillvalue=sentinel)):
        if q is sentinel or f is sentinel:
            raise ValueError("Different score lengths")
        for field in ("trajectory_index", "trajectory_id", "source_task", "response_position", "token_id"):
            if q[field] != f[field]:
                raise ValueError(f"Same-position alignment mismatch: {field}")
        if q["response_position"] != position:
            raise ValueError("Missing, repeated or unordered score positions")
        for field in ("trajectory_index", "trajectory_id", "source_task"):
            if q[field] != record[field]:
                raise ValueError(f"Score shard does not match its manifest: {field}")
        yield q, f


def nucleus_lookup(row, selected):
    ids, logps = row["nucleus_token_ids"], row["nucleus_logprobs"]
    if not ids or len(ids) != len(logps) or len(ids) != len(set(ids)):
        raise ValueError("Empty, duplicate or malformed saved nucleus")
    result = {}
    for key, value in zip(ids, logps):
        if type(key) is not int or key < 0 or not math.isfinite(value) or value > 0:
            raise ValueError("Invalid token ID or probability in saved nucleus")
        if key in selected:
            result[key] = float(value)
    return result


def extract_events(quantizer, qdir, fdir, pairs, selected, tail_probability, output):
    counts = {k: Counter() for k in selected}
    positions = 0
    event_path = output / "events" / f"{quantizer}.parquet"
    log_floor = math.log(tail_probability)
    with tempfile.TemporaryDirectory(prefix=f"median-{quantizer}-", dir=output) as temporary:
        values = ExactMedians(temporary, selected)
        pending = []
        with pq.ParquetWriter(event_path, EVENT_SCHEMA, compression="zstd") as writer:
            for ordinal, (qrecord, frecord) in enumerate(pairs):
                trajectory_counts = Counter()
                for q, f in aligned_rows(shard_path(qdir, qrecord), shard_path(fdir, frecord), qrecord):
                    qmap, fmap = nucleus_lookup(q, selected), nucleus_lookup(f, selected)
                    positions += 1
                    for key in sorted(qmap.keys() | fmap.keys()):
                        count = counts[key]
                        count["union_events"] += 1
                        qlp, flp = qmap.get(key, log_floor), fmap.get(key, log_floor)
                        value, exclusion = exact_correction(qlp, flp)
                        if exclusion == "nonpositive_shift":
                            count["nonpositive_events"] += 1
                            continue
                        count["positive_events_including_saturated"] += 1
                        if value is None:
                            count["saturated_positive_events_excluded"] += 1
                        else:
                            values.add(key, value)
                            trajectory_counts[key] += 1
                            count["valid_positive_events"] += 1
                            count["positive_q_imputed_events"] += int(key not in qmap)
                            count["positive_bf16_imputed_events"] += int(key not in fmap)
                            count["positive_both_nucleus_events"] += int(key in qmap and key in fmap)
                        pending.append({
                            "quantizer": quantizer, "trajectory_index": q["trajectory_index"],
                            "trajectory_id": q["trajectory_id"], "source_task": q["source_task"],
                            "response_position": q["response_position"], "token_id": key,
                            "q_in_nucleus": key in qmap, "bf16_in_nucleus": key in fmap,
                            "q_logprob_used": qlp, "bf16_logprob_used": flp, "delta_logp": qlp-flp,
                            "exact_logit_correction_magnitude": value, "used_for_calibration": value is not None,
                            "exclusion_reason": exclusion,
                        })
                        if len(pending) >= 8192:
                            writer.write_table(pa.Table.from_pylist(pending, schema=EVENT_SCHEMA))
                            pending.clear()
                for key, n in trajectory_counts.items():
                    counts[key]["positive_trajectory_support"] += 1
                    counts[key]["max_positive_events_one_trajectory"] = max(counts[key]["max_positive_events_one_trajectory"], n)
                if (ordinal + 1) % 100 == 0 or ordinal + 1 == len(pairs):
                    print(f"Calibration {quantizer}: {ordinal + 1}/{len(pairs)} trajectories; {positions} positions", flush=True)
            if pending:
                writer.write_table(pa.Table.from_pylist(pending, schema=EVENT_SCHEMA))
        medians = {k: values.median(k) for k in sorted(selected)}
    fields = ("union_events", "nonpositive_events", "positive_events_including_saturated", "valid_positive_events",
              "saturated_positive_events_excluded", "positive_trajectory_support", "max_positive_events_one_trajectory",
              "positive_q_imputed_events", "positive_bf16_imputed_events", "positive_both_nucleus_events")
    statistics = {}
    for key in sorted(selected):
        record = {field: int(counts[key][field]) for field in fields}
        if record["union_events"] != record["nonpositive_events"] + record["positive_events_including_saturated"]:
            raise ValueError("Union event accounting mismatch")
        if record["positive_events_including_saturated"] != record["valid_positive_events"] + record["saturated_positive_events_excluded"]:
            raise ValueError("Positive event accounting mismatch")
        record["max_positive_single_trajectory_fraction"] = record["max_positive_events_one_trajectory"] / record["valid_positive_events"]
        statistics[str(key)] = record
    return medians, {"trajectories": len(pairs), "aligned_positions": positions, "per_token": statistics}


def normalize(medians_awq, medians_gptq):
    if not medians_awq or set(medians_awq) != set(medians_gptq):
        raise ValueError("Both quantizers must supply exactly the nonempty final set")
    robust = {}
    for key in sorted(medians_awq):
        for value in (medians_awq[key], medians_gptq[key]):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid positive correction median for token {key}")
        robust[key] = min(medians_awq[key], medians_gptq[key])
    denominator = float(np.median(list(robust.values())))
    rows = []
    for key, value in robust.items():
        magnitude = value / denominator
        if not math.isfinite(magnitude):
            raise ValueError("Nonfinite normalized penalty")
        rounded = float(np.round(magnitude, 2))
        rows.append({"token_id": key, "median_exact_correction_awq": medians_awq[key],
                     "median_exact_correction_gptq": medians_gptq[key], "robust_median_exact_correction": value,
                     "normalization_denominator": denominator, "penalty_unrounded": magnitude,
                     "penalty": rounded, "raw_logit_bias": -rounded})
    return rows, denominator


def calibrate(selection_dir, output_dir, decision_file=None):
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse calibration directory: {output}")
    loaded = load_selection(selection_dir, decision_file)
    audit, selection_manifest = loaded["audit"], loaded["manifest"]
    inputs, decisions = loaded["inputs"], loaded["decisions"]
    retained = {r["token_id"]: r for r in decisions if r["decision"] == "retain"}
    selected = set(retained)
    floor = loaded["policy"]["tail_probability"]
    if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not 0 < floor < 1:
        raise ValueError("Invalid frozen selection probability floor")
    def frozen_input(filename):
        path = audit.add(filename)
        expected = selection_manifest["input_sha256"].get(str(path))
        if expected is None or expected != audit.files[str(path)]:
            raise ValueError(f"Raw calibration input changed or was not part of selection: {path}")
        return path
    print(f"Submitted {len(selected)} retained / {len(decisions)-len(selected)} excluded tokens "
          "(blank decisions retain; contextual review is user-owned). "
          "Validating Direction-2 sources (CPU only)...", flush=True)
    directories = {q: Path(p) for q, p in inputs["direction2"].items()}
    manifests = {q: unique_json(frozen_input(p / "manifest.json")) for q, p in directories.items()}
    pairs = {q: pair_manifests(manifests[q], manifests["bf16"], require_equal=True) for q in ("awq", "gptq")}
    for q, directory in directories.items():
        for record in manifests[q]["trajectories"]:
            frozen_input(shard_path(directory, record))
    tokenizer_path = Path(inputs["tokenizer"]).resolve()
    for filename in selection_manifest["input_sha256"]:
        if Path(filename).is_relative_to(tokenizer_path):
            frozen_input(filename)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    fingerprint = tokenizer_fingerprint(tokenizer)
    if fingerprint != manifests["bf16"]["tokenizer_vocab_sha256"]:
        raise ValueError("Tokenizer fingerprint differs from calibrated distributions")
    vocabulary = set(tokenizer.get_vocab().values())
    for key, row in retained.items():
        if key not in vocabulary or tokenizer.decode([key], skip_special_tokens=False, clean_up_tokenization_spaces=False) != row["decoded_piece"]:
            raise ValueError(f"Confirmed token does not match current tokenizer: {key}")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "status.json", {"state": "running"})
    try:
        (output / "events").mkdir()
        pd.DataFrame(decisions).to_csv(output / "confirmed_decisions.csv", index=False)
        pd.DataFrame(list(retained.values())).to_csv(output / "final_keywords.csv", index=False)
        medians, statistics = {}, {}
        for q in ("awq", "gptq"):
            medians[q], statistics[q] = extract_events(q, directories[q], directories["bf16"], pairs[q], selected, floor, output)
        rows, denominator = normalize(medians["awq"], medians["gptq"])
        for row in rows:
            row.update({k: retained[row["token_id"]][k] for k in ("decoded_piece", "route")})
            for q in ("awq", "gptq"):
                row.update({f"{field}_{q}": value for field, value in statistics[q]["per_token"][str(row["token_id"])].items()})
        pd.DataFrame(rows).to_csv(output / "penalty_table.csv", index=False)
        atomic_json(output / "event_statistics.json", statistics)
        payload = {
            "schema_version": 1, "kind": "ratio-v8-norm-noclip", "tokenizer_vocab_sha256": fingerprint,
            "token_count": len(rows), "shared_across_quantizers": ["awq", "gptq"],
            "logit_rule": "raw_logit[token_id] += token_biases[token_id]",
            "tail_probability_imputation": floor, "saturation_logprob_exclusive": SATURATION_LOGPROB,
            "probability_source": "temperature-scaled, pre-nucleus-renormalization nucleus_logprobs",
            "source_temperature": manifests["bf16"]["temperature"], "source_top_p": manifests["bf16"]["top_p"],
            "extra_temperature_multiplier": False, "normalization_denominator": denominator,
            "normalization_set": "all user-retained token IDs", "rounding_decimals": 2,
            "global_clip": None, "reliability_cap": None, "case_tying": False,
            "token_biases": {str(row["token_id"]): row["raw_logit_bias"] for row in rows},
            "tokens": [{"token_id": row["token_id"], "decoded_piece": row["decoded_piece"],
                        "raw_logit_bias": row["raw_logit_bias"]} for row in rows],
            "selection_manifest_sha256": audit.files[str(loaded["directory"] / "manifest.json")],
            "user_decisions_sha256": audit.files[str(loaded["decision_file"])],
        }
        audit.verify_unchanged()
        atomic_json(output / "token_bias_map.json", payload)
        atomic_json(output / "manifest.json", {
            "schema_version": 1, "kind": "user-confirmed-direction2-v8-calibration", "state": "complete",
            "selection_protocol": selection_manifest["protocol"],
            "custom_selection_policy": selection_manifest.get("custom_policy", False),
            "custom_selection_lexicons": selection_manifest.get("custom_lexicons", {}),
            "retained_count": len(rows), "excluded_count": len(decisions)-len(rows),
            "normalization_denominator": denominator, "input_sha256": audit.files,
            "source_selection_input_scope": "Frozen selection artifacts plus Direction-2 raw scores and tokenizer; no re-analysis of Direction 1",
            "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in [Path(__file__), ROOT / "keyword_selection/decisions.py", ROOT / "keyword_selection/inputs.py"]},
            "output_sha256": {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "status.json"},
        })
        atomic_json(output / "status.json", {"state": "complete"})
        print(f"Calibrated {len(rows)} tokens; inference map: {output / 'token_bias_map.json'}", flush=True)
        return payload
    except BaseException as error:
        atomic_json(output / "status.json", {"state": "failed", "error": str(error)})
        raise
