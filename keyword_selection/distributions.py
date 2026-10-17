"""Occurrence-weighted, same-prefix nucleus statistics for Direction 2."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from data_collection.io import sha256_file
from keyword_selection.inputs import pair_manifests, shard_path
SCORE_COLUMNS = ("trajectory_index", "trajectory_id", "source_task", "response_position", "token_id", "nucleus_token_ids", "nucleus_logprobs")
POSITION_BINS = (("p00_20",0,.2),("p20_40",.2,.4),("p40_60",.4,.6),("p60_80",.6,.8),("p80_100",.8,1.))

def token_labels(tokenizer, token_ids: np.ndarray) -> pd.DataFrame:
    special = set(tokenizer.all_special_ids)
    decodable_ids = set(tokenizer.get_vocab().values())
    rows = []
    for token_id in token_ids.tolist():
        piece = "" if int(token_id) not in decodable_ids else tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        rows.append(
            {
                "token_id": int(token_id),
                "token": tokenizer.convert_ids_to_tokens(int(token_id)),
                "decoded_piece": piece,
                "canonical_piece": piece.strip().lower(),
                "is_special_token": int(token_id) in special,
                "is_whitespace_piece": not bool(piece.strip()),
            }
        )
    return pd.DataFrame(rows, columns=["token_id", "token", "decoded_piece", "canonical_piece", "is_special_token", "is_whitespace_piece"])


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def analyze_pair(
    *,
    quant_dir: Path,
    bf16_dir: Path,
    quantizer: str,
    tokenizer,
    model_vocab_size: int,
    tail_probability: float,
    audit_modulus: int,
    max_audit_rows: int,
    exclude_unidentifiable_one_sided_events: bool,
) -> tuple[pd.DataFrame, list[dict], dict]:
    quant_manifest_path = quant_dir / "manifest.json"
    bf16_manifest_path = bf16_dir / "manifest.json"
    quant_manifest = json.loads(quant_manifest_path.read_text())
    bf16_manifest = json.loads(bf16_manifest_path.read_text())
    pairs = pair_manifests(quant_manifest, bf16_manifest, require_equal=True)
    categories = sorted({record["source_task"] for record, _ in pairs})
    category_index = {name: index for index, name in enumerate(categories)}

    # Some checkpoints pad the LM-head vocabulary beyond the tokenizer's
    # decodable IDs in some model families. Preserve those IDs in the
    # raw audit; the existing clean-token gate removes their empty pieces.
    vocab_size = max(len(tokenizer), int(tokenizer.vocab_size), model_vocab_size)
    events = np.zeros(vocab_size, dtype=np.int64)
    sum_q_lp = np.zeros(vocab_size, dtype=np.float64)
    sum_bf16_lp = np.zeros(vocab_size, dtype=np.float64)
    sum_delta = np.zeros(vocab_size, dtype=np.float64)
    sum_delta_sq = np.zeros(vocab_size, dtype=np.float64)
    positive = np.zeros(vocab_size, dtype=np.int64)
    negative = np.zeros(vocab_size, dtype=np.int64)
    both_events = np.zeros(vocab_size, dtype=np.int64)
    q_only_events = np.zeros(vocab_size, dtype=np.int64)
    bf16_only_events = np.zeros(vocab_size, dtype=np.int64)
    trajectory_support = np.zeros(vocab_size, dtype=np.int32)
    max_events_one_trajectory = np.zeros(vocab_size, dtype=np.int32)
    category_events = [np.zeros(vocab_size, dtype=np.int64) for _ in categories]
    category_sum = [np.zeros(vocab_size, dtype=np.float64) for _ in categories]
    position_events = [np.zeros(vocab_size, dtype=np.int64) for _ in POSITION_BINS]
    position_sum = [np.zeros(vocab_size, dtype=np.float64) for _ in POSITION_BINS]
    log_floor = math.log(tail_probability)
    audits: list[dict] = []
    positions = 0
    total_events = 0
    input_parquet_rows = 0
    input_parquet_bytes = 0
    raw_union_events = 0
    excluded_unidentifiable_events = 0

    for pair_number, (quant_record, bf16_record) in enumerate(pairs):
        quant_path = shard_path(quant_dir, quant_record)
        bf16_path = shard_path(bf16_dir, bf16_record)
        qmeta = pq.ParquetFile(quant_path).metadata
        bmeta = pq.ParquetFile(bf16_path).metadata
        if qmeta.num_rows != bmeta.num_rows or qmeta.num_rows != quant_record["response_token_count"] or not qmeta.num_rows:
            raise ValueError(f"Parquet row count mismatch: {quant_path} vs {bf16_path}")
        input_parquet_rows += qmeta.num_rows + bmeta.num_rows
        input_parquet_bytes += quant_path.stat().st_size + bf16_path.stat().st_size
        qtable = pq.read_table(quant_path, columns=list(SCORE_COLUMNS)).to_pylist()
        btable = pq.read_table(bf16_path, columns=list(SCORE_COLUMNS)).to_pylist()
        if len(qtable) != len(btable):
            raise ValueError("Materialized row count mismatch")
        response_length = len(qtable)
        group = category_index[quant_record["source_task"]]
        trajectory_ids: list[np.ndarray] = []

        for row_offset, (qrow, brow) in enumerate(zip(qtable, btable)):
            if qrow["response_position"] != row_offset:
                raise ValueError("Missing, repeated or unordered response positions")
            for field in ("trajectory_index", "trajectory_id", "source_task"):
                if qrow[field] != quant_record[field]:
                    raise ValueError(f"Score shard does not belong to manifest: {field}")
            for field in (
                "trajectory_index",
                "trajectory_id",
                "source_task",
                "response_position",
                "token_id",
            ):
                if qrow[field] != brow[field]:
                    raise ValueError(
                        f"Same-position alignment mismatch for {field}: {quant_path}"
                    )
            q_ids = np.asarray(qrow["nucleus_token_ids"], dtype=np.int32)
            bf16_ids = np.asarray(brow["nucleus_token_ids"], dtype=np.int32)
            q_lps = np.asarray(qrow["nucleus_logprobs"], dtype=np.float64)
            bf16_lps = np.asarray(brow["nucleus_logprobs"], dtype=np.float64)
            if len(q_ids) != len(q_lps) or len(bf16_ids) != len(bf16_lps):
                raise ValueError("Nucleus token/logprob length mismatch")
            if len(np.unique(q_ids)) != len(q_ids) or len(np.unique(bf16_ids)) != len(bf16_ids):
                raise ValueError("Duplicate token inside a nucleus")
            if not len(q_ids) or not len(bf16_ids) or not np.isfinite(q_lps).all() or not np.isfinite(bf16_lps).all():
                raise ValueError("Empty nucleus or nonfinite saved nucleus probabilities")
            if np.any(q_lps > 0) or np.any(bf16_lps > 0):
                raise ValueError("Positive log probability in saved nucleus")

            raw_union_size = len(np.union1d(q_ids, bf16_ids))
            raw_union_events += raw_union_size
            if exclude_unidentifiable_one_sided_events:
                bf16_id_set = set(bf16_ids.tolist())
                q_keep = (q_lps > log_floor) | np.fromiter(
                    (int(token_id) in bf16_id_set for token_id in q_ids),
                    dtype=bool,
                    count=len(q_ids),
                )
                q_ids = q_ids[q_keep]
                q_lps = q_lps[q_keep]
                q_id_set = set(q_ids.tolist())
                bf16_keep = (bf16_lps > log_floor) | np.fromiter(
                    (int(token_id) in q_id_set for token_id in bf16_ids),
                    dtype=bool,
                    count=len(bf16_ids),
                )
                bf16_ids = bf16_ids[bf16_keep]
                bf16_lps = bf16_lps[bf16_keep]
            union_ids = np.union1d(q_ids, bf16_ids).astype(np.int32, copy=False)
            excluded_unidentifiable_events += raw_union_size - len(union_ids)
            if len(union_ids) and (union_ids[0] < 0 or union_ids[-1] >= vocab_size):
                raise ValueError(f"Nucleus token ID outside model vocabulary: {quant_path}")
            q_lookup = dict(zip(q_ids.tolist(), q_lps.tolist()))
            bf16_lookup = dict(zip(bf16_ids.tolist(), bf16_lps.tolist()))
            q_present = np.fromiter((int(k) in q_lookup for k in union_ids), dtype=bool)
            bf16_present = np.fromiter((int(k) in bf16_lookup for k in union_ids), dtype=bool)
            q_used = np.fromiter(
                (q_lookup.get(int(k), log_floor) for k in union_ids), dtype=np.float64
            )
            bf16_used = np.fromiter(
                (bf16_lookup.get(int(k), log_floor) for k in union_ids), dtype=np.float64
            )
            direct_delta = q_used - bf16_used

            np.add.at(events, union_ids, 1)
            np.add.at(sum_q_lp, union_ids, q_used)
            np.add.at(sum_bf16_lp, union_ids, bf16_used)
            np.add.at(sum_delta, union_ids, direct_delta)
            np.add.at(sum_delta_sq, union_ids, direct_delta * direct_delta)
            np.add.at(positive, union_ids, direct_delta > 0.0)
            np.add.at(negative, union_ids, direct_delta < 0.0)
            both = q_present & bf16_present
            q_only = q_present & ~bf16_present
            bf16_only = ~q_present & bf16_present
            np.add.at(both_events, union_ids, both)
            np.add.at(q_only_events, union_ids, q_only)
            np.add.at(bf16_only_events, union_ids, bf16_only)
            np.add.at(category_events[group], union_ids, 1)
            np.add.at(category_sum[group], union_ids, direct_delta)
            relative_position = row_offset / max(response_length, 1)
            bin_index = min(int(relative_position * len(POSITION_BINS)), len(POSITION_BINS) - 1)
            np.add.at(position_events[bin_index], union_ids, 1)
            np.add.at(position_sum[bin_index], union_ids, direct_delta)
            trajectory_ids.append(union_ids)
            positions += 1
            total_events += len(union_ids)

            audit_key = (
                int(qrow["trajectory_index"]) * 1000003
                + int(qrow["response_position"]) * 9176
                + (0 if quantizer == "awq" else 1)
            )
            if audit_key % audit_modulus == 0 and len(audits) < max_audit_rows:
                for offset in range(min(3, len(union_ids))):
                    audits.append(
                        {
                            "quantizer": quantizer,
                            "quant_file": str(quant_path.resolve()),
                            "bf16_file": str(bf16_path.resolve()),
                            "trajectory_id": qrow["trajectory_id"],
                            "response_position": int(qrow["response_position"]),
                            "candidate_token_id": int(union_ids[offset]),
                            "q_in_nucleus": bool(q_present[offset]),
                            "bf16_in_nucleus": bool(bf16_present[offset]),
                            "q_candidate_logprob_used": float(q_used[offset]),
                            "bf16_candidate_logprob_used": float(bf16_used[offset]),
                            "direct_delta_logp": float(direct_delta[offset]),
                        }
                    )

        if trajectory_ids:
            ids, counts = np.unique(np.concatenate(trajectory_ids), return_counts=True)
            trajectory_support[ids] += 1
            max_events_one_trajectory[ids] = np.maximum(
                max_events_one_trajectory[ids], counts.astype(np.int32)
            )
        if (pair_number + 1) % 500 == 0:
            print(
                f"{quantizer}: {pair_number + 1}/{len(pairs)} trajectories, "
                f"{positions} positions, {total_events} events",
                flush=True,
            )

    token_ids = np.flatnonzero(events > 0).astype(np.int32)
    count = events[token_ids]
    mean_q = safe_divide(sum_q_lp[token_ids], count)
    mean_bf16 = safe_divide(sum_bf16_lp[token_ids], count)
    mean_delta = safe_divide(sum_delta[token_ids], count)
    variance = np.maximum(safe_divide(sum_delta_sq[token_ids], count) - mean_delta**2, 0.0)
    frame = token_labels(tokenizer, token_ids)
    frame["events"] = count
    frame["trajectory_support"] = trajectory_support[token_ids]
    frame["mean_logp_q_used"] = mean_q
    frame["mean_logp_bf16_used"] = mean_bf16
    frame["mean_direct_delta_logp"] = mean_delta
    frame["abs_mean_direct_delta_logp"] = np.abs(mean_delta)
    frame["std_direct_delta_logp"] = np.sqrt(variance)
    frame["positive_rate_direct_delta"] = safe_divide(positive[token_ids], count)
    frame["negative_rate_direct_delta"] = safe_divide(negative[token_ids], count)
    frame["both_events"] = both_events[token_ids]
    frame["q_only_events"] = q_only_events[token_ids]
    frame["bf16_only_events"] = bf16_only_events[token_ids]
    frame["q_imputed_rate"] = safe_divide(bf16_only_events[token_ids], count)
    frame["bf16_imputed_rate"] = safe_divide(q_only_events[token_ids], count)
    frame["both_rate"] = safe_divide(both_events[token_ids], count)
    frame["q_only_rate"] = safe_divide(q_only_events[token_ids], count)
    frame["bf16_only_rate"] = safe_divide(bf16_only_events[token_ids], count)
    frame["max_events_one_trajectory"] = max_events_one_trajectory[token_ids]
    frame["max_single_trajectory_fraction"] = safe_divide(
        max_events_one_trajectory[token_ids], count
    )

    source_task_support = np.zeros(len(token_ids), dtype=np.int32)
    source_means = []
    for category, cat_events, cat_sum in zip(categories, category_events, category_sum):
        cat_count = cat_events[token_ids]
        cat_mean = safe_divide(cat_sum[token_ids], cat_count)
        frame[f"{category}_events"] = cat_count
        frame[f"{category}_mean_direct_delta_logp"] = cat_mean
        source_task_support += cat_count > 0
        source_means.append(np.where(cat_count > 0, cat_mean, np.nan))
    source_matrix = np.stack(source_means, axis=1)
    frame["source_task_support"] = source_task_support
    frame["min_source_task_mean_direct_delta_logp"] = np.nanmin(source_matrix, axis=1)
    frame["max_source_task_mean_direct_delta_logp"] = np.nanmax(source_matrix, axis=1)

    for (bin_name, _, _), bin_events, bin_sum in zip(
        POSITION_BINS, position_events, position_sum
    ):
        bin_count = bin_events[token_ids]
        frame[f"{bin_name}_events"] = bin_count
        frame[f"{bin_name}_mean_direct_delta_logp"] = safe_divide(
            bin_sum[token_ids], bin_count
        )

    frame.insert(0, "quantizer", quantizer)
    frame = frame.sort_values(
        ["abs_mean_direct_delta_logp", "events", "trajectory_support"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    summary = {
        "quantizer": quantizer,
        "model": quant_manifest["model"],
        "bf16_model": bf16_manifest["model"],
        "temperature": quant_manifest["temperature"],
        "top_p": quant_manifest["top_p"],
        "trajectories": len(pairs),
        "positions": positions,
        "candidate_events": total_events,
        "raw_union_events_before_identifiability_filter": raw_union_events,
        "excluded_unidentifiable_one_sided_events": excluded_unidentifiable_events,
        "identifiability_filter_enabled": exclude_unidentifiable_one_sided_events,
        "table_event_sum": int(frame["events"].sum()),
        "unique_tokens": len(frame),
        "input_parquet_rows_both_sides": input_parquet_rows,
        "input_parquet_bytes_both_sides": input_parquet_bytes,
        "quant_score_manifest": str(quant_manifest_path.resolve()),
        "quant_score_manifest_sha256": sha256_file(quant_manifest_path),
        "bf16_score_manifest": str(bf16_manifest_path.resolve()),
        "bf16_score_manifest_sha256": sha256_file(bf16_manifest_path),
    }
    if summary["candidate_events"] != summary["table_event_sum"]:
        raise ValueError("Event accounting mismatch")
    return frame, audits, summary

