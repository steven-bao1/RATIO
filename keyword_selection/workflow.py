"""Raw-score analysis through auditable, user-reviewable candidate sets."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoConfig, AutoTokenizer

from evaluation import atomic_json
from data_collection.io import load_exact_trajectories, rollout_correctness, trajectory_id, sha256_file
from data_collection.materialize import tokenizer_fingerprint
from keyword_selection import evidence, stability
from keyword_selection.distributions import analyze_pair
from keyword_selection.gates import policy, lexical_families, select_candidates, REMINDER
from keyword_selection.inputs import Provenance, read_json, pair_manifests, shard_path, validate_inputs
from keyword_selection.repetition import lexical_tokens, repeated_ngram_fraction, duplicate_line_fraction, max_same_token_run
from keyword_selection.semantic_filter import SemanticFilter

ROOT = Path(__file__).resolve().parents[1]


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(frame, path):
    # Directory is exclusively created by the top-level runner, never reused.
    frame.to_csv(path, index=False, encoding="utf-8")


def direction2(inputs, tokenizer, settings, lexicon, output):
    frames, summaries = {}, {}
    vocab = int(AutoConfig.from_pretrained(inputs["tokenizer"], local_files_only=True, trust_remote_code=False).vocab_size)
    for ordinal, q in enumerate(("awq", "gptq"), 1):
        frame, audit, summary = analyze_pair(
            quant_dir=Path(inputs["direction2"][q]), bf16_dir=Path(inputs["direction2"]["bf16"]),
            quantizer=q, tokenizer=tokenizer, model_vocab_size=vocab,
            tail_probability=settings["tail_probability"], audit_modulus=10007, max_audit_rows=600,
            exclude_unidentifiable_one_sided_events=False)
        frames[q], summaries[q] = frame, summary
        write_csv(frame, output / f"table{ordinal}_{q}.csv")
        atomic_json(output / f"{q}_event_audit.json", json_safe(audit))
    a, g = frames["awq"], frames["gptq"]
    t3 = stability.build_common(a, g, settings["events_threshold_exclusive"])
    t4 = stability.add_shared_direction(t3)
    support_q25 = {q: float(t4[f"trajectory_support_{q}"].quantile(.25)) for q in ("awq", "gptq")}
    t5 = stability.add_min_support(t4[
        t4.trajectory_support_awq.ge(math.ceil(support_q25["awq"]) if len(t4) else math.inf)
        & t4.trajectory_support_gptq.ge(math.ceil(support_q25["gptq"]) if len(t4) else math.inf)])
    n = settings["trajectory_threshold_exclusive"]
    t6 = stability.add_min_support(t4[t4.trajectory_support_awq.gt(n) & t4.trajectory_support_gptq.gt(n)])
    stable = stability.add_stability_metrics(t6, a, g)
    t7, pos = stability.quantile_select(stable[stable.shared_direction.eq("positive")], use_concentration_gate=False)
    t8, neg = stability.quantile_select(stable[stable.shared_direction.eq("negative")], use_concentration_gate=False)
    t9 = pd.concat([t7, t8], ignore_index=True)
    t10 = t6.loc[t6.decoded_piece.map(lambda s: s.strip().casefold() in lexicon).astype(bool)].copy()
    ids9, ids10 = set(t9.token_id), set(t10.token_id)
    union = t6[t6.token_id.isin(ids9 | ids10)].copy().sort_values("token_id").reset_index(drop=True)
    union["source_table9"] = union.token_id.isin(ids9)
    union["source_table10"] = union.token_id.isin(ids10)
    for name, frame in (("table3_common_clean", t3), ("table4_same_direction", t4),
                        ("table5_support_q25_diagnostic", t5), ("table6_common_entry", t6),
                        ("table7_positive_stable", t7), ("table8_negative_stable", t8),
                        ("table9_numeric_union", t9), ("table10_lexical_recall", t10), ("candidate_union", union)):
        write_csv(frame, output / (name + ".csv"))
    atomic_json(output / "statistics.json", json_safe({"quantizers": summaries, "support_q25_diagnostic": support_q25,
                "stability_thresholds": {"positive": pos, "negative": neg}, "concentration_gate": False,
                "table10_rule": "table6 membership AND whole-piece lexical-family match",
                "counts": {"table3": len(t3), "table4": len(t4), "table5": len(t5), "table6": len(t6),
                           "table7": len(t7), "table8": len(t8), "table9": len(t9), "table10": len(t10), "union": len(union)}}))
    return union


def correctness_matches(quantized, full):
    """Use frozen question IDs, or exact prompt IDs for legacy no-question-ID data."""
    if not quantized or not full:
        raise ValueError("Empty correctness rollout")
    use_ids = all("question_id" in r for r in quantized + full)
    def key(row):
        return row["question_id"] if use_ids else tuple(row["prompt_token_ids"])
    lookup = {key(r): (i, r) for i, r in enumerate(full)}
    if len(lookup) != len(full) or len({key(r) for r in quantized}) != len(quantized):
        raise ValueError("Duplicate correctness identities")
    if set(lookup) != {key(r) for r in quantized}:
        raise ValueError("Missing BF16 correctness questions; cannot treat missing data as false")
    pairs = []
    for q in quantized:
        index, f = lookup[key(q)]
        if q["prompt_token_ids"] != f["prompt_token_ids"]:
            raise ValueError("Correctness pair has different exact prompt IDs")
        qm, _, qc = rollout_correctness(q)
        fm, _, fc = rollout_correctness(f)
        if qc is None or fc is None or qm != fm:
            raise ValueError("Missing or mismatched correctness metric")
        pairs.append((index, qc, fc))
    return pairs


def trajectory_labels(row, quant_correct, full_correct):
    if not isinstance(row.get("generated_text"), str):
        raise ValueError("Original generated_text is required for repetition and contextual review")
    ids = row["generated_token_ids"]
    if not ids:
        raise ValueError("Cannot analyze an empty response")
    words = lexical_tokens(row["generated_text"])
    repeat = 100 * repeated_ngram_fraction(words)
    duplicate = 100 * duplicate_line_fraction(row["generated_text"])
    run = max_same_token_run(words)
    loop = bool((repeat >= 80 and duplicate >= 50) or run >= 64)
    group = {(False, False): "both_wrong", (False, True): "q_wrong_fp_correct",
             (True, False): "q_correct_fp_wrong", (True, True): "both_correct"}[(quant_correct, full_correct)]
    return {"q_correct": int(quant_correct), "independent_bf16_correct": int(full_correct),
            "correctness_group": group, "output_tokens": len(ids), "is_long": int(len(ids) >= 60000),
            "loop": int(loop), "degenerate": int(loop or len(ids) >= 60000),
            "explicit_error_loop": int(loop and not quant_correct), "repeat4_pct": repeat,
            "duplicate_line_pct": duplicate, "max_same_lexical_token_run": run}


SCORE_FIELDS = ("event_count", "delta_sum", "anomaly_sum", "direction_match_count", "exact_event_count",
                "exact_magnitude_sum", "repeated4_event_count", "high_repeat4_event_count",
                "max_local_4gram_multiplicity", "max_same_token_id_run")
PREVIEW_FIELDS = ("event_count", "delta_sum", "anomaly_sum", "direction_match_count", "floor_event_count",
                  "late_event_count", "high_repeat_prefix_event_count")


def direction1(inputs, candidates, settings, output):
    directions = {int(r.token_id): -1 if r.shared_direction == "positive" else 1 for r in candidates.itertuples()}
    labels, scores, previews = [], [], []
    question_sets = {}
    for cell in sorted(inputs["direction1"], key=lambda c: (c["quantizer"], c["benchmark"])):
        qdir, fdir = Path(cell["quant_scores"]), Path(cell["bf16_scores"])
        qman, fman = read_json(qdir / "manifest.json"), read_json(fdir / "manifest.json")
        raw = load_exact_trajectories(Path(cell["rollouts"]))
        full = load_exact_trajectories(Path(cell["bf16_rollouts"]))
        correctness = correctness_matches(raw, full)
        prompt_set = {tuple(r["prompt_token_ids"]) for r in raw}
        if cell["benchmark"] in question_sets and prompt_set != question_sets[cell["benchmark"]]:
            raise ValueError("AWQ/GPTQ use different questions or prompt token IDs")
        question_sets[cell["benchmark"]] = prompt_set
        pairs = pair_manifests(qman, fman)
        print(f"Direction 1 {cell['quantizer']}/{cell['benchmark']}: {len(pairs)} selected trajectories", flush=True)
        for q, f in pairs:
            index = q["trajectory_index"]
            if type(index) is not int or not 0 <= index < len(raw):
                raise ValueError("Score trajectory index outside rollout array")
            row = raw[index]
            ids = row["generated_token_ids"]
            if trajectory_id(row["prompt_token_ids"], ids) != q["trajectory_id"]:
                raise ValueError("Score trajectory IDs do not match original prompt/response IDs")
            if len(ids) != q["response_token_count"] or len(row["prompt_token_ids"]) != q["prompt_token_count"]:
                raise ValueError("Score trajectory token count mismatch")
            bf_index, qc, fc = correctness[index]
            label = {"trajectory_key": len(labels), "trajectory_id": q["trajectory_id"],
                     "quantizer": cell["quantizer"], "dataset": cell["benchmark"], "trajectory_index": index,
                     "rollout_file": cell["rollouts"], "bf16_rollout_file": cell["bf16_rollouts"],
                     "bf16_source_index": bf_index, "question_id": row.get("question_id", ""),
                     **trajectory_labels(row, qc, fc)}
            labels.append(label)
            score, preview, _ = evidence.process_trajectory(shard_path(qdir, q), shard_path(fdir, f), ids,
                                    directions, 0.0, tail_probability=settings["tail_probability"])
            for events, fields, destination in ((score, SCORE_FIELDS, scores), (preview, PREVIEW_FIELDS, previews)):
                for event in events:
                    destination.append({**label, **dict.fromkeys(fields, 0), **event})
    labels = pd.DataFrame(labels)
    score = pd.DataFrame(scores, columns=[*labels.columns, "token_id", *SCORE_FIELDS])
    preview = pd.DataFrame(previews, columns=[*labels.columns, "token_id", *PREVIEW_FIELDS])
    write_csv(labels, output / "trajectory_labels.csv")
    write_csv(score, output / "scoring_trajectory_events.csv")
    write_csv(preview, output / "preview_trajectory_events.csv")
    if len(candidates):
        by_quantizer = pd.concat([
            evidence.aggregate_channel("scoring", labels, score, candidates, resamples=settings["bootstrap_resamples"], seed=settings["seed"]),
            evidence.aggregate_channel("preview", labels, preview, candidates, resamples=settings["bootstrap_resamples"], seed=settings["seed"] + 10000000),
        ], ignore_index=True)
        cross = evidence.cross_quantizer(by_quantizer, candidates)
    else:
        by_quantizer = pd.DataFrame(columns=["token_id", "channel", "quantizer"])
        cross = candidates.copy()
        for channel in ("scoring", "preview"):
            for metric in ("trajectory_support", "event_mean_anomaly", "trajectory_mean_anomaly",
                           "error_minus_control_anomaly", "error_loop_to_nonloop_rate_ratio", "high_repeat4_trajectory_support"):
                for q in ("awq", "gptq"):
                    cross[f"{channel}_{metric}_{q}"] = pd.Series(dtype=float)
    write_csv(by_quantizer, output / "candidate_evidence_by_quantizer.csv")
    write_csv(cross, output / "candidate_evidence_cross_quantizer.csv")
    atomic_json(output / "coverage.json", {
        "trajectory_counts": {f"{q}/{d}": int(n) for (q, d), n in labels.groupby(["quantizer", "dataset"]).size().items()},
        "correctness_groups": {f"{q}/{d}/{g}": int(n) for (q, d, g), n in labels.groupby(["quantizer", "dataset", "correctness_group"]).size().items()},
        "undefined_statistics": "Empty groups or zero observed events remain NaN/blank, never imputed to zero. They cannot supply positive evidence.",
    })
    return cross, labels


def contexts(decisions, labels, tokenizer, settings, audit, output):
    """Stratified, reproducible token-ID windows, not invented supporting examples."""
    wanted = set(decisions.loc[decisions.status.eq("context_pending"), "token_id"].astype(int))
    buckets, occurrence_totals = {}, dict.fromkeys(wanted, 0)
    by_source = labels.groupby("rollout_file", sort=True)
    for filename, selected in by_source:
        raw = load_exact_trajectories(Path(filename))
        for label in selected.to_dict("records"):
            row = raw[label["trajectory_index"]]
            ids, text = row["generated_token_ids"], row["generated_text"]
            positions = {}
            for position, token in enumerate(ids):
                if token in wanted:
                    positions.setdefault(token, []).append(position)
            for token, hits in positions.items():
                occurrence_totals[token] += len(hits)
                stratum = (token, label["quantizer"], label["dataset"], label["correctness_group"], label["loop"])
                rank = hashlib.sha256((str(settings["seed"]) + ":" + label["trajectory_id"]).encode()).hexdigest()
                bucket = buckets.setdefault(stratum, [])
                capacity = settings["context_trajectories_per_stratum"]
                if len(bucket) >= capacity and rank >= max(r[0] for r in bucket):
                    continue
                sampled = sorted(set(int(i) for i in np.linspace(0, len(hits)-1, min(settings["contexts_per_trajectory"], len(hits)))))
                examples = []
                for chosen in sampled:
                    position = hits[chosen]
                    start, end = max(0, position-settings["context_window_tokens"]), min(len(ids), position+settings["context_window_tokens"]+1)
                    window = tokenizer.decode(ids[start:end], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    offset = text.find(window) if window else -1
                    examples.append({
                        "token_id": token, "decoded_piece": tokenizer.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False),
                        "quantizer": label["quantizer"], "benchmark": label["dataset"], "trajectory_id": label["trajectory_id"],
                        "rollout_file": filename, "rollout_sha256": audit.files[str(Path(filename).resolve())],
                        "row_index": label["trajectory_index"], "question_id": label["question_id"],
                        "response_position": position, "window_start": start, "window_end_exclusive": end,
                        "window_token_ids": ids[start:end], "target_offset_in_window": position-start,
                        "decoded_token_window": window, "verbatim_raw_excerpt": text[offset:offset+len(window)] if offset >= 0 else None,
                        "raw_excerpt_first_char_offset": offset if offset >= 0 else None,
                        "raw_excerpt_note": "First exact text match, not a unique occurrence locator; token IDs/position are authoritative" if offset >= 0 else "Window decoding differs from stored raw text; inspect complete source row",
                        "q_correct": label["q_correct"], "bf16_correct": label["independent_bf16_correct"],
                        "correctness_group": label["correctness_group"], "loop": label["loop"],
                        "explicit_error_loop": label["explicit_error_loop"], "occurrences_in_trajectory": len(hits),
                    })
                bucket.append((rank, examples))
                bucket.sort(key=lambda pair: pair[0])
                del bucket[capacity:]
    totals = dict.fromkeys(wanted, 0)
    with (output / "occurrence_contexts.jsonl").open("x", encoding="utf-8") as stream:
        for stratum in sorted(buckets):
            for _, examples in buckets[stratum]:
                for example in examples:
                    stream.write(json.dumps(json_safe(example), ensure_ascii=False, allow_nan=False) + "\n")
                    totals[example["token_id"]] += 1
    return [{"token_id": k, "observed_occurrences": occurrence_totals[k], "exported_windows": totals[k]} for k in sorted(wanted)]


def run(inputs, output, *, policy_file=None, lexicon_file=None, exclusions_file=None, source_config=None,
        post_qsti_exclusions_file=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse existing selection directory: {output}")
    settings = policy(policy_file)
    lexicon_file = Path(lexicon_file or ROOT / "configs/selection/table10_lexicon.json")
    exclusions_file = Path(exclusions_file or ROOT / "configs/selection/d1_exclusions.json")
    post_qsti_exclusions_file = Path(post_qsti_exclusions_file or ROOT / "configs/selection/post_qsti_exclusions.json")
    audit = Provenance()
    for path in (source_config, policy_file, lexicon_file, exclusions_file, post_qsti_exclusions_file,
                 inputs.get("collection_manifest")):
        if path:
            audit.add(path)
    lexicon, exclusions = lexical_families(lexicon_file), lexical_families(exclusions_file)
    semantic_filter = SemanticFilter(post_qsti_exclusions_file)
    print("Validating raw-data provenance and hashing inputs (CPU only)...", flush=True)
    identity = validate_inputs(inputs, audit)
    tokenizer_path = Path(inputs["tokenizer"])
    for path in sorted(tokenizer_path.iterdir()):
        if path.is_file() and path.suffix in (".json", ".model", ".txt"):
            audit.add(path)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    if tokenizer_fingerprint(tokenizer) != identity["tokenizer_vocab_sha256"]:
        raise ValueError("Tokenizer vocabulary does not match raw score manifests")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "status.json", {"state": "running"})
    try:
        for name in ("direction2", "direction1", "review"):
            (output / name).mkdir()
        atomic_json(output / "policy.json", settings)
        atomic_json(output / "inputs.json", inputs)
        atomic_json(output / "review/table10_lexicon.json", read_json(lexicon_file))
        atomic_json(output / "review/d1_exclusions.json", read_json(exclusions_file))
        atomic_json(output / "review/post_qsti_exclusions.json", semantic_filter.config)
        print("Building Direction-2 tables...", flush=True)
        candidates = direction2(inputs, tokenizer, settings, lexicon, output / "direction2")
        candidates = semantic_filter.annotate(candidates)
        filter_summary = {"version": semantic_filter.config["version"], "union_count": len(candidates),
                          "excluded_count": int((~candidates.post_qsti_eligible).sum()),
                          "eligible_count": int(candidates.post_qsti_eligible.sum()),
                          "threshold_reference": "Full unfiltered Table9/Table10 union; exclusions affect eligibility only"}
        write_csv(candidates, output / "direction2/candidate_union.csv")
        statistics = read_json(output / "direction2/statistics.json")
        statistics["post_qsti_filter"] = filter_summary
        atomic_json(output / "direction2/statistics.json", statistics)
        print(f"Direction-2 union: {len(candidates)}; post-QSTI exclusions: {filter_summary['excluded_count']}; "
              f"eligible: {filter_summary['eligible_count']}. Building Direction-1 evidence with the full threshold reference...", flush=True)
        cross, labels = direction1(inputs, candidates, settings, output / "direction1")
        decisions, thresholds = select_candidates(cross, exclusions, settings)
        atomic_json(output / "direction1/strict_q25_thresholds.json", thresholds)
        write_csv(decisions, output / "all_candidate_decisions.csv")
        for status, filename in (("preliminary", "preliminary_keywords.csv"), ("context_pending", "review/context_review_candidates.csv"),
                                 ("lexically_excluded", "lexically_excluded.csv"), ("negative_watchlist", "negative_watchlist.csv")):
            write_csv(decisions[decisions.status.eq(status)], output / filename)
        review_counts = contexts(decisions, labels, tokenizer, settings, audit, output / "review")
        atomic_json(output / "review/context_coverage.json", review_counts)
        editable = decisions[decisions.status.isin(["preliminary", "context_pending"])][["token_id", "decoded_piece", "status"]].copy()
        editable["decision"] = ""
        write_csv(editable, output / "review/user_decisions.csv")
        counts = decisions.status.value_counts().to_dict()
        (output / "README.md").write_text(
            "# Preliminary selection\n\n" + REMINDER + "\n\n"
            + "Counts: `" + json.dumps(counts) + "`\n\n"
            + "Read preliminary_keywords.csv and review/context_review_candidates.csv. The latter is excluded from the preliminary set. "
            + "Audit all_candidate_decisions.csv, direction1/ evidence, and review/occurrence_contexts.jsonl. "
            + "Review windows are stratified samples, not exhaustive evidence or automatic retain/exclude recommendations. "
            + "In review/user_decisions.csv, leave decision blank to retain; enter no or 0 to exclude. Keep all rows and identity fields unchanged. There is no reason column. Run tools/calibrate_penalties.py --selection THIS_DIRECTORY --output-dir NEW_CALIBRATION_DIRECTORY to submit and calculate penalties.\n",
            encoding="utf-8")
        audit.verify_unchanged()
        from keyword_selection.gates import DEFAULTS
        manifest = {"schema_version": 1, "protocol": "public-lexical-q25-r1-ce-v3", "state": "complete_pending_user_review",
                    "post_qsti_filter": filter_summary,
                    "custom_policy": settings != DEFAULTS,
                    "custom_lexicons": {"table10": lexicon != lexical_families(ROOT / "configs/selection/table10_lexicon.json"),
                                        "exclusions": exclusions != lexical_families(ROOT / "configs/selection/d1_exclusions.json"),
                                        "post_qsti_exclusions": semantic_filter.config != read_json(ROOT / "configs/selection/post_qsti_exclusions.json")},
                    "counts": counts, "reminder": REMINDER, "policy": settings, "input_sha256": audit.files,
                    "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted((ROOT / "keyword_selection").glob("*.py"))},
                    "output_sha256": {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "status.json"}}
        atomic_json(output / "manifest.json", json_safe(manifest))
        atomic_json(output / "status.json", {"state": "complete_pending_user_review"})
        print(json.dumps({"output": str(output), "counts": counts, "finalized": False}, ensure_ascii=False), flush=True)
        return manifest
    except BaseException as error:
        atomic_json(output / "status.json", {"state": "failed", "error": str(error)})
        raise


