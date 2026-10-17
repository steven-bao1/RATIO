"""Actual-token and preview evidence, trajectory-weighted audits and bootstrap CIs."""
from pathlib import Path
from collections import Counter, defaultdict
import itertools
import math
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
QUANTIZERS = ("awq", "gptq")

def stable_logit_from_logp(value: float) -> float:
    if not math.isfinite(value) or value >= -1e-7:
        return math.nan
    return value - math.log(-math.expm1(value))


def repeat_metadata(token_ids: list[int], candidate_ids: set[int]) -> dict[int, dict]:
    positions: dict[int, list[int]] = defaultdict(list)
    same_run: Counter[int] = Counter()
    previous = None
    run = 0
    for position, token_id in enumerate(token_ids):
        if token_id in candidate_ids:
            positions[token_id].append(position)
        if token_id == previous:
            run += 1
        else:
            previous, run = token_id, 1
        if token_id in candidate_ids:
            same_run[token_id] = max(same_run[token_id], run)

    counts = Counter(
        tuple(token_ids[index : index + 4])
        for index in range(max(0, len(token_ids) - 3))
    )
    last_start = len(token_ids) - 4
    result = {}
    for token_id, token_positions in positions.items():
        repeated = high = 0
        maximum = 1
        for position in token_positions:
            local = 1
            for start in range(
                max(0, position - 3), min(position, last_start) + 1
            ):
                local = max(local, counts[tuple(token_ids[start : start + 4])])
            repeated += int(local >= 2)
            high += int(local >= 8)
            maximum = max(maximum, local)
        result[token_id] = {
            "repeated4_event_count": repeated,
            "high_repeat4_event_count": high,
            "max_local_4gram_multiplicity": maximum,
            "max_same_token_id_run": int(same_run[token_id]),
        }
    return result


def prefix_multiplicity(token_ids: list[int]) -> np.ndarray:
    result = np.zeros(len(token_ids), dtype=np.int32)
    if len(token_ids) < 4:
        return result
    counts = Counter(
        tuple(token_ids[index : index + 4])
        for index in range(len(token_ids) - 3)
    )
    for position in range(4, len(token_ids)):
        result[position] = counts[tuple(token_ids[position - 4 : position])]
    return result


def bootstrap_difference(
    first: np.ndarray, second: np.ndarray, *, resamples: int, seed: int
) -> tuple[float, float]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]
    if len(first) < 2 or len(second) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    values = (
        first[rng.integers(0, len(first), (resamples, len(first)))].mean(axis=1)
        - second[rng.integers(0, len(second), (resamples, len(second)))].mean(axis=1)
    )
    return tuple(float(value) for value in np.quantile(values, (0.025, 0.975)))


def compare_batches(q_path: Path, f_path: Path, batch_size: int = 2048):
    columns = [
        "response_position",
        "token_id",
        "raw_logprob",
        "nucleus_token_ids",
        "nucleus_logprobs",
    ]
    q_batches = pq.ParquetFile(q_path).iter_batches(columns=columns, batch_size=batch_size)
    f_batches = pq.ParquetFile(f_path).iter_batches(columns=columns, batch_size=batch_size)
    sentinel = object()
    for q_batch, f_batch in itertools.zip_longest(q_batches, f_batches, fillvalue=sentinel):
        if q_batch is sentinel or f_batch is sentinel:
            raise ValueError(f"Batch count mismatch: {q_path} / {f_path}")
        q = q_batch.to_pydict()
        f = f_batch.to_pydict()
        if q["response_position"] != f["response_position"] or q["token_id"] != f["token_id"]:
            raise ValueError(f"Same-position alignment mismatch: {q_path} / {f_path}")
        yield q, f


def process_trajectory(
    q_path: Path,
    f_path: Path,
    generated_ids: list[int],
    direction: dict[int, int],
    positive_delta_min: float,
    tail_probability: float = 0.001,
) -> tuple[list[dict], list[dict], dict]:
    log_floor = math.log(tail_probability)
    candidate_ids = set(direction)
    penalized_ids = {token_id for token_id, sign in direction.items() if sign == -1}
    boosted_ids = {token_id for token_id, sign in direction.items() if sign == 1}
    repeat = repeat_metadata(generated_ids, candidate_ids)
    prefix_mult = prefix_multiplicity(generated_ids)
    response_tokens = len(generated_ids)
    score = defaultdict(lambda: defaultdict(float))
    preview = defaultdict(lambda: defaultdict(float))
    rows_seen = 0

    for q, f in compare_batches(q_path, f_path):
        for offset, token_id in enumerate(q["token_id"]):
            position = int(q["response_position"][offset])
            if position != rows_seen or position >= len(generated_ids) or int(token_id) != generated_ids[position]:
                raise ValueError(f"Generated-token mismatch at {q_path}:{position}")
            token_id = int(token_id)
            q_lp = float(q["raw_logprob"][offset])
            f_lp = float(f["raw_logprob"][offset])
            if not math.isfinite(q_lp) or not math.isfinite(f_lp) or q_lp > 0 or f_lp > 0:
                raise ValueError("Missing or invalid actual-token raw log probability")
            if token_id in candidate_ids and math.isfinite(q_lp) and math.isfinite(f_lp):
                sign = direction[token_id]
                delta = q_lp - f_lp
                anomaly = -sign * delta
                acc = score[token_id]
                acc["event_count"] += 1
                acc["delta_sum"] += delta
                acc["anomaly_sum"] += anomaly
                acc["direction_match_count"] += int(anomaly > 0)
                q_logit = stable_logit_from_logp(q_lp)
                f_logit = stable_logit_from_logp(f_lp)
                correction = f_logit - q_logit
                aligned_correction = sign * correction
                if (
                    anomaly > positive_delta_min
                    and math.isfinite(aligned_correction)
                    and aligned_correction > 0
                ):
                    acc["exact_event_count"] += 1
                    acc["exact_magnitude_sum"] += aligned_correction

            q_ids = [int(value) for value in q["nucleus_token_ids"][offset]]
            f_ids = [int(value) for value in f["nucleus_token_ids"][offset]]
            q_lps = [float(value) for value in q["nucleus_logprobs"][offset]]
            f_lps = [float(value) for value in f["nucleus_logprobs"][offset]]
            for nucleus_ids, lps in ((q_ids, q_lps), (f_ids, f_lps)):
                if (not nucleus_ids or len(nucleus_ids) != len(lps) or len(set(nucleus_ids)) != len(nucleus_ids)
                        or any(not math.isfinite(lp) or lp > 0 for lp in lps)):
                    raise ValueError("Invalid saved preview nucleus")
            q_map = dict(zip(q_ids, q_lps))
            f_map = dict(zip(f_ids, f_lps))
            events: list[tuple[int, float, float, bool]] = []
            for preview_id in penalized_ids.intersection(q_map):
                events.append(
                    (preview_id, q_map[preview_id], f_map.get(preview_id, log_floor), preview_id not in f_map)
                )
            for preview_id in boosted_ids.intersection(f_map):
                events.append(
                    (preview_id, q_map.get(preview_id, log_floor), f_map[preview_id], preview_id not in q_map)
                )
            for preview_id, preview_q_lp, preview_f_lp, floor_event in events:
                sign = direction[preview_id]
                delta = preview_q_lp - preview_f_lp
                anomaly = -sign * delta
                acc = preview[preview_id]
                acc["event_count"] += 1
                acc["delta_sum"] += delta
                acc["anomaly_sum"] += anomaly
                acc["direction_match_count"] += int(anomaly > 0)
                acc["floor_event_count"] += int(floor_event)
                acc["late_event_count"] += int(position >= math.floor(0.8 * response_tokens))
                acc["high_repeat_prefix_event_count"] += int(prefix_mult[position] >= 8)
            rows_seen += 1

    if rows_seen != response_tokens:
        raise ValueError(f"Expected {response_tokens} positions, read {rows_seen}: {q_path}")

    score_rows = []
    for token_id, acc in score.items():
        metadata = repeat[token_id]
        score_rows.append(
            {
                "token_id": token_id,
                **{key: value for key, value in acc.items()},
                **metadata,
            }
        )
    preview_rows = [
        {"token_id": token_id, **dict(acc)} for token_id, acc in preview.items()
    ]
    return score_rows, preview_rows, {"positions": rows_seen}


def aggregate_channel(
    channel: str,
    labels: pd.DataFrame,
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    records = []
    token_ids = candidates["token_id"].astype(int).tolist()
    for quantizer_index, quantizer in enumerate(QUANTIZERS):
        q_labels = labels[labels["quantizer"] == quantizer].copy()
        q_events = events[events["quantizer"] == quantizer].copy()
        event_lookup = {
            (int(row.trajectory_key), int(row.token_id)): row
            for row in q_events.itertuples(index=False)
        }
        for token_id in token_ids:
            observed = q_events[q_events["token_id"] == token_id].copy()
            total_events = int(observed["event_count"].sum()) if len(observed) else 0
            total_anomaly = float(observed["anomaly_sum"].sum()) if len(observed) else 0.0
            total_delta = float(observed["delta_sum"].sum()) if len(observed) else 0.0
            observed["trajectory_mean_anomaly"] = observed["anomaly_sum"] / observed["event_count"]
            observed["trajectory_mean_delta"] = observed["delta_sum"] / observed["event_count"]
            rows = []
            for label in q_labels.itertuples(index=False):
                event = event_lookup.get((int(label.trajectory_key), token_id))
                count = int(event.event_count) if event is not None else 0
                rows.append(
                    {
                        "trajectory_key": int(label.trajectory_key),
                        "dataset": label.dataset,
                        "correctness_group": label.correctness_group,
                        "loop": int(label.loop),
                        "explicit_error_loop": int(label.explicit_error_loop),
                        "output_tokens": int(label.output_tokens),
                        "event_count": count,
                        "rate_per_1k": 1000.0 * count / max(1, int(label.output_tokens)),
                        "trajectory_mean_anomaly": (
                            float(event.anomaly_sum) / count if count else math.nan
                        ),
                    }
                )
            grid = pd.DataFrame(rows)
            error_grid = grid[grid["correctness_group"] == "q_wrong_fp_correct"]
            control_grid = grid[grid["correctness_group"] == "both_correct"]
            error_observed = error_grid[error_grid["event_count"] > 0]
            control_observed = control_grid[control_grid["event_count"] > 0]
            anomaly_shift = (
                error_observed["trajectory_mean_anomaly"].mean()
                - control_observed["trajectory_mean_anomaly"].mean()
            )
            rate_shift = error_grid["rate_per_1k"].mean() - control_grid["rate_per_1k"].mean()
            anomaly_low, anomaly_high = bootstrap_difference(
                error_observed["trajectory_mean_anomaly"].to_numpy(),
                control_observed["trajectory_mean_anomaly"].to_numpy(),
                resamples=resamples,
                seed=seed + quantizer_index * 1_000_000 + token_id,
            )
            rate_low, rate_high = bootstrap_difference(
                error_grid["rate_per_1k"].to_numpy(),
                control_grid["rate_per_1k"].to_numpy(),
                resamples=resamples,
                seed=seed + quantizer_index * 1_000_000 + 200_000 + token_id,
            )
            error_loop = grid[grid["explicit_error_loop"] == 1]
            nonloop = grid[grid["loop"] == 0]
            loop_rate = error_loop["rate_per_1k"].mean()
            nonloop_rate = nonloop["rate_per_1k"].mean()
            if error_loop.empty or nonloop.empty:
                loop_ratio = math.nan
            else:
                loop_ratio = (
                    loop_rate / nonloop_rate
                    if nonloop_rate > 0
                    else (math.inf if loop_rate > 0 else math.nan)
                )
            dataset_shift = []
            for dataset in sorted(q_labels["dataset"].unique()):
                left = error_observed[error_observed["dataset"] == dataset]
                right = control_observed[control_observed["dataset"] == dataset]
                if len(left) and len(right):
                    dataset_shift.append(
                        left["trajectory_mean_anomaly"].mean()
                        - right["trajectory_mean_anomaly"].mean()
                    )
            record = {
                "channel": channel,
                "quantizer": quantizer,
                "token_id": token_id,
                "trajectory_support": len(observed),
                "dataset_support": observed["dataset"].nunique() if len(observed) else 0,
                "events": total_events,
                "event_mean_delta": total_delta / total_events if total_events else math.nan,
                "event_mean_anomaly": total_anomaly / total_events if total_events else math.nan,
                "trajectory_mean_anomaly": observed["trajectory_mean_anomaly"].mean(),
                "direction_match_event_rate": (
                    observed["direction_match_count"].sum() / total_events
                    if total_events else math.nan
                ),
                "max_single_trajectory_fraction": (
                    observed["event_count"].max() / total_events if total_events else math.nan
                ),
                "qwrong_fpcorrect_trajectories": len(error_grid),
                "bothcorrect_trajectories": len(control_grid),
                "qwrong_fpcorrect_observed_support": len(error_observed),
                "bothcorrect_observed_support": len(control_observed),
                "min_correctness_observed_support": min(len(error_observed), len(control_observed)),
                "error_minus_control_anomaly": anomaly_shift,
                "error_minus_control_anomaly_ci_low": anomaly_low,
                "error_minus_control_anomaly_ci_high": anomaly_high,
                "error_minus_control_rate_per_1k": rate_shift,
                "error_minus_control_rate_ci_low": rate_low,
                "error_minus_control_rate_ci_high": rate_high,
                "comparable_correctness_datasets": len(dataset_shift),
                "positive_correctness_dataset_shifts": int(sum(value > 0 for value in dataset_shift)),
                "dataset_macro_error_minus_control_anomaly": (
                    float(np.mean(dataset_shift)) if dataset_shift else math.nan
                ),
                "explicit_error_loop_trajectories": len(error_loop),
                "explicit_error_loop_observed_support": int((error_loop["event_count"] > 0).sum()),
                "error_loop_rate_per_1k": loop_rate,
                "nonloop_rate_per_1k": nonloop_rate,
                "error_loop_to_nonloop_rate_ratio": loop_ratio,
            }
            if channel == "scoring":
                exact_count = int(observed["exact_event_count"].sum()) if len(observed) else 0
                record.update(
                    exact_correction_events=exact_count,
                    mean_exact_correction_magnitude=(
                        observed["exact_magnitude_sum"].sum() / exact_count
                        if exact_count else math.nan
                    ),
                    repeated4_events=int(observed["repeated4_event_count"].sum()) if len(observed) else 0,
                    high_repeat4_events=int(observed["high_repeat4_event_count"].sum()) if len(observed) else 0,
                    high_repeat4_trajectory_support=int((observed["high_repeat4_event_count"] > 0).sum()) if len(observed) else 0,
                    error_loop_high_repeat4_trajectory_support=int(
                        (
                            (observed["explicit_error_loop"] == 1)
                            & (observed["high_repeat4_event_count"] > 0)
                        ).sum()
                    ) if len(observed) else 0,
                    max_local_4gram_multiplicity=int(observed["max_local_4gram_multiplicity"].max()) if len(observed) else 0,
                    max_same_token_id_run=int(observed["max_same_token_id_run"].max()) if len(observed) else 0,
                )
            else:
                record.update(
                    floor_events=int(observed["floor_event_count"].sum()) if len(observed) else 0,
                    floor_event_fraction=(
                        observed["floor_event_count"].sum() / total_events
                        if total_events else math.nan
                    ),
                    late_events=int(observed["late_event_count"].sum()) if len(observed) else 0,
                    late_event_fraction=(
                        observed["late_event_count"].sum() / total_events
                        if total_events else math.nan
                    ),
                    high_repeat_prefix_events=int(observed["high_repeat_prefix_event_count"].sum()) if len(observed) else 0,
                    high_repeat_prefix_trajectory_support=int((observed["high_repeat_prefix_event_count"] > 0).sum()) if len(observed) else 0,
                    error_loop_high_repeat_prefix_trajectory_support=int(
                        (
                            (observed["explicit_error_loop"] == 1)
                            & (observed["high_repeat_prefix_event_count"] > 0)
                        ).sum()
                    ) if len(observed) else 0,
                )
            records.append(record)
    return pd.DataFrame(records)


def cross_quantizer(evidence: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for channel in ("scoring", "preview"):
        part = evidence[evidence["channel"] == channel].drop(columns="channel")
        wide = part.pivot(index="token_id", columns="quantizer")
        wide.columns = [f"{channel}_{metric}_{quantizer}" for metric, quantizer in wide.columns]
        pieces.append(wide.reset_index())
    result = candidates.merge(pieces[0], on="token_id", how="left", validate="one_to_one")
    result = result.merge(pieces[1], on="token_id", how="left", validate="one_to_one")
    for channel in ("scoring", "preview"):
        result[f"{channel}_observed_both_support20"] = (
            (result[f"{channel}_trajectory_support_awq"] >= 20)
            & (result[f"{channel}_trajectory_support_gptq"] >= 20)
        )
        result[f"{channel}_direction_reproduced_both"] = (
            result[f"{channel}_observed_both_support20"]
            & (result[f"{channel}_event_mean_anomaly_awq"] > 0)
            & (result[f"{channel}_event_mean_anomaly_gptq"] > 0)
        )
        result[f"{channel}_positive_error_shift_both"] = (
            (result[f"{channel}_error_minus_control_anomaly_awq"] > 0)
            & (result[f"{channel}_error_minus_control_anomaly_gptq"] > 0)
        )
        result[f"{channel}_loop_rate_enriched_both"] = (
            (result[f"{channel}_error_loop_to_nonloop_rate_ratio_awq"] > 1)
            & (result[f"{channel}_error_loop_to_nonloop_rate_ratio_gptq"] > 1)
        )
    return result

