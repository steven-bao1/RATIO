"""Direction-specific stability screens; concentration is diagnostic only."""
import re
import numpy as np
import pandas as pd
CLEAN_ENGLISH = re.compile(r"^[A-Za-z]+(?: [A-Za-z]+)*$")

def build_common(awq: pd.DataFrame, gptq: pd.DataFrame, threshold: int) -> pd.DataFrame:
    joined = awq.merge(gptq, on="token_id", how="inner", suffixes=("_awq", "_gptq"))
    same_piece = joined["decoded_piece_awq"] == joined["decoded_piece_gptq"]
    stripped = joined["decoded_piece_awq"].fillna("").str.strip()
    clean = stripped.str.fullmatch(CLEAN_ENGLISH)
    keep = (
        (joined["events_awq"] > threshold)
        & (joined["events_gptq"] > threshold)
        & same_piece
        & clean
        & ~joined["is_special_token_awq"].astype(bool)
        & ~joined["is_special_token_gptq"].astype(bool)
        & ~joined["is_whitespace_piece_awq"].astype(bool)
        & ~joined["is_whitespace_piece_gptq"].astype(bool)
    )
    joined = joined.loc[keep].copy()

    result = pd.DataFrame()
    result["decoded_piece"] = joined["decoded_piece_awq"]
    result["clean_text"] = joined["decoded_piece_awq"].str.strip()
    result["normalized_lower"] = result["clean_text"].str.lower()
    result["token_id"] = joined["token_id"].astype("int64")
    result["leading_space"] = joined["decoded_piece_awq"].str.startswith(" ")
    for field in ("events", "trajectory_support"):
        result[f"{field}_awq"] = joined[f"{field}_awq"].to_numpy()
        result[f"{field}_gptq"] = joined[f"{field}_gptq"].to_numpy()
    result["min_events"] = result[["events_awq", "events_gptq"]].min(axis=1)
    result["total_events"] = result["events_awq"] + result["events_gptq"]

    # Match the original column order: event summaries precede model scores.
    prefix = result[[
        "decoded_piece", "clean_text", "normalized_lower", "token_id", "leading_space",
        "events_awq", "events_gptq", "min_events", "total_events",
        "trajectory_support_awq", "trajectory_support_gptq",
    ]].copy()
    result = prefix
    for quantizer in ("awq", "gptq"):
        for field in (
            "mean_logp_q_used",
            "mean_logp_bf16_used",
            "mean_direct_delta_logp",
        ):
            result[f"{field}_{quantizer}"] = joined[f"{field}_{quantizer}"].to_numpy()

    awq_delta = result["mean_direct_delta_logp_awq"]
    gptq_delta = result["mean_direct_delta_logp_gptq"]
    result["mean_direct_delta_across_quantizers"] = (awq_delta + gptq_delta) / 2.0
    result["minimum_abs_mean_direct_delta"] = np.minimum(
        awq_delta.abs(), gptq_delta.abs()
    )
    result["same_delta_sign"] = (
        ((awq_delta > 0.0) & (gptq_delta > 0.0))
        | ((awq_delta < 0.0) & (gptq_delta < 0.0))
    )
    result["both_quantizers_over_bf16"] = (awq_delta > 0.0) & (gptq_delta > 0.0)

    for field in (
        "positive_rate_direct_delta",
        "std_direct_delta_logp",
        "both_rate",
        "q_only_rate",
        "bf16_only_rate",
        "q_imputed_rate",
        "bf16_imputed_rate",
        "max_single_trajectory_fraction",
        "source_task_support",
        "min_source_task_mean_direct_delta_logp",
        "max_source_task_mean_direct_delta_logp",
    ):
        for quantizer in ("awq", "gptq"):
            result[f"{field}_{quantizer}"] = joined[f"{field}_{quantizer}"].to_numpy()

    return result.sort_values(
        ["min_events", "total_events"], ascending=[False, False], kind="stable"
    ).reset_index(drop=True)


def add_shared_direction(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[frame["same_delta_sign"]].copy()
    insert_at = result.columns.get_loc("same_delta_sign") + 1
    result.insert(
        insert_at,
        "shared_direction",
        np.where(result["mean_direct_delta_logp_awq"] > 0.0, "positive", "negative"),
    )
    return result.reset_index(drop=True)


def add_min_support(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    insert_at = result.columns.get_loc("trajectory_support_gptq") + 1
    result.insert(
        insert_at,
        "min_trajectory_support",
        result[["trajectory_support_awq", "trajectory_support_gptq"]].min(axis=1),
    )
    return result


def add_stability_metrics(
    frame: pd.DataFrame, awq: pd.DataFrame, gptq: pd.DataFrame
) -> pd.DataFrame:
    result = frame.copy()
    negatives = (
        awq[["token_id", "negative_rate_direct_delta"]]
        .rename(columns={"negative_rate_direct_delta": "negative_rate_direct_delta_awq"})
        .merge(
            gptq[["token_id", "negative_rate_direct_delta"]].rename(
                columns={"negative_rate_direct_delta": "negative_rate_direct_delta_gptq"}
            ),
            on="token_id",
            how="inner",
            validate="one_to_one",
        )
    )
    result = result.merge(negatives, on="token_id", how="left", validate="one_to_one")
    result["robust_abs_delta"] = result[[
        "mean_direct_delta_logp_awq", "mean_direct_delta_logp_gptq"
    ]].abs().min(axis=1)
    positive = result["shared_direction"] == "positive"
    result["min_direction_event_rate"] = np.where(
        positive,
        result[["positive_rate_direct_delta_awq", "positive_rate_direct_delta_gptq"]].min(axis=1),
        result[["negative_rate_direct_delta_awq", "negative_rate_direct_delta_gptq"]].min(axis=1),
    )
    result["min_both_nucleus_rate"] = result[["both_rate_awq", "both_rate_gptq"]].min(axis=1)
    result["max_single_trajectory_concentration"] = result[[
        "max_single_trajectory_fraction_awq", "max_single_trajectory_fraction_gptq"
    ]].max(axis=1)
    result["conservative_signed_delta"] = np.where(
        positive, result["robust_abs_delta"], -result["robust_abs_delta"]
    )
    return result


def quantile_select(
    frame: pd.DataFrame, *, use_concentration_gate: bool = False,
    use_both_nucleus_gate: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    thresholds = {
        "robust_abs_delta_q25": float(frame["robust_abs_delta"].quantile(0.25)),
        "min_direction_event_rate_q25": float(frame["min_direction_event_rate"].quantile(0.25)),
        "min_both_nucleus_rate_q25": float(frame["min_both_nucleus_rate"].quantile(0.25)),
        "max_concentration_q75": float(frame["max_single_trajectory_concentration"].quantile(0.75)),
    }
    keep = (
        (frame["robust_abs_delta"] >= thresholds["robust_abs_delta_q25"])
        & (frame["min_direction_event_rate"] >= thresholds["min_direction_event_rate_q25"])
    )
    if use_both_nucleus_gate:
        keep &= (
            frame["min_both_nucleus_rate"]
            >= thresholds["min_both_nucleus_rate_q25"]
        )
    if use_concentration_gate:
        keep &= (
            frame["max_single_trajectory_concentration"]
            <= thresholds["max_concentration_q75"]
        )
    selected = frame.loc[keep].sort_values(
        "robust_abs_delta", ascending=False, kind="stable"
    ).reset_index(drop=True)
    return selected, thresholds

