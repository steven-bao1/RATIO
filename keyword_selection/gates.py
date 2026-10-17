"""Frozen core/R1/CE gates and deterministic public lexical policies."""
import math
from pathlib import Path
import pandas as pd

from keyword_selection.inputs import read_json

DEFAULTS = {
    "tail_probability": 0.001,
    "events_threshold_exclusive": 100,
    "trajectory_threshold_exclusive": 105,
    "strict_support": 20,
    "r1_support": 10,
    "ce_support": 5,
    "strict_quantile": 0.25,
    "bootstrap_resamples": 2000,
    "seed": 42,
    "context_window_tokens": 64,
    "context_trajectories_per_stratum": 2,
    "contexts_per_trajectory": 2,
}
REMINDER = (
    "Before calibration, review all candidates, check contexts for context-pending tokens, "
    "and exclude unsuitable overthinking markers; the fixed exclusion list is not exhaustive."
)


def policy(filename=None):
    values = DEFAULTS.copy()
    if filename:
        overrides = read_json(filename)
        if set(overrides) - set(values):
            raise ValueError(f"Unknown policy keys: {set(overrides) - set(values)}")
        values.update(overrides)
    for key, value in values.items():
        if isinstance(DEFAULTS[key], int):
            minimum = 0 if key in ("seed", "events_threshold_exclusive", "trajectory_threshold_exclusive") else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"Invalid integer policy value: {key}")
        elif not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"Invalid fractional policy value: {key}")
    if not values["strict_support"] >= values["r1_support"] >= values["ce_support"]:
        raise ValueError("Support gates must descend from strict to R1 to CE")
    return values


def lexical_families(path):
    obj = read_json(path)
    words = obj["families"]
    if obj.get("normalization") != "decoded_piece.strip().casefold()":
        raise ValueError("Unsupported lexical normalization")
    if len(set(words)) != len(words) or len(words) != obj["family_count"]:
        raise ValueError("Invalid lexical family count or duplicate entry")
    if any(not isinstance(w, str) or not w or w != w.strip().casefold() for w in words):
        raise ValueError("Lexical families must be normalized whole pieces")
    return set(words)


def select_candidates(evidence, exclusions, settings):
    """No LLM, target word count, downstream accuracy, or table10-only trigger."""
    e = evidence.copy()
    if e.token_id.duplicated().any():
        raise ValueError("Duplicate candidate token IDs")
    positive = e.shared_direction.eq("positive")
    sa, sg = e.scoring_trajectory_support_awq, e.scoring_trajectory_support_gptq
    aa, ag = e.scoring_event_mean_anomaly_awq, e.scoring_event_mean_anomaly_gptq
    reference = positive & sa.ge(settings["strict_support"]) & sg.ge(settings["strict_support"]) & aa.gt(0) & ag.gt(0)
    taus = {"awq": float(aa[reference].quantile(settings["strict_quantile"])),
            "gptq": float(ag[reference].quantile(settings["strict_quantile"]))}
    strict = reference & aa.ge(taus["awq"]) & ag.ge(taus["gptq"])
    def both(metric, threshold):
        return e[f"{metric}_awq"].gt(threshold) & e[f"{metric}_gptq"].gt(threshold)
    # Recompute from audited measurements, never coerce CSV strings/NaNs to bool.
    trajectory = both("scoring_trajectory_mean_anomaly", 0)
    preview = (e.preview_trajectory_support_awq.ge(20) & e.preview_trajectory_support_gptq.ge(20)
               & both("preview_event_mean_anomaly", 0))
    correct = both("scoring_error_minus_control_anomaly", 0) | both("preview_error_minus_control_anomaly", 0)
    loop = (both("scoring_error_loop_to_nonloop_rate_ratio", 1)
            | both("preview_error_loop_to_nonloop_rate_ratio", 1)
            | both("scoring_high_repeat4_trajectory_support", 0))
    count = trajectory.astype(int) + preview.astype(int) + correct.astype(int) + loop.astype(int)
    r1 = (positive & sa.ge(settings["r1_support"]) & sg.ge(settings["r1_support"])
          & (aa.gt(0) | ag.gt(0)) & count.ge(1) & ~strict)
    trigger = r1 & count.eq(1) & (trajectory | preview) & ~correct & ~loop
    ce = positive & sa.ge(settings["ce_support"]) & sg.ge(settings["ce_support"]) & count.ge(2) & ~(strict | r1)
    e["strict_q25_reference"] = reference
    e["strict_numeric"] = strict
    e["r1_numeric"] = r1
    e["r1_context_trigger"] = trigger
    e["ce_primary"] = ce & (correct | loop)
    e["ce_rescue"] = ce & ~correct & ~loop
    e["e_trajectory"], e["e_preview"], e["e_correct"], e["e_loop_repeat"] = trajectory, preview, correct, loop
    e["n_evidence_categories"] = count
    e["normalized_family"] = e.decoded_piece.map(lambda text: text.strip().casefold())
    lexical = e.normalized_family.isin(exclusions)
    post_qsti = (e.post_qsti_exclusion_reason.fillna("").ne("") if "post_qsti_exclusion_reason" in e
                 else pd.Series(False, index=e.index))
    excluded = lexical | post_qsti
    e["basic_semantic_status"] = excluded.map({True: "explicit_lexical_exclusion", False: "not_excluded_still_requires_final_review"})
    e.loc[post_qsti, "basic_semantic_status"] = "post_qsti_semantic_exclusion"
    e["route"] = "none"
    e.loc[strict, "route"] = "core"
    e.loc[r1, "route"] = "r1"
    e.loc[trigger, "route"] = "r1_context"
    e.loc[e.ce_primary, "route"] = "ce_primary"
    e.loc[e.ce_rescue, "route"] = "ce_rescue"
    e["status"] = "numeric_not_selected"
    e.loc[(strict | r1) & ~trigger & ~excluded, "status"] = "preliminary"
    e.loc[(trigger | ce) & ~excluded, "status"] = "context_pending"
    e.loc[excluded, "status"] = "lexically_excluded"
    e.loc[~positive, "status"] = "negative_watchlist"
    e["review_reason"] = ""
    e.loc[e.status.eq("preliminary"), "review_reason"] = "Passes numeric gate and not in exclusion list; final basic-semantic confirmation required"
    e.loc[trigger & ~excluded, "review_reason"] = "R1 has exactly one evidence category: trajectory or preview only"
    e.loc[e.ce_primary & ~excluded, "review_reason"] = "CE: at least two categories including correctness or loop/repeat; contextual review required"
    e.loc[e.ce_rescue & ~excluded, "review_reason"] = "CE rescue: trajectory + preview only; direct contextual evidence is required, not automatic acceptance"
    e.loc[excluded, "review_reason"] = "Whole normalized piece matched the frozen explicit exclusion list"
    if "post_qsti_exclusion_reason" in e:
        e.loc[post_qsti, "review_reason"] = "Post-QSTI exclusion: " + e.loc[post_qsti, "post_qsti_exclusion_reason"]
    e["user_decision"] = "pending"
    thresholds = {"reference_count": int(reference.sum()),
                  "quantile": settings["strict_quantile"],
                  "awq": taus["awq"] if math.isfinite(taus["awq"]) else None,
                  "gptq": taus["gptq"] if math.isfinite(taus["gptq"]) else None,
                  "empty_reference": not bool(reference.any())}
    return e, thresholds
