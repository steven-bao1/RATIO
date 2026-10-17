"""Generic raw-logit penalties; no keyword selection or calibration here."""
import json
import hashlib
import math
import re

# Positive sweep in Appendix A.3 of Lotfi et al., arXiv:2606.00206.
# Reference candidates, not a restriction on user-specified nonnegative strengths.
FIXED_PENALTY_CANDIDATES = (0.5, 0.7, 1.0, 1.5, 2.0, 2.6, 3.0, 4.0)


def load_bias_config(path):
    """Read biases and optional tokenizer binding; legacy plain maps still work."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream, object_pairs_hook=unique)
    if not isinstance(payload, dict) or not isinstance(payload.get("token_biases"), dict):
        raise ValueError('Expected {"token_biases": {"123": -1.0}}')
    biases = {}
    for key, value in payload["token_biases"].items():
        if not re.fullmatch(r"0|[1-9][0-9]*", key):
            raise ValueError(f"Invalid token ID: {key}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Non-finite or non-numeric bias: {key}")
        biases[int(key)] = float(value)
    if not biases:
        raise ValueError("The bias map must not be empty")
    bound = "tokenizer_vocab_sha256" in payload or payload.get("kind") == "ratio-v8-norm-noclip"
    metadata = ({key: payload[key] for key in ("kind", "tokenizer_vocab_sha256", "tokens") if key in payload}
                if bound else {})
    if payload.get("kind") == "ratio-v8-norm-noclip":
        if not isinstance(metadata.get("tokenizer_vocab_sha256"), str) or not isinstance(metadata.get("tokens"), list):
            raise ValueError("Calibrated RATIO maps require tokenizer binding and exact token pieces")
        if any(value > 0 for value in biases.values()):
            raise ValueError("RATIO calibration maps must not boost tokens")
    return biases, metadata or None


def load_bias_map(path):
    """Backward-compatible reader returning only signed token biases."""
    return load_bias_config(path)[0]


def validate_bias_tokenizer(tokenizer, biases, metadata):
    """Check a calibrated map before model weights are loaded, not just ID range."""
    if not metadata:
        return
    expected = metadata.get("tokenizer_vocab_sha256")
    if expected is not None:
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Malformed tokenizer fingerprint in bias map")
        digest = hashlib.sha256()
        for token, key in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
            digest.update(int(key).to_bytes(4, "little", signed=False))
            digest.update(token.encode("utf-8"))
            digest.update(b"\0")
        if digest.hexdigest() != expected:
            raise ValueError("Bias map was calibrated for a different tokenizer vocabulary")
    if "tokens" in metadata:
        tokens = metadata["tokens"]
        if not isinstance(tokens, list):
            raise ValueError("Bias-map tokens must be a list")
        seen = set()
        for row in tokens:
            key = row.get("token_id") if isinstance(row, dict) else None
            if type(key) is not int or key in seen or key not in biases:
                raise ValueError("Invalid or duplicate bias-map token metadata")
            seen.add(key)
            if row.get("raw_logit_bias") != biases[key]:
                raise ValueError("Token metadata and applied bias disagree")
            if tokenizer.decode([key], skip_special_tokens=False, clean_up_tokenization_spaces=False) != row.get("decoded_piece"):
                raise ValueError(f"Bias-map token piece differs from tokenizer: {key}")
        if seen != set(biases):
            raise ValueError("Bias-map token metadata is incomplete")


class TokenBias:
    """Biases are signed: a negative value suppresses the corresponding token."""
    def __init__(self, biases):
        self.biases = dict(biases)
        self._cache = None

    def __call__(self, prompt_token_ids, token_ids, logits=None):
        import torch
        if logits is None:  # vLLM's two-argument callback
            logits = token_ids
        if logits.ndim != 1 or any(k < 0 or k >= logits.shape[0] for k in self.biases):
            raise ValueError("Invalid logit shape or token ID outside vocabulary")
        key = (logits.device, logits.dtype)
        if self._cache is None or self._cache[0] != key:
            self._cache = (key, torch.tensor(list(self.biases), device=logits.device),
                           torch.tensor(list(self.biases.values()), device=logits.device, dtype=logits.dtype))
        logits[self._cache[1]] += self._cache[2]
        return logits


class FixedMarkerPenalty:
    """Penalize marker completions, including multi-token marker suffixes."""
    def __init__(self, tokenizer, magnitude):
        self.magnitude = magnitude
        self.sequences = set(tuple(tokenizer.encode(s, add_special_tokens=False)) for s in FIXED_MARKERS)
        self.sequences.discard(())

    def __call__(self, prompt_token_ids, token_ids, logits=None):
        if logits is None:
            logits, token_ids = token_ids, prompt_token_ids
        targets = {s[-1] for s in self.sequences
                   if len(s) == 1 or tuple(token_ids[-(len(s)-1):]) == s[:-1]}
        if targets:
            logits[list(targets)] -= self.magnitude
        return logits


def build_processor(tokenizer, fixed_penalty=0.0, token_biases=None, token_bias_metadata=None):
    if not math.isfinite(fixed_penalty) or fixed_penalty < 0:
        raise ValueError("fixed_penalty must be finite and non-negative")
    if fixed_penalty and token_biases is not None:
        raise ValueError("Choose fixed penalty OR a token bias map")
    if token_biases is not None:
        validate_bias_tokenizer(tokenizer, token_biases, token_bias_metadata)
        vocabulary = set(tokenizer.get_vocab().values())
        if not set(token_biases).issubset(vocabulary):
            raise ValueError("Bias map contains IDs absent from this tokenizer")
        return TokenBias(token_biases)
    return FixedMarkerPenalty(tokenizer, fixed_penalty) if fixed_penalty else None


# Table 8 of Lotfi et al., arXiv:2606.00206 (Meta); leading spaces are intentional.
FIXED_MARKERS = (' perhaps', ' maybe', ' wait', ' Wait', ' actually', ' hold', ' Hmm', ' hmm', ' Alternatively', ' alternatively', ' However', ' however', ' instead', ' Instead', ' But', ' but', ' though', ' although', ' yet', ' rather', ' unless', ' otherwise', ' nonetheless', ' nevertheless', ' regardless', ' still', ' anyway', ' Or', ' or', ' either', ' whether', ' uncertain', ' unsure', ' possibly', ' might', ' could', ' another', ' different', ' reconsider', ' rethink', ' backtrack', ' retry', ' recheck', ' revisit', ' doubt', ' confused', ' wrong', ' mistake', ' error', ' incorrect')
