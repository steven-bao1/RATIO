"""Lexical repetition metrics (not model-token metrics)."""

import math
import re
from collections import Counter


def lexical_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?|[^\w\s]", text.lower())


def repeated_ngram_fraction(tokens: list[str], n: int = 4) -> float:
    total = len(tokens) - n + 1
    if total <= 0:
        return math.nan
    counts = Counter(tuple(tokens[i : i + n]) for i in range(total))
    return (total - len(counts)) / total


def duplicate_line_fraction(text: str) -> float:
    lines = [
        re.sub(r"\s+", " ", line).strip().lower()
        for line in text.splitlines()
    ]
    lines = [line for line in lines if len(line) >= 24]
    if not lines:
        return 0.0
    return (len(lines) - len(set(lines))) / len(lines)


def max_same_token_run(tokens: list[str]) -> int:
    best = current = 0
    previous = None
    for token in tokens:
        if token == previous:
            current += 1
        else:
            previous, current = token, 1
        best = max(best, current)
    return best
