"""Conservative, whole-piece exclusions after the QSTI candidate union."""
import json
from pathlib import Path
import re


class SemanticFilter:
    def __init__(self, path):
        self.config = json.loads(Path(path).read_text())
        cfg = self.config
        if (cfg.get("schema_version") != 1
                or cfg.get("normalization") != "decoded_piece.strip().casefold()"
                or cfg.get("match") != "whole normalized piece, no substring or stemming"):
            raise ValueError("Unsupported post-QSTI exclusion policy")
        self.families = {}
        for category, words in cfg["categories"].items():
            if not isinstance(category, str) or not category or not isinstance(words, list):
                raise ValueError("Invalid exclusion category")
            for word in words:
                if not isinstance(word, str) or not word or word != word.strip().casefold() or word in self.families:
                    raise ValueError("Exclusion families must be unique normalized whole pieces")
                self.families[word] = category
        if len(self.families) != cfg["family_count"]:
            raise ValueError("Exclusion family count mismatch")
        protected = cfg["protected_families"]
        if (not isinstance(protected, list)
                or any(not isinstance(w, str) or not w or w != w.strip().casefold() for w in protected)
                or len(set(protected)) != len(protected)):
            raise ValueError("Invalid protected families")
        self.protected = set(protected)
        if self.protected.intersection(self.families):
            raise ValueError("A protected family cannot also be excluded")
        exact = cfg.get("exact_piece_exclusions", [])
        if (not isinstance(exact, list)
                or any(not isinstance(w, str) or not w.strip() for w in exact)
                or len(set(exact)) != len(exact)):
            raise ValueError("Exact exclusions must be unique nonempty decoded pieces")
        self.exact_pieces = set(exact)
        if cfg["rules"] != {"single_ascii_letter": True, "uppercase_variable_pattern": "[A-Z]{2,4}"}:
            raise ValueError("Unsupported token-shape exclusion rules")

    def reason(self, piece):
        if piece in self.exact_pieces:
            return "explicit_piece_exclusion"
        text = piece.strip()
        family = text.casefold()
        if family in self.protected:
            return ""
        if re.fullmatch(r"[A-Za-z]", text):
            return "single_letter"
        if family in self.families:
            return self.families[family]
        if re.fullmatch(r"[A-Z]{2,4}", text):
            return "uppercase_variable_or_abbreviation"
        return ""

    def annotate(self, candidates):
        result = candidates.copy()
        result["post_qsti_exclusion_reason"] = result.decoded_piece.map(self.reason)
        result["post_qsti_eligible"] = result.post_qsti_exclusion_reason.eq("")
        return result
