"""Validate user-owned decisions against immutable selection artifacts."""
import csv
import json
from pathlib import Path
import re

from keyword_selection.inputs import Provenance

EDITABLE = "review/user_decisions.csv"
ELIGIBLE = {"preliminary", "context_pending"}


def unique_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs)


def csv_rows(path):
    # utf-8-sig supports spreadsheet exports without changing leading token spaces.
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"Missing or duplicate CSV headers: {path}")
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row: {path}")
    return rows, set(reader.fieldnames)


def token_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise ValueError(f"Expected canonical nonnegative integer token ID, got {value!r}")
    return int(value)


def validate_decisions(candidates, submitted):
    all_ids = [token_id(row["token_id"]) for row in candidates]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate token IDs in selection artifacts")
    expected = {token_id(row["token_id"]): row for row in candidates if row["status"] in ELIGIBLE}
    if not expected:
        raise ValueError("Selection has no eligible candidates; no penalty map can be produced")
    result = []
    seen = set()
    for row in submitted:
        key = token_id(row["token_id"])
        if key in seen:
            raise ValueError(f"Duplicate submitted token ID: {key}")
        seen.add(key)
        if key not in expected:
            raise ValueError(f"Token {key} was not eligible for user review; re-run selection to change eligibility")
        original = expected[key]
        if original["shared_direction"] != "positive":
            raise ValueError("Negative-direction tokens cannot be penalized by this calibration route")
        for field in ("decoded_piece", "route", "status"):
            if field == "route" and field not in row:
                continue
            if row[field] != original[field]:
                raise ValueError(f"Immutable candidate field changed for token {key}: {field}")
        value = row["decision"].strip().lower()
        meanings = {"": "retain", "no": "exclude", "0": "exclude",
                    "retain": "retain", "exclude": "exclude"}
        if value not in meanings:
            raise ValueError(f"Unresolved decision for token {key}: leave blank to retain, or use no/0 to exclude")
        decision = meanings[value]
        result.append({"token_id": key, "decoded_piece": original["decoded_piece"],
                       "route": original["route"], "selection_status": original["status"],
                       "decision": decision})
    if seen != set(expected):
        raise ValueError(f"Missing decisions for token IDs: {sorted(set(expected) - seen)}; do not delete CSV rows")
    if not any(row["decision"] == "retain" for row in result):
        raise ValueError("All candidates excluded; use base inference without a bias map")
    return sorted(result, key=lambda row: row["token_id"])


def load_selection(directory, decision_file=None):
    """Hash all frozen outputs, except the deliberately editable decision CSV."""
    base = Path(directory).resolve()
    audit = Provenance()
    manifest_path = audit.add(base / "manifest.json")
    manifest = unique_json(manifest_path)
    status = unique_json(audit.add(base / "status.json"))
    if (manifest.get("protocol") not in ("public-lexical-q25-r1-ce-v1", "public-lexical-q25-r1-ce-v2", "public-lexical-q25-r1-ce-v3")
            or manifest.get("state") != "complete_pending_user_review"
            or status.get("state") != "complete_pending_user_review"):
        raise ValueError("Expected a completed public keyword selection, not a partial/failed run")
    outputs = manifest["output_sha256"]
    if manifest["protocol"] == "public-lexical-q25-r1-ce-v3" and "review/post_qsti_exclusions.json" not in outputs:
        raise ValueError("Selection manifest lacks the frozen post-QSTI policy")
    if not {"inputs.json", "policy.json", "all_candidate_decisions.csv", EDITABLE}.issubset(outputs):
        raise ValueError("Selection manifest lacks required frozen artifacts")
    for relative, digest in outputs.items():
        path = (base / relative).resolve()
        if not path.is_relative_to(base):
            raise ValueError("Selection artifact path escapes its directory")
        if relative == EDITABLE:
            continue  # User edits here are expected; identity is checked separately.
        audit.add(path)
        if audit.files[str(path)] != digest:
            raise ValueError(f"Frozen selection artifact changed: {relative}")
    path = audit.add(decision_file or base / EDITABLE)
    submitted, fields = csv_rows(path)
    required = {"token_id", "decoded_piece", "status", "decision"}
    if fields not in (required, required | {"route"}):
        raise ValueError("Decision CSV requires exactly token_id, decoded_piece, status, decision (optional: route)")
    candidates, _ = csv_rows(base / "all_candidate_decisions.csv")
    decisions = validate_decisions(candidates, submitted)
    return {"directory": base, "manifest": manifest, "audit": audit, "decision_file": path,
            "decisions": decisions, "inputs": unique_json(base / "inputs.json"),
            "policy": unique_json(base / "policy.json")}
