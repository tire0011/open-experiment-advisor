"""Privacy-preserving comparison of lm-evaluation-harness sample logs."""

from __future__ import annotations

import json
from pathlib import Path


def _read_samples(path: Path) -> list[dict]:
    if path.suffix.lower() not in {".json", ".jsonl"}:
        raise ValueError("sample input must be .json or .jsonl")
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        rows = json.loads(content)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("expected a list of sample objects")
    return rows


def _response(item: dict) -> str | None:
    response = item.get("filtered_resps", item.get("resps"))
    while isinstance(response, list):
        if not response:
            return ""
        response = response[0]
    return response if isinstance(response, str) else None


def _sample_key(item: dict) -> tuple:
    if "doc_id" not in item:
        raise ValueError("sample lacks doc_id")
    return (str(item["doc_id"]), str(item.get("filter", "none")))


def _index(rows: list[dict]) -> dict[tuple, dict]:
    mapping = {}
    for item in rows:
        key = _sample_key(item)
        if key in mapping:
            raise ValueError(f"duplicate sample key: {key}")
        mapping[key] = item
    return mapping


def compare_samples(
    before: str | Path, after: str | Path, *, metric: str = "f1"
) -> dict:
    """Compare same-ID outputs without exporting prompts, answers or raw documents.

    Requires identical doc/prompt/target hashes for matched scoring comparisons.
    A mismatch is counted separately, never silently scored as a model change.
    """
    left, right = _index(_read_samples(Path(before))), _index(_read_samples(Path(after)))
    matched = sorted(left.keys() & right.keys())
    mismatched = 0
    unverified = 0
    changed = 0
    drops = 0
    gains = 0
    comparable = 0
    empty_before = empty_after = 0
    multiline_before = multiline_after = 0
    changed_examples = []

    for key in matched:
        a, b = left[key], right[key]
        hashes = ("doc_hash", "prompt_hash", "target_hash")
        if any(a.get(field) != b.get(field) for field in hashes):
            mismatched += 1
            continue
        if any(a.get(field) is None for field in hashes):
            unverified += 1
            continue
        x, y = _response(a), _response(b)
        if x is None or y is None:
            unverified += 1
            continue
        comparable += 1
        empty_before += not x.strip()
        empty_after += not y.strip()
        multiline_before += "\n" in x.strip()
        multiline_after += "\n" in y.strip()
        if x != y:
            changed += 1
        old, new = a.get(metric), b.get(metric)
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            drops += new < old
            gains += new > old
            if new < old and len(changed_examples) < 20:
                changed_examples.append({
                    "doc_id": key[0], "filter": key[1],
                    "before_score": old, "after_score": new,
                    "before_empty": not x.strip(), "after_empty": not y.strip(),
                    "before_multiline": "\n" in x.strip(),
                    "after_multiline": "\n" in y.strip(),
                })

    return {
        "metric": metric,
        "before_samples": len(left), "after_samples": len(right),
        "matching_ids": len(matched),
        "missing_from_after": len(left.keys() - right.keys()),
        "new_in_after": len(right.keys() - left.keys()),
        "hash_mismatches": mismatched,
        "hashes_unavailable_or_response_missing": unverified,
        "comparable_samples": comparable,
        "changed_responses": changed,
        "score_decreases": drops, "score_increases": gains,
        "empty_before": empty_before, "empty_after": empty_after,
        "multiline_before": multiline_before, "multiline_after": multiline_after,
        "score_drop_examples": changed_examples,
        "limitations": [
            "No raw prompts, generated answers or reference texts are exported.",
            "Empty/multiline are formatting indicators, not proven scoring defects.",
            "Check evaluator configuration and repeatability before inferring a root cause.",
        ],
    }
