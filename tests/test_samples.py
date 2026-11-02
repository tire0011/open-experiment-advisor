import json

import pytest

from experiment_advisor.samples import compare_samples


def sample(doc_id, response, score, **changes):
    row = {
        "doc_id": doc_id,
        "filter": "none",
        "doc_hash": f"document-{doc_id}",
        "prompt_hash": f"prompt-{doc_id}",
        "target_hash": f"target-{doc_id}",
        "filtered_resps": [response],
        "f1": score,
        "doc": {"private": "do not export me"},
    }
    row.update(changes)
    return row


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(x) for x in records) + "\n", encoding="utf-8")


def test_score_changes_without_exporting_raw_answers(tmp_path):
    before, after = tmp_path / "before.jsonl", tmp_path / "after.jsonl"
    write_jsonl(before, [sample(1, "correct answer", 1.0), sample(2, "answer", 0.0)])
    write_jsonl(after, [sample(1, "Explanation:\nincorrect", 0.0), sample(2, "answer", 0.0)])
    report = compare_samples(before, after)
    assert report["comparable_samples"] == 2
    assert report["changed_responses"] == 1
    assert report["score_decreases"] == 1
    assert report["multiline_after"] == 1
    serialized = json.dumps(report)
    assert "correct answer" not in serialized
    assert "do not export me" not in serialized


def test_hash_mismatch_is_not_attributed_to_model(tmp_path):
    before, after = tmp_path / "before.jsonl", tmp_path / "after.jsonl"
    write_jsonl(before, [sample(1, "ok", 1.0)])
    write_jsonl(after, [sample(1, "bad", 0.0, prompt_hash="changed")])
    report = compare_samples(before, after)
    assert report["hash_mismatches"] == 1
    assert report["comparable_samples"] == 0
    assert report["score_decreases"] == 0


def test_missing_ids_and_duplicate_ids(tmp_path):
    before, after = tmp_path / "before.jsonl", tmp_path / "after.jsonl"
    write_jsonl(before, [sample(1, "a", 1.0), sample(2, "b", 0.0)])
    write_jsonl(after, [sample(1, "c", 0.0), sample(3, "d", 1.0)])
    report = compare_samples(before, after)
    assert report["missing_from_after"] == 1
    assert report["new_in_after"] == 1
    write_jsonl(after, [sample(1, "c", 0.0), sample(1, "d", 1.0)])
    with pytest.raises(ValueError, match="duplicate"):
        compare_samples(before, after)


def test_missing_hashes_are_not_considered_comparable(tmp_path):
    before, after = tmp_path / "before.jsonl", tmp_path / "after.jsonl"
    write_jsonl(before, [sample(1, "a", 1.0, prompt_hash=None)])
    write_jsonl(after, [sample(1, "b", 0.0, prompt_hash=None)])
    report = compare_samples(before, after)
    assert report["hashes_unavailable_or_response_missing"] == 1
    assert report["comparable_samples"] == 0


def test_json_array_supported(tmp_path):
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    before.write_text(json.dumps([sample(1, "a", 1.0)]), encoding="utf-8")
    after.write_text(json.dumps([sample(1, "a", 1.0)]), encoding="utf-8")
    assert compare_samples(before, after)["comparable_samples"] == 1
