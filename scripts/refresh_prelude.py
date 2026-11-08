"""Refresh public Prelude anomaly evidence from a pinned Hugging Face Parquet file.

Runs entirely inside GitHub Actions, without tokens, model inference or a GPU.
It delegates anomaly detection to the existing experiment-advisor CLI.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path


DATASET = "openeurollm/prelude-base-eval-scores"
API = f"https://huggingface.co/api/datasets/{DATASET}"
FILENAME = "scores.parquet"
MAX_BYTES = 16 * 1024 * 1024
USER_AGENT = "open-experiment-advisor-published-probe/1.0"


def json_response(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def bytes_response(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read(MAX_BYTES + 1)


def fetch_pinned_scores(*, get_json=json_response, get_bytes=bytes_response) -> tuple[bytes, dict]:
    """Verify a stable dataset SHA and fetch exactly the published scores file."""
    before = get_json(API)
    sha = before.get("sha")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Dataset did not return a valid commit SHA")
    tree = get_json(f"{API}/tree/{sha}?recursive=true")
    if not isinstance(tree, list) or not any(
        item.get("type") == "file" and item.get("path") == FILENAME for item in tree
    ):
        raise ValueError(f"Published {FILENAME} not present at dataset revision")
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{sha}/{FILENAME}"
    data = get_bytes(url)
    if len(data) > MAX_BYTES or len(data) < 8:
        raise ValueError(f"Parquet file size outside expected bound: {len(data)} bytes")
    if not (data.startswith(b"PAR1") and data.endswith(b"PAR1")):
        raise ValueError("Downloaded content does not have Parquet magic bytes")
    after = get_json(API)
    if after.get("sha") != sha:
        raise ValueError("Dataset changed during download; refusing mixed revisions")
    return data, {
        "dataset": DATASET, "dataset_sha": sha,
        "last_modified": after.get("lastModified"),
        "source_url": url, "file": FILENAME,
        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        "collected_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def refresh(output: Path, *, fetch=fetch_pinned_scores, run=subprocess.run) -> dict:
    """Use the existing analyzer, then make its published outputs source-pinned."""
    data, manifest = fetch()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prelude-") as directory:
        source = Path(directory) / FILENAME
        source.write_bytes(data)
        json_path = output / "latest.json"
        md_path = output / "COQA_INVESTIGATION.md"
        run([
            "experiment-advisor", str(source), "--output", str(json_path),
            "--markdown", str(md_path), "--focus-task", "coqa_gen",
        ], check=True)
        report = json.loads(json_path.read_text(encoding="utf-8"))
        report["source"] = manifest["source_url"]
        report["source_provenance"] = manifest
        json_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        md = md_path.read_text(encoding="utf-8")
        old_source = f"Source: \u0060{source}\u0060"
        new_source = (
            f"Source: [published scores.parquet]({manifest['source_url']}) "
            f"(dataset commit {manifest['dataset_sha']}, "
            f"file SHA256 {manifest['sha256']})"
        )
        if old_source not in md:
            raise ValueError("Unexpected analyzer Markdown format; provenance not applied")
        md_path.write_text(md.replace(old_source, new_source, 1), encoding="utf-8")
        (output / "SOURCE_PROVENANCE.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    manifest = refresh(args.output)
    print(f"Published Prelude investigation: {manifest['dataset_sha']} / {manifest['sha256']}")


if __name__ == "__main__":
    main()
