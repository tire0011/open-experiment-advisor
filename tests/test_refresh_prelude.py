import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.refresh_prelude import API, FILENAME, fetch_pinned_scores, refresh


SHA = "a" * 40
PARQUET = b"PAR1" + b"data" * 10 + b"PAR1"


def fake_json(url):
    if url == API:
        return {"sha": SHA, "lastModified": "2026-11-08"}
    if url == f"{API}/tree/{SHA}?recursive=true":
        return [{"type": "file", "path": FILENAME}]
    raise AssertionError(url)


class PublishedPreludeTests(unittest.TestCase):
    def test_fetch_pins_dataset_and_records_digest(self):
        seen = []

        def download(url):
            seen.append(url)
            return PARQUET

        blob, manifest = fetch_pinned_scores(get_json=fake_json, get_bytes=download)
        self.assertEqual(blob, PARQUET)
        self.assertEqual(manifest["dataset_sha"], SHA)
        self.assertEqual(manifest["sha256"], hashlib.sha256(PARQUET).hexdigest())
        self.assertEqual(seen, [manifest["source_url"]])
        self.assertIn(f"/resolve/{SHA}/scores.parquet", manifest["source_url"])

    def test_bad_parquet_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "magic bytes"):
            fetch_pinned_scores(get_json=fake_json, get_bytes=lambda _: b"not-a-parquet-file")

    def test_revision_drift_is_rejected(self):
        calls = 0

        def changing_json(url):
            nonlocal calls
            if url == API:
                calls += 1
                return {"sha": SHA if calls == 1 else "b" * 40}
            return [{"type": "file", "path": FILENAME}]

        with self.assertRaisesRegex(ValueError, "changed during download"):
            fetch_pinned_scores(get_json=changing_json, get_bytes=lambda _: PARQUET)

    def test_refresh_reuses_analyzer_and_publishes_pinned_sources(self):
        manifest = {
            "source_url": f"https://huggingface.co/datasets/example/resolve/{SHA}/scores.parquet",
            "dataset_sha": SHA, "sha256": hashlib.sha256(PARQUET).hexdigest(),
            "bytes": len(PARQUET),
        }
        calls = []

        def fake_run(args, *, check):
            calls.append(args)
            self.assertTrue(check)
            self.assertEqual(Path(args[1]).read_bytes(), PARQUET)
            json_target = Path(args[args.index("--output") + 1])
            markdown_target = Path(args[args.index("--markdown") + 1])
            json_target.write_text(json.dumps({
                "source": args[1], "matched_control_divergences": [],
            }))
            markdown_target.write_text(
                "# Investigation\n\nSource: " + chr(96) + args[1] + chr(96) + "\n"
            )

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            refreshed = refresh(target, fetch=lambda: (PARQUET, manifest), run=fake_run)
            self.assertEqual(refreshed["dataset_sha"], SHA)
            self.assertEqual(calls[0][0], "experiment-advisor")
            report = json.loads((target / "latest.json").read_text())
            self.assertEqual(report["source"], manifest["source_url"])
            self.assertEqual(report["source_provenance"], manifest)
            markdown = (target / "COQA_INVESTIGATION.md").read_text()
            self.assertIn(manifest["sha256"], markdown)
            self.assertIn(manifest["source_url"], markdown)
            self.assertEqual(
                json.loads((target / "SOURCE_PROVENANCE.json").read_text()), manifest
            )


if __name__ == "__main__":
    unittest.main()
