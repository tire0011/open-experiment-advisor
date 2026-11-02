"""Compare two lm-eval sample log files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .samples import compare_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two lm-eval sample logs by document ID")
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--metric", default="f1")
    parser.add_argument("--output", type=Path, help="Write privacy-preserving JSON report")
    args = parser.parse_args()
    try:
        report = compare_samples(args.before, args.after, metric=args.metric)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Comparable samples: {report['comparable_samples']}; "
          f"changed answers: {report['changed_responses']}; "
          f"score drops: {report['score_decreases']}; "
          f"hash mismatches: {report['hash_mismatches']}")


if __name__ == "__main__":
    main()
