"""CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import analyze, write_json
from .report import write_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only checkpoint evaluation anomaly triage")
    parser.add_argument("scores", type=Path, help="Parquet file with checkpoint evaluation scores")
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--minimum-change", type=float, default=0.02)
    parser.add_argument("--output", type=Path, help="Write full JSON report")
    parser.add_argument("--markdown", type=Path, help="Write evidence-first Markdown investigation")
    parser.add_argument("--focus-task", help="Restrict Markdown findings to this exact task name")
    args = parser.parse_args()
    try:
        report = analyze(args.scores, window=args.window, minimum_change=args.minimum_change)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.output:
        write_json(report, args.output)
    if args.markdown:
        write_markdown(report, args.markdown, focus_task=args.focus_task)
    print(f"Comparable series: {report['series_tested']}; flagged: {len(report['candidates'])}")
    for item in report["candidates"][:10]:
        print(f"{item['model']} / {item['task']} / {item['metric']}: "
              f"{item['previous_mean']:.3f} -> {item['latest_mean']:.3f} "
              f"(drop {item['drop']:.3f})")
    print("Top volatile series:")
    for item in report["volatility"][:5]:
        print(f"{item['model']} / {item['task']} / {item['metric']}: "
              f"median adjacent change {item['median_absolute_step_change']:.3f}")
    print("Top matched-control divergences:")
    for item in report["matched_control_divergences"][:5]:
        print(f"{item['model']} / {item['task']} / {item['metric']} "
              f"({item['from_step']} -> {item['to_step']}): "
              f"change {item['change']:+.3f}; "
              f"control median {item['control_median_change']:+.3f} "
              f"(n={item['matched_controls']})")


if __name__ == "__main__":
    main()
