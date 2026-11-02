"""Human-readable, evidence-first investigation notes."""

from __future__ import annotations

from pathlib import Path


def investigation_markdown(
    report: dict, *, max_items: int = 10, focus_task: str | None = None
) -> str:
    """Create a standalone Markdown triage report without claiming root causes."""
    findings = [
        item for item in report["matched_control_divergences"]
        if focus_task is None or item["task"] == focus_task
    ]
    lines = [
        "# Evaluation investigation report",
        "",
        f"Source: `{report['source']}`  ",
        f"Comparable series: {report['series_tested']}  ",
        f"Recent-decline candidates: {len(report['candidates'])}  ",
        f"Matched-control divergences (top 100): {len(findings)}",
        f"Task filter: `{focus_task}`" if focus_task else "Task filter: none",
        "",
        "## Task-specific checkpoint changes",
        "",
        "| Model / task / metric | Step transition | Score change | Control median "
        "| Controls | Config changes |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in findings[:max_items]:
        flags = []
        for key, label in (("task_config_changed", "task"), ("eval_config_changed", "eval")):
            status = item.get(key)
            flags.append(f"{label}: {'yes' if status else 'no' if status is False else 'unknown'}")
        lines.append(
            f"| {item['model']} / {item['task']} / {item['metric']} "
            f"| {item['from_step']} → {item['to_step']} "
            f"| {item['change']:+.4f} | {item['control_median_change']:+.4f} "
            f"| {item['matched_controls']} | {', '.join(flags)} |"
        )
    lines += [
        "",
        "## Investigation hypotheses (unverified)",
        "",
        "1. **Evaluation/configuration drift:** inspect changed task or backend parameters. "
        "A `no` flag means only the *recorded* config is unchanged.",
        "2. **Generated output changes:** inspect raw answers, response formatting, truncation "
        "and stop reasons for the same prompts across both checkpoints.",
        "3. **Inference variability:** repeat the same checkpoint with identical settings "
        "to assess nondeterministic output.",
        "4. **Genuine task-specific model behavior:** compare per-sample errors and "
        "additional related benchmarks before attributing a model regression.",
        "",
        "## Suggested reproducible experiment",
        "",
        "- Select the highest-priority checkpoint pair above.",
        "- Re-run both checkpoints with the **same task, evaluator version, prompts, "
        "dataset revision, inference backend, and generation settings**.",
        "- Enable per-sample logging; diff matching sample IDs and normalized scores.",
        "- Verify whether the discrepancy persists in an independently repeated run.",
        "- Record artifact hashes and runtime settings in the follow-up report.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    lines += ["", "**This is automated triage, not an identified software defect.**", ""]
    return "\n".join(lines)


def write_markdown(
    report: dict, destination: str | Path, *, focus_task: str | None = None
) -> None:
    Path(destination).write_text(
        investigation_markdown(report, focus_task=focus_task), encoding="utf-8"
    )
