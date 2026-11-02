"""Read-only analysis of checkpoint-level evaluation results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SERIES_KEY = (
    "model", "task", "metric", "metric_filter", "num_fewshot", "language", "task_formulation"
)
REQUIRED = set(SERIES_KEY) | {"step", "score", "score_stderr", "higher_is_better", "n_samples"}


def _canonical_config(value: object, *, evaluation: bool = False) -> str | None:
    """Normalize serialized configs while excluding per-run metadata only."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return value
    if evaluation and isinstance(value, dict):
        value = {key: item for key, item in value.items()
                 if key not in ("model_name", "total_eval_time_seconds")}
    return json.dumps(value, sort_keys=True, default=str)


def matched_controls(transitions: pd.DataFrame, *, minimum_change: float) -> list[dict]:
    """Find large task-specific changes with same-step, same-metric controls.

    Controls must share the exact model, metric and checkpoint pair. Their
    scores are never compared directly: only differences between steps are.
    """
    if transitions.empty:
        return []
    results = []
    keys = ["model", "metric", "from_step", "to_step"]
    for _, group in transitions.groupby(keys, dropna=False):
        if len(group) < 4:
            continue
        for row in group.itertuples(index=False):
            if abs(row.change) < minimum_change:
                continue
            controls = group[group["task"] != row.task]
            if len(controls) < 3:
                continue
            median_change = float(controls["change"].median())
            median_absolute = float(controls["change"].abs().median())
            # Require the reference set to be substantially calmer than the target.
            if abs(row.change - median_change) < minimum_change or (
                median_absolute > abs(row.change) / 4
            ):
                continue
            results.append({
                "model": row.model,
                "task": row.task,
                "metric": row.metric,
                "from_step": int(row.from_step),
                "to_step": int(row.to_step),
                "change": round(float(row.change), 5),
                "matched_controls": len(controls),
                "control_median_change": round(median_change, 5),
                "control_median_absolute_change": round(median_absolute, 5),
                "task_config_changed": row.task_config_changed,
                "eval_config_changed": row.eval_config_changed,
                "interpretation": "task-specific divergence; cause unknown",
            })
    results.sort(
        key=lambda item: abs(item["change"] - item["control_median_change"]),
        reverse=True,
    )
    return results[:100]


def analyze(path: str | Path, *, window: int = 3, minimum_change: float = 0.02) -> dict:
    """Find declines and adjacent-checkpoint volatility; never infer causality.

    Scores are grouped only with identical benchmark identifiers and sample counts.
    The noise ratio is descriptive, not a significance test.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    if minimum_change < 0:
        raise ValueError("minimum_change must be non-negative")
    path = Path(path)
    if path.suffix.lower() != ".parquet":
        raise ValueError("input must be a Parquet file")
    data = pd.read_parquet(path)
    missing = REQUIRED - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = data[(data["higher_is_better"] == True) & data["score"].notna()].copy()  # noqa: E712
    candidates: list[dict] = []
    volatility: list[dict] = []
    transition_rows: list[dict] = []
    tested = 0

    for labels, group in data.groupby(list(SERIES_KEY), dropna=False):
        group = group.sort_values("step")
        if group["step"].isna().any() or group["step"].duplicated().any():
            continue
        if group["n_samples"].isna().any() or group["n_samples"].nunique() != 1:
            continue
        identity = {
            key: None if pd.isna(value) else str(value)
            for key, value in zip(SERIES_KEY, labels)
        }
        if len(group) >= 2:
            records = group.to_dict("records")
            for before_point, after_point in zip(records[:-1], records[1:]):
                def config_change(field: str, *, evaluation: bool = False) -> bool | None:
                    if field not in group:
                        return None
                    previous_config = _canonical_config(before_point[field], evaluation=evaluation)
                    current_config = _canonical_config(after_point[field], evaluation=evaluation)
                    if previous_config is None or current_config is None:
                        return None
                    return previous_config != current_config

                transition_rows.append({
                    "model": identity["model"], "task": identity["task"],
                    "metric": identity["metric"],
                    "from_step": before_point["step"], "to_step": after_point["step"],
                    "change": float(after_point["score"] - before_point["score"]),
                    "task_config_changed": config_change("task_config"),
                    "eval_config_changed": config_change("eval_config", evaluation=True),
                })
        if len(group) >= 3:
            jumps = group["score"].diff().abs().dropna()
            volatility.append({
                **identity,
                "checkpoints": len(group),
                "median_absolute_step_change": round(float(jumps.median()), 5),
                "maximum_absolute_step_change": round(float(jumps.max()), 5),
            })
        if len(group) < 2 * window:
            continue
        previous = group.iloc[-2 * window : -window]
        latest = group.iloc[-window:]
        if previous["score_stderr"].isna().any() or latest["score_stderr"].isna().any():
            continue
        tested += 1
        before, after = float(previous["score"].mean()), float(latest["score"].mean())
        decline = before - after
        noise = float(
            ((previous["score_stderr"] ** 2).sum() + (latest["score_stderr"] ** 2).sum())
            ** 0.5 / window
        )
        ratio = decline / noise if noise > 0 else 0.0
        if decline >= minimum_change and ratio >= 2:
            candidates.append({
                **identity,
                "previous_mean": round(before, 5),
                "latest_mean": round(after, 5),
                "drop": round(decline, 5),
                "signal_to_reported_noise": round(ratio, 2),
                "previous_steps": [int(v) for v in previous["step"]],
                "latest_steps": [int(v) for v in latest["step"]],
                "n_samples": int(group["n_samples"].iloc[0]),
            })

    candidates.sort(key=lambda item: (item["signal_to_reported_noise"], item["drop"]), reverse=True)
    volatility.sort(key=lambda item: item["median_absolute_step_change"], reverse=True)
    divergences = matched_controls(
        pd.DataFrame(transition_rows), minimum_change=max(minimum_change, 0.05)
    )
    return {
        "source": str(path),
        "rows": len(data),
        "series_tested": tested,
        "window": window,
        "minimum_change": minimum_change,
        "candidates": candidates,
        "volatility": volatility,
        "matched_control_divergences": divergences,
        "limitations": [
            "Heuristics, not statistical tests: adjacent checkpoints are correlated.",
            "Multiple comparisons may produce false positives.",
            "Reported score standard errors do not measure evaluation-run variability.",
            "Changed task configuration, model behavior, or sampling can explain a signal.",
            "Aggregate scores alone cannot establish root cause.",
            "A stable configuration does not guarantee deterministic inference.",
            "Matched controls share metric names, but not necessarily task difficulty or variance.",
        ],
    }


def write_json(report: dict, destination: str | Path) -> None:
    Path(destination).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
