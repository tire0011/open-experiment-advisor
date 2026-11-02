import pandas as pd
import pytest

from experiment_advisor import analyze


def sample_rows(scores, *, task="coqa_gen", samples=100):
    return [
        dict(model="test-model", task=task, metric="f1", metric_filter="none",
             num_fewshot=0, language="en", task_formulation="gen", step=i,
             score=score, score_stderr=0.01, higher_is_better=True, n_samples=samples)
        for i, score in enumerate(scores)
    ]


def run_analysis(tmp_path, rows, **kwargs):
    source = tmp_path / "scores.parquet"
    pd.DataFrame(rows).to_parquet(source)
    return analyze(source, **kwargs)


def test_finds_drop_and_volatility(tmp_path):
    result = run_analysis(tmp_path, sample_rows([0.8, 0.82, 0.81, 0.4, 0.41, 0.39]))
    assert result["series_tested"] == 1
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["drop"] == 0.41
    assert result["volatility"][0]["maximum_absolute_step_change"] == 0.41


def test_does_not_mix_tasks(tmp_path):
    rows = sample_rows([0.7] * 6) + sample_rows([0.3] * 6, task="squad_gen")
    result = run_analysis(tmp_path, rows)
    assert result["series_tested"] == 2
    assert not result["candidates"]


def test_changed_sample_counts_excluded(tmp_path):
    rows = sample_rows([0.8] * 3 + [0.4] * 3)
    for row in rows[3:]:
        row["n_samples"] = 50
    result = run_analysis(tmp_path, rows)
    assert result["series_tested"] == 0
    assert not result["volatility"]


def test_duplicate_steps_excluded(tmp_path):
    rows = sample_rows([0.8] * 6)
    rows[-1]["step"] = rows[-2]["step"]
    assert run_analysis(tmp_path, rows)["series_tested"] == 0


def test_missing_column_is_actionable(tmp_path):
    source = tmp_path / "bad.parquet"
    pd.DataFrame({"score": [0.5]}).to_parquet(source)
    with pytest.raises(ValueError, match="Missing required columns"):
        analyze(source)


def test_invalid_parameters(tmp_path):
    with pytest.raises(ValueError, match="window"):
        run_analysis(tmp_path, sample_rows([0.5] * 6), window=1)
    with pytest.raises(ValueError, match="minimum_change"):
        run_analysis(tmp_path, sample_rows([0.5] * 6), minimum_change=-1)


def test_matched_controls_detect_task_specific_jump(tmp_path):
    rows = sample_rows([0.8, 0.8, 0.8, 0.3, 0.3, 0.3])
    for task in ("squad_gen", "drop_gen", "naturalqs_gen"):
        rows.extend(sample_rows([0.6, 0.6, 0.6, 0.61, 0.61, 0.61], task=task))
    report = run_analysis(tmp_path, rows)
    coqa = [item for item in report["matched_control_divergences"]
            if item["task"] == "coqa_gen"]
    assert len(coqa) == 1
    assert coqa[0]["change"] == -0.5
    assert coqa[0]["matched_controls"] == 3
    assert coqa[0]["control_median_change"] == 0.01


def test_no_control_claim_when_all_tasks_fall(tmp_path):
    rows = []
    for task in ("coqa_gen", "squad_gen", "drop_gen", "naturalqs_gen"):
        rows.extend(sample_rows([0.8, 0.8, 0.8, 0.3, 0.3, 0.3], task=task))
    assert not run_analysis(tmp_path, rows)["matched_control_divergences"]


def test_no_control_claim_with_only_two_controls(tmp_path):
    rows = sample_rows([0.8, 0.8, 0.8, 0.3, 0.3, 0.3])
    for task in ("squad_gen", "drop_gen"):
        rows.extend(sample_rows([0.6] * 6, task=task))
    assert not run_analysis(tmp_path, rows)["matched_control_divergences"]
