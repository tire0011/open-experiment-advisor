# Open Experiment Advisor

Open-source, **read-only** anomaly triage for LLM evaluation checkpoints. It uses deterministic statistics and requires **no LLM API, GPU, HPC credentials, or agent runtime**.

> **Status:** experimental v0.1.0. Flags are hypotheses for human review, not verified model regressions or causal diagnoses.

## Quick start

Python 3.10+:

```powershell
python -m pip install -e ".[dev]"
experiment-advisor .\scores.parquet --output .\report.json
python -m pytest -q
```

The input must be a Parquet table containing the columns `model`, `task`, `metric`, `metric_filter`, `num_fewshot`, `language`, `task_formulation`, `step`, `score`, `score_stderr`, `higher_is_better`, and `n_samples`.

### Public OpenEuroLLM example

Download the [Prelude base evaluation scores](https://huggingface.co/datasets/openeurollm/prelude-base-eval-scores) (`scores.parquet`) and run the command above.

The analyzer groups equivalent task/metric definitions, rejects duplicate steps or changing sample counts, and reports:

- **Recent declines:** last N checkpoints versus preceding N checkpoints (`--window 3` by default), ranked by the drop relative to the *reported per-evaluation* standard errors.
- **Checkpoint volatility:** median and maximum absolute adjacent-checkpoint score changes per series.
- **Machine-readable JSON** for external tools, CI workflows, dashboards, or agent integrations.

All scoring is descriptive and checkpoint samples are correlated: **the reported noise ratio is not a statistical p-value or significance test**. The tool deliberately does not guess a root cause from aggregate scores.

### Why it exists

The [OpenEuroLLM CoQA investigation (#163)](https://github.com/OpenEuroLLM/oellm-eval/issues/163) is the first motivating example: generative CoQA F1 changes dramatically across adjacent Prelude checkpoints while multiple-choice CoQA remains comparatively stable. The dataset alone cannot establish whether this is a model, prompt, backend, or evaluation problem.

## Roadmap

1. Dataset adapters and configuration-drift diagnostics.
2. Automatic matched-task controls and a compact Markdown report.
3. Optional sample-level output comparison.
4. Optional LLM explanations and MCP interface, with all underlying numeric evidence preserved.

## Contributing

Issues and PRs welcome. Add a small anonymized fixture and an automated test for each new detector. Avoid including private model outputs or checkpoint locations in reports.

## License

Apache-2.0. See [LICENSE](LICENSE).
