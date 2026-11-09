# Evaluation investigation report

Source: [published scores.parquet](https://huggingface.co/datasets/openeurollm/prelude-base-eval-scores/resolve/88f335f943f34bbf1fbba681719b2750f02340f7/scores.parquet) (dataset commit 88f335f943f34bbf1fbba681719b2750f02340f7, file SHA256 d8d2f75aecd874ded2830dd02de7c8bcc4ace85243d846aa4952d132fa72bf5a)  
Comparable series: 1776  
Recent-decline candidates: 29  
Matched-control divergences (top 100): 10
Task filter: `coqa_gen`

## Task-specific checkpoint changes

| Model / task / metric | Step transition | Score change | Control median | Controls | Config changes |
| --- | --- | ---: | ---: | ---: | --- |
| prelude_anneal2t / coqa_gen / f1 | 1092000 → 1094400 | +0.4244 | +0.0034 | 7 | task: no, eval: no |
| prelude_anneal2t / coqa_gen / em | 1092000 → 1094400 | +0.3672 | +0.0019 | 6 | task: no, eval: no |
| prelude_anneal2t / coqa_gen / f1 | 1036800 → 1039200 | -0.3513 | -0.0059 | 7 | task: no, eval: no |
| prelude_anneal2t / coqa_gen / em | 1024800 → 1027200 | +0.3028 | -0.0098 | 6 | task: no, eval: no |
| prelude_anneal2t / coqa_gen / f1 | 1024800 → 1027200 | +0.2996 | -0.0068 | 7 | task: no, eval: no |
| prelude_anneal2t / coqa_gen / em | 1036800 → 1039200 | -0.2955 | -0.0056 | 6 | task: no, eval: no |
| prelude_anneal2t / coqa_gen / f1 | 967200 → 969600 | +0.2678 | +0.0043 | 7 | task: no, eval: no |
| prelude_anneal2t / coqa_gen / f1 | 1065600 → 1067939 | -0.2621 | -0.0125 | 7 | task: no, eval: no |
| prelude_anneal2t / coqa_gen / f1 | 1067939 → 1068000 | +0.2542 | +0.0098 | 7 | task: no, eval: no |
| prelude_anneal2t / coqa_gen / f1 | 1089600 → 1092000 | -0.2344 | +0.0077 | 7 | task: no, eval: no |

## Investigation hypotheses (unverified)

1. **Evaluation/configuration drift:** inspect changed task or backend parameters. A `no` flag means only the *recorded* config is unchanged.
2. **Generated output changes:** inspect raw answers, response formatting, truncation and stop reasons for the same prompts across both checkpoints.
3. **Inference variability:** repeat the same checkpoint with identical settings to assess nondeterministic output.
4. **Genuine task-specific model behavior:** compare per-sample errors and additional related benchmarks before attributing a model regression.

## Suggested reproducible experiment

- Select the highest-priority checkpoint pair above.
- Re-run both checkpoints with the **same task, evaluator version, prompts, dataset revision, inference backend, and generation settings**.
- Enable per-sample logging; diff matching sample IDs and normalized scores.
- Verify whether the discrepancy persists in an independently repeated run.
- Record artifact hashes and runtime settings in the follow-up report.

## Limitations

- Heuristics, not statistical tests: adjacent checkpoints are correlated.
- Multiple comparisons may produce false positives.
- Reported score standard errors do not measure evaluation-run variability.
- Changed task configuration, model behavior, or sampling can explain a signal.
- Aggregate scores alone cannot establish root cause.
- A stable configuration does not guarantee deterministic inference.
- Matched controls share metric names, but not necessarily task difficulty or variance.

**This is automated triage, not an identified software defect.**
