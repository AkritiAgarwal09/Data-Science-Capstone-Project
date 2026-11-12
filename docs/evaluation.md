# Evaluation and reporting agent

`agents.evaluation.run_evaluation(request, run, client=...)` implements
`EvaluationInput -> EvaluationOutput`. The default graph calls it after training.
`EvaluationAgent(client=...)` is an injectable callable for tests or custom pipelines.

The dependency tools and tests were imported without unrelated branch changes from
`origin/pH7/sprint3-eval-tools-and-llm-diagnosis`, commit
`86c56f679bc709ef0f20bcad8bd7828f9627b8d8` (the evaluation-tools work for #75).

The agent creates structured results from the training handoff, calls `compare_arms`
and `reliable_gain`, and evaluates selected models' test predictions with
`segment_metrics`. It compares arms against original and cleaned data, and GAN
against SMOTENC at matched fill levels. Selection comes from validation; the report
does not select a new model, threshold, or fill level from test results. An unavailable
GAN arm is omitted from the comparisons. Missing arms are valid.

The arm comparisons report PR-AUC, AUROC, F1, accuracy, precision and recall.
The random baseline is 0.5 for AUROC and the observed positive rate of each split
for PR-AUC. For accuracy it is the expected accuracy of random labels drawn at
that split's positive rate. The report prints the rates and baselines alongside
the scores, and number verification reads them from `comparisons.csv`.

Segment metrics use source-row joins to the source dataset, with observed binary
labels and real-row provenance checks. Available age, job, marital, education and
Default demographic columns are evaluated. Numeric segment bands use evaluated
rows only. Interval labels also export numeric `segment_lower_bound` and
`segment_upper_bound` evidence, so a quoted boundary can pass number verification
without allowing arbitrary numbers. Missing segment values remain excluded.
Small test sets and small positive-count groups are flagged. Missing
segment columns are reported as a limitation, not evidence of fairness.

## Verified narrative and fallback

The shared LLM runtime requests a Pydantic response at temperature zero with a fixed
outline: summary, improvements, segments, negative/inconclusive results and limitations.
The response cannot change rules or thresholds. Configuration, tool rules and result
artifact hashes are checked around the call.

The response model validates the **complete rendered report** with
`verify_report_numbers`. A made-up number, missing section, unknown field or failed
call consumes the shared retry budget: at most two calls in total. After repeated
failure, the runtime logs `LLM_FALLBACK` and returns a verified plain template.
The final rendered report is verified again before publication. If even the template
or final report fails verification, the output is `FAILED` with
`numbers_verified=False`; an unverified report is never returned as successful.
All LLM calls and number-rejection/final-verification decisions are logged.

The report always includes negative/inconclusive results, small-sample caveats,
augmentation availability, the recorded volume decision (or its absence), audit
references, and ethical limitations. Verdicts and tables come from tools, not LLM
calculations. The JSON evidence retains skipped-fit reasons, the quality report,
GAN-unavailability reason, volume rationale and audit paths. Constant validation-score
models excluded by training remain in the metrics and evidence, and are called out
in the limitations section; they do not enter selected-model comparisons.

## Artifacts

All files are written under `report/` and returned for manifest registration:

- `metrics.csv`: the training handoff's reported metrics, including selection/status.
- `comparisons.csv`: means, seed spread, paired differences by reference arm, and split baselines.
- `segments.csv`: per-model, per-seed segment metrics and small-group flags.
- `evidence.json`: deterministic verdicts, test counts, rule constants and audit context.
- `final_report.md`: verified narrative, comparison/verdict tables and caveats.

The available #75 tools cover PR-AUC, F1, precision, recall, seed-based comparisons,
segments and number checking. They do not supply SHAP analysis, calibration analysis
or PDF export; the agent does not claim to have computed those. The reliable-gain rule
uses paired mean PR-AUC lift versus seed-to-seed spread on both validation and test,
with at least three paired seeds. This is descriptive evidence, not a confidence interval
or significance test. Selected models may differ across arms.

Number verification establishes that numeric tokens occur in the result artifacts,
with the tool's rounding/percentage conventions. It does not prove that every prose
claim associates a valid number with the correct metric, nor validate causal or
fairness interpretations; the fixed tables remain the authoritative evidence.

## Bank Marketing deciles

Export the selected cleaned model's deciles from a saved Bank training-agent run.
Choose a seed explicitly for a run with more than one selected cleaned model.

```bash
uv run python -m agentic_gan.tools.bank_deciles runs/<run_id> --seed 42
```

The exporter reads the validation and test prediction CSVs and scores the
cleaned training rows with the saved estimator and its frozen threshold. It
writes three CSVs and three response-rate charts under `results/`, along with
business takeaways and any held-out response-rate increases between deciles.
Train results are in-sample. Equal scores are ordered by source-row ID.
For runs with a fitted treatment plan, the estimator uses the saved raw split
to clean the retained training rows with the same fitted encoder.

## Offline tests

```bash
uv run pytest tests/test_evaluation_agent.py tests/tools/test_evaluation.py
```

Tests use `FakeLLM` and no credentials. They cover accepted narratives, fabricated
numbers and regeneration, exhausted retries, network/schema failures, changed rules,
failed final checks, absent GAN/volume decisions, artifact manifests, and the default
pipeline reaching a real report. There is no live LLM test.
