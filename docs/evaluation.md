# Reference evaluation

Use `gesture-annotator evaluate-run RUN_DIRECTORY REFERENCE.eaf` after a run completes. You can also supply `annotations.json` directly. For single-pass probability runs, evaluation groups the saved raw LLM estimates using the current local output rules (`Q = L`). It does not rerun preprocessing or inference, and does not alter the saved run. Compact runs need only `annotations.json` and `run.json`. If a manifest is present, its completed status and annotation hash are checked.

This public command evaluates one completed run. It does not apply the
[performance report](performance.md)'s shot-value filter, aggregate multiple
videos or compute its bootstrap comparisons. Those results use a separate local
analysis workflow, described in the report; that workflow is not bundled with
this release. Single-run command output therefore need not match the filtered
tables even when it reads the same recording.

```powershell
gesture-annotator evaluate-run "C:\gesture-runs\test-001" "C:\references\video.eaf" --reference-tier "Manual_GUnit" --positive-value "G-unit"
```

The reference must be an EAF 3.0 file with an alignable time tier. Tier and positive-value matching are exact and case-sensitive (surrounding annotation whitespace is stripped). Within reference-assessed time, other G-unit-tier values and gaps in that tier count as negative. The reference is read-only and is not included in this distribution.

## Comparison and exclusions

The program uses the same sample-count calculation as the development activity plot: a sample is reference-positive when its center lies in a positive half-open interval `[start, end)`. Predictions use LLM-only candidate creation and final grouping, including the rest/departure rules and exclusion barriers; they are not a simple probability threshold. A run originally finalized with motion fusion can therefore yield different evaluation/export decisions from its stored `samples[].is_g_unit`. The original results remain unchanged. Samples are 100 ms; a shorter terminal sample counts as one sample, and its actual duration is used for duration totals.

Saved Phase1 exclusions, time-range exclusions, and unlabelled reference visibility intervals are removed from the denominator. A sample is reference-assessed only when its center falls inside a nonblank annotation in `Spker_vis`; gaps, blank values, and leading/trailing unlabelled portions are excluded. Use `--reference-visibility-tier "Your tier"` for another tier. An empty or missing tier yields no scorable samples; a missing tier reports “Phase1 cannot be scored.” Unknown motion or an uncertain LLM score alone does not exclude a sample. Nonblank visibility label values, including “No visible data”, do not themselves exclude detection samples. When Phase1 was enabled, a warning notes that visibility errors among admitted samples may contribute to the results. Report the evaluated and excluded durations alongside performance: these statistics do not measure scene-selection recall over the whole video.

Run and reference timestamps must refer to the same recording and time origin. Processing with `--from-second` retains original-video timestamps. Use `--alignment-offset-us 1000000` only if an explicit alignment correction is needed; this example adds one second to every reference interval. Aligned intervals must remain nonnegative. No automatic alignment or boundary tolerance is applied.

## Metrics and files

| Metric | Formula |
| --- | --- |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| Miss rate | FN / (TP + FN) |
| False-positive rate | FP / (FP + TN) |
| F1 | 2 TP / (2 TP + FP + FN) |
| Accuracy | (TP + TN) / (TP + FP + FN + TN) |
| Balanced accuracy | (TP / (TP + FN) + TN / (TN + FP)) / 2 |

TP and TN are correctly classified positive and negative samples; FP are false alarms and FN are misses. A zero denominator yields `null` in JSON and “undefined” in text. Balanced accuracy requires both reference classes. JSON ratios range from zero to one; text displays percentages. Confusion entries also include sample counts, durations and percentages of evaluated time. These percentages differ from false-positive rate: their denominator is all evaluated duration.

The command replaces `reference-statistics.json` and `reference-statistics.txt` in the run directory. Use `--output "C:\reports\test-001.json"` to change both destinations. Reports contain the reference mapping, input hashes, exclusion totals (including `unlabelled_reference`), metric definitions and warnings. They summarize errors without listing individual error intervals.

Add `--export-eaf` to write or replace `annotations.eaf` using the same LLM-only grouping. The EAF header records its decision basis and output hash; standalone public-format exports also have a provenance sidecar. This does not modify canonical decisions or the reference. Output destinations may not overwrite input files or protected run artifacts. The separate `export-eaf` command remains available without evaluation.

## Decision provenance and older formats

JSON/text reports identify their decision basis. Probability runs require valid
saved raw LLM estimates for every scored sample; a missing or invalid estimate
is an error, including when all probability records are missing. It never
substitutes a motion/fused value or makes a model call. Fully excluded or genuinely
unscored timelines remain valid. Phase1/range exclusions and unscored gaps remain
respected.

Historical non-probability formats retain their saved decisions and are explicitly
labelled `saved_legacy_decisions`; they are not reported as LLM-only results.
Both original and reprocessed completed-run manifests are hash-checked.
