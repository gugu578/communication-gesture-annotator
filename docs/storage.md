# Storage and resume

Minimal storage is the default. An explicit `[storage].mode` in TOML overrides
that default, and `--storage-mode` overrides TOML. Existing configurations that
set `mode = "compact"` continue to use compact storage until changed.

| Mode | Retained data |
| --- | --- |
| `minimal` | Reusable preprocessing, validated answers, phase results and small audit/checkpoint records; bounded derived image caches. |
| `compact` | More cached output for inspection and reuse. |
| `debug` | Full diagnostics, including request details and intermediate images. |

Storage mode does not change model inputs or decisions. All modes support resume,
time ranges, preprocessing reuse, Phase1 skipping, evaluation and EAF export.

## Cache lifecycle

In minimal mode, Phase2 crops are limited to the current context; retry images
are released after checkpointing. Phase1 sheets are kept for fast resume and
removed at finalization. Large phase evidence files use lossless `.json.gz`;
canonical `annotations.json` and EAF remain directly readable.

Existing phase files keep their encoding when resumed. Changing mode does not
clean up already completed runs or old debug diagnostics. Saved media and model
answers are restored before rebuilding request images; large pose caches still
require validation. Keep the identical source video and inference settings for
`--resume`. If the original command included `--use-pre-processing-from` or
`--reuse-preprocessing-from`, remove that option when resuming; the imported
preprocessing is already in the run. Use a new run directory for a new experiment; `--force` replaces an
existing run's generated outputs and is usually unnecessary.

## Free space

Keep free space available throughout processing, resume and finalization. Atomic
saves can temporarily need both the previous and replacement file. For one
measured 17 min 42 s recording, minimal retention is estimated at 2.7–3.0 GB
instead of 8.2 GB in compact mode; this is not a measured peak or a bound for
other recordings. Budget at least 6 GB free for a comparable run, plus source
media and model weights, with a reserve.

Resolution, duration and scene complexity affect requirements. Test a short
interval before a batch and select a sufficiently large drive with `--run-dir`.
Keep the source video outside the run directory.

Use only one running process per run directory. Concurrent writers are not
supported; give parallel experiments separate directories.
