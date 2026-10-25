# Configuration

Start with `configs/config.example.toml`. Fill `open_webui.base_url` and `open_webui.model`, then set the environment variable named by `api_key_env`. Unknown tables and option names are rejected. Internal contract versions are recorded by the program and are not configuration switches.

## Main controls

| Setting | Default | Meaning |
| --- | --- | --- |
| `storage.mode` | `"minimal"` | Bound derived image caches and compress large phase results. `"compact"` retains more cached output; `"debug"` keeps full diagnostics; `--storage-mode` overrides TOML. |
| `inference.phase1_enabled` | `true` | Select the main speaker and classify scene membership before Phase2. `--skip-phase1` overrides this for a run. |
| `inference.visibility_max_frames` | `5` | Maximum CURRENT frames per scene-membership call. Reference exemplars are additional images/cells. |
| `detection.probability_target_frames` | `3` | Number of 100 ms samples scored per Phase2 call; at least one. |
| `detection.probability_history_frames` | `47` | Maximum preceding context frames; zero is allowed. |
| `detection.probability_future_frames` | `10` | Maximum following context frames; zero is allowed. |
| `detection.probability_history_enabled` | `true` | Include previous raw LLM estimates for already-scored history, explicitly labelled as fallible estimates. |
| `detection.probability_score_source` | `"llm"` | Use the LLM output directly for candidates; `"motion_fusion"` combines it with motion. Both modes provide motion evidence to the LLM. |
| `detection.probability_motion_trace_enabled` | `true` | Add the labelled physical-motion trace to the same request. |
| `detection.probability_context_strength` | `4.0` | Contextual contribution to the combined probability; permitted range 0–6. |
| `analysis.minimum_separating_rest_us` | `300000` | Supported rest duration required for candidate separation, in microseconds. |
| `detection.communication_minimum_departure_us` | `300000` | Discard shorter activity excursions when surrounded by supported rest. |
| `detection.finalization_merge_gap_ms` | `300` | Join final G-unit intervals separated by at most this many milliseconds; zero disables. |

The 10 Hz annotation grid is fixed in this release. Native pose observations retain decoded source-frame timestamps. At 10 Hz, 300 ms corresponds to three complete sample intervals; the final merge comparison uses elapsed microseconds, not an assumed source frame rate.

The default Phase2 context is at most six seconds: 4.7 seconds of history, 300 ms of targets and one second of look-ahead. Clip/range edges and confirmed Phase1 exclusions shorten it. Targets and look-ahead stop at the next detected camera shot. Preceding same-speaker RGB can cross a camera cut, with the cut marked explicitly; motion derivatives cannot cross it. Previous estimates from another shot, or estimates produced without the full configured same-shot history, are withheld from later requests. The original estimates remain outputs. Pose-only discontinuities suppress unsuitable measurements without automatically discarding same-speaker visual history. No additional model pass is introduced by these options.

Schedule and separating-rest values are inserted into the Phase2 prompt from the resolved configuration. Native preprocessing remains at source timestamps. Audio is disabled by default and is not used by the model.

Set `phase1_enabled = false` in `[inference]`, or add `--skip-phase1`, when the selected video interval already contains only the intended speaker scene. This removes the overview, exemplar selection and scene-membership model calls. It does not remove preprocessing or Phase2. Scene membership is assumed in this mode, so irrelevant shots cannot be filtered by Phase1. Leave it enabled if the recording includes slides, audience views or other speakers.

## Candidate-score selection

In `[detection]`, `probability_score_source = "llm"` is the default even
if the option is omitted. It uses the LLM score directly. Use `"motion_fusion"`
to combine motion and LLM scores using [the pipeline formula](pipeline.md).
An existing explicit `"motion_fusion"` setting continues to select that mode. Both modes retain RGB, motion
measurements and the motion trace in the same model requests. Rest thresholds
remain 0.5 and 0.65, with unchanged temporal rules. The source is saved in run
configuration; `non_g_unit_probability` is the selected score.

Changing only this setting can reuse valid model answers when resuming an
unfinished run; scores and candidates are recomputed locally. Completed runs
remain immutable. Evaluation and EAF export always use a local LLM-only projection
for probability runs, independently of the saved fusion choice. Choose a fresh directory to preserve a separate comparison.

## Model transport

The supported interface is Open WebUI: `GET /api/models` for discovery and `POST /api/chat/completions` for image/text messages and JSON-schema responses. Set the server root as `base_url`, not `/v1` or a completions URL. The selected VLM must accept multiple images and structured output; the vLLM serving software is optional. Qwen36-27b has been tested through this interface. Model names/aliases and available context depend on the deployment.

`timeout_seconds` defaults to 900 seconds. Retryable transport/server failures have at most three extra attempts with delays of 30, 60 and 60 seconds. `maximum_transient_http_retries = 0` disables those retries. The delay list must cover the configured retries; edit both together. The optional `transient_http_retry_backoff_seconds` supplies the fallback schedule when no explicit list is configured. Invalid structured responses have a separate `inference.maximum_repair_attempts` limit, default one. Authentication, context-capacity errors and other non-retryable failures stop the run with saved diagnostics. Phase2 does not silently shorten the configured context.

`inter_request_cooldown_seconds = 0.5` controls the pause between requests. Model response time, transport retries and semantic repairs are separate from this cooldown. These values do not change your server's own timeout or context limit.

The program checks image, pixel, byte, prompt and output limits before sending. Phase2 packages crops and sheets as PNG; Phase1 uses JPEG sheets. `model_context_window_tokens` must match the backend's usable context. The configured `maximum_output_tokens` is an upper limit; small tasks reserve smaller outputs. If capacity remains insufficient, try a shorter context or smaller image dimensions, using a new run directory.

For advanced image layout and transport limits, see the typed settings in `src/communication_gesture_annotator/config.py`. The default Phase2 layout has three 640×360 cells per row; person crops preserve aspect ratio without restoring large black canvases. Frame and time labels identify each cell's role.

## Processing ranges, reuse and output location

`--from-second` and `--to-second` use original-video seconds. They round half-up to the nearest 100 ms and bound preprocessing, scene selection, targets and context. `20.44` becomes `20.4`; `20.45` becomes `20.5`. The end is exclusive. Original PTS-derived timestamps remain in output. A small structural timeline is retained outside the selection, marked as excluded, without processing those images or pose observations.

`--run-dir` chooses where generated artifacts are stored. `--resume` continues an incomplete run with matching settings and successful request checkpoints. Remove any preprocessing-reuse option from the original command when resuming; these options are mutually exclusive. `--use-pre-processing-from OLD_RUN` starts a new inference run with compatible preprocessing from another run. When a range is supplied, only the relevant preprocessing is copied; missing selected data triggers recomputation for the selected interval. It does not copy old model decisions.

All storage modes support preprocessing reuse; minimal is the default. Storage
mode is excluded from model-input identities and does not change the algorithm.
See [storage and resume](storage.md) for retained data, cleanup and space guidance.
Use runs from this release; compatibility with separate development versions is
not promised.

## Pose models and data handling

The defaults use ONNX Runtime on CPU, a YOLOX person detector and an RTMW3D model. The configured URLs identify external model files; no weights are redistributed here. Set `detector_model` and `pose_model` to compatible local files to avoid downloading them again. Check the model providers' terms separately from this package's Apache 2.0 license. GPU execution requires an appropriate ONNX Runtime installation and matching device libraries; it is not required by the default setup.

Credentials are read from the process environment. The generic data-rights sidecar checks are retained so restrictions follow copied preprocessing and model requests. Remote processing of restricted media requires explicit operator authorization and HTTPS; do not copy someone else's authorization settings. Generated traces contain source-derived content and belong with the run, not the source distribution.

HTTP 413 indicates a server request-body limit, distinct from the model context
limit and the client's byte budget. Phase2 first tries lossless PNG compression.
If that is insufficient, it tries video sheets at 85%, then 70%, of their original
width and height, using area resampling. These two fallbacks reduce visual detail;
they preserve every frame, chronological label, target/history/future sample,
prompt and numeric motion value. The motion chart retains its original resolution.
No level is sent unless its encoded images are smaller than the rejected version.
The console warns when spatial reduction is used, and the run records its policy,
dimensions, image identities and coordinate transforms.

Further rejection stops the run. `--resume` keeps successful checkpoints and
skips recorded rejected larger representations; the smallest level may be retried
after a server-side correction. A run interrupted under an older release may
repeat its saved lossless request once before entering the new fallback. Normal
successful requests keep their original representation. Raising the client's byte
budget cannot increase a server's limit.
