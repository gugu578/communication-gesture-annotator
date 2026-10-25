# Communication Gesture Annotator

This program detects communicative hand gestures in video using reconstructed motion and a vision-language model. The program produces gesture-unit (G-unit) intervals, per-sample scores and optional ELAN annotations. It processes one selected visible speaker and uses pictures and motion data; audio is not analyzed.

## Supported videos

This proof of concept is intended for ordinary speaking situations with **one recurring main speaker**, such as presentations, lectures and interviews focused on one person. The program works best when the speaker's upper body, hands and fingers are sufficiently visible for pose reconstruction and visual assessment in one continuous camera shot. Use video files that FFmpeg can decode; MP4 with H.264 is a practical starting point.

The program can exclude slides, audience shots and scenes outside the selected speaker's main scene through optional visual inspection (Phase1) by a vision-language model before G-unit detection. The pipeline marks detected camera discontinuities and instructs the model to distinguish visual jumps from hand movement; a cut does not automatically force a gesture boundary. Cuts, occlusion, small hands and unreliable pose estimates can still cause errors. Multi-person gesture annotation, sign-language recognition and general action recognition are outside this release's scope. Evaluation so far is exploratory; accuracy across different speakers, recordings and model deployments has not been established.

**If the entire selected interval already shows the intended speaker and nothing needs to be excluded, add `--skip-phase1`.** This skips the speaker overview and scene-membership calls. Pose preprocessing and the single Phase2 pass still run. Keep Phase1 enabled for recordings containing unrelated scenes or people; skipping it assumes scene membership rather than checking it.

## Requirements

- Python 3.12 or newer.
- FFmpeg and FFprobe on `PATH`, with H.264 encoding (`libx264`).
- **Access to an LLM API:** this release supports the [Open WebUI API](https://docs.openwebui.com/reference/api-endpoints/), using `GET /api/models` and `POST /api/chat/completions`. An API key and permission to use the selected model are required.
- **A vision-language model (VLM)** accepting multiple pictures and structured JSON responses. A text-only model is insufficient. The program sends labelled frames/contact sheets and a motion trace, not a video stream. It has been tested with **Qwen36-27b** through Open WebUI; use your server's exact model identifier. The vLLM serving software is not a requirement, and direct provider APIs are not supported by this client.

- **No task-specific model training or fine-tuning was performed for this project.** The prompt and pipeline were developed using the evaluation videos.

- **Free disk space throughout processing, resume and finalization**, for frames, pose caches, model weights and temporary output replacement. See [storage guidance](docs/storage.md). Begin with a short interval to check runtime and disk use.

Pose reconstruction uses [rtmlib](https://github.com/Tau-J/rtmlib), ONNX Runtime, a YOLOX person detector and RTMW3D. The default local preprocessing runs on CPU. Model weights download on first use unless local paths are configured; no weights are included here. Access to a separate vision-language model remains necessary.

## Install and configure

The distribution is named `communication-gesture-annotator`, the command is
`gesture-annotator`, and the Python package is `communication_gesture_annotator`.
You can also run the command as `python -m communication_gesture_annotator`.

Run these commands in the folder containing `pyproject.toml`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -c constraints.txt .
Copy-Item configs/config.example.toml configs/config.toml
```

The tested environment is **Windows with Python 3.12**. Linux/macOS command
equivalents and upgrade instructions are in [installation notes](docs/installation.md);
those platforms have not been verified for this release.

Fill the blank `open_webui.base_url` and `open_webui.model` in `configs/config.toml`. The URL is the Open WebUI server root, such as `http://127.0.0.1:3000`, without an API path. Keep credentials in the environment:

```powershell
$env:GESTURE_ANNOTATOR_OPEN_WEBUI_API_KEY = "YOUR_API_KEY"
.\.venv\Scripts\gesture-annotator.exe model-check --phase2-config configs/config.toml
```

On Linux/macOS, set the key with `export GESTURE_ANNOTATOR_OPEN_WEBUI_API_KEY="YOUR_API_KEY"`. Public network endpoints require HTTPS and `allow_non_private_network = true`. Match the configured context and image limits to your server; a model's advertised maximum may differ from its deployed limit.

## Run

```powershell
.\.venv\Scripts\gesture-annotator.exe run-llm "C:\media\video.mp4" --phase2-config configs/config.toml --run-dir "C:\gesture-runs\test-001" --from-second 20.4 --to-second 26.4 --export-eaf
```

Omit the time limits to process the full recording. They apply to preprocessing and all model stages, round to the nearest 100 ms, and use an exclusive end. Output timestamps remain aligned to the original video. Keep the source video outside the run directory.

- `annotations.json` is the canonical result; `annotations.eaf` is the optional single-tier `G-unit` presence export, without hand labels.
- **Minimal storage and no audio are defaults.** Minimal runs retain only necessary results and caches needed for resume and preprocessing reuse. Use `--storage-mode compact` for more output or `--storage-mode debug` for full diagnostics.
- Repeat an interrupted command with `--resume`, keeping the identical input video and inference settings. If the original command used `--use-pre-processing-from` (or `--reuse-preprocessing-from`), remove that option when adding `--resume`; reuse initializes a new run and cannot be combined with resume. For a new experiment or a shortened/re-encoded video, choose a new `--run-dir`.
- Add `--use-pre-processing-from "C:\gesture-runs\test-001"` to reuse compatible preprocessing in a new run. Selected missing or incompatible preprocessing is recomputed; old model decisions are not copied. Compact and minimal runs support this reuse.
- `--stop-after preprocessing`, `phase1` or `phase2` saves a stage boundary for later `--resume`.

To export an already completed run without model calls, use `gesture-annotator export-eaf "C:\gesture-runs\test-001"`; add `--force` to replace an existing EAF.

For an already selected speaker clip, the corresponding command is:

```powershell
.\.venv\Scripts\gesture-annotator.exe run-llm "C:\media\speaker.mp4" --phase2-config configs/config.toml --run-dir "C:\gesture-runs\test-002" --skip-phase1 --export-eaf
```

Alternatively, set `phase1_enabled = false` in the existing `[inference]` table of your configuration file.


## Evaluate against a reference EAF

Compare a completed run with your reference annotations; no model calls, API key, source video or pose files are needed:

```powershell
.\.venv\Scripts\gesture-annotator.exe evaluate-run "C:\gesture-runs\test-001" "C:\references\video.eaf"
```

The default reference tier is **`Manual_GUnit`**, with **`G-unit`** as the positive
value. The command groups saved LLM estimates locally and writes JSON/text statistics on the 100 ms
grid, excluding Phase1-rejected time and unlabelled reference `Spker_vis` intervals.
See [evaluation details](docs/evaluation.md) for mappings, alignment and metrics.
Evaluation and EAF export use the same LLM-only grouping, including for older
fused runs, without changing their saved JSON or making model calls.
This single-run command does not reproduce the performance report's shot filtering,
pooled results or bootstrap comparisons; that separate analysis is described there.

## Method and attribution

Preprocessing reconstructs source-rate body and hand motion. Phase1 selects the speaker and relevant scenes. One chronological Phase2 pass receives labelled images, body-relative hand transport, torso-relative palm rotation, finger motion and quality information. It estimates whether each target sample lies outside a communicative hand phrase. By default, the LLM estimate directly drives local candidate grouping and finalization. Numerical fusion with motion remains optional; both modes provide the same motion evidence to the LLM and add no model pass.

The gesture-unit concept follows Kendon's work and McNeill's *Hand and Mind* (1992). We used M3D's guidance regarding the distinction between communicative hand activity, contextual rest and holds, including the ambiguous-pause criterion. The reconstruction, reliability handling, request design, probability fusion and automatic grouping are this project's computational choices; they are not algorithms prescribed by M3D. This is an independent application, not an implementation of the full M3D framework. [Method, formulas and references](docs/pipeline.md)

These are experimental machine predictions and heuristic scores, not calibrated probabilities or reference annotations. Performance depends on the model, visibility and pose quality.

## Citation

Hládek, Ľ. (2026). *Communication Gesture Annotator* (Version 0.2.5) [Software]. [GitHub repository](https://github.com/gugu578/communication-gesture-annotator).

Machine-readable citation: [CITATION.cff](CITATION.cff). The publication citation will be added when available.

## Documentation and tests

The [exact Phase2 prompt](docs/phase2-prompt.md) includes its full default wording, configuration substitutions, user-message structure and response schema, with links to the implementation.

See [preliminary performance on M3D_EN](docs/performance.md), [configuration](docs/configuration.md) and [pipeline/output](docs/pipeline.md). The performance report evaluates default LLM-only scoring on selected shots with visible hands and upper body from five videos, alongside simple baselines and bootstrap confidence intervals with adjusted p-values. After installation, run `python -m unittest discover -s tests -v` with the virtual environment's Python. Tests use synthetic media and scripted model responses; FFmpeg is needed for integration tests. They do not download weights or call an LLM.

Version 0.2.5 defaults to LLM-only scoring and minimal storage. The `v0.2.5` tag identifies this release; `RELEASE.json` records its runtime, configuration template and prompt hashes. Use the same configuration and model deployment across videos; server-side model changes and nondeterminism can change outputs even with identical client code.

## AI-assisted development

Development of this software and documentation was assisted by **OpenAI Codex**,
an AI coding tool. AI assistance does not establish correctness or scientific
validity. Responsibility for review, testing and releases remains with the
maintainer; users should independently validate results for their intended use.
This disclosure does not imply endorsement or certification by OpenAI.

## License

Original code is [Apache-2.0](LICENSE). Dependencies, external model weights and input media retain their own licenses and terms. No datasets, reference annotations or weights are distributed. Only send media to a model service you are authorized to use; generated frames, motion caches and traces can contain sensitive source content.
