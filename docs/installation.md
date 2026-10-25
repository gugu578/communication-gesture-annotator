# Installation and compatibility

Follow the [README installation commands](../README.md#install-and-configure).
The tested environment is Windows with Python 3.12, FFmpeg/FFprobe and the
versions in `constraints.txt`. Linux and macOS have not been verified for this
release. Their command equivalents are `python3.12`, `.venv/bin/python`,
`.venv/bin/gesture-annotator` and `cp`; set credentials with
`export GESTURE_ANNOTATOR_OPEN_WEBUI_API_KEY="YOUR_API_KEY"`.

## Dependencies and upgrades

Install into a dedicated virtual environment. `constraints.txt` records the
release's runtime versions. If those wheels are unavailable on your platform,
`pip install .` uses the declared dependency ranges instead; that creates a
different, unverified environment. Record its versions when comparing results.

When upgrading, use a new dedicated virtual environment and install this source
tree there. Keep your existing local configuration and run directories. Python
scripts must import `communication_gesture_annotator`; the command remains
`gesture-annotator`. No alias for the previous Python namespace is installed.
Existing saved runs remain readable for evaluation and EAF export.

`RELEASE.json` records the prompt and runtime identities. Existing saved runs
remain unchanged when evaluated or exported; those operations replay the current
LLM-only grouping as described in [evaluation details](evaluation.md).

The current pose dependency set installs both `opencv-python` and
`opencv-contrib-python`, which share the `cv2` package. The tested environment
works, but overlapping files are an unresolved installation/upgrade risk. Removing
one constraint alone does not resolve the upstream dependencies.

The default pose-weight URLs are not pinned to immutable revisions or verified
against expected download hashes. For repeatable preprocessing, keep the same
local model files and configure their paths. Recorded hashes identify the files
used; they do not guarantee that a future download will return the same weights.

## Verification scope

Automated tests cover synthetic media and scripted model responses on
Windows/Python 3.12, without live LLM calls or pose-weight downloads. These checks
validate software behavior; accuracy is assessed separately in the performance
report. After installation, run the test command in the README and `model-check`
against your configured model deployment.
