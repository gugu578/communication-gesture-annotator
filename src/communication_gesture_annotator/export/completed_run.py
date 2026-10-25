# SPDX-License-Identifier: Apache-2.0
"""Read completed canonical results and export EAF without rerunning inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import xml.etree.ElementTree as ET

from .eaf import export_eaf_document
from .json_export import read_json_snapshot, validate_analysis_document, write_json


@dataclass(frozen=True)
class CompletedRun:
    path: Path
    document: dict[str, Any]
    sha256: str
    manifest: dict[str, Any] | None


def paths_alias(first: Path, second: Path) -> bool:
    return first.resolve() == second.resolve() or (
        first.exists() and second.exists() and first.samefile(second)
    )


def check_outputs(
    outputs: Sequence[Path], *, protected: Sequence[Path], overwrite: bool
) -> None:
    """Check all destinations before writing any output, including hard links."""
    for index, target in enumerate(outputs):
        if any(
            paths_alias(target, source) for source in (*protected, *outputs[:index])
        ):
            raise ValueError(
                "output may not overwrite an input or another run artifact"
            )
        if target.exists() and (not overwrite or not target.is_file()):
            raise FileExistsError(target)


def load_completed_run(path: str | Path) -> CompletedRun:
    supplied = Path(path).resolve()
    source = supplied / "annotations.json" if supplied.is_dir() else supplied
    document, digest = read_json_snapshot(source)
    validate_analysis_document(document)
    manifest_path = source.parent / "run.json"
    manifest = None
    if manifest_path.exists():
        manifest, _ = read_json_snapshot(manifest_path)
        if not isinstance(manifest, dict) or manifest.get("run_id") != document.get(
            "run_id"
        ):
            raise ValueError("run.json does not belong to the exported analysis")
        if manifest.get("status") != "completed":
            raise ValueError("run.json must declare a completed run")
        if "artifacts" in manifest:
            artifacts = manifest.get("artifacts")
            hashes = manifest.get("artifact_sha256")
            if not isinstance(artifacts, list) or not isinstance(hashes, dict):
                raise ValueError("run.json artifact provenance is malformed")
            expected = hashes.get(source.name) if source.name in artifacts else None
        else:
            expected = (
                manifest.get("annotations_sha256")
                if manifest.get("annotations") == source.name
                else None
            )
        if expected != digest:
            raise ValueError(
                "canonical analysis does not match its recorded run artifact hash"
            )
    elif supplied.is_dir():
        raise ValueError("completed run directory requires run.json")
    return CompletedRun(source, document, digest, manifest)


def protected_run_paths(run: CompletedRun) -> tuple[Path, ...]:
    names = {
        "run.json",
        "completion.json",
        "source.json",
        "preprocessing.json",
        "profile.json",
        "frame-evidence.json",
        "activity-decisions.json",
        "phase1-admission.json",
        "phase2-evidence.json",
        "scene-usability.json",
        "candidates.json",
        "candidate-assessments.json",
        "window-responses.json",
        "prompt-envelopes.json",
        "decision-trace.json",
        "pipeline-status.json",
    }
    if run.manifest is not None and "artifacts" in run.manifest:
        names.update(
            name
            for name in run.manifest["artifacts"]
            if isinstance(name, str) and not name.lower().endswith(".eaf")
        )
    names.update(name + ".gz" for name in tuple(names) if name.endswith(".json"))
    return (run.path, *(run.path.parent / name for name in names))


def eaf_destinations(run: CompletedRun, output: Path) -> tuple[Path, ...]:
    if output.suffix.lower() != ".eaf":
        raise ValueError("optional EAF output must use the .eaf extension")
    if (
        run.manifest is not None
        and "artifacts" in run.manifest
        and output.parent.resolve() == run.path.parent
    ):
        return (output,)
    return (output, output.with_name(output.name + ".provenance.json"))


def export_loaded_run(
    run: CompletedRun,
    output: Path,
    *,
    tier_id: str = "G-Units",
    overwrite: bool = False,
) -> Path:
    targets = eaf_destinations(run, output)
    check_outputs(targets, protected=protected_run_paths(run), overwrite=overwrite)
    export_eaf_document(run.document, output, tier_id, overwrite=overwrite)
    eaf_digest = sha256(output.read_bytes()).hexdigest()
    if len(targets) == 1:
        assert run.manifest is not None
        manifest = dict(run.manifest)
        manifest["artifacts"] = list(manifest["artifacts"])
        manifest["artifact_sha256"] = dict(manifest["artifact_sha256"])
        if output.name not in manifest["artifacts"]:
            manifest["artifacts"].append(output.name)
        manifest["artifact_sha256"][output.name] = eaf_digest
        write_json(run.path.parent / "run.json", manifest)
    else:
        write_json(
            targets[1],
            {
                "schema_version": "0.2.0",
                "analysis_run_id": run.document["run_id"],
                "analysis_sha256": run.sha256,
                "eaf_sha256": eaf_digest,
                "projection": "one-tier presence-only G-unit EAF",
                "decision_basis": json.loads(ET.parse(output).findtext("./HEADER/PROPERTY[@NAME='decision_basis']") or "{}"),
                "tier_id": tier_id,
            },
        )
    return output.resolve()


def export_completed_run(
    path: str | Path,
    output: str | Path | None = None,
    *,
    tier_id: str = "G-Units",
    overwrite: bool = False,
) -> Path:
    run = load_completed_run(path)
    target = Path(output) if output is not None else run.path.with_suffix(".eaf")
    return export_loaded_run(run, target, tier_id=tier_id, overwrite=overwrite)
