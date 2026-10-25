# SPDX-License-Identifier: Apache-2.0
"""Deterministic selection and export of positive visual representatives."""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence
from uuid import uuid4
from ..rights import read_rights_sidecar, rights_sidecar_path

SELECTION_POLICY_VERSION = "stratified-positive-representatives-v0.1"
EXPORT_SCHEMA_VERSION = "representative-image-export-v0.2"


@dataclass(frozen=True, slots=True)
class PositiveRepresentative:
    """One positive record selected as a visual representative."""

    sample_index: int
    positive_label: str
    probability: float | None
    view: str | None
    category: str | None
    rank_within_stratum: int | None
    rationale: str
    metadata: Mapping[str, Any]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_source_path(source: Path, source_root: Path) -> str:
    resolved_source = source.resolve()
    resolved_root = source_root.resolve()
    if not resolved_source.is_relative_to(resolved_root):
        raise ValueError("representative source frame is outside source_root")
    return resolved_source.relative_to(resolved_root).as_posix()


def _validated_frame(
    selection: PositiveRepresentative,
    frame_by_index: Mapping[int, Mapping[str, Any]],
    *,
    source_root: Path,
) -> tuple[Path, str, int, int]:
    frame = frame_by_index.get(selection.sample_index)
    if not isinstance(frame, Mapping):
        raise ValueError("representative source frame is missing")
    declared_index = frame.get("sample_index", selection.sample_index)
    if declared_index != selection.sample_index:
        raise ValueError("representative source frame index does not match")
    source_value = frame.get("path")
    if not isinstance(source_value, (str, Path)):
        raise ValueError("representative source frame path is invalid")
    source = Path(source_value)
    if not source.is_absolute():
        source = source_root / source
    if not source.is_file() or source.suffix.casefold() not in {".jpg", ".jpeg"}:
        raise ValueError("representative source frame must be an existing JPEG")
    _relative_source_path(source, source_root)
    declared_digest = frame.get("sha256")
    if (
        not isinstance(declared_digest, str)
        or len(declared_digest) != 64
        or _file_sha256(source) != declared_digest
    ):
        raise ValueError("representative source frame digest does not match")
    start_us = frame.get("start_us")
    end_us = frame.get("end_us")
    if type(start_us) is not int or type(end_us) is not int or start_us < 0 or (end_us <= start_us):
        raise ValueError("representative source frame timestamps are invalid")
    return (source, declared_digest, start_us, end_us)


def export_positive_representatives(
    selections: Sequence[PositiveRepresentative],
    *,
    frame_by_index: Mapping[int, Mapping[str, Any]],
    destination: str | Path,
    source_root: str | Path,
    export_metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Copy selected JPEGs and rights companions into a traceable export.

    Only helper-managed ``sample-*.jpg`` files, their rights companions, and
    ``manifest.json`` participate in replacement. Other files in the requested
    directory are left untouched.
    """
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be Boolean")
    output = Path(destination)
    root = Path(source_root)
    if not root.is_dir():
        raise ValueError("source_root must be an existing directory")
    metadata = {} if export_metadata is None else export_metadata
    if not isinstance(metadata, Mapping):
        raise ValueError("export_metadata must be an object")
    ordered = list(selections)
    if any((not isinstance(item, PositiveRepresentative) for item in ordered)):
        raise ValueError("selections must contain PositiveRepresentative values")
    if [item.sample_index for item in ordered] != sorted({item.sample_index for item in ordered}):
        raise ValueError("selections must be unique and in sample-index order")
    json.dumps(
        {"metadata": metadata, "selection_metadata": [item.metadata for item in ordered]},
        allow_nan=False,
    )
    validated = [
        (selection, *_validated_frame(selection, frame_by_index, source_root=root))
        for selection in ordered
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and (not output.is_dir()):
        raise ValueError("representative export destination must be a directory")
    managed_existing = (
        []
        if not output.exists()
        else [
            *output.glob("sample-*.jpg"),
            *output.glob("sample-*.jpeg"),
            *output.glob("sample-*.jpg.rights.json"),
            *output.glob("sample-*.jpeg.rights.json"),
            *(path for path in [output / "manifest.json"] if path.exists()),
        ]
    )
    managed_existing = sorted(set(managed_existing), key=lambda path: path.name)
    if managed_existing and (not overwrite):
        raise FileExistsError(managed_existing[0])
    with TemporaryDirectory(prefix=".representative-export-", dir=output.parent) as staging_text:
        staging = Path(staging_text)
        manifest_records: list[dict[str, Any]] = []
        for selection, source, digest, start_us, end_us in validated:
            filename = f"sample-{selection.sample_index:08d}.jpg"
            target = staging / filename
            if (output / filename).resolve() == source.resolve():
                raise ValueError("representative export may not overwrite its source")
            shutil.copyfile(source, target)
            if _file_sha256(target) != digest:
                raise ValueError("copied representative frame digest does not match")
            source_sidecar = rights_sidecar_path(source)
            sidecar_record: dict[str, Any] | None = None
            if source_sidecar.exists():
                read_rights_sidecar(source)
                target_sidecar = rights_sidecar_path(target)
                shutil.copyfile(source_sidecar, target_sidecar)
                sidecar_record = {
                    "filename": target_sidecar.name,
                    "source_path": _relative_source_path(source_sidecar, root),
                    "sha256": _file_sha256(target_sidecar),
                }
            manifest_records.append(
                {
                    "sample_index": selection.sample_index,
                    "start_us": start_us,
                    "end_us": end_us,
                    "center_us": start_us + (end_us - start_us) // 2,
                    "filename": filename,
                    "sha256": digest,
                    "source_path": _relative_source_path(source, root),
                    "source_sha256": digest,
                    "positive_label": selection.positive_label,
                    "probability": selection.probability,
                    "view": selection.view,
                    "category": selection.category,
                    "rank_within_stratum": selection.rank_within_stratum,
                    "representative_rationale": selection.rationale,
                    "metadata": deepcopy(dict(selection.metadata)),
                    "rights_sidecar": sidecar_record,
                }
            )
        manifest = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "selection_policy_version": SELECTION_POLICY_VERSION,
            "representative_count": len(manifest_records),
            "metadata": deepcopy(dict(metadata)),
            "representatives": manifest_records,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        output.mkdir(parents=True, exist_ok=True)
        token = uuid4().hex
        backups: list[tuple[Path, Path]] = []
        committed: list[Path] = []
        try:
            for existing in managed_existing:
                backup = output / f".{existing.name}.{token}.backup"
                existing.replace(backup)
                backups.append((existing, backup))
            for staged in sorted(staging.iterdir(), key=lambda path: path.name):
                target = output / staged.name
                if target.exists():
                    raise FileExistsError(target)
                staged.replace(target)
                committed.append(target)
        except Exception:
            for target in committed:
                target.unlink(missing_ok=True)
            for original, backup in reversed(backups):
                if backup.exists():
                    backup.replace(original)
            raise
        else:
            for _, backup in backups:
                backup.unlink(missing_ok=True)
    return output / "manifest.json"
