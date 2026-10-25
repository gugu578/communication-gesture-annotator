# SPDX-License-Identifier: Apache-2.0
"""Check source identity before resume can replace derived media."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..export.json_export import read_json
from .ffmpeg import file_sha256


def _source_identity(
    value: Any, *, digest_key: str, size_key: str
) -> tuple[str, int | None] | None:
    if not isinstance(value, Mapping):
        raise ValueError("resume source provenance must be an object")
    digest = value.get(digest_key)
    if digest is None:
        return None
    size = value.get(size_key)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or (size is not None and (type(size) is not int or size <= 0))
    ):
        raise ValueError("resume source provenance has an invalid identity")
    return digest, size


def validate_resume_source(run_dir: Path, source_path: Path) -> None:
    """Reject another video without modifying the run or contacting a model.

    Durable stage and pose records take precedence over the regenerable media
    manifest. This also identifies the original video after an older version
    accidentally overwrote that manifest during an unsuccessful resume.
    Remaining lifecycle, configuration and artifact checks still run normally.
    """
    identities: list[tuple[str, int | None]] = []
    status_path = run_dir / "pipeline-status.json"
    if status_path.exists():
        identity = _source_identity(
            read_json(status_path), digest_key="source_sha256", size_key="source_byte_size"
        )
        if identity is not None:
            identities.append(identity)
    pose_path = run_dir / "pose" / "manifest.json"
    if pose_path.exists():
        pose = read_json(pose_path)
        if not isinstance(pose, Mapping):
            raise ValueError("resume pose provenance must be an object")
        identity = _source_identity(
            pose.get("input"), digest_key="source_sha256", size_key="source_byte_size"
        )
        if identity is not None:
            identities.append(identity)
    preprocessing_path = run_dir / "preprocessing.json"
    if not identities and preprocessing_path.exists():
        preprocessing = read_json(preprocessing_path)
        if not isinstance(preprocessing, Mapping):
            raise ValueError("resume preprocessing provenance must be an object")
        identity = _source_identity(
            preprocessing.get("source"), digest_key="sha256", size_key="byte_size"
        )
        if identity is not None:
            identities.append(identity)
    if not identities:
        # The existing lifecycle validator handles missing/unsupported runs.
        return
    if len({digest for digest, _ in identities}) != 1 or len(
        {size for _, size in identities if size is not None}
    ) > 1:
        raise ValueError("resume source identities disagree between saved checkpoints; run files were not changed")
    initial_stat = source_path.stat()
    digest = file_sha256(source_path)
    final_stat = source_path.stat()
    if (initial_stat.st_size, initial_stat.st_mtime_ns) != (
        final_stat.st_size, final_stat.st_mtime_ns
    ):
        raise ValueError("source video changed while resume identity was checked; run files were not changed")
    if any(
        digest != expected_digest
        or (expected_size is not None and initial_stat.st_size != expected_size)
        for expected_digest, expected_size in identities
    ):
        raise ValueError(
            "--resume source video does not match the original run; use the original "
            "video or a new --run-dir for another video. Run files were not changed"
        )
