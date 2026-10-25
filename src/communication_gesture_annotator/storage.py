# SPDX-License-Identifier: Apache-2.0
"""Operational storage policy; never part of model-input configuration."""
from __future__ import annotations

import gzip
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TextIO

VERSION = "minimal-resumable-storage-v0.2"
_MODE: ContextVar[str] = ContextVar("run_storage_mode", default="debug")


def compact_enabled() -> bool:
    return _MODE.get() in {"compact", "minimal"}


def minimal_enabled() -> bool:
    return _MODE.get() == "minimal"


def current_mode() -> str:
    return _MODE.get()


@contextmanager
def storage_mode(mode: str) -> Iterator[None]:
    if mode not in {"compact", "minimal", "debug"}:
        raise ValueError("storage mode must be compact, minimal or debug")
    token = _MODE.set(mode)
    try:
        yield
    finally:
        _MODE.reset(token)


def data_path(path: Path) -> Path:
    """Find an existing stream without changing a saved bundle's encoding."""
    compressed = path.with_name(path.name + ".gz")
    if path.is_file() and compressed.is_file():
        raise ValueError(f"ambiguous plain/compressed cache: {path.name}")
    return compressed if compressed.is_file() else path


def output_data_path(path: Path) -> Path:
    return path.with_name(path.name + ".gz") if compact_enabled() else path


def phase_result_path(path: Path) -> Path:
    """Compress large new phase results; preserve existing encoding on resume.

    Retaining an existing filename avoids invalidating lifecycle receipts if
    storage mode changes during resume. Canonical annotations stay plain JSON.
    """
    existing = data_path(path)
    if existing.exists():
        return existing
    return path.with_name(path.name + ".gz") if minimal_enabled() else path


@contextmanager
def open_data(path: Path, mode: str = "r") -> Iterator[TextIO]:
    """Stream lossless JSONL; fixed gzip headers keep regeneration deterministic."""
    if path.suffix != ".gz":
        with path.open(mode, encoding="utf-8", newline="\n") as handle:
            yield handle
    elif mode == "r":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield handle
    elif mode == "w":
        from io import TextIOWrapper
        with path.open("wb") as raw, gzip.GzipFile(
            fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=3
        ) as encoded, TextIOWrapper(encoded, encoding="utf-8", newline="\n") as handle:
            yield handle
    else:
        raise ValueError("unsupported cache stream mode")


def relative_data_path(path: Path, relative: str) -> str:
    if path.name.endswith(".gz.rights.json") and not relative.endswith(".gz.rights.json"):
        return relative.removesuffix(".rights.json") + ".gz.rights.json"
    return relative + ".gz" if path.suffix == ".gz" and not relative.endswith(".gz") else relative


def overlay_omitted(manifest: object) -> bool:
    return isinstance(manifest, dict) and manifest.get("diagnostic_overlay") == "omitted-compact-storage-v0.1"


def release_image(path: Path, root: Path) -> None:
    """Remove only an owned regenerable PNG and its two known sidecars."""
    resolved_root = root.resolve()
    absolute = path.absolute()
    if ".." in absolute.parts or not absolute.resolve().is_relative_to(resolved_root):
        raise ValueError("temporary image is outside its cache root")
    for parent in (absolute, *absolute.parents):
        if parent == resolved_root.parent:
            break
        if parent.is_symlink() or getattr(parent, "is_junction", lambda: False)():
            raise ValueError("temporary image cache may not contain links")
    if not re.fullmatch(r"(?:[0-9a-f]{64}|trace-[0-9a-f]{24})\.png", path.name):
        raise ValueError("unrecognized temporary image filename")
    # Invalidate the cache first. An interruption must not leave a manifest
    # advertising a PNG which was already removed; absent manifests regenerate.
    targets = (path.with_suffix(".json"), path, path.with_name(path.name + ".rights.json"))
    for target in targets:
        if target.is_symlink() or getattr(target, "is_junction", lambda: False)():
            raise ValueError("temporary image sidecar may not be a link")
    for target in targets:
        target.unlink(missing_ok=True)


def release_cache_files(root: Path, paths: list[Path]) -> None:
    """Unlink an explicit list of owned files, never follow cache links.

    Callers provide only known regenerable filenames, with manifests first.
    Validate the entire list before removing anything; never recursively delete.
    """
    absolute_root = root.absolute()
    for path in [absolute_root, *paths]:
        absolute = path.absolute()
        if ".." in absolute.parts or not absolute.is_relative_to(absolute_root):
            raise ValueError("temporary cache file is outside its cache root")
        for parent in (absolute, *absolute.parents):
            if parent.is_symlink() or getattr(parent, "is_junction", lambda: False)():
                raise ValueError("temporary cache may not contain links")
        if absolute != absolute_root and absolute.exists() and not absolute.is_file():
            raise ValueError("temporary cache entry must be a file")
    for path in paths:
        path.unlink(missing_ok=True)


def release_phase1_images(run_root: Path) -> None:
    """Release completed Phase1 sheets; inputs/checkpoints can regenerate them."""
    if not minimal_enabled():
        return
    root = run_root / "model-inputs"
    release_cache_files(root, [])
    if not root.is_dir():
        return
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        release_cache_files(directory, [])
        images = sorted(p for p in directory.iterdir() if re.fullmatch(
            r"contact-sheet-[0-9]+-[0-9a-f]+\.jpg(?:\.rights\.json)?", p.name))
        # The manifest is invalidated before its images disappear.
        if images:
            release_cache_files(root, [directory / "image-packages.json",
                directory / "image-packages.json.rights.json", directory / "rights-manifest.json", *images])
