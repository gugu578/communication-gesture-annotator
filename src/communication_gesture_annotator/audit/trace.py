# SPDX-License-Identifier: Apache-2.0
"""Sanitized event logs and sensitive, content-addressed model traces."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from hashlib import sha256
import json
import logging
from pathlib import Path
import re
from threading import Lock
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

TRACE_SCHEMA_VERSION = "0.1.0"
_LOGGER = logging.getLogger(__name__)
_REDACTED = "<redacted>"
_SENSITIVE_KEYS = frozenset(
    {
        "access_token",
        "api-key",
        "api_key",
        "apikey",
        "auth_token",
        "authorization",
        "bearer_token",
        "cookie",
        "credential",
        "credentials",
        "key",
        "password",
        "proxy-authorization",
        "refresh_token",
        "secret",
        "token",
        "user_name",
        "username",
        "x-api-key",
    }
)
_SENSITIVE_QUERY_KEYS = frozenset(
    {"access_token", "api-key", "api_key", "apikey", "key", "password", "secret", "token"}
)
_AUTH_PATTERN = re.compile("(?i)\\b(?:bearer|basic)\\s+[A-Za-z0-9._~+/=-]+")
_API_KEY_PATTERN = re.compile("\\bsk-[A-Za-z0-9_-]{8,}\\b")
_DATA_URL_PATTERN = re.compile("(?is)data:[^,;\\s]+(?:;[^,\\s]+)*;base64,[A-Za-z0-9+/=\\s]+")
_BASE64_PATTERN = re.compile("^[A-Za-z0-9+/]+={0,2}$")
_CONSOLE_PROGRESS_KEYS = (
    "planned_calls",
    "cached_calls_available",
    "planned_uncached_calls",
    "completed_calls",
    "remaining_calls",
    "reused_calls",
    "model_request_count",
    "target_frames",
    "history_frames",
    "future_frames",
    "completed_stage",
    "request_id",
    "subject_id",
    "candidate_id",
    "window_id",
    "sample_range",
    "range",
    "canonical_range_us",
    "mode",
    "source_sample_count",
    "sample_count",
    "frame_count",
    "analysis_rate_hz",
    "duration_us",
    "audio_supplied",
    "image_count",
    "segment_count",
    "reduction_level",
    "scale_percent",
    "input_profile_count",
    "selected_subject_count",
    "window_count",
    "candidate_count",
    "g_unit_count",
    "artifact_count",
    "latency_seconds",
    "wall_seconds",
    "completed_count",
    "total_count",
    "repaired",
    "contract_normalized",
    "normalization_event_count",
    "evidence_scope_policy_version",
    "direct_evidence_sample_count",
    "inherited_evidence_sample_count",
    "allowed_evidence_sample_count",
    "error_type",
    "semantic_attempt",
    "transport_attempt",
    "retry_count",
    "http_status",
    "category",
    "retry_backoff_seconds",
    "cooldown_seconds",
    "reused",
)
_CONSOLE_IDENTIFIER_KEYS = frozenset(
    {
        "completed_stage",
        "request_id",
        "subject_id",
        "candidate_id",
        "window_id",
        "mode",
        "evidence_scope_policy_version",
        "error_type",
        "category",
    }
)
_CONSOLE_INTEGER_KEYS = frozenset(
    {
        "planned_calls",
        "cached_calls_available",
        "planned_uncached_calls",
        "completed_calls",
        "remaining_calls",
        "reused_calls",
        "model_request_count",
        "target_frames",
        "history_frames",
        "future_frames",
        "completed_count",
        "total_count",
        "source_sample_count",
        "sample_count",
        "frame_count",
        "analysis_rate_hz",
        "image_count",
        "segment_count",
        "reduction_level",
        "scale_percent",
        "input_profile_count",
        "selected_subject_count",
        "window_count",
        "candidate_count",
        "g_unit_count",
        "artifact_count",
        "normalization_event_count",
        "direct_evidence_sample_count",
        "inherited_evidence_sample_count",
        "allowed_evidence_sample_count",
        "semantic_attempt",
        "transport_attempt",
        "retry_count",
        "http_status",
    }
)
_CONSOLE_IDENTIFIER_PATTERN = re.compile("^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_INFO_PROGRESS_EVENTS = frozenset(
    {
        ("pipeline", "analysis"),
        ("pipeline", "requested_stop"),
        ("phase1", "speaker_scene_admission"),
        *(
            ("preprocessing", event)
            for event in (
                "media_proxy",
                "analysis_frames",
                "whole_body_pose",
                "body_relative_pose_cues",
                "pose_conditioning",
            )
        ),
    }
)


def _console_progress_value(key: str, value: Any) -> str:
    if key in _CONSOLE_IDENTIFIER_KEYS:
        if isinstance(value, str) and _CONSOLE_IDENTIFIER_PATTERN.fullmatch(value):
            return value
        return "<invalid>"
    if key in _CONSOLE_INTEGER_KEYS:
        return str(value) if type(value) is int and value >= 0 else "<invalid>"
    if (
        key in {"sample_range", "range"}
        and isinstance(value, list)
        and (len(value) == 2)
        and all((type(item) is int for item in value))
    ):
        return f"[{value[0]},{value[1]})"
    if key in {"sample_range", "range"}:
        return "<invalid>"
    if (
        key == "canonical_range_us"
        and isinstance(value, list)
        and (len(value) == 2)
        and all((type(item) is int for item in value))
    ):
        return f"[{int(value[0]) / 1000000:.3f}s,{int(value[1]) / 1000000:.3f}s)"
    if key == "canonical_range_us":
        return "<invalid>"
    if key == "duration_us" and type(value) is int and (value >= 0):
        return f"{value / 1000000:.3f}s"
    if key == "duration_us":
        return "<invalid>"
    if key in {
        "latency_seconds",
        "wall_seconds",
        "retry_backoff_seconds",
        "cooldown_seconds",
    } and isinstance(value, (int, float)):
        return f"{value:.3f}s"
    if key in {"latency_seconds", "wall_seconds", "retry_backoff_seconds", "cooldown_seconds"}:
        return "<invalid>"
    if key in {"audio_supplied", "repaired", "contract_normalized", "reused"} and isinstance(
        value, bool
    ):
        return str(value).lower()
    return "<invalid>"


def _log_console_progress(record: Mapping[str, Any]) -> None:
    """Render only allowlisted operational metadata from a sanitized event."""
    details = record["details"]
    console_keys = _CONSOLE_PROGRESS_KEYS
    if record["event"] == "step":
        omitted = (
            {"completed_calls", "remaining_calls", "reused_calls"}
            if record["status"] == "started"
            else {"target_frames", "history_frames", "future_frames"}
        )
        if record["status"] == "completed":
            omitted.add("remaining_calls")
        console_keys = tuple((key for key in console_keys if key not in omitted))
    safe_details = " ".join(
        (
            f"{key}={_console_progress_value(key, details[key])}"
            for key in console_keys
            if key in details and details[key] is not None
        )
    )

    def label(value: Any) -> str:
        if isinstance(value, str) and _CONSOLE_IDENTIFIER_PATTERN.fullmatch(value):
            return value
        return "<invalid>"

    stage = label(record["stage"])
    event = label(record["event"])
    status = label(record["status"])
    message = f"stage={stage} event={event} status={status}"
    if safe_details:
        message = f"{message} {safe_details}"
    if status == "failed":
        _LOGGER.error(message)
    elif event == "transport_retry" and status == "scheduled":
        _LOGGER.warning(message)
    elif event == "spatial_png_size_recovery" and status == "started":
        _LOGGER.warning(message)
    elif event == "step" or (stage, event) in _INFO_PROGRESS_EVENTS:
        _LOGGER.info(message)
    else:
        _LOGGER.debug(message)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": sha256(value).hexdigest()}
    return value


def _looks_like_base64(value: str) -> bool:
    compact = "".join(value.split())
    return (
        len(compact) >= 128
        and len(compact) % 4 == 0
        and (_BASE64_PATTERN.fullmatch(compact) is not None)
    )


def _sanitize_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return value
    try:
        hostname = parsed.hostname or ""
        port = parsed.port
    except ValueError:
        return "<redacted-invalid-url>"
    if ":" in hostname and (not hostname.startswith("[")):
        hostname = f"[{hostname}]"
    netloc = hostname
    if port is not None:
        netloc += f":{port}"
    query = urlencode(
        [
            (key, _REDACTED if key.casefold() in _SENSITIVE_QUERY_KEYS else item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        ]
    )
    return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))


def _sanitize_string(value: str) -> str:
    if _DATA_URL_PATTERN.search(value) or _looks_like_base64(value):
        return "<redacted-binary-payload>"
    sanitized = _AUTH_PATTERN.sub("<redacted-authorization>", value)
    sanitized = _API_KEY_PATTERN.sub("<redacted-api-key>", sanitized)
    if sanitized.startswith(("http://", "https://")):
        sanitized = _sanitize_url(sanitized)
    return sanitized


def _sanitize(value: Any) -> Any:
    value = _json_value(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized = key.casefold().replace(" ", "_")
            if (
                normalized in _SENSITIVE_KEYS
                or normalized.endswith(("-key", "_key"))
                or normalized.endswith(("-token", "_token"))
                or any(
                    (
                        fragment in normalized
                        for fragment in (
                            "api_key",
                            "authorization",
                            "password",
                            "secret",
                            "credential",
                        )
                    )
                )
            ):
                result[key] = _REDACTED
            else:
                result[key] = _sanitize(item)
        return result
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return _sanitize_string(value)
    return value


def _encoded_json(value: Any) -> bytes:
    return (
        json.dumps(
            _json_value(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256(payload).hexdigest()


def _artifact_payload(value: Any, *, raw: bool = False) -> tuple[str, bytes]:
    if raw and isinstance(value, bytes):
        return ("bin", value)
    if raw and isinstance(value, str):
        return ("txt", value.encode("utf-8"))
    return ("json", _encoded_json(value))


class TraceWriter:
    """Write one sanitized JSONL event stream and separate sensitive artifacts."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        overwrite: bool = False,
        append_exchanges: bool = False,
        storage_mode: str = "debug",
    ) -> None:
        if storage_mode not in {"compact", "debug"}:
            raise ValueError("storage mode must be compact or debug")
        self.storage_mode = storage_mode
        if overwrite and append_exchanges:
            raise ValueError("overwrite and append_exchanges are mutually exclusive")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.event_log_path = self.run_dir / "run-events.jsonl"
        self.exchange_dir = self.run_dir / "model-traces"
        self._lock = Lock()
        self._overwrite = overwrite
        self._append_exchanges = append_exchanges
        if overwrite:
            self.event_log_path.unlink(missing_ok=True)
            self._clear_owned_exchanges()

    def _clear_owned_exchanges(self) -> None:
        """Remove only exchange directories created by this writer.

        Unknown files and directories are deliberately preserved. This makes a
        forced rerun discard stale sensitive model exchanges without treating
        the run directory as a generally disposable location.
        """
        if not self.exchange_dir.is_dir():
            return
        owned_exchange = re.compile("^exchange-[0-9a-f]{24}(?:-attempt-[0-9]{4})?$")
        owned_staging = re.compile(
            "^\\.exchange-[0-9a-f]{24}(?:-attempt-[0-9]{4})?\\.[0-9a-f]{32}\\.tmp$"
        )
        owned_targets = [
            target
            for target in self.exchange_dir.iterdir()
            if owned_exchange.fullmatch(target.name) or owned_staging.fullmatch(target.name)
        ]
        for target in owned_targets:
            is_junction = getattr(target, "is_junction", lambda: False)()
            if target.is_symlink() or is_junction or (not target.is_dir()):
                raise ValueError("owned model-trace path is not a directory")
            for child in target.iterdir():
                if not child.is_file():
                    raise ValueError("model trace contains an unexpected nested path")
        for target in owned_targets:
            for child in target.iterdir():
                child.unlink()
            target.rmdir()

    def log_event(
        self, stage: str, event: str, status: str, details: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Append a timestamped event after recursively redacting sensitive values."""
        if any(
            (not isinstance(value, str) or not value.strip() for value in (stage, event, status))
        ):
            raise ValueError("trace stage, event, and status must be non-empty strings")
        if details is not None and (not isinstance(details, Mapping)):
            raise ValueError("trace event details must be an object or null")
        record = {
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "timestamp": datetime.now(UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "stage": stage,
            "event": event,
            "status": status,
            "details": _sanitize(details or {}),
        }
        line = (
            json.dumps(
                record, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
            + b"\n"
        )
        with self._lock:
            with self.event_log_path.open("ab") as handle:
                handle.write(line)
                handle.flush()
        _log_console_progress(record)
        return record

    def write_model_exchange(
        self,
        request_id: str,
        sanitized_request: Any,
        raw_response: Any,
        parsed_response: Any,
        validation: Any,
        image_refs: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Persist a local model exchange and return content-addressed references.

        Requests are defensively sanitized again. Raw and parsed responses live
        only in the sensitive ``model-traces`` area; their contents are never
        copied into the ordinary event stream.
        """
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(image_refs, Sequence) or isinstance(image_refs, (str, bytes)):
            raise ValueError("image_refs must be an array")
        if any((not isinstance(item, Mapping) for item in image_refs)):
            raise ValueError("every image reference must be an object")
        exchange_key = sha256(request_id.encode("utf-8")).hexdigest()[:24]
        target = self.exchange_dir / f"exchange-{exchange_key}"
        physical_attempt = 1
        if target.exists():
            if self._append_exchanges:
                physical_attempt = 2
                while True:
                    candidate = (
                        self.exchange_dir
                        / f"exchange-{exchange_key}-attempt-{physical_attempt:04d}"
                    )
                    if not candidate.exists():
                        target = candidate
                        break
                    physical_attempt += 1
            elif not self._overwrite:
                raise FileExistsError(f"model trace already exists for request {request_id!r}")
            else:
                for child in target.iterdir():
                    if not child.is_file():
                        raise ValueError("model trace contains an unexpected nested path")
                    child.unlink()
                target.rmdir()
        staging = self.exchange_dir / f".exchange-{exchange_key}.{uuid4().hex}.tmp"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            values = {
                "request": (_sanitize(sanitized_request), False),
                "raw-response": (raw_response, True),
                "parsed-response": (parsed_response, False),
                "validation": (validation, False),
                "image-references": ([_sanitize(item) for item in image_refs], False),
            }
            if self.storage_mode == "compact":
                for name in ("request", "raw-response", "image-references"):
                    value, raw = values[name]
                    _, payload = _artifact_payload(value, raw=raw)
                    values[name] = (
                        {
                            "storage_policy": "compact-resumable-storage-v0.1",
                            "content_retained": False,
                            "original_byte_size": len(payload),
                            "original_sha256": sha256(payload).hexdigest(),
                        },
                        False,
                    )
            artifacts: dict[str, dict[str, Any]] = {}
            for name, (value, raw) in values.items():
                suffix, payload = _artifact_payload(value, raw=raw)
                path = staging / f"{name}.{suffix}"
                digest = _atomic_write(path, payload)
                artifacts[name.replace("-", "_")] = {
                    "path": (target.relative_to(self.run_dir) / path.name).as_posix(),
                    "sha256": digest,
                    "byte_size": len(payload),
                }
            manifest = {
                "trace_schema_version": TRACE_SCHEMA_VERSION,
                "request_id": request_id,
                "physical_attempt": physical_attempt,
                "artifacts": artifacts,
            }
            manifest_payload = _encoded_json(manifest)
            manifest_digest = _atomic_write(staging / "manifest.json", manifest_payload)
            target.parent.mkdir(parents=True, exist_ok=True)
            staging.replace(target)
        except Exception:
            if staging.exists():
                for child in staging.iterdir():
                    child.unlink(missing_ok=True)
                staging.rmdir()
            raise
        result = {
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "request_id": request_id,
            "physical_attempt": physical_attempt,
            "manifest": {
                "path": (target.relative_to(self.run_dir) / "manifest.json").as_posix(),
                "sha256": manifest_digest,
                "byte_size": len(manifest_payload),
            },
            "artifacts": artifacts,
        }
        self.log_event(
            "model",
            "exchange_trace_written",
            "ok",
            {"request_id": request_id, "manifest": result["manifest"], "artifacts": artifacts},
        )
        return result
