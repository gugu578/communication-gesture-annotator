# SPDX-License-Identifier: Apache-2.0
"""Task-specific Open WebUI transport for Research Stage 2 experiments.

This module deliberately does not expose a generic completion gateway. The
only public operation accepts the validated Research Stage 2 model contract and returns
a structured domain payload with non-secret transport provenance.
"""

from __future__ import annotations
from base64 import b64encode
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from http.client import HTTPException, HTTPMessage, IncompleteRead, RemoteDisconnected
import ipaddress
import json
import mimetypes
import os
from pathlib import Path
import re
import socket
import ssl
from threading import Lock
from time import perf_counter, sleep
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener
from ..config import InferenceConfig, OpenWebUIConfig
from ..pipeline.llm_contracts import (
    ModelCallResult,
    ModelContextLengthError,
    PROFILE_NORMALIZATION_VERSION,
    TaskContractViolation,
    validate_profile_normalization_events,
)
from ..audit import TraceWriter
from .image_recovery import (
    ImageRecovery, POLICY_VERSION as IMAGE_RECOVERY_POLICY_VERSION, SIZE_POLICY_VERSION,
    SPATIAL_SIZE_POLICY_VERSION, SPATIAL_SIZE_PERCENTAGES, recovery_trace_context,
)


class OpenWebUIError(RuntimeError):
    """A safe, non-secret Open WebUI transport or response failure."""


class OpenWebUIContextLengthError(OpenWebUIError, ModelContextLengthError):
    """A traced context-limit rejection eligible for evidence-window retry."""


class _HTTPStatusError(OpenWebUIError):
    """A bounded HTTP failure whose body is safe only for sensitive tracing."""

    def __init__(
        self,
        *,
        path: str,
        status: int,
        category: str,
        response_body: bytes,
        response_body_truncated: bool,
    ) -> None:
        self.path = path
        self.status = status
        self.category = category
        self.response_body = response_body
        self.response_body_bytes = len(response_body)
        self.response_body_sha256 = sha256(response_body).hexdigest()
        self.response_body_truncated = response_body_truncated
        self.transport_attempts: tuple[dict[str, Any], ...] = ()
        super().__init__(
            f"Open WebUI {path} returned HTTP {status} (category={category}, response_bytes={self.response_body_bytes}, response_sha256={self.response_body_sha256}, response_truncated={str(response_body_truncated).lower()})"
        )


class _TransportRequestError(OpenWebUIError):
    """Content-free diagnostics for an unavailable complete response body."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None,
        category: str,
        phase: str,
        transport_attempts: Sequence[Mapping[str, Any]],
    ) -> None:
        self.status = status
        self.category = category
        self.phase = phase
        self.transport_attempts = tuple((dict(item) for item in transport_attempts))
        self.response_body = b""
        self.response_body_bytes = 0
        self.response_body_sha256 = sha256(b"").hexdigest()
        self.response_body_truncated = False
        super().__init__(message)


class _ReasoningBudgetExhausted(OpenWebUIError):
    """The model used its output budget before emitting answer content."""


class _ResponseBudgetExhausted(OpenWebUIError):
    """The model used its output budget before completing answer content."""


class _TaskContractValidationError(OpenWebUIError):
    """A task failure carrying only allowlisted, content-free diagnostics."""

    def __init__(self, violation: TaskContractViolation) -> None:
        self.contract_failure_code = violation.code
        self.contract_failure_path = violation.path
        super().__init__(
            f"model JSON failed the task-specific response contract (contract_failure_code={violation.code}, contract_failure_path={violation.path})"
        )


class _NoRedirects(HTTPRedirectHandler):
    """Prevent an allowed local origin from redirecting media elsewhere."""

    def redirect_request(
        self,
        request: Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: HTTPMessage,
        new_url: str,
    ) -> None:
        return None


_IPV4_PRIVATE_NETWORKS = tuple(
    (ipaddress.ip_network(value) for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))
)
_IPV6_PRIVATE_NETWORK = ipaddress.ip_network("fc00::/7")
_HTTP_ERROR_CATEGORIES = frozenset(
    {
        "unsupported_json_schema",
        "vision_input_limit",
        "context_length",
        "request_too_large",
        "model_not_found",
        "backend_internal_error",
        "generic_http_error",
    }
)


def _classify_http_error(status: int, body: bytes) -> str:
    """Classify a bounded response without returning any server-supplied text."""
    text = body.decode("utf-8", errors="ignore").casefold()
    wrapped_internal_error = False
    if status == 400 and len(body) <= 256:
        try:
            wrapped_internal_error = json.loads(body) == {"detail": "Internal server error"}
        except (ValueError, UnicodeError):
            pass
    if status == 413:
        category = "request_too_large"
    elif wrapped_internal_error:
        category = "backend_internal_error"
    elif status in {400, 404} and any(
        (marker in text for marker in ("model not found", "model_not_found"))
    ):
        category = "model_not_found"
    elif any(
        (
            marker in text
            for marker in (
                "json_schema",
                "json schema",
                "response_format",
                "guided_json",
                "structured output",
            )
        )
    ):
        category = "unsupported_json_schema"
    elif any(
        (
            marker in text
            for marker in (
                "context length",
                "context_length",
                "context window",
                "maximum context",
                "max context",
                "num_ctx",
                "token limit",
                "too many tokens",
            )
        )
    ):
        category = "context_length"
    # Match image, vision and modality terminology used by inference servers.
    elif any((marker in text for marker in ("image", "vision", "modal"))) and any(
        (
            marker in text
            for marker in ("limit", "maximum", "too many", "too large", "exceed", "input length")
        )
    ):
        category = "vision_input_limit"
    elif any(
        (
            marker in text
            for marker in (
                "request too large",
                "payload too large",
                "body too large",
                "content too large",
                "request entity too large",
            )
        )
    ):
        category = "request_too_large"
    else:
        category = "generic_http_error"
    assert category in _HTTP_ERROR_CATEGORIES
    return category


_OPEN_WEBUI_ADAPTER_VERSION = "open-webui-research-v0.13"
_OPEN_WEBUI_ENDPOINT_MODE = "open-webui-chat-completions-v0.4"
_TRANSIENT_HTTP_RETRY_STATUSES = frozenset({502, 503, 504})
_COMPLETION_TIMEOUT_RETRY_POLICY_VERSION = "completion-timeout-retry-v0.1"
_COMPLETION_DISCONNECT_RETRY_POLICY_VERSION = "completion-disconnect-retry-v0.1"
_WRAPPED_INTERNAL_ERROR_RETRY_POLICY_VERSION = "wrapped-internal-error-retry-v0.1"
_EXPLICIT_RETRY_SCHEDULE_VERSION = "explicit-transport-retry-delays-v0.1"
_VLLM_XGRAMMAR_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset({"uniqueItems"})


def _provider_safe_response_schema(response_schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep wire copy without known xgrammar-incompatible keywords."""
    wire_schema = deepcopy(dict(response_schema))

    def remove_unsupported(value: Any) -> None:
        if isinstance(value, dict):
            for keyword in _VLLM_XGRAMMAR_UNSUPPORTED_SCHEMA_KEYWORDS:
                value.pop(keyword, None)
            for child in value.values():
                remove_unsupported(child)
        elif isinstance(value, list):
            for child in value:
                remove_unsupported(child)

    remove_unsupported(wire_schema)
    return wire_schema


def _local_response_validation_schema(
    *, task: str, response_schema: Mapping[str, Any], has_response_validator: bool
) -> Mapping[str, Any]:
    """Permit only the visibility view escape hatch owned by its validator.

    The provider still receives the exact recurring-view enum.  Locally, an
    otherwise bounded string may reach the Phase 0b task validator, which is
    solely responsible for replacing an unrecognized auxiliary view label with
    ``null`` and recording versioned normalization provenance.  No relaxation
    is applied without that task validator or to any other response field.
    """
    if task != "visibility" or not has_response_validator:
        return response_schema
    schema = deepcopy(dict(response_schema))
    try:
        view_schema = schema["properties"]["samples"]["items"]["properties"]["view"]
    except (KeyError, TypeError):
        return response_schema
    if not isinstance(view_schema, dict):
        return response_schema
    choices = view_schema.get("enum")
    if (
        not isinstance(choices, list)
        or None not in choices
        or any((item is not None and (not isinstance(item, str)) for item in choices))
    ):
        return response_schema
    view_schema.pop("enum")
    return schema


def _address_is_local_or_private(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if address.is_loopback:
        return True
    if isinstance(address, ipaddress.IPv4Address):
        return any((address in network for network in _IPV4_PRIVATE_NETWORKS))
    if address.ipv4_mapped is not None:
        return _address_is_local_or_private(address.ipv4_mapped)
    return address in _IPV6_PRIVATE_NETWORK


def _resolved_addresses(
    host: str, port: int
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    try:
        records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise OpenWebUIError("Open WebUI host could not be resolved") from exc
    addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    for record in records:
        try:
            addresses.add(ipaddress.ip_address(record[4][0].split("%", 1)[0]))
        except ValueError as exc:
            raise OpenWebUIError("Open WebUI host resolved to an invalid address") from exc
    if not addresses:
        raise OpenWebUIError("Open WebUI host resolved to no usable address")
    return tuple(sorted(addresses, key=str))


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not permitted: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def _load_strict_json(payload: bytes | str, label: str) -> Any:
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        return json.loads(
            text, parse_constant=_reject_json_constant, object_pairs_hook=_reject_duplicate_pairs
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise OpenWebUIError(f"{label} is not strict UTF-8 JSON") from exc


def _resolve_json_pointer(root: Mapping[str, Any], reference: str) -> Mapping[str, Any]:
    if not reference.startswith("#/"):
        raise OpenWebUIError("response schema contains an unsupported reference")
    current: Any = root
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or part not in current:
            raise OpenWebUIError("response schema contains an unresolved reference")
        current = current[part]
    if not isinstance(current, Mapping):
        raise OpenWebUIError("response schema reference does not identify an object")
    return current


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return type(value) is bool
    if expected == "integer":
        return type(value) is int
    if expected == "number":
        return isinstance(value, (int, float)) and (not isinstance(value, bool))
    raise OpenWebUIError(f"response schema uses unsupported type {expected!r}")


def _json_values_equal(left: Any, right: Any) -> bool:
    """Compare parsed JSON values using JSON rather than Python equality."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            (_json_values_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            (
                _json_values_equal(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)
            )
        )
    if isinstance(left, (int, float)) and (not isinstance(left, bool)):
        return isinstance(right, (int, float)) and (not isinstance(right, bool)) and (left == right)
    return type(left) is type(right) and left == right


def _validate_schema_value(
    value: Any, schema: Mapping[str, Any], root: Mapping[str, Any], path: str = "$"
) -> None:
    reference = schema.get("$ref")
    if reference is not None:
        if not isinstance(reference, str):
            raise OpenWebUIError("response schema $ref must be a string")
        _validate_schema_value(value, _resolve_json_pointer(root, reference), root, path)
        return
    expected = schema.get("type")
    expected_types = [expected] if isinstance(expected, str) else expected
    if expected_types is not None:
        if not isinstance(expected_types, list) or any(
            (not isinstance(item, str) for item in expected_types)
        ):
            raise OpenWebUIError("response schema type is invalid")
        if not any((_matches_type(value, item) for item in expected_types)):
            raise OpenWebUIError(f"model JSON {path} has the wrong type")
        if value is None:
            return
    choices = schema.get("enum")
    if choices is not None and value not in choices:
        raise OpenWebUIError(f"model JSON {path} is outside the permitted enum")
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise OpenWebUIError("response schema properties must be an object")
        required = schema.get("required", [])
        if not isinstance(required, list) or any((not isinstance(item, str) for item in required)):
            raise OpenWebUIError("response schema required must be a string array")
        missing = set(required) - set(value)
        if missing:
            raise OpenWebUIError(
                f"model JSON {path} is missing required members: {', '.join(sorted(missing))}"
            )
        unknown = set(value) - set(properties)
        if schema.get("additionalProperties") is False and unknown:
            raise OpenWebUIError(
                f"model JSON {path} has unexpected members: {', '.join(sorted(unknown))}"
            )
        for key, item in value.items():
            property_schema = properties.get(key)
            if isinstance(property_schema, Mapping):
                _validate_schema_value(item, property_schema, root, f"{path}.{key}")
    if isinstance(value, list):
        minimum_items = schema.get("minItems")
        if minimum_items is not None and len(value) < int(minimum_items):
            raise OpenWebUIError(f"model JSON {path} has too few items")
        maximum_items = schema.get("maxItems")
        if maximum_items is not None:
            if type(maximum_items) is not int or maximum_items < 0:
                raise OpenWebUIError("response schema maxItems must be a non-negative integer")
            if len(value) > maximum_items:
                raise OpenWebUIError(f"model JSON {path} has too many items")
        unique_items = schema.get("uniqueItems")
        if unique_items is not None and type(unique_items) is not bool:
            raise OpenWebUIError("response schema uniqueItems must be a boolean")
        if unique_items is True:
            for index, item in enumerate(value):
                if any((_json_values_equal(item, prior_item) for prior_item in value[:index])):
                    raise OpenWebUIError(f"model JSON {path} contains duplicate array items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_schema_value(item, item_schema, root, f"{path}[{index}]")
    if isinstance(value, str):
        minimum_length = schema.get("minLength")
        if minimum_length is not None:
            if type(minimum_length) is not int or minimum_length < 0:
                raise OpenWebUIError("response schema minLength must be a non-negative integer")
            if len(value) < minimum_length:
                raise OpenWebUIError(f"model JSON {path} is too short")
        maximum_length = schema.get("maxLength")
        if maximum_length is not None:
            if type(maximum_length) is not int or maximum_length < 0:
                raise OpenWebUIError("response schema maxLength must be a non-negative integer")
            if len(value) > maximum_length:
                raise OpenWebUIError(f"model JSON {path} is too long")
    if isinstance(value, (int, float)) and (not isinstance(value, bool)):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            raise OpenWebUIError(f"model JSON {path} is below its minimum")
        if maximum is not None and value > maximum:
            raise OpenWebUIError(f"model JSON {path} is above its maximum")


def _extract_message_text(response: Any) -> str:
    if not isinstance(response, Mapping):
        raise OpenWebUIError("Open WebUI completion response must be an object")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise OpenWebUIError("Open WebUI completion must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping) or not isinstance(choice.get("message"), Mapping):
        raise OpenWebUIError("Open WebUI completion choice has no message")
    message = choice["message"]
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if (
                not isinstance(block, Mapping)
                or block.get("type") not in {"text", "output_text"}
                or (not isinstance(block.get("text"), str))
            ):
                raise OpenWebUIError("Open WebUI completion has unsupported content blocks")
            parts.append(block["text"])
        if parts:
            return "".join(parts)
    reasoning = next(
        (
            message.get(name)
            for name in ("reasoning", "reasoning_content", "thinking")
            if isinstance(message.get(name), str) and message.get(name)
        ),
        None,
    )
    if choice.get("finish_reason") == "length" and reasoning is not None:
        raise _ReasoningBudgetExhausted(
            "model exhausted maximum_output_tokens in reasoning before producing JSON content; disable or reduce model thinking for this experiment"
        )
    raise OpenWebUIError("Open WebUI completion message has no text content")


def _completion_reasoning_observed(response: Any) -> bool:
    """Return whether a completion contains a non-empty reasoning channel."""
    if not isinstance(response, Mapping):
        return False
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return False
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return False
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return False
    return any(
        (
            isinstance(message.get(name), str) and bool(message.get(name))
            for name in ("reasoning", "reasoning_content", "thinking")
        )
    )


_KNOWN_FINISH_REASONS = frozenset(
    {"stop", "length", "content_filter", "tool_calls", "function_call"}
)


def _completion_finish_reason(response: Any) -> str | None:
    """Return an allowlisted finish reason suitable for validation metadata."""
    if not isinstance(response, Mapping):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return None
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return None
    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str):
        return None
    return finish_reason if finish_reason in _KNOWN_FINISH_REASONS else "other"


def _completion_attempt_metrics(response: Any) -> dict[str, Any]:
    """Return content-free response metrics safe for compact validation traces."""
    metrics: dict[str, Any] = {}
    finish_reason = _completion_finish_reason(response)
    if finish_reason is not None:
        metrics["finish_reason"] = finish_reason
    if not isinstance(response, Mapping):
        return metrics
    usage = response.get("usage")
    if isinstance(usage, Mapping):
        completion_tokens = usage.get("completion_tokens")
        if type(completion_tokens) is int and completion_tokens >= 0:
            metrics["completion_tokens"] = completion_tokens
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return metrics
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return metrics
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return metrics
    content = message.get("content")
    parts: list[str]
    if isinstance(content, str):
        parts = [content]
    elif isinstance(content, list) and all(
        (
            isinstance(block, Mapping)
            and block.get("type") in {"text", "output_text"}
            and isinstance(block.get("text"), str)
            for block in content
        )
    ):
        parts = [str(block["text"]) for block in content]
    else:
        return metrics
    joined = "".join(parts)
    metrics["content_characters"] = len(joined)
    metrics["content_sha256"] = sha256(joined.encode("utf-8", errors="surrogatepass")).hexdigest()
    return metrics


def _validation_attempt_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project an attempt into content-free validation metadata."""
    return {
        key: record[key]
        for key in (
            "attempt",
            "valid",
            "failure",
            "finish_reason",
            "completion_tokens",
            "content_characters",
            "content_sha256",
            "contract_failure_code",
            "contract_failure_path",
            "contract_normalization_count",
            "contract_normalization_policy_version",
        )
        if key in record
    }


def _contract_normalization_events(value: Any) -> tuple[dict[str, Any], ...]:
    """Copy trusted validator events without altering the canonical mapping."""
    raw_events = getattr(value, "normalization_events", ())
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes, bytearray)):
        raise OpenWebUIError("task validator returned invalid normalization provenance")
    policy_version = getattr(value, "normalization_policy_version", None)
    if policy_version is None and (not raw_events):
        return ()
    try:
        if policy_version == PROFILE_NORMALIZATION_VERSION:
            events = validate_profile_normalization_events(raw_events)
        elif policy_version == "exemplar-probability-normalization-v0.1":
            from ..pipeline.exemplar_normalization import validate_exemplar_normalization_events
            events = validate_exemplar_normalization_events(raw_events)
        else:
            raise ValueError("normalization policy is unsupported")
    except ValueError as exc:
        raise OpenWebUIError("task validator returned invalid normalization provenance") from exc
    return tuple(events)


def _contract_normalization_policy_version(value: Any) -> str | None:
    policy_version = getattr(value, "normalization_policy_version", None)
    if policy_version is None:
        return None
    if not isinstance(policy_version, str) or not policy_version:
        raise OpenWebUIError("task validator returned invalid normalization provenance")
    return policy_version


def _structured_payload_sha256(value: Mapping[str, Any]) -> str:
    """Hash a structured payload independently of the HTTP response envelope."""
    encoded = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _explicit_vision_capability(value: Any) -> bool | None:
    """Find only explicit capability declarations; never infer from a model name."""
    if not isinstance(value, Mapping):
        return None
    capabilities = value.get("capabilities")
    if isinstance(capabilities, list):
        lowered = {item.lower() for item in capabilities if isinstance(item, str)}
        if "vision" in lowered:
            return True
    elif isinstance(capabilities, Mapping):
        vision = capabilities.get("vision")
        if type(vision) is bool:
            return vision
    for key in ("info", "meta", "details", "model_info"):
        nested = value.get(key)
        explicit = _explicit_vision_capability(nested)
        if explicit is not None:
            return explicit
    return None


def _positive_integer(value: Any) -> int | None:
    if type(value) is int and value > 0:
        return value
    return None


def _ollama_declared_limits(value: Any) -> dict[str, int | None]:
    """Extract stable model limits from an Ollama ``/api/show`` document."""
    context_window_tokens: int | None = None
    vision_tokens_per_image: int | None = None
    default_context_window_tokens: int | None = None
    if not isinstance(value, Mapping):
        return {
            "context_window_tokens": None,
            "default_context_window_tokens": None,
            "vision_tokens_per_image": None,
        }
    model_info = value.get("model_info")
    if isinstance(model_info, Mapping):
        architecture = model_info.get("general.architecture")
        if isinstance(architecture, str) and architecture:
            context_window_tokens = _positive_integer(
                model_info.get(f"{architecture}.context_length")
            )
        if context_window_tokens is None:
            candidates = [
                parsed
                for key, item in model_info.items()
                if isinstance(key, str) and key.endswith(".context_length")
                if (parsed := _positive_integer(item)) is not None
            ]
            if candidates:
                context_window_tokens = max(candidates)
        image_token_candidates = [
            parsed
            for key, item in model_info.items()
            if isinstance(key, str) and key.endswith(".mm.tokens_per_image")
            if (parsed := _positive_integer(item)) is not None
        ]
        if image_token_candidates:
            vision_tokens_per_image = max(image_token_candidates)
    parameters = value.get("parameters")
    if isinstance(parameters, Mapping):
        default_context_window_tokens = _positive_integer(parameters.get("num_ctx"))
    elif isinstance(parameters, str):
        match = re.search("(?m)^\\s*num_ctx(?:\\s+|\\s*=\\s*)([0-9]+)\\s*$", parameters)
        if match is not None:
            default_context_window_tokens = _positive_integer(int(match.group(1)))
    return {
        "context_window_tokens": context_window_tokens,
        "default_context_window_tokens": default_context_window_tokens,
        "vision_tokens_per_image": vision_tokens_per_image,
    }


def _model_entry_context_limit(value: Any) -> int | None:
    """Read common Open WebUI context-limit metadata when Ollama show is absent."""
    if not isinstance(value, Mapping):
        return None
    candidates: list[int] = []
    for key in ("context_length", "context_window", "max_context_length", "max_model_len"):
        parsed = _positive_integer(value.get(key))
        if parsed is not None:
            candidates.append(parsed)
    for key in ("info", "meta", "details", "params"):
        nested = _model_entry_context_limit(value.get(key))
        if nested is not None:
            candidates.append(nested)
    return max(candidates) if candidates else None


def _message_instruction_characters(
    messages: Sequence[Mapping[str, Any]], response_schema: Mapping[str, Any]
) -> int:
    """Count instruction text and structured-schema characters in a request."""
    count = len(
        json.dumps(response_schema, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    )
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            count += len(content)
        elif isinstance(content, Sequence) and (not isinstance(content, (str, bytes, bytearray))):
            for block in content:
                if isinstance(block, Mapping) and isinstance(block.get("text"), str):
                    count += len(block["text"])
    return count


def _reasoning_control_record(think: bool | str) -> dict[str, Any]:
    """Describe requested reasoning controls without claiming server compliance."""
    return {
        "requested": think,
        "transport_fields": [
            "params.think",
            "chat_template_kwargs.enable_thinking",
            *(
                ["params.reasoning_effort", "reasoning_effort"]
                if think is False or isinstance(think, str)
                else []
            ),
        ],
        "server_compliance_verified": False,
    }


def _reasoning_effort(think: bool | str) -> str | None:
    """Map the research control to vLLM's OpenAI-compatible field."""
    if think is False:
        return "none"
    if isinstance(think, str):
        return think
    return None


def _safe_schema_name(task: str) -> str:
    normalized = "".join((character if character.isalnum() else "_" for character in task))
    normalized = normalized.strip("_")[:48]
    return normalized or "phase2_response"


def _effective_maximum_output_tokens(
    task: str, configured: int, trace_context: Mapping[str, Any] | None = None
) -> int:
    """Bound compact task responses without changing their semantic contracts."""
    if task == "non_g_probability":
        return min(
            configured, max(256, 16 * len((trace_context or {}).get("core_sample_indices", [])))
        )
    if task == "main_scene_visibility":
        return min(
            configured, max(256, 40 * len((trace_context or {}).get("core_sample_indices", [])))
        )
    return configured


def _repair_instruction(
    task: str,
    *,
    trace_context: Mapping[str, Any] | None = None,
    contract_failure_code: str | None = None,
    failed_model_output: Mapping[str, Any] | None = None,
) -> str:
    if task == "non_g_probability":
        from ..pipeline.probability_pass import repair_instruction

        return repair_instruction(trace_context)
    instruction = "Return one corrected JSON object only, with every field and no Markdown."
    overview_relations = "; merge repeated sightings of the same target across camera views; subject_id must be unique. Copy only authoritative global evidence indices, sorted and unique. This stage selects the recording target and recurring live views only: annotation_relevance_spans and every typical_rest_postures array must be empty."
    if task in {"overview_segment", "overview_reduce"}:
        instruction += (
            ' Overview: selected_subjects=[]=>target_mode="none"; 1=>"individual"; 2-8=>"group"'
            + overview_relations
        )
    elif task == "overview":
        instruction += (
            ' Overview: select exactly one subject and set target_mode="individual"; never "none" or "group"'
            + overview_relations
            + " For the final overview, evidence_sample_indices contains zero to eight ordinary supporting citations from the directly supplied final storyboard images. They are not certified positive exemplars; a separate exact task owns that decision. Never pad the array and never use inherited indices."
        )
    elif task == "overview_exemplar_verification":
        instruction += " Copy target_subject_ids exactly. Output exactly one sample object for every candidate_sample_index, in that order, with no additions or omissions. A positive_main_scene sample requires scene_type=live_target and selected_target_visibility=visible. Slides, static pictures, embedded videos, audience/cutaways, other live scenes, or selected_target_visibility=not_visible are not_positive_main_scene; remaining ambiguous combinations are uncertain. A speaker beside a slide is positive only while the selected live target is physically visible. A person depicted inside a slide, picture, screen, or clip is not the live selected target. Probabilities contain exactly positive_main_scene, not_positive_main_scene, and uncertain, sum to 1, and give the declared classification a maximum probability. Every reason is concise and nonempty."
        if isinstance(trace_context, Mapping):
            candidate_indices = trace_context.get("candidate_sample_indices")
            if (
                isinstance(candidate_indices, list)
                and candidate_indices
                and all((type(index) is int and index >= 0 for index in candidate_indices))
            ):
                instruction += f" This request requires exactly {len(candidate_indices)} ordered sample objects."
    return instruction


def _transport_failure_message(path: str, error: BaseException, *, timeout_seconds: float) -> str:
    """Return a bounded, non-secret category for a network failure."""
    reason: BaseException = error
    if isinstance(error, URLError) and isinstance(error.reason, BaseException):
        reason = error.reason
    windows_error = getattr(reason, "winerror", None)
    if isinstance(reason, (TimeoutError, socket.timeout)) or windows_error == 10060:
        return f"Open WebUI {path} timed out after {timeout_seconds:g} seconds"
    if isinstance(reason, socket.gaierror):
        return f"Open WebUI {path} failed during DNS resolution"
    if isinstance(reason, ConnectionRefusedError) or windows_error == 10061:
        return f"Open WebUI {path} connection was refused"
    if isinstance(reason, ConnectionResetError) or windows_error == 10054:
        return f"Open WebUI {path} connection was reset"
    if isinstance(reason, ssl.SSLError):
        return f"Open WebUI {path} TLS connection failed"
    return f"Open WebUI {path} transport failed ({type(reason).__name__})"


def _is_transport_timeout(error: BaseException) -> bool:
    reason = error.reason if isinstance(error, URLError) else error
    return (
        isinstance(reason, (TimeoutError, socket.timeout))
        or getattr(reason, "winerror", None) == 10060
    )


def _is_transport_disconnect(error: BaseException) -> bool:
    """Recognize transient connection loss without retrying arbitrary network errors."""
    reason = error.reason if isinstance(error, URLError) else error
    return isinstance(
        reason,
        (
            ConnectionResetError,
            ConnectionAbortedError,
            BrokenPipeError,
            RemoteDisconnected,
            IncompleteRead,
        ),
    ) or getattr(reason, "winerror", None) in {10053, 10054}


class OpenWebUIClient:
    """Private research implementation of the task-specific ``Phase2Model`` port."""

    def __init__(
        self,
        config: OpenWebUIConfig,
        inference: InferenceConfig,
        *,
        environ: Mapping[str, str] | None = None,
        trace_writer: TraceWriter | None = None,
    ) -> None:
        self.config = config
        self.inference = inference
        self._environ = os.environ if environ is None else environ
        self._trace_writer = trace_writer
        self._base_url = config.base_url.rstrip("/")
        self._parsed_url = urlsplit(self._base_url)
        self._capability_snapshot: Mapping[str, Any] | None = None
        self._completion_lock = Lock()
        self._last_completion_finished_at: float | None = None
        self._last_completion_request_id: str | None = None
        context = (
            ssl.create_default_context() if config.verify_tls else ssl._create_unverified_context()
        )
        self._opener = build_opener(ProxyHandler({}), _NoRedirects(), HTTPSHandler(context=context))
        self._assert_allowed_destination()

    def redacted_configuration(self) -> dict[str, Any]:
        """Return the exact serializable settings, which contain no key value."""
        return self.config.to_dict()

    def _assert_allowed_destination(self) -> None:
        scope = self.destination_scope()
        if scope != "loopback" and (
            self._parsed_url.scheme != "https" or not self.config.verify_tls
        ):
            raise OpenWebUIError(
                "non-loopback Open WebUI endpoints require HTTPS certificate verification"
            )
        if self.config.allow_non_private_network:
            return
        if scope == "non_private":
            raise OpenWebUIError(
                "Open WebUI destination is not loopback or a private LAN address; set allow_non_private_network=true only for an authorized HTTPS endpoint"
            )

    def destination_scope(self) -> str:
        """Return loopback, private_lan, or non_private after DNS resolution."""
        assert self._parsed_url.hostname is not None
        port = self._parsed_url.port or (443 if self._parsed_url.scheme == "https" else 80)
        addresses = _resolved_addresses(self._parsed_url.hostname, port)
        if all((address.is_loopback for address in addresses)):
            return "loopback"
        if all((_address_is_local_or_private(address) for address in addresses)):
            return "private_lan"
        return "non_private"

    def _credential(self) -> str:
        value = self._environ.get(self.config.api_key_env)
        if not isinstance(value, str) or not value.strip():
            raise OpenWebUIError(
                f"Open WebUI API key environment variable {self.config.api_key_env!r} is not set"
            )
        return value.strip()

    def _headers(self) -> dict[str, str]:
        key = self._credential()
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.config.auth_mode == "bearer":
            headers["Authorization"] = f"Bearer {key}"
        else:
            headers[self.config.api_key_header] = key
        return headers

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        tolerated_statuses: frozenset[int] = frozenset(),
        request_id: str | None = None,
        semantic_attempt: int | None = None,
    ) -> tuple[int, Any | None, str | None, tuple[dict[str, Any], ...]]:
        encoded = None
        if payload is not None:
            try:
                encoded = json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise OpenWebUIError("Open WebUI request is not serializable JSON") from exc
            if len(encoded) > self.config.maximum_request_bytes:
                raise OpenWebUIError("Open WebUI request exceeds maximum_request_bytes")
        headers = self._headers()
        transport_attempts: list[dict[str, Any]] = []
        retry_count = 0

        def wait_for_retry() -> None:
            nonlocal retry_count
            retry_count += 1
            delays = self.config.transient_http_retry_delays_seconds
            if delays is not None:
                backoff_seconds = delays[retry_count - 1]
                transport_attempts[-1]["retry_schedule_policy_version"] = (
                    _EXPLICIT_RETRY_SCHEDULE_VERSION
                )
            else:
                backoff_seconds = min(
                    60.0, self.config.transient_http_retry_backoff_seconds * 2 ** (retry_count - 1)
                )
            transport_attempts[-1]["retry_backoff_seconds"] = backoff_seconds
            if self._trace_writer is not None:
                self._trace_writer.log_event(
                    "model",
                    "transport_retry",
                    "scheduled",
                    {
                        "request_id": request_id,
                        "path": path,
                        "semantic_attempt": semantic_attempt,
                        **transport_attempts[-1],
                        "maximum_transient_http_retries": self.config.maximum_transient_http_retries,
                    },
                )
            sleep(backoff_seconds)

        while True:
            self._assert_allowed_destination()
            request = Request(self._base_url + path, data=encoded, headers=headers, method=method)
            transport_started = perf_counter()
            phase = "connect"
            status: int | None = None
            try:
                response = self._opener.open(request, timeout=self.config.timeout_seconds)
                phase = "response_read"
                with response:
                    status = response.status
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None:
                        try:
                            declared_size = int(content_length)
                        except ValueError as exc:
                            raise OpenWebUIError(
                                "Open WebUI returned an invalid Content-Length"
                            ) from exc
                        if declared_size > self.config.maximum_response_bytes:
                            raise OpenWebUIError(
                                "Open WebUI response exceeds maximum_response_bytes"
                            )
                    body = response.read(self.config.maximum_response_bytes + 1)
                    if len(body) > self.config.maximum_response_bytes:
                        raise OpenWebUIError("Open WebUI response exceeds maximum_response_bytes")
            except HTTPError as exc:
                try:
                    body = exc.read(self.config.maximum_response_bytes + 1)
                    if not isinstance(body, bytes):
                        body = b""
                except (OSError, ValueError, HTTPException):
                    body = b""
                finally:
                    try:
                        exc.close()
                    except (OSError, ValueError, HTTPException):
                        pass
                body_digest = sha256(body).hexdigest()
                category = _classify_http_error(exc.code, body)
                transport_attempts.append(
                    {
                        "transport_attempt": len(transport_attempts) + 1,
                        "http_status": exc.code,
                        "category": category,
                        "response_body_bytes": len(body),
                        "response_body_sha256": body_digest,
                        "response_body_truncated": len(body) > self.config.maximum_response_bytes,
                        "latency_seconds": perf_counter() - transport_started,
                        **(
                            {"retry_policy_version": _WRAPPED_INTERNAL_ERROR_RETRY_POLICY_VERSION}
                            if category == "backend_internal_error"
                            else {}
                        ),
                    }
                )
                if exc.code in tolerated_statuses:
                    return (exc.code, None, body_digest, tuple(transport_attempts))
                retryable = (
                    exc.code in _TRANSIENT_HTTP_RETRY_STATUSES
                    and category == "generic_http_error"
                    or (
                        category == "backend_internal_error"
                        and method == "POST"
                        and (path == "/api/chat/completions")
                    )
                ) and retry_count < self.config.maximum_transient_http_retries
                if not retryable:
                    error = _HTTPStatusError(
                        path=path,
                        status=exc.code,
                        category=category,
                        response_body=body,
                        response_body_truncated=len(body) > self.config.maximum_response_bytes,
                    )
                    error.transport_attempts = tuple(transport_attempts)
                    raise error from None
                wait_for_retry()
                continue
            except (URLError, TimeoutError, OSError, HTTPException) as exc:
                is_timeout = _is_transport_timeout(exc)
                category = "transport_timeout" if is_timeout else "transport_failure"
                retry_policy = None
                if method == "POST" and path == "/api/chat/completions":
                    if is_timeout:
                        retry_policy = _COMPLETION_TIMEOUT_RETRY_POLICY_VERSION
                    elif _is_transport_disconnect(exc):
                        retry_policy = _COMPLETION_DISCONNECT_RETRY_POLICY_VERSION
                transport_attempts.append(
                    {
                        "transport_attempt": len(transport_attempts) + 1,
                        "http_status": status,
                        "category": category,
                        "phase": phase,
                        "response_body_available": False,
                        "response_body_bytes": 0,
                        "response_body_sha256": sha256(b"").hexdigest(),
                        "response_body_truncated": False,
                        "latency_seconds": perf_counter() - transport_started,
                        "timeout_seconds": self.config.timeout_seconds,
                        **({"retry_policy_version": retry_policy} if retry_policy else {}),
                    }
                )
                if (
                    retry_policy is not None
                    and retry_count < self.config.maximum_transient_http_retries
                ):
                    wait_for_retry()
                    continue
                raise _TransportRequestError(
                    _transport_failure_message(
                        path, exc, timeout_seconds=self.config.timeout_seconds
                    ),
                    status=status,
                    category=category,
                    phase=phase,
                    transport_attempts=transport_attempts,
                ) from None
            break
        body_digest = sha256(body).hexdigest()
        transport_attempts.append(
            {
                "transport_attempt": len(transport_attempts) + 1,
                "http_status": status,
                "category": None,
                "response_body_bytes": len(body),
                "response_body_sha256": body_digest,
                "response_body_truncated": False,
                "latency_seconds": perf_counter() - transport_started,
            }
        )
        if retry_count and self._trace_writer is not None:
            self._trace_writer.log_event(
                "model",
                "transport_retry",
                "recovered",
                {
                    "request_id": request_id,
                    "path": path,
                    "semantic_attempt": semantic_attempt,
                    "transport_attempt": len(transport_attempts),
                    "http_status": status,
                    "retry_count": retry_count,
                    "latency_seconds": transport_attempts[-1]["latency_seconds"],
                },
            )
        return (
            status,
            _load_strict_json(body, f"Open WebUI {path} response"),
            body_digest,
            tuple(transport_attempts),
        )

    def check_capabilities(self) -> Mapping[str, Any]:
        """Verify endpoint access, exact model availability, and explicit vision support."""
        if self._capability_snapshot is not None:
            return dict(self._capability_snapshot)
        version_status, version, _, _ = self._request_json(
            "GET", "/api/version", tolerated_statuses=frozenset({404})
        )
        ollama_version_status, ollama_version, _, _ = self._request_json(
            "GET", "/ollama/api/version", tolerated_statuses=frozenset({400, 404, 405, 503})
        )
        _, models_response, _, _ = self._request_json("GET", "/api/models")
        if not isinstance(models_response, Mapping) or not isinstance(
            models_response.get("data"), list
        ):
            raise OpenWebUIError("Open WebUI /api/models response is malformed")
        matches = [
            item
            for item in models_response["data"]
            if isinstance(item, Mapping)
            and (item.get("id") == self.config.model or item.get("name") == self.config.model)
        ]
        if len(matches) != 1:
            raise OpenWebUIError(
                "configured Open WebUI model is unavailable or not uniquely identified"
            )
        model_entry = dict(matches[0])
        show_status, show, _, _ = self._request_json(
            "POST",
            "/ollama/api/show",
            {"model": self.config.model},
            tolerated_statuses=frozenset({400, 404, 405, 503}),
        )
        vision = _explicit_vision_capability(show)
        vision_source = "ollama-api-show"
        if vision is None:
            vision = _explicit_vision_capability(model_entry)
            vision_source = "api-models" if vision is not None else "not-declared"
        if self.config.require_vision and vision is not True:
            state = "not supported" if vision is False else "not explicitly declared"
            raise OpenWebUIError(f"configured model vision capability is {state}")
        declared_limits = _ollama_declared_limits(show)
        declared_context = declared_limits["context_window_tokens"]
        declared_context_source = "ollama-api-show"
        if declared_context is None:
            declared_context = _model_entry_context_limit(model_entry)
            declared_context_source = (
                "api-models" if declared_context is not None else "not-declared"
            )
        if (
            declared_context is not None
            and self.inference.model_context_window_tokens > declared_context
        ):
            raise OpenWebUIError(
                "configured model_context_window_tokens exceeds the context length declared by Ollama"
            )
        model_entry_encoded = json.dumps(
            model_entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        show_digest = None
        if isinstance(show, Mapping):
            show_digest = sha256(
                json.dumps(
                    show, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
                ).encode("utf-8")
            ).hexdigest()
        snapshot: dict[str, Any] = {
            "adapter": _OPEN_WEBUI_ADAPTER_VERSION,
            "base_url": self._base_url,
            "destination_scope": self.destination_scope(),
            "username": self.config.username,
            "server_version_status": "available"
            if version_status == 200
            else "endpoint-unavailable",
            "server_version": version if version_status == 200 else None,
            "model": self.config.model,
            "resolved_model": str(model_entry.get("id") or model_entry.get("name")),
            "model_entry_sha256": sha256(model_entry_encoded).hexdigest(),
            "model_details_sha256": show_digest,
            "vision": vision,
            "vision_evidence": vision_source,
            "ollama_show_status": "available" if show_status == 200 else f"http-{show_status}",
            "ollama_version_status": "available"
            if ollama_version_status == 200
            else f"http-{ollama_version_status}",
            "ollama_version": ollama_version if ollama_version_status == 200 else None,
            "structured_output": "json-schema requested; response locally validated on each task call",
            "model_input_limits": {
                "contract_version": "model-input-limits-v0.1",
                "configured_context_window_tokens": self.inference.model_context_window_tokens,
                "configured_maximum_output_tokens": self.inference.maximum_output_tokens,
                "configured_maximum_prompt_characters": self.inference.maximum_prompt_characters,
                "configured_maximum_images_per_request": self.inference.maximum_images_per_request,
                "configured_maximum_image_bytes_per_request": self.inference.maximum_image_bytes_per_request,
                "configured_maximum_total_image_pixels": self.inference.maximum_total_image_pixels,
                "configured_maximum_image_width": self.inference.maximum_image_width,
                "configured_maximum_image_height": self.inference.maximum_image_height,
                "declared_context_window_tokens": declared_context,
                "declared_context_window_source": declared_context_source,
                "ollama_declared_context_window_tokens": declared_limits["context_window_tokens"],
                "ollama_default_context_window_tokens": declared_limits[
                    "default_context_window_tokens"
                ],
                "ollama_declared_vision_tokens_per_image": declared_limits[
                    "vision_tokens_per_image"
                ],
            },
            "authentication": self.config.auth_mode,
            "api_key_env": self.config.api_key_env,
            "credential_serialized": False,
        }
        self._capability_snapshot = snapshot
        return snapshot

    def _image_block(self, path: Path) -> dict[str, Any]:
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise OpenWebUIError("model image input is not readable") from exc
        media_type = mimetypes.guess_type(path.name)[0]
        if media_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise OpenWebUIError("model images must be JPEG, PNG, or WebP")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64encode(payload).decode('ascii')}"},
        }

    def _completion_payload(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        response_schema: Mapping[str, Any],
        trace_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        reasoning_effort = _reasoning_effort(self.inference.think)
        maximum_output_tokens = _effective_maximum_output_tokens(
            task, self.inference.maximum_output_tokens, trace_context
        )
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "params": {
                "temperature": self.inference.temperature,
                "think": self.inference.think,
                "max_tokens": maximum_output_tokens,
                "num_ctx": self.inference.model_context_window_tokens,
                **({"seed": self.inference.seed} if self.inference.seed is not None else {}),
                **({"reasoning_effort": reasoning_effort} if reasoning_effort is not None else {}),
            },
            "chat_template_kwargs": {"enable_thinking": self.inference.think is not False},
            "temperature": self.inference.temperature,
            "max_tokens": maximum_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": _safe_schema_name(task),
                    "strict": True,
                    "schema": _provider_safe_response_schema(response_schema),
                },
            },
        }
        if self.inference.seed is not None:
            payload["seed"] = self.inference.seed
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        return payload

    def complete_json(
        self,
        *,
        request_id: str,
        task: str,
        system_prompt: str,
        user_prompt: str,
        images: Sequence[Path],
        response_schema: Mapping[str, Any],
        trace_context: Mapping[str, Any],
        response_validator: Callable[[Any], Mapping[str, Any]] | None = None,
    ) -> ModelCallResult:
        """Pace and serialize one logical structured completion request."""
        with self._completion_lock:
            cooldown_seconds = self.config.inter_request_cooldown_seconds
            previous_request_id = self._last_completion_request_id
            if self._last_completion_finished_at is not None and cooldown_seconds > 0:
                remaining_seconds = max(
                    0.0, cooldown_seconds - (perf_counter() - self._last_completion_finished_at)
                )
                if remaining_seconds > 0:
                    if self._trace_writer is not None:
                        self._trace_writer.log_event(
                            "model",
                            "inter_request_cooldown",
                            "scheduled",
                            {
                                "request_id": request_id,
                                "previous_request_id": previous_request_id,
                                "cooldown_seconds": remaining_seconds,
                            },
                        )
                    wait_started = perf_counter()
                    sleep(remaining_seconds)
                    waited_seconds = perf_counter() - wait_started
                    if self._trace_writer is not None:
                        self._trace_writer.log_event(
                            "model",
                            "inter_request_cooldown",
                            "completed",
                            {
                                "request_id": request_id,
                                "previous_request_id": previous_request_id,
                                "cooldown_seconds": waited_seconds,
                            },
                        )
            try:
                return self._complete_json_with_size_recovery(
                    request_id=request_id,
                    task=task,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    images=images,
                    response_schema=response_schema,
                    trace_context=trace_context,
                    response_validator=response_validator,
                )
            finally:
                self._last_completion_finished_at = perf_counter()
                self._last_completion_request_id = request_id

    def _complete_json_with_size_recovery(self, **arguments: Any) -> ModelCallResult:
        """Recover HTTP 413 with a bounded PNG ladder, retaining every sample."""
        images = tuple(Path(path) for path in arguments["images"])
        if (
            arguments["task"] != "non_g_probability"
            or not images
            or self._trace_writer is None
            or any(path.suffix.lower() != ".png" for path in images)
        ):
            return self._complete_json_with_image_recovery(**arguments)
        request_id = arguments["request_id"]
        run_dir = self._trace_writer.run_dir

        def recovery_for_input() -> ImageRecovery:
            if len(images) > self.inference.maximum_images_per_request:
                raise OpenWebUIError("request exceeds maximum_images_per_request")
            records = []
            for path in images:
                content = path.read_bytes()
                records.append({
                    "sha256": sha256(content).hexdigest(), "byte_size": len(content),
                    "mime_type": "image/png",
                })
            if sum(item["byte_size"] for item in records) > self.inference.maximum_image_bytes_per_request:
                raise OpenWebUIError("request exceeds maximum_image_bytes_per_request")
            settings = self._completion_payload(
                task=arguments["task"],
                messages=[
                    {"role": "system", "content": arguments["system_prompt"]},
                    {"role": "user", "content": arguments["user_prompt"]},
                ],
                response_schema=arguments["response_schema"],
                trace_context=arguments["trace_context"],
            )
            return ImageRecovery(run_dir, request_id, {
                "request_id": request_id, "endpoint": self._base_url,
                "settings_and_text": settings, "images": records,
                "trace_context": dict(arguments["trace_context"]),
                "policy_version": SIZE_POLICY_VERSION,
            }, optimize_size=True)

        recovery = (
            recovery_for_input()
            if ImageRecovery.request_directory(run_dir, request_id).exists() else None
        )
        reused = recovery is not None and recovery.saved()
        started = perf_counter()
        if not reused:
            try:
                return self._complete_json_with_image_recovery(**arguments)
            except OpenWebUIError as exc:
                cause = exc.__cause__
                if not isinstance(cause, _HTTPStatusError) or cause.status != 413:
                    raise
                recovery = recovery_for_input()
                recovery.record()
        assert recovery is not None
        try:
            compressed, manifest = recovery.prepare(
                images, maximum_bytes=self.inference.maximum_image_bytes_per_request,
                maximum_width=self.inference.maximum_image_width,
                maximum_height=self.inference.maximum_image_height,
            )
        except (ValueError, OSError) as exc:
            raise OpenWebUIError("lossless PNG size recovery preparation failed") from exc
        before = sum(item["byte_size"] for item in recovery.identity["images"])
        after = sum(path.stat().st_size for path in compressed)
        if after >= before:
            # No reason to resend identical bytes after the original 413.
            recovery.record_size_rejection()
        if recovery.size_rejected():
            result = self._complete_json_with_spatial_size_recovery(arguments, recovery, min(before, after))
            return replace(result, request_id=request_id, latency_seconds=perf_counter() - started)
        self._trace_writer.log_event("model", "lossless_png_size_recovery", "started", {
            "request_id": request_id, "policy_version": SIZE_POLICY_VERSION,
            "reused": reused, "image_count": len(images),
            "original_image_bytes": before, "compressed_image_bytes": after,
        })
        try:
            result = self._complete_json_unpaced(**{
                **arguments, "request_id": request_id + "-png-size-v1",
                "images": compressed,
                "trace_context": recovery_trace_context(
                    arguments["trace_context"], manifest, request_id,
                ),
            })
        except OpenWebUIError as exc:
            if isinstance(exc.__cause__, _HTTPStatusError) and exc.__cause__.status == 413:
                recovery.record_size_rejection()
                result = self._complete_json_with_spatial_size_recovery(arguments, recovery, after)
                return replace(result, request_id=request_id, latency_seconds=perf_counter() - started)
            else:
                raise
        self._trace_writer.log_event("model", "lossless_png_size_recovery", "completed", {
            "request_id": request_id, "policy_version": SIZE_POLICY_VERSION,
            "reused": reused, "image_count": len(images),
            "original_image_bytes": before, "compressed_image_bytes": after,
        })
        return replace(result, request_id=request_id, latency_seconds=perf_counter() - started)

    def _complete_json_with_spatial_size_recovery(
        self, arguments: Mapping[str, Any], lossless: ImageRecovery, rejected_bytes: int,
    ) -> ModelCallResult:
        """Only HTTP 413 permits spatial loss; prompts and temporal scope stay fixed.

        Each level is rendered from the original images, never from another
        resized level. Successful checkpoints retain the logical request ID.
        """
        assert self._trace_writer is not None
        request_id = arguments["request_id"]
        images = tuple(Path(path) for path in arguments["images"])
        motion_trace = arguments["trace_context"].get("motion_trace") or {}
        chart_ordinal = motion_trace.get("image_ordinal")
        preserve = (chart_ordinal,) if type(chart_ordinal) is int and 0 <= chart_ordinal < len(images) else ()
        for percent in SPATIAL_SIZE_PERCENTAGES:
            recovery = ImageRecovery(
                self._trace_writer.run_dir, request_id, lossless.identity,
                optimize_size=True, scale_percent=percent, preserve_ordinals=preserve,
            )
            reused = recovery.saved()
            recovery.record()
            try:
                resized, manifest = recovery.prepare(
                    images, maximum_bytes=self.inference.maximum_image_bytes_per_request,
                    maximum_width=self.inference.maximum_image_width,
                    maximum_height=self.inference.maximum_image_height,
                )
                size = sum(path.stat().st_size for path in resized)
                # Resume skips rejected levels; the final level can be retried
                # explicitly after the server configuration has been corrected.
                if recovery.size_rejected() and percent != SPATIAL_SIZE_PERCENTAGES[-1]:
                    rejected_bytes = min(rejected_bytes, size)
                    continue
            except (ValueError, OSError) as exc:
                raise OpenWebUIError("spatial PNG size recovery preparation failed") from exc
            if size >= rejected_bytes:
                continue
            details = {
                "request_id": request_id, "policy_version": SPATIAL_SIZE_POLICY_VERSION,
                "scale_percent": percent, "image_count": len(images), "reused": reused,
                "image_bytes": size, "preserved_image_ordinals": list(preserve),
                "spatially_lossy": True, "temporal_samples_removed": 0,
            }
            self._trace_writer.log_event("model", "spatial_png_size_recovery", "started", details)
            try:
                result = self._complete_json_unpaced(**{
                    **arguments, "request_id": f"{request_id}-png-spatial-v1-{percent}",
                    "images": resized,
                    "trace_context": recovery_trace_context(arguments["trace_context"], manifest, request_id),
                })
            except OpenWebUIError as exc:
                if not isinstance(exc.__cause__, _HTTPStatusError) or exc.__cause__.status != 413:
                    raise
                recovery.record_size_rejection()
                rejected_bytes = size
                if percent == SPATIAL_SIZE_PERCENTAGES[-1]:
                    raise OpenWebUIError(
                        f"{exc}; bounded PNG size recovery exhausted (minimum spatial scale=70%); "
                        "no frames or temporal context were removed. The server request-body limit "
                        "must be increased or smaller visual inputs configured. Earlier checkpoints remain reusable."
                    ) from exc.__cause__
                continue
            self._trace_writer.log_event("model", "spatial_png_size_recovery", "completed", details)
            return result
        raise OpenWebUIError(
            "HTTP 413: bounded PNG size recovery cannot reduce this request further; "
            "no frames or temporal context were removed. Check the server request-body limit."
        )

    def _complete_json_with_image_recovery(self, **arguments: Any) -> ModelCallResult:
        """Try one PNG representation after a traced wrapped backend failure.

        The logical caller/checkpoint identity remains unchanged. Physical PNG
        exchanges have distinct IDs and trace the exact source/raster mapping.
        """
        images = tuple((Path(path) for path in arguments["images"]))
        request_id = arguments["request_id"]
        if (
            arguments["task"] not in {"non_g_probability"}
            or not images
            or self._trace_writer is None
            or any((path.suffix.lower() not in {".jpg", ".jpeg"} for path in images))
        ):
            return self._complete_json_unpaced(**arguments)
        run_dir = self._trace_writer.run_dir

        def recovery_for_input() -> ImageRecovery:
            if len(images) > self.inference.maximum_images_per_request:
                raise OpenWebUIError("request exceeds maximum_images_per_request")
            records = []
            for path in images:
                content = path.read_bytes()
                records.append(
                    {
                        "sha256": sha256(content).hexdigest(),
                        "byte_size": len(content),
                        "mime_type": mimetypes.guess_type(path.name)[0],
                    }
                )
            if (
                sum((item["byte_size"] for item in records))
                > self.inference.maximum_image_bytes_per_request
            ):
                raise OpenWebUIError("request exceeds maximum_image_bytes_per_request")
            settings = self._completion_payload(
                task=arguments["task"],
                messages=[
                    {"role": "system", "content": arguments["system_prompt"]},
                    {"role": "user", "content": arguments["user_prompt"]},
                ],
                response_schema=arguments["response_schema"],
                trace_context=arguments["trace_context"],
            )
            return ImageRecovery(
                run_dir,
                request_id,
                {
                    "request_id": request_id,
                    "endpoint": self._base_url,
                    "settings_and_text": settings,
                    "images": records,
                    "trace_context": dict(arguments["trace_context"]),
                    "policy_version": IMAGE_RECOVERY_POLICY_VERSION,
                },
            )

        recovery = (
            recovery_for_input()
            if ImageRecovery.request_directory(run_dir, request_id).exists()
            else None
        )
        reused = recovery is not None and recovery.saved()
        started = perf_counter()
        if not reused:
            try:
                return self._complete_json_unpaced(**arguments)
            except OpenWebUIError as exc:
                cause = exc.__cause__
                if (
                    not isinstance(cause, _HTTPStatusError)
                    or cause.category != "backend_internal_error"
                ):
                    raise
                recovery = recovery_for_input()
                recovery.record()
        assert recovery is not None
        self._trace_writer.log_event(
            "model",
            "lossless_png_recovery",
            "started",
            {
                "request_id": request_id,
                "policy_version": IMAGE_RECOVERY_POLICY_VERSION,
                "reused": reused,
                "image_count": len(images),
            },
        )
        try:
            png_images, manifest = recovery.prepare(
                images,
                maximum_bytes=self.inference.maximum_image_bytes_per_request,
                maximum_width=self.inference.maximum_image_width,
                maximum_height=self.inference.maximum_image_height,
            )
        except (ValueError, OSError) as exc:
            raise OpenWebUIError(
                "lossless PNG recovery preparation failed; inspect local recovery artifacts"
            ) from exc
        context = recovery_trace_context(arguments["trace_context"], manifest, request_id)
        result = self._complete_json_unpaced(
            **{
                **arguments,
                "request_id": request_id + "-png-recovery-v1",
                "images": png_images,
                "trace_context": context,
            }
        )
        self._trace_writer.log_event(
            "model",
            "lossless_png_recovery",
            "completed",
            {
                "request_id": request_id,
                "policy_version": IMAGE_RECOVERY_POLICY_VERSION,
                "reused": reused,
                "image_count": len(png_images),
                "repaired": result.repaired,
            },
        )
        return replace(result, request_id=request_id, latency_seconds=perf_counter() - started)

    def _complete_json_unpaced(
        self,
        *,
        request_id: str,
        task: str,
        system_prompt: str,
        user_prompt: str,
        images: Sequence[Path],
        response_schema: Mapping[str, Any],
        trace_context: Mapping[str, Any],
        response_validator: Callable[[Any], Mapping[str, Any]] | None = None,
    ) -> ModelCallResult:
        """Complete one logical request without applying inter-request pacing."""
        if not request_id.strip() or not task.strip():
            raise ValueError("request_id and task must be non-blank")
        if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
            raise ValueError("model prompts must be strings")
        if not isinstance(response_schema, Mapping) or not isinstance(trace_context, Mapping):
            raise ValueError("response_schema and trace_context must be mappings")
        if len(images) > self.inference.maximum_images_per_request:
            raise OpenWebUIError("request exceeds maximum_images_per_request")
        image_paths = tuple((Path(path) for path in images))
        image_bytes = 0
        for path in image_paths:
            try:
                if not path.is_file():
                    raise OSError
                image_bytes += path.stat().st_size
            except OSError as exc:
                raise OpenWebUIError("model image input is not readable") from exc
            if image_bytes > self.inference.maximum_image_bytes_per_request:
                raise OpenWebUIError("request exceeds maximum_image_bytes_per_request")
        if self._capability_snapshot is None:
            self.check_capabilities()
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        content.extend((self._image_block(path) for path in image_paths))
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        reasoning_control = _reasoning_control_record(self.inference.think)
        started = perf_counter()
        repaired = False
        final_body_sha256: str | None = None
        last_validation_error: OpenWebUIError | None = None
        attempts = self.inference.maximum_repair_attempts + 1
        raw_text = ""
        attempt_records: list[dict[str, Any]] = []
        transport_attempt_records: list[dict[str, Any]] = []
        repair_instruction_records: list[dict[str, Any]] = []
        model_payload: Mapping[str, Any] | None = None
        contract_normalization_events: tuple[dict[str, Any], ...] = ()
        contract_normalization_policy_version: str | None = None
        instruction_characters = 0
        local_validation_schema = _local_response_validation_schema(
            task=task,
            response_schema=response_schema,
            has_response_validator=response_validator is not None,
        )
        effective_maximum_output_tokens = _effective_maximum_output_tokens(
            task, self.inference.maximum_output_tokens, trace_context
        )
        output_budget_trace: dict[str, Any] = {
            "maximum_output_tokens": effective_maximum_output_tokens
        }
        for attempt in range(attempts):
            instruction_characters = _message_instruction_characters(messages, response_schema)
            if instruction_characters > self.inference.maximum_prompt_characters:
                raise OpenWebUIError("request exceeds maximum_prompt_characters")
            try:
                request_result = self._request_json(
                    "POST",
                    "/api/chat/completions",
                    self._completion_payload(
                        task=task,
                        messages=messages,
                        response_schema=response_schema,
                        trace_context=trace_context,
                    ),
                    request_id=request_id,
                    semantic_attempt=attempt + 1,
                )
                if len(request_result) == 3:
                    _, response, response_digest = request_result
                    transport_attempts = ()
                else:
                    _, response, response_digest, transport_attempts = request_result
                transport_attempt_records.extend(
                    ({"semantic_attempt": attempt + 1, **item} for item in transport_attempts)
                )
            except (_HTTPStatusError, _TransportRequestError) as exc:
                transport_failure = isinstance(exc, _TransportRequestError)
                failure_kind = "transport_error" if transport_failure else "http_status_error"
                transport_diagnostics = (
                    {
                        "phase": exc.phase,
                        "response_body_available": False,
                        "timeout_seconds": self.config.timeout_seconds,
                        **(
                            {
                                "retry_policy_version": exc.transport_attempts[-1][
                                    "retry_policy_version"
                                ]
                            }
                            if exc.transport_attempts
                            and "retry_policy_version" in exc.transport_attempts[-1]
                            else {}
                        ),
                    }
                    if transport_failure
                    else {}
                )
                transport_attempt_records.extend(
                    ({"semantic_attempt": attempt + 1, **item} for item in exc.transport_attempts)
                )
                trace_reference = None
                if self._trace_writer is not None:
                    image_refs = trace_context.get("image_refs", [])
                    if not isinstance(image_refs, Sequence) or isinstance(image_refs, (str, bytes)):
                        image_refs = []
                    prior_attempts = [_validation_attempt_record(item) for item in attempt_records]
                    trace_reference = self._trace_writer.write_model_exchange(
                        request_id,
                        {
                            "task": task,
                            "model": self.config.model,
                            "system_prompt": system_prompt,
                            "user_prompt": user_prompt,
                            "image_count": len(image_paths),
                            "image_bytes": image_bytes,
                            "instruction_characters": instruction_characters,
                            "message_count": len(messages),
                            "attempt": attempt + 1,
                            "trace_context": dict(trace_context),
                            "response_schema": dict(response_schema),
                            "temperature": self.inference.temperature,
                            "seed": self.inference.seed,
                            "model_context_window_tokens": self.inference.model_context_window_tokens,
                            **output_budget_trace,
                            "reasoning_control": reasoning_control,
                            **(
                                {"repair_instructions": repair_instruction_records}
                                if repair_instruction_records
                                else {}
                            ),
                        },
                        exc.response_body,
                        None,
                        {
                            "valid": False,
                            "repaired": repaired,
                            "attempt_count": attempt + 1,
                            "transport_attempt_count": len(transport_attempt_records),
                            "transport_retry_count": sum(
                                (
                                    1
                                    for item in transport_attempt_records
                                    if "retry_backoff_seconds" in item
                                )
                            ),
                            "transport_attempts": transport_attempt_records,
                            "failure": failure_kind,
                            "category": exc.category,
                            "http_status": exc.status,
                            "response_body_bytes": exc.response_body_bytes,
                            "response_body_sha256": exc.response_body_sha256,
                            "response_body_truncated": exc.response_body_truncated,
                            "maximum_response_bytes": self.config.maximum_response_bytes,
                            **transport_diagnostics,
                            "attempts": [
                                *prior_attempts,
                                {
                                    "attempt": attempt + 1,
                                    "valid": False,
                                    "failure": failure_kind,
                                    "category": exc.category,
                                },
                            ],
                            "endpoint_mode": _OPEN_WEBUI_ENDPOINT_MODE,
                        },
                        list(image_refs),
                    )
                trace_suffix = ""
                if trace_reference is not None:
                    validation_path = trace_reference["artifacts"]["validation"]["path"]
                    trace_suffix = f", validation_trace={validation_path}"
                if transport_failure:
                    raise OpenWebUIError(
                        f"{exc} (task={task}, category={exc.category}, request_id={request_id}, phase={exc.phase}, transport_attempts={len(transport_attempt_records)}{trace_suffix})"
                    ) from exc
                error_type = (
                    OpenWebUIContextLengthError
                    if exc.category == "context_length"
                    else OpenWebUIError
                )
                raise error_type(
                    f"Open WebUI completion request failed (task={task}, category={exc.category}, request_id={request_id}, http_status={exc.status}, response_bytes={exc.response_body_bytes}, response_sha256={exc.response_body_sha256}, response_truncated={str(exc.response_body_truncated).lower()}{trace_suffix})"
                    + (
                        "; the server could not resolve the requested model. Check the exact [open_webui].model ID, account access and backing model connection."
                        if exc.category == "model_not_found"
                        else ""
                    )
                    + (
                        "; the completion backend reported an internal server error. Inspect Open WebUI and backing inference-server logs for this request; the response does not identify the underlying cause."
                        if exc.category == "backend_internal_error"
                        else ""
                    )
                ) from exc
            final_body_sha256 = response_digest
            failure = "response_text_extraction"
            try:
                try:
                    raw_text = _extract_message_text(response)
                except OpenWebUIError:
                    raw_text = ""
                    raise
                failure = "strict_json_parsing"
                try:
                    parsed = _load_strict_json(raw_text, "model message content")
                except OpenWebUIError as exc:
                    if _completion_finish_reason(response) == "length":
                        raise _ResponseBudgetExhausted(
                            "model exhausted maximum_output_tokens before completing JSON content; reduce response cardinality or increase the configured output budget"
                        ) from exc
                    raise
                failure = "json_schema_validation"
                _validate_schema_value(parsed, local_validation_schema, local_validation_schema)
                if not isinstance(parsed, Mapping):
                    raise OpenWebUIError("model JSON root must be an object")
                model_payload = dict(parsed)
                if response_validator is not None:
                    failure = "task_contract_validation"
                    try:
                        parsed = response_validator(parsed)
                        contract_normalization_events = _contract_normalization_events(parsed)
                        contract_normalization_policy_version = (
                            _contract_normalization_policy_version(parsed)
                        )
                    except TaskContractViolation as exc:
                        raise _TaskContractValidationError(exc) from None
                    except ValueError:
                        raise OpenWebUIError(
                            "model JSON failed the task-specific response contract"
                        ) from None
            except OpenWebUIError as exc:
                if isinstance(exc, _ReasoningBudgetExhausted) and self.inference.think is False:
                    exc = _ReasoningBudgetExhausted(
                        "model ignored configured think=false and exhausted maximum_output_tokens in reasoning before producing JSON; use an Instruct/non-thinking model or a serving runtime version that honors the reasoning control"
                    )
                last_validation_error = exc
                if isinstance(exc, _ReasoningBudgetExhausted):
                    failure = "reasoning_budget_exhausted"
                elif isinstance(exc, _ResponseBudgetExhausted):
                    failure = "response_budget_exhausted"
                attempt_record = {
                    "attempt": attempt + 1,
                    "response": response,
                    "response_sha256": response_digest,
                    "valid": False,
                    "failure": failure,
                    **_completion_attempt_metrics(response),
                }
                if isinstance(exc, _TaskContractValidationError):
                    attempt_record.update(
                        {
                            "contract_failure_code": exc.contract_failure_code,
                            "contract_failure_path": exc.contract_failure_path,
                        }
                    )
                attempt_records.append(attempt_record)
                if isinstance(exc, _ReasoningBudgetExhausted):
                    break
                if attempt + 1 >= attempts or isinstance(exc, _ResponseBudgetExhausted):
                    break
                repaired = True
                contract_failure_code = (
                    exc.contract_failure_code
                    if isinstance(exc, _TaskContractValidationError)
                    else None
                )
                repair_instruction = _repair_instruction(
                    task, trace_context=trace_context, contract_failure_code=contract_failure_code
                )
                repair_instruction_records.append(
                    {
                        "semantic_attempt": attempt + 2,
                        "instruction": repair_instruction,
                        "instruction_sha256": sha256(
                            repair_instruction.encode("utf-8")
                        ).hexdigest(),
                        "contract_failure_code": contract_failure_code,
                    }
                )
                repair_content: list[dict[str, Any]] = [
                    {"type": "text", "text": user_prompt + "\n\n" + repair_instruction}
                ]
                repair_content.extend((self._image_block(path) for path in image_paths))
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": repair_content},
                ]
                continue
            if not isinstance(parsed, Mapping) or model_payload is None:
                raise OpenWebUIError("model JSON root must be an object")
            reasoning_observed = _completion_reasoning_observed(response)
            attempt_records.append(
                {
                    "attempt": attempt + 1,
                    "response": response,
                    "response_sha256": response_digest,
                    "valid": True,
                    "failure": None,
                    "contract_normalization_count": len(contract_normalization_events),
                    "contract_normalization_policy_version": contract_normalization_policy_version,
                    **_completion_attempt_metrics(response),
                }
            )
            assert final_body_sha256 is not None
            model_payload_sha256 = _structured_payload_sha256(model_payload)
            canonical_payload_sha256 = _structured_payload_sha256(parsed)
            trace_reference = None
            if self._trace_writer is not None:
                image_refs = trace_context.get("image_refs", [])
                if not isinstance(image_refs, Sequence) or isinstance(image_refs, (str, bytes)):
                    image_refs = []
                trace_reference = self._trace_writer.write_model_exchange(
                    request_id,
                    {
                        "task": task,
                        "model": self.config.model,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "image_count": len(image_paths),
                        "image_bytes": image_bytes,
                        "instruction_characters": instruction_characters,
                        "trace_context": dict(trace_context),
                        "response_schema": dict(response_schema),
                        "temperature": self.inference.temperature,
                        "seed": self.inference.seed,
                        "model_context_window_tokens": self.inference.model_context_window_tokens,
                        **output_budget_trace,
                        "reasoning_control": reasoning_control,
                        **(
                            {"repair_instructions": repair_instruction_records}
                            if repair_instruction_records
                            else {}
                        ),
                    },
                    attempt_records,
                    {
                        "model_output": model_payload,
                        "model_output_sha256": model_payload_sha256,
                        "contract_normalized_output": parsed,
                        "contract_normalized_output_sha256": canonical_payload_sha256,
                        "contract_normalization_events": list(contract_normalization_events),
                        "contract_normalization_policy_version": contract_normalization_policy_version,
                    },
                    {
                        "valid": True,
                        "repaired": repaired,
                        "attempt_count": len(attempt_records),
                        "transport_attempt_count": len(transport_attempt_records),
                        "transport_retry_count": sum(
                            (
                                1
                                for item in transport_attempt_records
                                if "retry_backoff_seconds" in item
                            )
                        ),
                        "transport_attempts": transport_attempt_records,
                        "reasoning_observed": reasoning_observed,
                        "contract_normalization_count": len(contract_normalization_events),
                        "contract_normalization_policy_versions": sorted(
                            [contract_normalization_policy_version]
                            if contract_normalization_policy_version is not None
                            else []
                        ),
                        "contract_normalization_policy_version": contract_normalization_policy_version,
                        "model_output_sha256": model_payload_sha256,
                        "contract_normalized_output_sha256": canonical_payload_sha256,
                        "attempts": [_validation_attempt_record(item) for item in attempt_records],
                        "endpoint_mode": _OPEN_WEBUI_ENDPOINT_MODE,
                    },
                    list(image_refs),
                )
            return ModelCallResult(
                payload=dict(parsed),
                request_id=request_id,
                response_sha256=final_body_sha256,
                latency_seconds=perf_counter() - started,
                endpoint_mode=_OPEN_WEBUI_ENDPOINT_MODE,
                repaired=repaired,
                trace_reference=trace_reference,
                model_payload=model_payload,
                reasoning_observed=reasoning_observed,
                contract_normalization_events=contract_normalization_events,
                model_payload_sha256=model_payload_sha256,
                payload_sha256=canonical_payload_sha256,
                contract_normalization_policy_version=contract_normalization_policy_version,
            )
        assert last_validation_error is not None
        trace_reference = None
        if self._trace_writer is not None:
            image_refs = trace_context.get("image_refs", [])
            if not isinstance(image_refs, Sequence) or isinstance(image_refs, (str, bytes)):
                image_refs = []
            trace_reference = self._trace_writer.write_model_exchange(
                request_id,
                {
                    "task": task,
                    "model": self.config.model,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "image_count": len(image_paths),
                    "image_bytes": image_bytes,
                    "instruction_characters": instruction_characters,
                    "trace_context": dict(trace_context),
                    "response_schema": dict(response_schema),
                    "temperature": self.inference.temperature,
                    "seed": self.inference.seed,
                    "model_context_window_tokens": self.inference.model_context_window_tokens,
                    **output_budget_trace,
                    "reasoning_control": reasoning_control,
                    **(
                        {"repair_instructions": repair_instruction_records}
                        if repair_instruction_records
                        else {}
                    ),
                },
                attempt_records,
                {
                    "last_schema_valid_model_output": model_payload,
                    "contract_normalized_output": None,
                }
                if model_payload is not None
                else None,
                {
                    "valid": False,
                    "repaired": repaired,
                    "attempt_count": len(attempt_records),
                    "transport_attempt_count": len(transport_attempt_records),
                    "transport_retry_count": sum(
                        (1 for item in transport_attempt_records if "retry_backoff_seconds" in item)
                    ),
                    "transport_attempts": transport_attempt_records,
                    "failure": failure,
                    **(
                        {
                            "contract_failure_code": last_validation_error.contract_failure_code,
                            "contract_failure_path": last_validation_error.contract_failure_path,
                        }
                        if isinstance(last_validation_error, _TaskContractValidationError)
                        else {}
                    ),
                    "attempts": [_validation_attempt_record(item) for item in attempt_records],
                    "endpoint_mode": _OPEN_WEBUI_ENDPOINT_MODE,
                },
                list(image_refs),
            )
        trace_suffix = ""
        if trace_reference is not None:
            validation_path = trace_reference["artifacts"]["validation"]["path"]
            trace_suffix = f"; validation_trace={validation_path}"
        raise OpenWebUIError(
            f"model response failed structured validation (task={task}, category={failure}, request_id={request_id}{trace_suffix}): "
            + str(last_validation_error)
        ) from last_validation_error
