# SPDX-License-Identifier: Apache-2.0
"""Durable rights labels for source and derived research media."""

from __future__ import annotations
from hashlib import sha256
from ipaddress import ip_address
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit
from uuid import uuid4

RIGHTS_CONTRACT_VERSION = "0.2.0"
RESTRICTED_RESEARCH = "restricted_research"
NOT_DECLARED = "not_declared"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rights_sidecar_path(path: str | Path) -> Path:
    """Return the companion path that travels with one derived media file."""
    media = Path(path)
    return media.with_name(media.name + ".rights.json")


def unrestricted_or_unspecified_rights() -> dict[str, Any]:
    """Return the conservative no-declaration record for ordinary input."""
    return {
        "contract_version": RIGHTS_CONTRACT_VERSION,
        "restriction": NOT_DECLARED,
        "redistribution_allowed": None,
        "model_processing_scope": "configured_endpoint",
        "basis": "no_repository_restriction_detected",
    }


def restricted_research_rights(*, basis: str) -> dict[str, Any]:
    """Return the repository label for nonredistributable research material."""
    if not basis.strip():
        raise ValueError("rights basis must be non-blank")
    return {
        "contract_version": RIGHTS_CONTRACT_VERSION,
        "restriction": RESTRICTED_RESEARCH,
        "redistribution_allowed": False,
        "model_processing_scope": "operator_authorization_required",
        "basis": basis,
    }


def validate_rights(value: Any) -> dict[str, Any]:
    """Validate and normalize one rights record without broadening permission."""
    if not isinstance(value, Mapping):
        raise ValueError("media rights must be an object")
    required = {
        "contract_version",
        "restriction",
        "redistribution_allowed",
        "model_processing_scope",
        "basis",
    }
    if set(value) != required:
        raise ValueError("media rights must contain the complete contract")
    restriction = value.get("restriction")
    if restriction not in {RESTRICTED_RESEARCH, NOT_DECLARED}:
        raise ValueError("media rights restriction is unsupported")
    contract_version = value.get("contract_version")
    if contract_version != RIGHTS_CONTRACT_VERSION:
        raise ValueError("media rights contract_version is unsupported")
    if not isinstance(value.get("basis"), str) or not value["basis"].strip():
        raise ValueError("media rights basis must be non-blank")
    if restriction == RESTRICTED_RESEARCH:
        expected_scope = "operator_authorization_required"
        if (
            value.get("redistribution_allowed") is not False
            or value.get("model_processing_scope") != expected_scope
        ):
            raise ValueError("restricted research rights may not be broadened")
    elif (
        value.get("redistribution_allowed") is not None
        or value.get("model_processing_scope") != "configured_endpoint"
    ):
        raise ValueError("undeclared rights must not assert redistribution permission")
    return dict(value)


def is_restricted_rights(value: Any) -> bool:
    return validate_rights(value)["restriction"] == RESTRICTED_RESEARCH


def _is_within(path: Path, roots: Sequence[Path]) -> bool:
    resolved = path.resolve()
    return any((resolved.is_relative_to(root.resolve()) for root in roots))


def repository_restricted_roots() -> tuple[Path, ...]:
    """Return the working-tree and installed-source interpretations of ``ext``."""
    return tuple(
        sorted(
            {
                (Path.cwd() / "ext").resolve(),
                (Path(__file__).resolve().parents[2] / "ext").resolve(),
            },
            key=str,
        )
    )


def is_literal_loopback_url(value: str) -> bool:
    """Return whether a URL names localhost or a literal loopback address.

    A hostname that merely resolves to loopback is intentionally rejected for
    restricted input because DNS is consulted again by the HTTP transport.
    """
    hostname = urlsplit(value).hostname
    if hostname is None:
        return False
    if hostname.casefold() == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


def restricted_transport_mode(
    *,
    restricted: bool,
    base_url: str,
    destination_scope: str,
    verify_tls: bool,
    allow_non_private_network: bool,
    operator_asserts_local_provider: bool,
    operator_authorizes_restricted_remote_processing: bool,
) -> str | None:
    """Return the explicit transport authorization mode for restricted input.

    Rights remain attached to every derivative regardless of transport. This
    function records an operator decision; it does not determine that the
    configured use is permitted by a source license, contract, or law.
    """
    if not restricted:
        return None
    if destination_scope not in {"loopback", "private_lan", "non_private"}:
        raise ValueError("model destination scope is unsupported")
    if destination_scope == "loopback":
        if not is_literal_loopback_url(base_url):
            raise ValueError(
                "restricted input using a local model requires localhost or a literal loopback Open WebUI hostname"
            )
        if not operator_asserts_local_provider:
            raise ValueError(
                "restricted local processing requires the operator's explicit local-provider assertion"
            )
        return "operator_asserted_local_provider"
    parsed = urlsplit(base_url)
    if parsed.scheme != "https" or not verify_tls:
        raise ValueError("restricted remote processing requires HTTPS certificate verification")
    if destination_scope == "non_private" and (not allow_non_private_network):
        raise ValueError("restricted remote processing requires allow_non_private_network=true")
    if operator_asserts_local_provider:
        raise ValueError("operator_asserts_local_provider cannot authorize a non-loopback endpoint")
    if not operator_authorizes_restricted_remote_processing:
        raise ValueError(
            "restricted remote processing requires operator_authorizes_restricted_remote_processing=true; this records operator responsibility and is not legal clearance"
        )
    return "operator_authorized_remote_processing"


def read_rights_sidecar(path: str | Path) -> dict[str, Any] | None:
    """Read a hash-bound rights companion, rejecting stale or copied-alone labels."""
    media = Path(path)
    companion = rights_sidecar_path(media)
    if not companion.is_file():
        return None
    try:
        document = json.loads(companion.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("media rights sidecar is not valid UTF-8 JSON") from exc
    if not isinstance(document, Mapping) or set(document) != {
        "schema_version",
        "artifact_sha256",
        "rights",
    }:
        raise ValueError("media rights sidecar is malformed")
    if document.get("schema_version") != RIGHTS_CONTRACT_VERSION:
        raise ValueError("media rights sidecar version is unsupported")
    digest = document.get("artifact_sha256")
    if not isinstance(digest, str) or digest != _file_sha256(media):
        raise ValueError("media rights sidecar does not match its artifact")
    return validate_rights(document.get("rights"))


def classify_media_rights(
    path: str | Path, *, restricted_roots: Sequence[str | Path] = ()
) -> dict[str, Any]:
    """Classify direct restricted paths and hash-bound derived media."""
    media = Path(path)
    inherited = read_rights_sidecar(media)
    direct_restricted = _is_within(media, tuple((Path(root) for root in restricted_roots)))
    if direct_restricted:
        return restricted_research_rights(basis="repository_ext_path_policy")
    if inherited is not None:
        if is_restricted_rights(inherited):
            inherited["basis"] = "hash_bound_derived_media_sidecar"
        return inherited
    return unrestricted_or_unspecified_rights()


def combine_rights(*values: Mapping[str, Any]) -> dict[str, Any]:
    """Propagate the strictest source label into a derived artifact."""
    normalized = [validate_rights(value) for value in values]
    if any((is_restricted_rights(value) for value in normalized)):
        return restricted_research_rights(basis="inherited_from_restricted_input")
    return unrestricted_or_unspecified_rights()


def write_rights_sidecar(
    path: str | Path, rights: Mapping[str, Any], *, overwrite: bool = False
) -> Path:
    """Atomically bind a rights record to the current bytes of one artifact."""
    media = Path(path)
    if not media.is_file():
        raise FileNotFoundError(media)
    target = rights_sidecar_path(media)
    if target.exists() and (not overwrite):
        raise FileExistsError(target)
    document = {
        "schema_version": RIGHTS_CONTRACT_VERSION,
        "artifact_sha256": _file_sha256(media),
        "rights": validate_rights(rights),
    }
    encoded = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
