# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import json
import gzip
import zlib
from io import BytesIO
from ..storage import data_path, open_data
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4
from ..domain import AnalysisResult, to_json_value
from ..rights import validate_rights


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not permitted: {value}")


def write_json(path: str | Path, payload: Any, *, sort_keys: bool = True) -> Path:
    """Atomically write UTF-8 JSON; preserve report order when explicitly requested."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp" + (".gz" if target.suffix == ".gz" else ""))
    try:
        # Large phase evidence must not become a list of encoded fragments plus
        # one complete string in memory. Keep the existing bytes and atomic save.
        stream = open_data(temporary, "w") if target.suffix == ".gz" else temporary.open("w", encoding="utf-8")
        with stream as handle:
            json.dump(
                to_json_value(payload), handle, indent=2, sort_keys=sort_keys,
                ensure_ascii=False, allow_nan=False,
            )
            handle.write("\n")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def read_json(path: str | Path) -> Any:
    return read_json_snapshot(path)[0]


def read_json_snapshot(path: str | Path) -> tuple[Any, str]:
    """Parse and hash one immutable byte snapshot of a JSON artifact."""
    try:
        actual = data_path(Path(path))
        payload = actual.read_bytes()
        if actual.suffix == ".gz":
            with gzip.GzipFile(fileobj=BytesIO(payload), mode="rb") as handle:
                document = json.load(handle, parse_constant=_reject_nonfinite)
        else:
            document = json.loads(payload.decode("utf-8"), parse_constant=_reject_nonfinite)
    except (gzip.BadGzipFile, zlib.error) as exc:
        raise ValueError(f"JSON gzip artifact is invalid (path={Path(path)})") from exc
    except EOFError as exc:
        raise ValueError(f"JSON gzip artifact is truncated (path={Path(path)})") from exc
    except OSError as exc:
        # The CLI prints ValueError without its chained traceback. Preserve the
        # actual OS failure here instead of misreporting it as malformed JSON.
        raise ValueError(
            f"Cannot read JSON artifact (path={Path(path)}, cause={type(exc).__name__}, "
            f"errno={exc.errno}, winerror={getattr(exc, 'winerror', None)}, "
            f"reason={exc.strerror}); this is a file-read failure, not a JSON validation failure"
        ) from exc
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"JSON artifact contains invalid UTF-8 (path={Path(path)}, "
            f"byte_offset={exc.start}, reason={exc.reason})"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON artifact contains invalid JSON (path={Path(path)}, "
            f"line={exc.lineno}, column={exc.colno}, reason={exc.msg})"
        ) from exc
    except ValueError as exc:
        raise ValueError(f"JSON artifact contains invalid JSON (path={Path(path)}, reason={exc})") from exc
    return (document, sha256(payload).hexdigest())


def write_analysis(path: str | Path, result: AnalysisResult) -> Path:
    result.validate()
    document = to_json_value(result)
    validate_analysis_document(document)
    return write_json(path, document)


def validate_analysis_document(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict) or document.get("schema_version") != "0.2.0":
        raise ValueError("Expected current annotation schema 0.2.0")
    required = {
        "run_id",
        "source",
        "preprocessing",
        "config",
        "participant_profiles",
        "samples",
        "activity_candidates",
        "g_units",
        "provenance",
    }
    if required - set(document):
        raise ValueError("Incomplete annotation document")
    if len(document["participant_profiles"]) != 1:
        raise ValueError("Exactly one selected participant is required")
    samples = document["samples"]
    previous = 0
    for index, item in enumerate(samples):
        sample = item["sample"]
        if (
            sample["index"] != index
            or sample["start_us"] != previous
            or (not previous < sample["end_us"] <= previous + 100000)
        ):
            raise ValueError("Invalid canonical sample timeline")
        if type(item["is_g_unit"]) is not bool:
            raise ValueError("Membership must be Boolean")
        previous = sample["end_us"]
    ids = {c["candidate_id"] for c in document["activity_candidates"]}
    if len(ids) != len(document["activity_candidates"]):
        raise ValueError("Duplicate candidate IDs")
    membership = set()
    unit_ids = set()
    for unit in document["g_units"]:
        a, b = (unit["start_sample"], unit["end_sample_exclusive"])
        if not 0 <= a < b <= len(samples) or unit["candidate_id"] not in ids:
            raise ValueError("Invalid G-unit range or candidate")
        if unit["g_unit_id"] in unit_ids:
            raise ValueError("Duplicate G-unit ID")
        unit_ids.add(unit["g_unit_id"])
        if (
            unit["start_us"] != samples[a]["sample"]["start_us"]
            or unit["end_us"] != samples[b - 1]["sample"]["end_us"]
        ):
            raise ValueError("G-unit timestamps do not match samples")
        if membership.intersection(range(a, b)):
            raise ValueError("G-units overlap")
        membership.update(range(a, b))
    if membership != {i for i, s in enumerate(samples) if s["is_g_unit"]}:
        raise ValueError("G-units do not match final sample membership")
    if document["provenance"].get("policy_version") == "probability-finalization-v0.4":
        from ..domain import AnalysisSample, JointActivityEvidence, JointActivityState
        from ..pipeline.output_guards import excluded_from_output
        for item in samples:
            value = dict(item["joint_evidence"])
            value["state"] = JointActivityState(value["state"])
            if value.get("resolved_state") is not None:
                value["resolved_state"] = JointActivityState(value["resolved_state"])
            if item["is_g_unit"] and excluded_from_output(AnalysisSample(**item["sample"]), JointActivityEvidence(**value)):
                raise ValueError("final membership may not include excluded or barrier samples")
    validate_rights(document["source"]["rights"])
    return document
