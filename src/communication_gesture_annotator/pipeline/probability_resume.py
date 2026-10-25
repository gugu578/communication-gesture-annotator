# SPDX-License-Identifier: Apache-2.0
"""Restore probability checkpoints before preparing their expensive visual inputs.

New sidecars bind the source evidence, policies, schedule and previous estimates.
For older compact v13 runs, first and last prefix requests still undergo exact
ordinary replay. Only their intervening validated answers use the migration path;
no new request is sent until the last boundary has passed ordinary validation.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..export.json_export import read_json, read_json_snapshot, write_json

VERSION = "probability-checkpoint-restore-v0.1"


class ProbabilityReplay:
    def __init__(self, model: Any, request_ids: Sequence[str], identity: Mapping[str, Any]) -> None:
        from .llm_runner import _digest
        self.model = model
        self.root = getattr(model, "_checkpoint_root", None)
        self.resuming = bool(getattr(model, "_resume", False))
        self.binding = _digest({"version": VERSION,
            "scope": getattr(model, "_scope_sha256", None), "evidence": identity})
        self.prefix: set[str] = set()
        self.boundaries: set[str] = set()
        self.verified = False
        if self.root is not None and self.resuming:
            for request_id in request_ids:
                if not self.path(request_id).is_file():
                    break
                self.prefix.add(request_id)
            prefix = [r for r in request_ids if r in self.prefix]
            if prefix:
                self.boundaries = {prefix[0], prefix[-1]}

    def path(self, request_id: str) -> Path:
        return self.root / ("checkpoint-" + sha256(request_id.encode()).hexdigest()[:24] + ".json")

    def sidecar(self, request_id: str) -> Path:
        path = self.path(request_id)
        return path.with_name(path.name.replace("checkpoint-", "replay-", 1))

    def key(self, *, core: Sequence[int], context: Sequence[int], history: Any, policy: Any) -> str:
        from .llm_runner import _digest
        return _digest({"binding": self.binding, "core": list(core), "context": list(context),
                        "history": history, "previous_estimate_admission": policy})

    def restore(self, request_id: str, key: str, validator: Any) -> tuple[Any, dict[str, Any]] | None:
        from . import llm_runner as runtime
        from .llm_contracts import ModelCallResult
        if self.root is None or not self.resuming or request_id not in self.prefix:
            return None
        path = self.path(request_id)
        saved, checkpoint_hash = read_json_snapshot(path)
        metadata = None
        if self.sidecar(request_id).is_file():
            metadata = read_json(self.sidecar(request_id))
            if (metadata.get("version") != VERSION or metadata.get("key") != key
                    or metadata.get("checkpoint_sha256") != checkpoint_hash):
                raise ValueError("probability resume evidence, history or checkpoint changed")
        elif request_id in self.boundaries or not self.verified:
            return None
        if saved.get("request_id") != request_id or saved.get("task") != "non_g_probability":
            raise ValueError("probability checkpoint identity does not match")
        if hasattr(self.model, "_load_checkpoint"):
            # The sidecar (or legacy boundary validation) provides the input
            # binding; retain all existing result/trace/contract validation.
            call = self.model._load_checkpoint(path, request_id=request_id,
                task="non_g_probability", input_record={},
                input_sha256=saved["input_sha256"], response_validator=validator)
        else:
            result = saved.get("call")
            if not isinstance(result, dict) or runtime._digest(result) != saved.get("call_sha256"):
                raise ValueError("probability checkpoint result hash does not match")
            if result.get("request_id") != request_id:
                raise ValueError("probability checkpoint result identity does not match")
            validator(result["payload"])
            call = ModelCallResult(**result)
            self.model.reused_count += 1
            runtime._record_event(self.model._trace, "model", "checkpoint", "reused",
                                  request_id=request_id, images_prepared=False)
        details = deepcopy(metadata["trace"]) if metadata is not None else {
            "model_view_sha256": None, "native_input_artifact_sha256": None,
            "motion_trace": None, "checkpoint_restore": {
                "policy_version": VERSION, "original_input_sha256": saved["input_sha256"],
                "legacy_compact_metadata": "original request metadata was not retained",
                "checkpoint_sha256": checkpoint_hash}}
        self.verified = True
        return call, details

    def remember(self, request_id: str, key: str, trace: Mapping[str, Any]) -> None:
        if self.root is None or not self.path(request_id).is_file():
            return
        # Called only after successful ordinary checkpoint validation/persistence
        # or restore. A failed validation never enables the legacy fast path.
        self.verified = True
        details = {k: deepcopy(trace[k]) for k in ("model_view_sha256",
            "native_input_artifact_sha256", "motion_trace", "checkpoint_restore") if k in trace}
        value = {"version": VERSION, "key": key,
                 "checkpoint_sha256": sha256(self.path(request_id).read_bytes()).hexdigest(),
                 "trace": details}
        path = self.sidecar(request_id)
        if not path.exists() or read_json(path) != value:
            write_json(path, value)
