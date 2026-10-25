# SPDX-License-Identifier: Apache-2.0
"""Command-line interface for video G-unit detection."""

from __future__ import annotations
from .storage import data_path
import argparse
from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha256
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import logging
import os
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory
from typing import Sequence
import zlib
from .config import load_research_config
from .adapters import OpenWebUIClient, OpenWebUIError
from .audit import TraceWriter
from .audit.progress import StepProgress
from .domain import AnalysisSample
from .export.json_export import write_json
from .media import (
    MediaRuntimeError,
    PREPROCESSING_REUSE_CONTRACT_VERSION,
    extract_analysis_frames,
    load_reused_preprocessing,
    preprocess_video,
    reuse_preprocessing,
    run_pose_conditioning,
    run_pose_preprocessing,
)
from .media.resume_guard import validate_resume_source
from .pipeline.llm_contracts import PROFILE_SEGMENT_SCHEMA, ModelCallResult
from .pipeline.front_end_contracts import validate_overview_response
from .pipeline.llm_runner import (
    PIPELINE_STAGE_ORDER,
    run_phase2_pipeline,
    validate_pipeline_stage,
    validate_model_run_resume_state,
    write_pipeline_status,
)
from .rights import (
    classify_media_rights,
    combine_rights,
    is_restricted_rights,
    repository_restricted_roots,
    restricted_transport_mode,
)

_LOG_NAMESPACE = "communication_gesture_annotator"
_LOGGER = logging.getLogger(__name__)
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _analysis_time_us(value: str) -> int:
    """Round user seconds to the nearest 10 Hz boundary; exact ties round up."""
    try:
        seconds = Decimal(value)
        if not seconds.is_finite() or seconds < 0:
            raise ValueError
        return int((seconds * 10).to_integral_value(rounding=ROUND_HALF_UP)) * 100000
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError("seconds must be finite and nonnegative") from exc


def _configure_cli_logging(level_name: str) -> tuple[logging.Logger, logging.Handler, int, bool]:
    """Attach one invocation-scoped handler for safe package progress logs."""
    package_logger = logging.getLogger(_LOG_NAMESPACE)
    previous_level = package_logger.level
    previous_propagate = package_logger.propagate
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    package_logger.addHandler(handler)
    package_logger.setLevel(getattr(logging, level_name))
    package_logger.propagate = False
    return (package_logger, handler, previous_level, previous_propagate)


def _restore_cli_logging(state: tuple[logging.Logger, logging.Handler, int, bool]) -> None:
    package_logger, handler, previous_level, previous_propagate = state
    package_logger.removeHandler(handler)
    handler.close()
    package_logger.setLevel(previous_level)
    package_logger.propagate = previous_propagate


def _existing_analysis_artifacts(run_dir: Path) -> list[Path]:
    names = (
        "source.json",
        "profile.json",
        "frame-evidence.json",
        "candidates.json",
        "prompt-envelopes.json",
        "window-responses.json",
        "candidate-assessments.json",
        "activity-decisions.json",
        "decision-trace.json",
        "phase2-evidence.json",
        "preprocessing-reuse.json",
        "annotations.json",
        "annotations.eaf",
        "run.json",
        "pipeline-status.json",
    )
    return [data_path(run_dir / name) for name in names if data_path(run_dir / name).exists()]


_CLOUD_SYNC_ENVIRONMENT_VARIABLES = (
    "OneDrive",
    "OneDriveConsumer",
    "OneDriveCommercial",
    "DROPBOX",
    "GOOGLE_DRIVE",
    "ICLOUDDRIVE",
)
_CLOUD_SYNC_DIRECTORY_NAMES = frozenset(
    {"onedrive", "dropbox", "google drive", "googledrive", "icloud drive", "iclouddrive"}
)


def _cloud_sync_roots(environ: Mapping[str, str]) -> tuple[Path, ...]:
    """Return configured synchronization roots without reading their content."""
    roots: set[Path] = set()
    for name in _CLOUD_SYNC_ENVIRONMENT_VARIABLES:
        value = environ.get(name)
        if isinstance(value, str) and value.strip():
            roots.add(Path(value).expanduser().resolve())
    user_profile = environ.get("USERPROFILE")
    if isinstance(user_profile, str) and user_profile.strip():
        roots.add((Path(user_profile).expanduser() / "OneDrive").resolve())
    return tuple(sorted(roots, key=str))


def _path_is_cloud_synchronized(
    path: str | Path, *, environ: Mapping[str, str] | None = None
) -> bool:
    """Conservatively identify common synchronized folders on Windows."""
    resolved = Path(path).expanduser().resolve()
    environment = os.environ if environ is None else environ
    if any((resolved.is_relative_to(root) for root in _cloud_sync_roots(environment))):
        return True
    for part in resolved.parts:
        normalized = part.casefold().replace("_", " ").strip()
        if normalized in _CLOUD_SYNC_DIRECTORY_NAMES or normalized.startswith("onedrive - "):
            return True
    return False


def _assert_restricted_run_destination(run_dir: str | Path) -> None:
    """Keep sensitive derivatives of restricted inputs out of sync services."""
    if _path_is_cloud_synchronized(run_dir):
        raise ValueError(
            "restricted research input requires a run directory outside OneDrive, Dropbox, Google Drive, iCloud Drive, or another synchronized folder"
        )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 4294967295
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _synthetic_visual_challenge_png() -> bytes:
    """Create a rights-safe image whose answer is not disclosed to the model."""
    width, height = (384, 256)
    pixels = bytearray([248, 248, 248] * width * height)

    def rectangle(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        for y in range(y0, y1):
            for x in range(x0, x1):
                offset = (y * width + x) * 3
                pixels[offset : offset + 3] = bytes(color)

    for left in (38, 150, 262):
        rectangle(left, 30, left + 72, 102, (30, 100, 230))
    for left in (94, 218):
        rectangle(left, 126, left + 64, 190, (238, 126, 32))
    rectangle(34, 218, 350, 240, (30, 156, 76))
    scanlines = bytearray()
    row_bytes = width * 3
    for y in range(height):
        scanlines.append(0)
        start = y * row_bytes
        scanlines.extend(pixels[start : start + row_bytes])
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(scanlines), level=9))
        + _png_chunk(b"IEND", b"")
    )


_PROFILE_SCHEMA_PROBE_IMAGE_COUNT = 4
_PROFILE_SCHEMA_PROBE_IMAGE_DIMENSIONS_PX = (1814, 2046)


def _synthetic_profile_schema_probe_png() -> bytes:
    """Create a rights-safe, Phase 0a-sized image with no depicted person."""
    width, height = _PROFILE_SCHEMA_PROBE_IMAGE_DIMENSIONS_PX
    compressor = zlib.compressobj(level=9)
    compressed = bytearray()
    neutral_bands = ((242, 244, 247), (229, 233, 239), (236, 240, 244))
    for y in range(height):
        color = neutral_bands[y // 192 % len(neutral_bands)]
        scanline = b"\x00" + bytes(color) * width
        compressed.extend(compressor.compress(scanline))
    compressed.extend(compressor.flush())
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", bytes(compressed))
        + _png_chunk(b"IEND", b"")
    )


def _validate_visual_probe_response(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or value != {
        "blue_square_count": 3,
        "orange_square_count": 2,
        "green_bar_position": "bottom",
    }:
        raise ValueError("synthetic visual challenge was answered incorrectly")
    return value


def _probe_model_request(client: OpenWebUIClient) -> dict[str, object]:
    """Exercise vision plus the exact Phase 0a wire schema with synthetic input."""
    payload = _synthetic_visual_challenge_png()
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["blue_square_count", "orange_square_count", "green_bar_position"],
        "properties": {
            "blue_square_count": {"type": "integer", "minimum": 0, "maximum": 8},
            "orange_square_count": {"type": "integer", "minimum": 0, "maximum": 8},
            "green_bar_position": {
                "type": "string",
                "enum": ["top", "bottom", "left", "right", "absent"],
            },
        },
    }
    with TemporaryDirectory(prefix="gesture-annotator-model-check-") as directory:
        image = Path(directory) / "synthetic-probe.png"
        image.write_bytes(payload)
        _LOGGER.debug(
            "stage=preflight event=synthetic_vision_probe status=started request_id=model-check-synthetic-vision"
        )
        visual_result = client.complete_json(
            request_id="model-check-synthetic-vision",
            task="capability_probe",
            system_prompt="You are testing the local Communication Gesture Annotator connection. Inspect the supplied synthetic image itself and return only the requested JSON.",
            user_prompt="Count the solid blue squares, count the solid orange squares, and report whether the solid green bar is at the top, bottom, left, right, or absent.",
            images=[image],
            response_schema=schema,
            trace_context={"input": "generated-rights-safe-visual-challenge"},
            response_validator=_validate_visual_probe_response,
        )
        _reject_observed_probe_reasoning(client, visual_result, "simple visual")
        _LOGGER.debug(
            "stage=preflight event=synthetic_vision_probe status=completed request_id=model-check-synthetic-vision"
        )
        _LOGGER.debug(
            f"stage=preflight event=profile_schema_probe status=started request_id=model-check-profile-schema image_count={_PROFILE_SCHEMA_PROBE_IMAGE_COUNT}"
        )
        profile_payload = _synthetic_profile_schema_probe_png()
        profile_images = [
            Path(directory) / f"synthetic-profile-sheet-{index:02d}.png"
            for index in range(1, _PROFILE_SCHEMA_PROBE_IMAGE_COUNT + 1)
        ]
        for profile_image in profile_images:
            profile_image.write_bytes(profile_payload)
        profile_result = client.complete_json(
            request_id="model-check-profile-schema",
            task="overview_segment",
            system_prompt="You are testing the local Communication Gesture Annotator connection. Inspect only the four supplied generated synthetic images and return only JSON matching the requested Phase 0a overview schema.",
            user_prompt="Return a valid empty intermediate profile because this generated image set contains no person. Set target_mode to none; keep selected_subjects, ignored_transient_people, and annotation_relevance_spans empty; give a concise scene_summary; use confidence from 0 through 1; and use only evidence sample index 0.",
            images=profile_images,
            response_schema=PROFILE_SEGMENT_SCHEMA,
            trace_context={
                "input": "generated-rights-safe-profile-schema-probe",
                "image_count": _PROFILE_SCHEMA_PROBE_IMAGE_COUNT,
                "image_dimensions_px": list(_PROFILE_SCHEMA_PROBE_IMAGE_DIMENSIONS_PX),
            },
            response_validator=lambda value: validate_overview_response(
                value, {0}, allow_empty=True, require_nested_evidence_in_root=False
            ),
        )
        _reject_observed_probe_reasoning(client, profile_result, "profile schema")
        _LOGGER.debug(
            f"stage=preflight event=profile_schema_probe status=completed request_id=model-check-profile-schema image_count={_PROFILE_SCHEMA_PROBE_IMAGE_COUNT}"
        )
    return {
        "simple_visual_request": _probe_result_summary(
            visual_result, image_count=1, image_dimensions_px=(384, 256)
        ),
        "profile_schema_request": _probe_result_summary(
            profile_result,
            image_count=_PROFILE_SCHEMA_PROBE_IMAGE_COUNT,
            image_dimensions_px=_PROFILE_SCHEMA_PROBE_IMAGE_DIMENSIONS_PX,
        ),
    }


def _reject_observed_probe_reasoning(
    client: OpenWebUIClient, result: ModelCallResult, probe_name: str
) -> None:
    """Reject a provider that ignores an explicit non-thinking configuration."""
    configured_think = getattr(getattr(client, "inference", None), "think", None)
    if configured_think is False and result.reasoning_observed:
        raise OpenWebUIError(
            f"configured think=false was not honored: model-check observed reasoning in the {probe_name} response; use an Instruct/non-thinking model or a serving runtime version that honors reasoning suppression"
        )


def _probe_result_summary(
    result: ModelCallResult, *, image_count: int, image_dimensions_px: tuple[int, int]
) -> dict[str, object]:
    """Return only non-content capability evidence for one synthetic request."""
    return {
        "status": "passed",
        "response_sha256": result.response_sha256,
        "endpoint_mode": result.endpoint_mode,
        "repaired": result.repaired,
        "image_count": image_count,
        "image_dimensions_px": list(image_dimensions_px),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gesture-annotator", description="Communication Gesture Annotator research backend"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    model_check = subparsers.add_parser(
        "model-check", help="verify the configured Open WebUI vision model"
    )
    model_check.add_argument("--phase2-config", type=Path, required=True)
    model_check.add_argument("--output", type=Path)
    model_check.add_argument(
        "--log-level",
        choices=_LOG_LEVELS,
        default="INFO",
        help="console progress level (default: INFO)",
    )
    llm = subparsers.add_parser(
        "run-llm", help="run preprocessing, scene selection and G-unit detection"
    )
    llm.add_argument("video", type=Path)
    llm.add_argument("--run-dir", type=Path, required=True)
    llm.add_argument("--phase2-config", type=Path, required=True)
    llm.add_argument("--ffmpeg", default="ffmpeg")
    llm.add_argument("--ffprobe", default="ffprobe")
    llm.add_argument("--audio", type=Path, help=argparse.SUPPRESS)
    llm.add_argument("--audio-offset-us", type=int, default=0, help=argparse.SUPPRESS)
    llm.add_argument(
        "--no-audio",
        action="store_true",
        default=True,
        help="compatibility option; audio is disabled by default and unsupported by run-llm",
    )
    llm_existing = llm.add_mutually_exclusive_group()
    llm_existing.add_argument("--force", action="store_true")
    llm_existing.add_argument(
        "--resume", action="store_true", help="resume a validated stopped or incomplete run"
    )
    llm_existing.add_argument(
        "--reuse-preprocessing-from",
        "--use-pre-processing-from",
        type=Path,
        metavar="RUN_DIR",
        help="copy a validated proxy, 10 Hz frames, and raw pose bundle from another run into a fresh run; model-stage artifacts are never reused",
    )
    llm.add_argument("--export-eaf", action="store_true")
    llm.add_argument(
        "--storage-mode",
        choices=("compact", "minimal", "debug"),
        help="minimal (default): bounded image caches and compressed phase results; compact: retain resumable image caches; debug: retain full diagnostics (overrides TOML [storage].mode)",
    )
    llm.add_argument(
        "--skip-phase1",
        action="store_true",
        help="skip scene/speaker selection; preprocessing still runs",
    )
    llm.add_argument(
        "--from-seconds",
        "--from-second",
        type=_analysis_time_us,
        metavar="SECONDS",
        help="whole-program start, rounded to nearest 0.1 s; preprocessing and LLM context stay inside the range",
    )
    llm.add_argument(
        "--to-seconds",
        "--to-second",
        type=_analysis_time_us,
        metavar="SECONDS",
        help="exclusive whole-program end, rounded to nearest 0.1 s; preprocessing and LLM context stay inside the range",
    )
    llm.add_argument(
        "--stop-after",
        choices=PIPELINE_STAGE_ORDER,
        default="phase3",
        help="stop after a validated stage artifact boundary (default: phase3)",
    )
    llm.add_argument(
        "--log-level",
        choices=_LOG_LEVELS,
        default="INFO",
        help="console progress level (default: INFO)",
    )
    export = subparsers.add_parser(
        "export-eaf", help="export LLM-only G-units from saved estimates without model calls"
    )
    export.add_argument("analysis", type=Path, help="completed run directory or annotations.json")
    export.add_argument(
        "--output", type=Path, help="default: annotations.eaf beside annotations.json"
    )
    export.add_argument("--tier-id", default="G-Units")
    export.add_argument("--force", action="store_true", help="replace an existing EAF")
    evaluate = subparsers.add_parser(
        "evaluate-run", help="compare completed results with a reference EAF on the 100 ms grid"
    )
    evaluate.add_argument("--reference-visibility-tier", default="Spker_vis", help="tier defining labelled reference coverage (default: Spker_vis)")
    evaluate.add_argument("analysis", type=Path, help="completed run directory or annotations.json")
    evaluate.add_argument("reference", type=Path, help="reference EAF")
    evaluate.add_argument(
        "--reference-tier", default="Manual_GUnit", help="exact tier ID (default: Manual_GUnit)"
    )
    evaluate.add_argument(
        "--positive-value",
        "--reference-value",
        default="G-unit",
        help="exact positive value (default: G-unit)",
    )
    evaluate.add_argument(
        "--alignment-offset-us",
        type=int,
        default=0,
        help="offset added to reference timestamps, in microseconds",
    )
    evaluate.add_argument(
        "--output",
        type=Path,
        help="JSON path; matching .txt is also written; existing reports are replaced",
    )
    evaluate.add_argument(
        "--export-eaf",
        action="store_true",
        help="also export LLM-only G-units to annotations.eaf",
    )
    return parser


def _run_command(args: argparse.Namespace) -> int:
    if args.command == "evaluate-run":
        from .evaluation.completed_run import evaluate_completed_run

        report = evaluate_completed_run(
            args.analysis,
            args.reference,
            reference_tier=args.reference_tier,
            positive_value=args.positive_value,
            visibility_tier=args.reference_visibility_tier,
            alignment_offset_us=args.alignment_offset_us,
            output=args.output,
            export_eaf=args.export_eaf,
        )
        metrics = report["evaluation"]
        for warning in report["warnings"]:
            print(f"warning: {warning}", file=sys.stderr)
        print(
            f"Evaluated {metrics['interval_count']} samples; excluded {report['scope']['excluded_sample_count']}."
        )
        print(
            "; ".join(
                f"{name}: {100 * metrics[name]:.2f}%"
                if metrics[name] is not None
                else f"{name}: undefined"
                for name in ("precision", "recall", "f1", "false_positive_rate", "accuracy")
            )
        )
        print(
            f"Statistics: {report['output_files']['json']}; text report: {report['output_files']['text']}"
        )
        if args.export_eaf:
            print(f"optional EAF projection written: {report['exported_eaf']}")
        return 0
    if args.command == "export-eaf":
        from .export.completed_run import export_completed_run

        output = export_completed_run(
            args.analysis, args.output, tier_id=args.tier_id, overwrite=args.force
        )
        print(f"optional EAF projection written: {output}")
        return 0
    if args.command == "model-check":
        _LOGGER.info("stage=preflight event=step status=started planned_calls=2")
        research = load_research_config(args.phase2_config)
        client = OpenWebUIClient(research.open_webui, research.inference)
        _LOGGER.debug("stage=preflight event=capability_check status=started")
        capability = dict(client.check_capabilities())
        _LOGGER.debug("stage=preflight event=capability_check status=completed")
        _LOGGER.debug("stage=preflight event=synthetic_request_probes status=started")
        capability["synthetic_request_probe"] = _probe_model_request(client)
        _LOGGER.debug("stage=preflight event=synthetic_request_probes status=completed")
        _LOGGER.info("stage=preflight event=step status=completed completed_calls=2")
        if args.output:
            write_json(args.output, capability)
        print(f"model available: {capability.get('model')} (vision={capability.get('vision')})")
        return 0
    if args.command == "run-llm":
        if args.audio is not None or args.audio_offset_us != 0:
            raise ValueError(
                "audio input is not supported by run-llm; omit --audio and --audio-offset-us"
            )
        _LOGGER.info(
            "stage=preflight event=step status=started planned_calls=%s",
            0 if args.stop_after == "preprocessing" else 2,
        )
        research = load_research_config(args.phase2_config)
        if args.skip_phase1:
            research = replace(
                research, inference=replace(research.inference, phase1_enabled=False)
            )
        if args.from_seconds is not None or args.to_seconds is not None:
            overrides = {
                name: value
                for name, value in (
                    ("phase2_start_us", args.from_seconds),
                    ("phase2_end_us", args.to_seconds),
                )
                if value is not None
            }
            research = replace(research, detection=replace(research.detection, **overrides))
        if research.detection.phase2_start_us or research.detection.phase2_end_us is not None:
            _LOGGER.info(
                "stage=preflight event=processing_time_selection status=configured start_us=%s end_us=%s end_exclusive=true",
                research.detection.phase2_start_us,
                research.detection.phase2_end_us,
            )
        research.validate_export(args.export_eaf, detection_run=True)
        stop_after = validate_pipeline_stage(args.stop_after)
        if not research.inference.phase1_enabled and stop_after in {"phase0a", "phase0b"}:
            raise ValueError(
                "Phase1 is disabled; stop after preprocessing, phase1 (skipped), phase2 or phase3"
            )
        if args.export_eaf and stop_after != "phase3":
            raise ValueError("EAF export requires --stop-after phase3")
        video_path = args.video.resolve()
        if args.resume:
            validate_resume_source(args.run_dir, video_path)
        audio_path = args.audio.resolve() if args.audio else None
        restricted_roots = repository_restricted_roots()
        video_rights = classify_media_rights(video_path, restricted_roots=restricted_roots)
        audio_rights = (
            classify_media_rights(audio_path, restricted_roots=restricted_roots)
            if audio_path is not None
            else None
        )
        combined_rights = combine_rights(
            video_rights, *([audio_rights] if audio_rights is not None else [])
        )
        resolved_run_dir = args.run_dir.resolve()
        if video_path.is_relative_to(resolved_run_dir) or (
            audio_path is not None and audio_path.is_relative_to(resolved_run_dir)
        ):
            raise ValueError(
                "Research Stage 2 source media must stay outside its generated run directory"
            )
        restricted_input = is_restricted_rights(combined_rights)
        transport_mode: str | None = None
        if stop_after != "preprocessing":
            destination_probe = OpenWebUIClient(research.open_webui, research.inference)
            transport_mode = restricted_transport_mode(
                restricted=restricted_input,
                base_url=research.open_webui.base_url,
                destination_scope=destination_probe.destination_scope(),
                verify_tls=research.open_webui.verify_tls,
                allow_non_private_network=research.open_webui.allow_non_private_network,
                operator_asserts_local_provider=research.open_webui.operator_asserts_local_provider,
                operator_authorizes_restricted_remote_processing=research.open_webui.operator_authorizes_restricted_remote_processing,
            )
        _LOGGER.debug(
            "stage=preflight event=transport_policy status=completed mode=%s",
            "no_model_transport"
            if stop_after == "preprocessing"
            else transport_mode or "ordinary_input",
        )
        if transport_mode == "operator_authorized_remote_processing":
            print(
                "warning: restricted media derivatives will be sent to the configured remote model endpoint under the recorded operator authorization; the software does not determine whether that use is permitted",
                file=sys.stderr,
            )
        if restricted_input:
            _assert_restricted_run_destination(resolved_run_dir)
        root_owned = (
            args.run_dir / "recording-overview.json",
            args.run_dir / "scene-usability.json",
            args.run_dir / "recording-profile.json",
            args.run_dir / "model-capabilities.json",
            data_path(args.run_dir / "phase2-evidence.json"),
            args.run_dir / "inference-plan.json",
            args.run_dir / "analysis-index.json",
            args.run_dir / "pipeline-status.json",
        )
        if args.resume:
            try:
                validate_model_run_resume_state(args.run_dir)
            except ValueError:
                reuse_manifest = args.run_dir / "preprocessing-reuse.json"
                recovery_blockers = [
                    path
                    for path in (
                        *_existing_analysis_artifacts(args.run_dir),
                        *(path for path in root_owned if path.exists()),
                    )
                    if path.name != "preprocessing-reuse.json"
                ]
                checkpoint_root = args.run_dir / "model-checkpoints"
                checkpoint_present = checkpoint_root.is_dir() and any(
                    checkpoint_root.glob("checkpoint-*.json")
                )
                if (
                    not reuse_manifest.is_file()
                    or (args.run_dir / "pipeline-status.json").exists()
                    or recovery_blockers
                    or checkpoint_present
                ):
                    raise
        else:
            existing = _existing_analysis_artifacts(args.run_dir)
            existing.extend((path for path in root_owned if path.exists()))
            if existing and (not args.force):
                raise FileExistsError(existing[0])
        if args.reuse_preprocessing_from is not None and args.run_dir.exists():
            is_junction = getattr(args.run_dir, "is_junction", lambda: False)()
            if (
                not args.run_dir.is_dir()
                or args.run_dir.is_symlink()
                or is_junction
                or any(args.run_dir.iterdir())
            ):
                raise FileExistsError(args.run_dir)
        scoped_reuse_root = None
        scoped_processing = bool(
            research.detection.phase2_start_us or research.detection.phase2_end_us is not None
        )
        if scoped_processing and args.reuse_preprocessing_from is not None:
            scoped_reuse_root = args.reuse_preprocessing_from.resolve()
            args.reuse_preprocessing_from = None
        reused_preprocessing = None
        if args.resume and (args.run_dir / "preprocessing-reuse.json").is_file():
            if scoped_processing:
                raise ValueError(
                    "this run imported recording-wide preprocessing before whole-program ranges; use a new --run-dir with --reuse-preprocessing-from to create the selected excerpt"
                )
            _LOGGER.debug("stage=preprocessing event=local_reuse_validation status=started")
            reused_preprocessing = load_reused_preprocessing(
                args.run_dir,
                source_path=args.video,
                analysis_config=research.analysis,
                inference_config=research.inference,
                pose_config=research.pose_preprocessing,
                source_rights=video_rights,
                audio_path=args.audio,
                audio_offset_us=args.audio_offset_us,
                include_audio=not args.no_audio,
                sidecar_audio_rights=audio_rights,
                ffmpeg=args.ffmpeg,
            )
            _LOGGER.debug("stage=preprocessing event=local_reuse_validation status=completed")
        if args.reuse_preprocessing_from is not None:
            _LOGGER.debug("stage=preprocessing event=cross_run_reuse status=started")
            reused_preprocessing = reuse_preprocessing(
                args.reuse_preprocessing_from,
                args.run_dir,
                source_path=args.video,
                analysis_config=research.analysis,
                inference_config=research.inference,
                pose_config=research.pose_preprocessing,
                source_rights=video_rights,
                audio_path=args.audio,
                audio_offset_us=args.audio_offset_us,
                include_audio=not args.no_audio,
                sidecar_audio_rights=audio_rights,
                ffmpeg=args.ffmpeg,
            )
            _LOGGER.debug("stage=preprocessing event=cross_run_reuse status=completed")
        from .storage import VERSION as storage_version, compact_enabled, current_mode, minimal_enabled, overlay_omitted

        trace = TraceWriter(
            args.run_dir,
            overwrite=args.force,
            append_exchanges=args.resume,
            storage_mode="compact" if compact_enabled() else "debug",
        )
        write_json(
            args.run_dir / "storage-policy.json",
            {
                "policy_version": storage_version,
                "mode": current_mode(),
                "resume_supported": True,
                "model_inputs_unchanged": True,
                "regenerable_phase2_images_retained": not compact_enabled(),
                "bounded_crop_cache": minimal_enabled(),
                "large_phase_results_compressed": minimal_enabled(),
            },
        )
        if args.reuse_preprocessing_from is not None:
            trace.log_event(
                "preprocessing",
                "cross_run_reuse",
                "completed",
                {"source_path_logged": False, "model_artifacts_reused": False},
            )
        client: OpenWebUIClient | None = None
        capability: dict[str, object] = {}
        if stop_after != "preprocessing":
            client = OpenWebUIClient(research.open_webui, research.inference, trace_writer=trace)
            _LOGGER.debug("stage=preflight event=capability_check status=started")
            capability = dict(client.check_capabilities())
            _LOGGER.debug("stage=preflight event=capability_check status=completed")
            _LOGGER.debug("stage=preflight event=synthetic_request_probes status=started")
            capability["synthetic_request_probe"] = _probe_model_request(client)
            _LOGGER.debug("stage=preflight event=synthetic_request_probes status=completed")
            trace.log_event(
                "preflight", "model_capability", "completed", {"capability": capability}
            )
        trace.log_event(
            "preflight",
            "step",
            "completed",
            {"completed_calls": 0 if stop_after == "preprocessing" else 2},
        )
        media_cache_times = {
            name: (args.run_dir / name).stat().st_mtime_ns
            if args.resume and (args.run_dir / name).is_file() else None
            for name in ("preprocessing.json", "analysis-frames/rights-manifest.json")
        }
        preprocessing_progress = StepProgress(trace, "preprocessing", 0)
        trace.log_event(
            "preprocessing",
            "media_proxy",
            "started",
            {
                "source_path_logged": False,
                "source_kind": "configured local video",
                "reused": reused_preprocessing is not None,
            },
        )
        if reused_preprocessing is None:
            manifest = preprocess_video(
                args.video,
                args.run_dir,
                config=research.analysis,
                audio_path=args.audio,
                audio_offset_us=args.audio_offset_us,
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
                include_audio=not args.no_audio,
                overwrite=args.force or args.resume,
                resume=args.resume,
                source_rights=video_rights,
                sidecar_audio_rights=audio_rights,
                start_us=research.detection.phase2_start_us,
                end_us=research.detection.phase2_end_us,
            )
            samples = [AnalysisSample(**item) for item in manifest["proxy"]["frame_mapping"]]
        else:
            manifest = dict(reused_preprocessing.manifest)
            samples = list(reused_preprocessing.samples)
        processing_bounds = manifest.get(
            "processing_scope", {"start_us": 0, "end_us": manifest["source"]["duration_us"]}
        )
        processed_sample_count = sum(
            (s.source_gap_reason != "outside_processing_range" for s in samples)
        )
        trace.log_event(
            "preprocessing",
            "media_proxy",
            "completed",
            {
                "source_id": manifest["source"]["source_id"],
                "duration_us": processing_bounds["end_us"] - processing_bounds["start_us"],
                "original_duration_us": manifest["source"]["duration_us"],
                "audio_supplied": manifest["proxy"]["audio_supplied"],
                "reused": reused_preprocessing is not None or (
                    media_cache_times["preprocessing.json"] is not None
                    and (args.run_dir / "preprocessing.json").stat().st_mtime_ns
                    == media_cache_times["preprocessing.json"]
                ),
            },
        )
        trace.log_event(
            "preprocessing",
            "analysis_frames",
            "started",
            {
                "sample_count": processed_sample_count,
                "timeline_sample_count": len(samples),
                "analysis_rate_hz": research.analysis.sample_rate_hz,
                "reused": reused_preprocessing is not None,
            },
        )
        if reused_preprocessing is None:
            frames = extract_analysis_frames(
                args.run_dir / manifest["proxy"]["relative_path"],
                samples,
                args.run_dir / "analysis-frames",
                ffmpeg=args.ffmpeg,
                maximum_width=research.inference.frame_width,
                jpeg_quality=research.inference.jpeg_quality,
                overwrite=args.force or args.resume,
                resume=args.resume,
                reuse_from=scoped_reuse_root,
            )
        else:
            frames = [dict(item) for item in reused_preprocessing.frame_records]
        trace.log_event(
            "preprocessing",
            "analysis_frames",
            "completed",
            {
                "frame_count": processed_sample_count,
                "canonical_range_us": [processing_bounds["start_us"], processing_bounds["end_us"]],
                "reused": reused_preprocessing is not None or (
                    media_cache_times["analysis-frames/rights-manifest.json"] is not None
                    and (args.run_dir / "analysis-frames/rights-manifest.json").stat().st_mtime_ns
                    == media_cache_times["analysis-frames/rights-manifest.json"]
                ),
            },
        )
        trace.log_event(
            "preprocessing",
            "whole_body_pose",
            "started",
            {
                "source_frame_count": sum(
                    (
                        f.get("processing_status") != "outside_processing_range"
                        for f in manifest["source"]["video_frames"]
                    )
                ),
                "analysis_sample_count": processed_sample_count,
                "input": "original_source_video_at_full_decoded_frame_rate",
                "consumed_by_model_stages": [],
                "reused": reused_preprocessing is not None,
            },
        )
        preprocessing_path = args.run_dir / "preprocessing.json"
        if reused_preprocessing is None:
            pose_result = run_pose_preprocessing(
                args.video,
                manifest["source"]["video_frames"],
                samples,
                args.run_dir,
                config=research.pose_preprocessing,
                rights=manifest["source"]["rights"],
                expected_source_sha256=manifest["source"]["sha256"],
                expected_source_byte_size=manifest["source"]["byte_size"],
                preprocessing_sha256=sha256(preprocessing_path.read_bytes()).hexdigest(),
                ffmpeg=args.ffmpeg,
                overwrite=args.force,
                resume=args.resume,
                reuse_from=scoped_reuse_root,
            )
        else:
            pose_result = reused_preprocessing.pose_result
        pose_coverage = pose_result.manifest["coverage"]
        trace.log_event(
            "preprocessing",
            "whole_body_pose",
            "completed",
            {
                "source_frame_count": pose_coverage["source_frame_count"],
                "analysis_sample_count": pose_coverage["analysis_sample_count"],
                "frames_with_people": pose_coverage["frames_with_people"],
                "person_observation_count": pose_coverage["person_observation_count"],
                "reused": pose_result.reused,
                "consumed_by_model_stages": [],
            },
        )
        trace.log_event(
            "preprocessing",
            "pose_conditioning",
            "started",
            {
                "source_rate_smoothing": True,
                "projection_rate_hz": research.analysis.sample_rate_hz,
                "analysis_sample_count": len(samples),
                "raw_pose_consumed_by_model_stages": [],
                "derived_conditioning_consumed_by_model_stages": ["phase1", "phase2"],
            },
        )
        pose_conditioning = run_pose_conditioning(
            pose_result,
            samples,
            frames,
            args.run_dir,
            rights=manifest["source"]["rights"],
            overwrite=args.force,
            resume=args.resume,
            ffmpeg=args.ffmpeg,
            source_path=video_path,
            progress=lambda event, status, details: trace.log_event(
                "preprocessing", event, status, details
            ),
        )
        conditioning_coverage = pose_conditioning.manifest["coverage"]
        conditioning_policy = pose_conditioning.manifest["policy"]
        trace.log_event(
            "preprocessing",
            "pose_conditioning",
            "completed",
            {
                "source_frame_count": conditioning_coverage["source_frame_count"],
                "analysis_sample_count": conditioning_coverage["analysis_sample_count"],
                "feature_record_count": conditioning_coverage["feature_record_count"],
                "pose_guided_crop_count": conditioning_coverage["unique_person_crop_count"],
                "full_frame_fallback_count": conditioning_coverage["full_frame_fallback_count"],
                "smoothing_policy_version": conditioning_policy["smoothing_policy_version"],
                "reused": pose_conditioning.reused,
            },
        )
        preprocessing_status_path = write_pipeline_status(
            args.run_dir,
            requested_stop_after=stop_after,
            completed_stage="preprocessing",
            artifacts=(
                preprocessing_path,
                args.run_dir / manifest["proxy"]["relative_path"],
                pose_result.manifest_path,
                pose_result.observations_path,
                pose_result.observations_rights_path,
                *(
                    ()
                    if overlay_omitted(pose_result.manifest)
                    else (
                        pose_result.overlay_path,
                        pose_result.overlay_path.with_name(
                            pose_result.overlay_path.name + ".rights.json"
                        ),
                    )
                ),
                pose_conditioning.manifest_path,
                pose_conditioning.features_path,
                pose_conditioning.features_rights_path,
                pose_conditioning.focus_rights_manifest_path,
                *(
                    ()
                    if overlay_omitted(pose_conditioning.manifest)
                    else (
                        pose_conditioning.conditioned_overlay_path,
                        pose_conditioning.conditioned_overlay_rights_path,
                    )
                ),
                *(
                    (reused_preprocessing.reuse_manifest_path,)
                    if reused_preprocessing is not None
                    else ()
                ),
                *(
                    (args.run_dir / "analysis-frames" / "rights-manifest.json",)
                    if (args.run_dir / "analysis-frames" / "rights-manifest.json").is_file()
                    else ()
                ),
            ),
            source_sha256=manifest["source"]["sha256"],
            preprocessing_sha256=sha256(preprocessing_path.read_bytes()).hexdigest(),
        )
        preprocessing_progress.finish(sample_count=processed_sample_count)
        if stop_after == "preprocessing":
            trace.log_event(
                "pipeline",
                "requested_stop",
                "completed",
                {
                    "completed_stage": "preprocessing",
                    "pipeline_status": preprocessing_status_path.name,
                },
            )
            print(
                f"completed preprocessing: raw and body-relative pose evidence, smoothed focus evidence and focus frames are in {pose_result.directory}"
            )
            return 0
        assert client is not None
        preprocessing_reuse_identity = (
            {
                "contract_version": PREPROCESSING_REUSE_CONTRACT_VERSION,
                "copy_policy": "validated-preprocessing-only-v0.1",
                "manifest": "preprocessing-reuse.json",
                "manifest_sha256": sha256(
                    reused_preprocessing.reuse_manifest_path.read_bytes()
                ).hexdigest(),
                "source_bundle_sha256": reused_preprocessing.source_bundle_sha256,
                "model_artifacts_reused": False,
            }
            if reused_preprocessing is not None
            else None
        )
        result = run_phase2_pipeline(
            run_dir=args.run_dir,
            source=manifest["source"],
            preprocessing=manifest,
            analysis_samples=samples,
            extracted_frames=frames,
            config=research,
            model=client,
            audio_supplied=bool(manifest["proxy"]["audio_supplied"]),
            ffmpeg=args.ffmpeg,
            export_eaf=args.export_eaf,
            stop_after=stop_after,
            overwrite=args.force,
            resume=args.resume,
            trace=trace,
            preflight_capabilities=capability,
            pose_conditioning=pose_conditioning,
            preprocessing_reuse=preprocessing_reuse_identity,
        )
        if result.completed_stage == "phase3":
            print(
                f"completed run {result.run_id}: {len(result.analyses)} selected subject analysis file(s)"
            )
        else:
            print(
                f"completed through {result.completed_stage}; status recorded in {result.status_path}"
            )
        return 0
    raise AssertionError("unhandled command")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging_state = _configure_cli_logging(args.log_level) if hasattr(args, "log_level") else None
    try:
        from .storage import storage_mode

        mode = (
            args.storage_mode or load_research_config(args.phase2_config).storage.mode
            if args.command == "run-llm"
            else "debug"
        )
        with storage_mode(mode):
            return _run_command(args)
    except (
        FileNotFoundError,
        FileExistsError,
        MediaRuntimeError,
        OpenWebUIError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if logging_state is not None:
            _restore_cli_logging(logging_state)
