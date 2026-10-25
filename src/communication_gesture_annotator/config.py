# SPDX-License-Identifier: Apache-2.0
"""Current pipeline settings. Credentials are resolved from the environment."""

from __future__ import annotations
from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path
import re
from typing import Any, ClassVar, Literal, Mapping
import tomllib
from urllib.parse import urlsplit

_ENVIRONMENT_VARIABLE = re.compile("[A-Za-z_][A-Za-z0-9_]*")
_HTTP_HEADER_NAME = re.compile("[!#$%&'*+.^_`|~0-9A-Za-z-]+")


def _positive(value: Any, name: str, *, zero: bool = False) -> None:
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f"{name} must be an integer >= {(0 if zero else 1)}")


def _number(value: Any, name: str, low: float, high: float) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or (not low <= value <= high):
        raise ValueError(f"{name} must be finite in [{low}, {high}]")


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Settings for analysis."""

    sample_rate_hz: int = 10
    minimum_separating_rest_us: int = 300000
    maximum_bridged_missing_samples: int = 1
    maximum_proxy_width: int = 960
    eaf_tier_id: str = "G-Units"

    @property
    def sample_period_us(self) -> int:
        return 1000000 // self.sample_rate_hz

    contract_version: ClassVar[str] = "0.1.0"

    def __post_init__(self) -> None:
        if type(self.sample_rate_hz) is not int or self.sample_rate_hz != 10:
            raise ValueError("The current annotation grid is fixed at 10 Hz")
        _positive(self.minimum_separating_rest_us, "minimum_separating_rest_us")
        _positive(self.maximum_proxy_width, "maximum_proxy_width")
        if self.maximum_proxy_width < 2:
            raise ValueError("maximum_proxy_width must be at least 2")
        if (
            type(self.maximum_bridged_missing_samples) is not int
            or not 0 <= self.maximum_bridged_missing_samples <= 2
        ):
            raise ValueError("maximum_bridged_missing_samples must be 0, 1 or 2")
        if not isinstance(self.eaf_tier_id, str) or not self.eaf_tier_id.strip():
            raise ValueError("eaf_tier_id is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OpenWebUIConfig:
    """Settings for open webui."""

    base_url: str = "http://127.0.0.1:3000"
    model: str = ""
    username: str | None = None
    api_key_env: str = "GESTURE_ANNOTATOR_OPEN_WEBUI_API_KEY"
    auth_mode: str = "bearer"
    api_key_header: str = "x-api-key"
    timeout_seconds: float = 900.0
    maximum_transient_http_retries: int = 3
    transient_http_retry_backoff_seconds: float = 30.0
    transient_http_retry_delays_seconds: tuple[float, ...] | None = (30.0, 60.0, 60.0)
    inter_request_cooldown_seconds: float = 0.5
    maximum_response_bytes: int = 16777216
    maximum_request_bytes: int = 67108864
    verify_tls: bool = True
    allow_non_private_network: bool = False
    require_vision: bool = True
    operator_asserts_local_provider: bool = False
    operator_authorizes_restricted_remote_processing: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("Open WebUI base_url is required")
        parsed_url = urlsplit(self.base_url)
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.hostname
            or parsed_url.username is not None
            or (parsed_url.password is not None)
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ValueError(
                "Open WebUI base_url must be an HTTP(S) URL without credentials, query, or fragment"
            )
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Open WebUI model is required")
        if self.username is not None and (
            not isinstance(self.username, str) or not self.username.strip()
        ):
            raise ValueError("Open WebUI username must be a non-blank string or null")
        if not isinstance(self.api_key_env, str) or not _ENVIRONMENT_VARIABLE.fullmatch(
            self.api_key_env
        ):
            raise ValueError("api_key_env must be a valid environment-variable name")
        if self.auth_mode not in {"bearer", "custom_header"}:
            raise ValueError("Open WebUI auth_mode must be bearer or custom_header")
        if not isinstance(self.api_key_header, str) or not _HTTP_HEADER_NAME.fullmatch(
            self.api_key_header
        ):
            raise ValueError("api_key_header must be a valid HTTP header name")
        forbidden_headers = {
            "connection",
            "content-length",
            "content-type",
            "host",
            "proxy-authorization",
            "transfer-encoding",
        }
        if self.auth_mode == "custom_header" and self.api_key_header.lower() in forbidden_headers:
            raise ValueError("api_key_header cannot replace a transport-controlled header")
        if (
            not isinstance(self.timeout_seconds, (int, float))
            or isinstance(self.timeout_seconds, bool)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("Open WebUI timeout_seconds must be positive")
        if (
            type(self.maximum_transient_http_retries) is not int
            or not 0 <= self.maximum_transient_http_retries <= 3
        ):
            raise ValueError(
                "Open WebUI maximum_transient_http_retries must be an integer from 0 to 3"
            )
        if (
            not isinstance(self.transient_http_retry_backoff_seconds, (int, float))
            or isinstance(self.transient_http_retry_backoff_seconds, bool)
            or (not math.isfinite(self.transient_http_retry_backoff_seconds))
            or (not 0 <= self.transient_http_retry_backoff_seconds <= 60)
        ):
            raise ValueError(
                "Open WebUI transient_http_retry_backoff_seconds must be finite and from 0 to 60"
            )
        if (
            not isinstance(self.inter_request_cooldown_seconds, (int, float))
            or isinstance(self.inter_request_cooldown_seconds, bool)
            or (not math.isfinite(self.inter_request_cooldown_seconds))
            or (not 0 <= self.inter_request_cooldown_seconds <= 60)
        ):
            raise ValueError(
                "Open WebUI inter_request_cooldown_seconds must be finite and from 0 to 60"
            )
        delays = self.transient_http_retry_delays_seconds
        if delays is not None:
            if (
                not isinstance(delays, (list, tuple))
                or not self.maximum_transient_http_retries <= len(delays) <= 3
                or any(
                    (
                        isinstance(delay, bool)
                        or not isinstance(delay, (int, float))
                        or (not math.isfinite(delay))
                        or (not 0 <= delay <= 3600)
                        for delay in delays
                    )
                )
            ):
                raise ValueError(
                    "transient_http_retry_delays_seconds must contain up to three finite delays from 0 to 3600 seconds, with an entry for each enabled retry"
                )
            object.__setattr__(
                self,
                "transient_http_retry_delays_seconds",
                tuple((float(delay) for delay in delays)),
            )
        for name in ("maximum_response_bytes", "maximum_request_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Open WebUI {name} must be a positive integer")
        for name in (
            "verify_tls",
            "allow_non_private_network",
            "require_vision",
            "operator_asserts_local_provider",
            "operator_authorizes_restricted_remote_processing",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"Open WebUI {name} must be Boolean")
        if (
            self.operator_asserts_local_provider
            and self.operator_authorizes_restricted_remote_processing
        ):
            raise ValueError(
                "local-provider assertion and restricted-remote authorization are mutually exclusive"
            )
        if self.allow_non_private_network and (parsed_url.scheme != "https" or not self.verify_tls):
            raise ValueError(
                "non-private Open WebUI endpoints require HTTPS certificate verification"
            )

    contract_version: ClassVar[str] = "0.5.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PosePreprocessingConfig:
    """Settings for pose preprocessing."""

    backend: str = "onnxruntime"
    device: str = "cpu"
    detector_model: str = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
    pose_model: str = "https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx"
    detector_input_width: int = 640
    detector_input_height: int = 640
    pose_input_width: int = 288
    pose_input_height: int = 384
    keypoint_confidence_threshold: float = 0.4
    tracking_iou_threshold: float = 0.3
    maximum_track_gap_us: int = 200000
    model_depth_range: float = 2.1744869
    overlay_crf: int = 20

    def __post_init__(self) -> None:
        if self.backend != "onnxruntime":
            raise ValueError("pose preprocessing backend must be onnxruntime")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("pose preprocessing device must be non-blank")
        if self.device not in {"cpu", "cuda", "rocm", "mps"} and (
            not re.fullmatch("cuda:[0-9]+", self.device)
        ):
            raise ValueError("pose preprocessing device must be cpu, cuda, cuda:N, rocm, or mps")
        for name in ("detector_model", "pose_model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a local path or HTTPS URL")
            parsed = urlsplit(value)
            if "://" in value and (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username is not None
                or (parsed.password is not None)
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    f"{name} URL must use HTTPS without credentials, query, or fragment"
                )
        for name in (
            "detector_input_width",
            "detector_input_height",
            "pose_input_width",
            "pose_input_height",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("keypoint_confidence_threshold", "tracking_iou_threshold"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or (not math.isfinite(value))
                or (not 0.0 <= value <= 1.0)
            ):
                raise ValueError(f"{name} must be a finite probability in [0, 1]")
        if self.tracking_iou_threshold == 0:
            raise ValueError("tracking_iou_threshold must be greater than zero")
        if (
            type(self.maximum_track_gap_us) is not int
            or not 0 <= self.maximum_track_gap_us <= 10000000
        ):
            raise ValueError("maximum_track_gap_us must be an integer in [0, 10000000]")
        if (
            not isinstance(self.model_depth_range, (int, float))
            or isinstance(self.model_depth_range, bool)
            or (not math.isfinite(self.model_depth_range))
            or (self.model_depth_range <= 0)
        ):
            raise ValueError("model_depth_range must be finite and positive")
        if type(self.overlay_crf) is not int or not 0 <= self.overlay_crf <= 51:
            raise ValueError("overlay_crf must be an integer in [0, 51]")

    contract_version: ClassVar[str] = "0.2.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Settings for inference."""

    phase1_enabled: bool = True
    temperature: float = 0.1
    seed: int | None = 0
    think: bool | Literal["low", "medium", "high", "max"] = False
    model_context_window_tokens: int = 131072
    maximum_output_tokens: int = 65536
    maximum_prompt_characters: int = 2000000
    maximum_repair_attempts: int = 1
    profile_samples: int = 24
    profile_sample_period_us: int = 2000000
    profile_window_us: int = 240000000
    profile_overlap_us: int = 60000000
    profile_consolidation_batch_size: int = 8
    profile_summary_maximum_characters: int = 1000
    visibility_max_frames: int = 5
    maximum_images_per_request: int = 72
    maximum_image_bytes_per_request: int = 47185920
    maximum_total_image_pixels: int = 50000000
    maximum_image_width: int = 2048
    maximum_image_height: int = 2048
    frame_width: int = 640
    jpeg_quality: int = 85
    profile_image_mode: str = "contact_sheet"
    profile_tile_width: int = 452
    profile_tile_height: int = 254
    profile_sheet_columns: int = 4
    profile_sheet_rows: int = 8
    profile_jpeg_quality: int = 85
    visibility_image_mode: str = "contact_sheet"
    visibility_tile_width: int = 640
    visibility_tile_height: int = 360
    visibility_sheet_columns: int = 3
    visibility_sheet_rows: int = 1
    visibility_jpeg_quality: int = 85
    activity_image_mode: str = "contact_sheet"
    activity_tile_width: int = 640
    activity_tile_height: int = 360
    activity_sheet_columns: int = 3
    activity_sheet_rows: int = 1
    activity_jpeg_quality: int = 85
    contact_sheet_gutter: int = 2

    def contact_sheet_dimensions(self, stage: str) -> tuple[int, int]:
        """Return the exact configured contact-sheet output dimensions."""
        prefix = self._stage_prefix(stage)
        columns = getattr(self, f"{prefix}_sheet_columns")
        rows = getattr(self, f"{prefix}_sheet_rows")
        width = getattr(self, f"{prefix}_tile_width")
        height = getattr(self, f"{prefix}_tile_height")
        return (
            columns * width + (columns - 1) * self.contact_sheet_gutter,
            rows * height + (rows - 1) * self.contact_sheet_gutter,
        )

    def stage_sample_capacity(self, stage: str) -> int:
        """Return how many samples one request can carry for a visual stage."""
        prefix = self._stage_prefix(stage)
        if getattr(self, f"{prefix}_image_mode") == "individual":
            return self.maximum_images_per_request
        return (
            self.maximum_images_per_request
            * getattr(self, f"{prefix}_sheet_columns")
            * getattr(self, f"{prefix}_sheet_rows")
        )

    def packaged_image_count(self, stage: str, sample_count: int) -> int:
        """Return the image count required to transport ``sample_count`` samples."""
        prefix = self._stage_prefix(stage)
        if type(sample_count) is not int or sample_count <= 0:
            raise ValueError("sample_count must be a positive integer")
        if getattr(self, f"{prefix}_image_mode") == "individual":
            return sample_count
        cells = getattr(self, f"{prefix}_sheet_columns") * getattr(self, f"{prefix}_sheet_rows")
        return (sample_count + cells - 1) // cells

    def packaged_pixel_count(self, stage: str, sample_count: int) -> int | None:
        """Return planned contact-sheet pixels, or null for individual images.

        Individual-frame height depends on source aspect ratio and is therefore
        checked when the images are prepared rather than inferred here.
        """
        prefix = self._stage_prefix(stage)
        if getattr(self, f"{prefix}_image_mode") == "individual":
            return None
        width, height = self.contact_sheet_dimensions(prefix)
        return self.packaged_image_count(prefix, sample_count) * width * height

    contract_version: ClassVar[str] = "0.8.0"

    def _stage_prefix(self, stage: str) -> str:
        if stage not in {"profile", "visibility", "activity"}:
            raise ValueError("unsupported visual stage")
        return stage

    def __post_init__(self) -> None:
        if type(self.phase1_enabled) is not bool:
            raise ValueError("phase1_enabled must be Boolean")
        _number(self.temperature, "temperature", 0, 2)
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if not (type(self.think) is bool or self.think in ("low", "medium", "high", "max")):
            raise ValueError("invalid think setting")
        zero = {"seed", "maximum_repair_attempts", "profile_overlap_us", "contact_sheet_gutter"}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name not in {"phase1_enabled", "temperature", "seed", "think"} and (
                not f.name.endswith("_image_mode")
            ):
                _positive(value, f.name, zero=f.name in zero)
        if (
            self.profile_sample_period_us % 100000
            or self.profile_overlap_us >= self.profile_window_us
        ):
            raise ValueError(
                "profile sampling must align to 10 Hz and overlap must be shorter than the window"
            )
        if self.maximum_output_tokens >= self.model_context_window_tokens:
            raise ValueError("output budget must leave input context")
        if self.profile_consolidation_batch_size < 2:
            raise ValueError("profile_consolidation_batch_size must be >= 2")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be 1..100")
        if self.contact_sheet_gutter > 128:
            raise ValueError("contact_sheet_gutter must be <= 128")
        for stage in ("profile", "visibility", "activity"):
            if getattr(self, f"{stage}_image_mode") not in {"individual", "contact_sheet"}:
                raise ValueError("invalid image mode")
            w, h = self.contact_sheet_dimensions(stage)
            if w > self.maximum_image_width or h > self.maximum_image_height:
                raise ValueError(f"{stage} sheet exceeds image dimensions")
            for side in ("width", "height"):
                v = getattr(self, f"{stage}_tile_{side}")
                if v < 16 or v % 2:
                    raise ValueError("tile dimensions must be even and >=16")
            if not 1 <= getattr(self, f"{stage}_jpeg_quality") <= 100:
                raise ValueError("image quality must be 1..100")
            if getattr(self, f"{stage}_sheet_columns") * getattr(self, f"{stage}_sheet_rows") > 400:
                raise ValueError("too many sheet cells")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Settings for detection."""

    motion_transport_scale: float = 1.0
    motion_orientation_scale: float = 3.0
    motion_finger_scale: float = 1.0
    motion_articulation_scale: float = 5.0
    motion_hand_weight: float = 0.25
    communication_rest_threshold: float = 0.5
    communication_rest_seed_threshold: float = 0.65
    communication_minimum_departure_us: int = 300000
    # The model still sees motion; this selects the score used after its response.
    probability_score_source: str = "llm"
    probability_context_strength: float = 4.0
    probability_target_frames: int = 3
    probability_history_frames: int = 47
    probability_future_frames: int = 10
    probability_history_enabled: bool = True
    probability_motion_trace_enabled: bool = True
    motion_outlier_window_us: int = 1500000
    motion_outlier_z: float = 8.0
    motion_outlier_ratio: float = 6.0
    phase2_start_us: int = 0
    phase2_end_us: int | None = None
    finalization_merge_gap_ms: int = 300

    def model_input_dict(self) -> dict[str, Any]:
        result = self.to_dict()
        result.pop("finalization_merge_gap_ms", None)
        result.pop("probability_score_source", None)
        return result

    contract_version: ClassVar[str] = "0.2.0"

    @property
    def effective_finalization_merge_gap_ms(self) -> int:
        return self.finalization_merge_gap_ms

    def __post_init__(self) -> None:
        for key in (
            "probability_target_frames",
            "probability_history_frames",
            "probability_future_frames",
            "finalization_merge_gap_ms",
            "communication_minimum_departure_us",
            "phase2_start_us",
        ):
            _positive(getattr(self, key), key, zero=key != "probability_target_frames")
        for key in ("probability_history_enabled", "probability_motion_trace_enabled"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be Boolean")
        if self.probability_score_source not in ("llm", "motion_fusion"):
            raise ValueError("probability_score_source must be llm or motion_fusion")
        _number(self.probability_context_strength, "probability_context_strength", 0, 6)
        for key in (
            "motion_transport_scale",
            "motion_orientation_scale",
            "motion_finger_scale",
            "motion_articulation_scale",
        ):
            _number(getattr(self, key), key, 1e-06, 1000000.0)
        _number(self.motion_hand_weight, "motion_hand_weight", 1e-06, 1)
        if not 0 < self.communication_rest_threshold <= self.communication_rest_seed_threshold < 1:
            raise ValueError("rest thresholds must satisfy 0 < low <= seed < 1")
        for key in ("motion_outlier_z", "motion_outlier_ratio"):
            _number(getattr(self, key), key, 2, 30)
        if (
            type(self.motion_outlier_window_us) is not int
            or not 900000 <= self.motion_outlier_window_us <= 10000000
            or self.motion_outlier_window_us % 100000
        ):
            raise ValueError("outlier window must be 0.9..10 s on the 10 Hz grid")
        if self.phase2_start_us % 100000:
            raise ValueError("start must align to 10 Hz")
        if self.phase2_end_us is not None and (
            type(self.phase2_end_us) is not int
            or self.phase2_end_us <= self.phase2_start_us
            or self.phase2_end_us % 100000
        ):
            raise ValueError("exclusive end must exceed start and align to 10 Hz")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StorageConfig:
    """Disk retention only; excluded from inference and checkpoint identities."""

    mode: str = "minimal"

    def __post_init__(self) -> None:
        if self.mode not in {"compact", "minimal", "debug"}:
            raise ValueError("storage mode must be compact, minimal or debug")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    open_webui: OpenWebUIConfig
    storage: StorageConfig = field(default_factory=StorageConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    pose_preprocessing: PosePreprocessingConfig = field(default_factory=PosePreprocessingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)

    def __post_init__(self) -> None:
        budget = (
            4 * ((self.inference.maximum_image_bytes_per_request + 2) // 3)
            + self.inference.maximum_prompt_characters
            + 65536
        )
        if budget > self.open_webui.maximum_request_bytes:
            raise ValueError("prompt and image limits exceed maximum_request_bytes")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name).to_dict() for f in fields(self)}

    def detection_dict(self) -> dict[str, Any]:
        snapshot = self.to_dict()
        snapshot.pop("storage")
        return snapshot

    def model_input_dict(self) -> dict[str, Any]:
        snapshot = self.detection_dict()
        snapshot["detection"] = self.detection.model_input_dict()
        return snapshot

    def validate_export(self, export_eaf: bool, *, detection_run: bool = False) -> None:
        if type(export_eaf) is not bool:
            raise ValueError("export_eaf must be Boolean")


def _construct_config(cls: type, values: Mapping[str, Any], name: str) -> Any:
    if not isinstance(values, dict):
        raise ValueError(f"[{name}] must be a table")
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown [{name}] settings: " + ", ".join(sorted(unknown)))
    return cls(**values)


def load_research_config(path: str | Path) -> ResearchConfig:
    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    classes = {
        "storage": StorageConfig,
        "open_webui": OpenWebUIConfig,
        "analysis": AnalysisConfig,
        "inference": InferenceConfig,
        "pose_preprocessing": PosePreprocessingConfig,
        "detection": DetectionConfig,
    }
    if set(payload) - set(classes):
        raise ValueError(
            "Unknown configuration tables: " + ", ".join(sorted(set(payload) - set(classes)))
        )
    if "open_webui" not in payload:
        raise ValueError("[open_webui] with base_url and model is required")
    return ResearchConfig(
        **{key: _construct_config(cls, payload.get(key, {}), key) for key, cls in classes.items()}
    )
