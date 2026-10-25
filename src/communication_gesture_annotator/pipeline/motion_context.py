# SPDX-License-Identifier: Apache-2.0
"""Several-second evidence scope independent of model output ownership."""

from __future__ import annotations
from typing import Any, Mapping, Sequence

MOTION_CONTEXT_POLICY_VERSION = "synchronized-hand-context-v0.2"


def synchronized_evidence(
    context: Sequence[int],
    core: Sequence[int],
    frames: Mapping[int, Mapping[str, Any]],
    image_refs: Sequence[Mapping[str, Any]],
    packet: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Join actual delivered RGB references to the common source timeline."""
    mapping = []
    for index in context:
        frame = frames[index]
        selected = frame.get("selected_source_frame")
        mapping.append(
            {
                "sample_index": index,
                "source_frame": selected,
                "start_us": frame.get("start_us"),
                "end_us": frame.get("end_us"),
                "image_references": [
                    {"image_ordinal": ref.get("image_ordinal"), "cell": cell}
                    for ref in image_refs
                    for cell in (
                        ref.get("cells", []) if ref.get("kind") == "contact_sheet" else [ref]
                    )
                    if cell.get("sample_index") == index
                ],
            }
        )
    return {
        "policy_version": MOTION_CONTEXT_POLICY_VERSION,
        "context_sample_span": [context[0], context[-1] + 1],
        "output_core_sample_span": [core[0], core[-1] + 1],
        "pose_artifact_sha256": packet.get("artifact_sha256") if packet else None,
        "video_representation": "chronological delivered RGB frames/contact sheets; not continuous encoded video",
        "same_frame_overlay_requirement": True,
        "sample_to_visual_mapping": mapping,
        "interpretation_rule": "Read hand transport, palm orientation and finger articulation over the entire time window together with RGB. Cite global sample intervals and supplied visual references. Do not infer motion absence from missing finger detail.",
    }
