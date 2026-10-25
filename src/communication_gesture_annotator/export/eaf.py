# SPDX-License-Identifier: Apache-2.0
"""Dependency-free, presence-only EAF export and output validation."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4
import xml.etree.ElementTree as ET
import json
from ..pipeline.llm_projection import project_llm_output, decision_basis
from .json_export import validate_analysis_document

EAF_SCHEMA_URL = "https://www.mpi.nl/tools/elan/EAFv3.0.xsd"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _validate_eaf_root(root: ET.Element) -> ET.Element:
    if _local_name(root.tag) != "ANNOTATION_DOCUMENT":
        raise ValueError("EAF root must be ANNOTATION_DOCUMENT")
    if root.get("FORMAT") != "3.0" or root.get("VERSION") != "3.0":
        raise ValueError("EAF FORMAT/VERSION must be 3.0")
    headers = root.findall("./HEADER")
    if len(headers) != 1:
        raise ValueError("EAF must contain exactly one HEADER")
    if headers[0].get("TIME_UNITS", "milliseconds") != "milliseconds":
        raise ValueError("only millisecond EAF time units are supported")
    return headers[0]


def _selected_tier(root: ET.Element, tier_id: str) -> ET.Element:
    matches = [item for item in root.findall("./TIER") if item.get("TIER_ID") == tier_id]
    if len(matches) != 1:
        detail = "does not exist" if not matches else "is duplicated"
        raise ValueError(f"EAF tier {tier_id!r} {detail}")
    tier = matches[0]
    for wrapper in list(tier):
        if _local_name(wrapper.tag) != "ANNOTATION":
            raise ValueError(f"selected tier {tier_id!r} contains an unsupported child")
        children = list(wrapper)
        if len(children) != 1:
            raise ValueError(f"selected tier {tier_id!r} has a malformed ANNOTATION wrapper")
        child_name = _local_name(children[0].tag)
        if child_name == "REF_ANNOTATION":
            raise ValueError(f"selected tier {tier_id!r} contains unsupported REF_ANNOTATIONs")
        if child_name != "ALIGNABLE_ANNOTATION":
            raise ValueError(f"selected tier {tier_id!r} has an unsupported annotation type")
    return tier


def _time_slots(root: ET.Element) -> dict[str, int]:
    slots: dict[str, int] = {}
    for element in root.findall("./TIME_ORDER/TIME_SLOT"):
        identifier = element.get("TIME_SLOT_ID")
        value = element.get("TIME_VALUE")
        if identifier and value is not None:
            if identifier in slots:
                raise ValueError(f"EAF contains duplicate time slot ID: {identifier}")
            slots[identifier] = int(value)
    return slots


def _tier_intervals(
    root: ET.Element, tier_id: str, slots: Mapping[str, int]
) -> list[dict[str, Any]]:
    tier = _selected_tier(root, tier_id)
    intervals: list[dict[str, Any]] = []
    annotation_ids: set[str] = set()
    for annotation in tier.findall("./ANNOTATION/ALIGNABLE_ANNOTATION"):
        annotation_id = annotation.get("ANNOTATION_ID")
        if not annotation_id or annotation_id in annotation_ids:
            raise ValueError(f"selected tier {tier_id!r} has a missing or duplicate annotation ID")
        annotation_ids.add(annotation_id)
        start_ref = annotation.get("TIME_SLOT_REF1")
        end_ref = annotation.get("TIME_SLOT_REF2")
        if start_ref not in slots or end_ref not in slots:
            raise ValueError(f"selected tier {tier_id!r} has an unresolved time slot")
        value_node = annotation.find("./ANNOTATION_VALUE")
        value = "" if value_node is None else "".join(value_node.itertext()).strip()
        start_ms, end_ms = (slots[start_ref], slots[end_ref])
        start_us = start_ms * 1000
        end_us = end_ms * 1000
        if start_ms < 0 or end_ms <= start_ms or start_us < 0:
            raise ValueError(f"selected tier {tier_id!r} has an invalid interval")
        intervals.append(
            {
                "annotation_id": annotation_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "start_us": start_us,
                "end_us": end_us,
                "source_value": value,
            }
        )
    intervals.sort(key=lambda item: (item["start_us"], item["end_us"]))
    previous_end = -1
    for interval in intervals:
        if interval["start_us"] < previous_end:
            raise ValueError(f"selected tier {tier_id!r} contains overlapping intervals")
        previous_end = interval["end_us"]
    return intervals


def _round_us_to_ms(value_us: int) -> int:
    if value_us < 0:
        raise ValueError("canonical timestamps cannot be negative")
    return (value_us + 500) // 1000


def export_eaf_document(
    analysis: Mapping[str, Any],
    output_path: str | Path,
    tier_id: str = "G-Units",
    *,
    overwrite: bool = False,
) -> Path:
    """Create the deliberately lossy one-tier G-unit projection."""
    document = validate_analysis_document(dict(analysis))
    if not tier_id.strip():
        raise ValueError("EAF tier ID cannot be blank")
    projection = project_llm_output(document)
    units = sorted(projection.units if projection else document["g_units"],
                   key=lambda item: (item["start_us"], item["end_us"]))
    previous_end = -1
    for unit in units:
        if unit["start_us"] < previous_end:
            raise ValueError("G-units must not overlap for one-tier EAF export")
        previous_end = unit["end_us"]
    xsi = "http://www.w3.org/2001/XMLSchema-instance"
    root = ET.Element(
        "ANNOTATION_DOCUMENT",
        {
            "AUTHOR": "",
            "DATE": "1970-01-01T00:00:00+00:00",
            "FORMAT": "3.0",
            "VERSION": "3.0",
            f"{{{xsi}}}noNamespaceSchemaLocation": EAF_SCHEMA_URL,
        },
    )
    header = ET.SubElement(root, "HEADER", {"MEDIA_FILE": "", "TIME_UNITS": "milliseconds"})
    ET.SubElement(header, "PROPERTY", {"NAME": "decision_basis"}).text = json.dumps(
        decision_basis(projection), sort_keys=True, separators=(",", ":"))
    ET.SubElement(header, "PROPERTY", {"NAME": "lastUsedAnnotationId"}).text = str(len(units))
    time_order = ET.SubElement(root, "TIME_ORDER")
    tier = ET.SubElement(
        root,
        "TIER",
        {
            "TIER_ID": tier_id,
            "LINGUISTIC_TYPE_REF": "g-unit-presence-v0.2",
            "PARTICIPANT": document["participant_profiles"][0]["participant_id"],
        },
    )
    slot_number = 0
    for annotation_number, unit in enumerate(units, start=1):
        start_ms = _round_us_to_ms(unit["start_us"])
        end_ms = max(start_ms + 1, _round_us_to_ms(unit["end_us"]))
        slot_number += 1
        start_slot = f"ts{slot_number}"
        ET.SubElement(
            time_order, "TIME_SLOT", {"TIME_SLOT_ID": start_slot, "TIME_VALUE": str(start_ms)}
        )
        slot_number += 1
        end_slot = f"ts{slot_number}"
        ET.SubElement(
            time_order, "TIME_SLOT", {"TIME_SLOT_ID": end_slot, "TIME_VALUE": str(end_ms)}
        )
        wrapper = ET.SubElement(tier, "ANNOTATION")
        alignable = ET.SubElement(
            wrapper,
            "ALIGNABLE_ANNOTATION",
            {
                "ANNOTATION_ID": f"a{annotation_number}",
                "TIME_SLOT_REF1": start_slot,
                "TIME_SLOT_REF2": end_slot,
            },
        )
        ET.SubElement(alignable, "ANNOTATION_VALUE").text = "G-unit"
    ET.SubElement(
        root,
        "LINGUISTIC_TYPE",
        {
            "GRAPHIC_REFERENCES": "false",
            "LINGUISTIC_TYPE_ID": "g-unit-presence-v0.2",
            "TIME_ALIGNABLE": "true",
        },
    )
    ET.indent(root, space="  ")
    target = Path(output_path)
    if target.exists() and (not overwrite):
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = ET.tostring(root, encoding="UTF-8", xml_declaration=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(encoded)
        validate_exported_eaf(temporary, tier_id=tier_id)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def validate_exported_eaf(path: str | Path, *, tier_id: str = "G-Units") -> None:
    root = ET.parse(path).getroot()
    header = _validate_eaf_root(root)
    if root.findall("./PROPERTY") or not header.findall("./PROPERTY"):
        raise ValueError("EAF properties must be stored inside HEADER")
    if root.findall("./CONTROLLED_VOCABULARY"):
        raise ValueError("Presence-only EAF export cannot contain a controlled vocabulary")
    tiers = root.findall("./TIER")
    if len(tiers) != 1 or tiers[0].get("TIER_ID") != tier_id:
        raise ValueError("exported EAF must contain exactly the configured tier")
    slots = _time_slots(root)
    intervals = _tier_intervals(root, tier_id, slots)
    kind = tiers[0].get("LINGUISTIC_TYPE_REF")
    allowed = {"G-unit"} if kind == "g-unit-presence-v0.2" else set()
    if not allowed or any((item["source_value"] not in allowed for item in intervals)):
        raise ValueError("exported EAF contains an unsupported type or annotation value")
