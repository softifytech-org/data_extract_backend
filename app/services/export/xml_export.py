"""XML export.

IMPORTANT: the schema below is a DEFAULT, not the target system's real
schema - that was never provided (see docs/ discussion). It is a
reasonable, self-describing structure (one <Record> per detected user,
fields in a stable order, values plus their review status/confidence for
traceability) that can be re-shaped once the real target schema is
available, without touching OCR, extraction, or validation - this file
is the only thing that would need to change.

Uses only the standard library (xml.etree.ElementTree) - no new dependency.
"""
from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from app.models.document import DocumentRecord
from app.services.extraction.fields import FIELD_ORDER, FIELD_SPEC_BY_NAME


def build_xml_element(record: DocumentRecord) -> ET.Element:
    root = ET.Element(
        "Document",
        {
            "id": record.document_id,
            "status": record.status.value,
            "sourceFile": record.source_file.original_filename,
            "recordCount": str(record.record_count),
            "overallConfidence": f"{record.overall_confidence:.1f}" if record.overall_confidence is not None else "",
            "approvedAt": record.audit.approved_at.isoformat() if record.audit.approved_at else "",
            "approvedBy": record.audit.approved_by or "",
        },
    )

    for entry in record.records:
        record_el = ET.SubElement(
            root,
            "Record",
            {
                "index": str(entry.record_index),
                "page": str(entry.page),
                "confidence": f"{entry.overall_confidence:.1f}" if entry.overall_confidence is not None else "",
            },
        )
        for name in FIELD_ORDER:
            field = entry.fields.get(name)
            spec = FIELD_SPEC_BY_NAME[name]
            field_el = ET.SubElement(
                record_el,
                "Field",
                {
                    "name": name,
                    "label": spec.label,
                    "status": field.status.value if field else "needs_review",
                    "confidence": f"{field.confidence:.1f}" if field else "0.0",
                    "source": field.source.value if field else "ocr",
                },
            )
            field_el.text = field.value if field and field.value is not None else ""

        if entry.cross_field_warnings:
            warnings_el = ET.SubElement(record_el, "Warnings")
            for message in entry.cross_field_warnings:
                w = ET.SubElement(warnings_el, "Warning")
                w.text = message

    return root


def serialize_to_xml(record: DocumentRecord) -> str:
    root = build_xml_element(record)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def export_record_to_xml(record: DocumentRecord, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{record.document_id}.xml"
    xml_text = serialize_to_xml(record)
    output_path.write_text(xml_text, encoding="utf-8")
    return output_path
