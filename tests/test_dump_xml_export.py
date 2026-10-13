"""Formalizes the XML serialization behavior manually verified during
development (see project history) as a repeatable automated test matrix:
N.A convention, tag order/completeness, special-character escaping,
CreateDate immutability, and UpdateDate-only-on-genuine-edit."""
from __future__ import annotations

import copy
from xml.etree import ElementTree as ET

from app.services.export.dump_xml_export import (
    DUMP_TAG_ORDER,
    FIELD_TO_DUMP_TAG,
    METADATA_TAGS,
    submit_document,
)

EXPECTED_TAGS = ["ImageName", *FIELD_TO_DUMP_TAG.values(), *METADATA_TAGS]


def _build_record(repo, document_id="DOC-AAAAAAAAAAAA"):
    from app.models.document import (
        AuditInfo, DataType, DocumentRecord, DocumentStatus, ExtractedField,
        FieldSource, FieldStatus, RecordEntry, SourceFileInfo,
    )

    fields = {}
    for name in FIELD_TO_DUMP_TAG:
        fields[name] = ExtractedField(
            field_name=name, label=name, value="Sample Value" if name != "customer_name" else None,
            data_type=DataType.TEXT, confidence=90.0, status=FieldStatus.VERIFIED,
            source=FieldSource.OCR,
        )
    record = DocumentRecord(
        document_id=document_id, status=DocumentStatus.NEEDS_REVIEW,
        source_file=SourceFileInfo(
            original_filename="test.jpeg", stored_filename="x.jpeg",
            file_type="image", content_type="image/jpeg",
        ),
        audit=AuditInfo(),
        records=[RecordEntry(record_index=1, page=1, fields=fields, overall_confidence=90.0)],
    )
    repo.save(record)
    return record


def test_populated_field_not_replaced(repo, test_settings):
    record = _build_record(repo)
    xml_path, _ = submit_document(record, test_settings.submissions_dir, test_settings.submissions_db_path)
    text = xml_path.read_bytes().decode("utf-8-sig")
    root = ET.fromstring(text.split("?>", 1)[1])
    first = root.find("DataM")
    assert first.findtext("ResAddress") == "Sample Value"


def test_missing_field_becomes_na(repo, test_settings):
    record = _build_record(repo, "DOC-BBBBBBBBBBBB")
    xml_path, _ = submit_document(record, test_settings.submissions_dir, test_settings.submissions_db_path)
    text = xml_path.read_bytes().decode("utf-8-sig")
    root = ET.fromstring(text.split("?>", 1)[1])
    assert root.find("DataM").findtext("CustomerName") == "N.A"


def test_every_field_tag_present_in_order(repo, test_settings):
    record = _build_record(repo, "DOC-CCCCCCCCCCCC")
    xml_path, _ = submit_document(record, test_settings.submissions_dir, test_settings.submissions_db_path)
    text = xml_path.read_bytes().decode("utf-8-sig")
    root = ET.fromstring(text.split("?>", 1)[1])
    tags = [c.tag for c in root.find("DataM")]
    assert tags == EXPECTED_TAGS


def test_root_declaration_and_bom(repo, test_settings):
    record = _build_record(repo, "DOC-DDDDDDDDDDDD")
    xml_path, _ = submit_document(record, test_settings.submissions_dir, test_settings.submissions_db_path)
    raw = xml_path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    assert text.startswith('<?xml version="1.0" encoding="utf-8"?>\r\n<doc>')


def test_special_characters_escaped_and_recoverable(repo, test_settings):
    record = _build_record(repo, "DOC-EEEEEEEEEEEE")
    record.records[0].fields["customer_name"].value = "A&B <Test> \"Quoted\" 'Apos'"
    xml_path, _ = submit_document(record, test_settings.submissions_dir, test_settings.submissions_db_path)
    text = xml_path.read_bytes().decode("utf-8-sig")
    assert "A&amp;B &lt;Test&gt;" in text
    root = ET.fromstring(text.split("?>", 1)[1])
    recovered = root.find("DataM").findtext("CustomerName")
    assert recovered == "A&B <Test> \"Quoted\" 'Apos'"


def test_createdate_immutable_across_resubmit(repo, test_settings):
    record = _build_record(repo, "DOC-FFFFFFFFFFFF")
    xml_path1, _ = submit_document(record, test_settings.submissions_dir, test_settings.submissions_db_path)
    text1 = xml_path1.read_bytes().decode("utf-8-sig")

    xml_path2, _ = submit_document(record, test_settings.submissions_dir, test_settings.submissions_db_path)
    text2 = xml_path2.read_bytes().decode("utf-8-sig")
    assert text1 == text2


def test_updatedate_set_only_on_genuine_edit(repo, test_settings):
    record = _build_record(repo, "DOC-111111111111")
    submit_document(record, test_settings.submissions_dir, test_settings.submissions_db_path)

    edited = copy.deepcopy(record)
    edited.records[0].fields["res_address"].value = "CHANGED VALUE"
    xml_path, _ = submit_document(edited, test_settings.submissions_dir, test_settings.submissions_db_path)
    text = xml_path.read_bytes().decode("utf-8-sig")
    root = ET.fromstring(text.split("?>", 1)[1])
    datam = root.find("DataM")
    assert datam.findtext("ResAddress") == "CHANGED VALUE"
    assert datam.findtext("UpdateDate").strip() != ""


def test_no_data_fabricated_for_none_values(repo, test_settings):
    """Every field in FIELD_TO_DUMP_TAG must round-trip to either the real
    value or the documented N.A placeholder - nothing else."""
    record = _build_record(repo, "DOC-222222222222")
    for f in record.records[0].fields.values():
        f.value = None
    xml_path, _ = submit_document(record, test_settings.submissions_dir, test_settings.submissions_db_path)
    text = xml_path.read_bytes().decode("utf-8-sig")
    root = ET.fromstring(text.split("?>", 1)[1])
    for tag in DUMP_TAG_ORDER:
        assert root.find("DataM").findtext(tag) == "N.A"
