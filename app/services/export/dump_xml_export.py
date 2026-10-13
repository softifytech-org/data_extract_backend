"""Submit export: writes each record in the SAME XML schema as the legacy
system's own data dump (<DataM><RecordNo>...</RecordNo>...</DataM>, one
<DataM> per record) - not the self-describing <Document>/<Record>/<Field>
schema in xml_export.py, which is a different, separate export format kept
for its own existing download link.

Every structural detail here - root element, declaration quoting, BOM,
field order, the UserID value, and which CreateDate/UpdateDate shapes are
real - was checked directly against the 25 real production DataM.xml files
under data/ground_truth/softify_dump/ (90,321 real <DataM> records), not
guessed or taken from a single pasted example. See the comments at each
decision point below for what was actually found there.

Storage: every submission is written to two places -
  1. A flat XML file under data/submissions/<document_id>.xml (human-
     readable, easy to hand off, no server/setup required).
  2. A row per record in a local SQLite database (data/submissions.db) -
     zero new dependency (sqlite3 is in the Python standard library),
     zero infrastructure to stand up before today's deployment, and a
     straightforward later migration to Postgres (same column shapes)
     if/when that becomes the production datastore.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from app.models.document import DocumentRecord, RecordEntry

# our field name -> legacy dump XML tag name (matches DataM.xml exactly,
# the same mapping this project's own scoring tools use against the real
# ground-truth dump).
FIELD_TO_DUMP_TAG: dict[str, str] = {
    "record_no": "RecordNo",
    "customer_name": "CustomerName",
    "email_address": "EmailAddress",
    "res_address": "ResAddress",
    "city_1": "City_1",
    "state_1": "State_1",
    "zip_1": "Zip_1",
    "phno_1": "PhNo_1",
    "country_1": "Country_1",
    "sex_1": "Sex_1",
    "d_birth": "D_Birth",
    "height": "Height",
    "weight": "Weight",
    "blood_group": "Blood_Group",
    "billing_name": "BillingName",
    "shipper_name": "ShipperName",
    "city_2": "City_2",
    "state_2": "State_2",
    "zip_2": "Zip_2",
    "country_2": "Country_2",
    "phno_2": "PhNo_2",
    "alcoholic": "Alcoholic",
    "smoker": "Smoker",
    "past_surg": "PastSug",
    "diabetic": "Diabetic",
    "allergies": "Allergiesd",
    "policy_no": "PloicyNo",
    "d_b_life_assure": "D_B_Life_Assure",
    "p_inst": "P_Inst",
    "name_p_holder": "Name_P_Holder",
    "stm_name": "STM_Name",
    "stm_code": "STM_Code",
    "dob": "DOB",
    "sex_2": "Sex_2",
    "card_name": "CardName",
    "medicine": "Medicine",
    "dosage": "Dosage",
    "tablets": "Tablets",
    "pill_rate": "PillRate",
    "cost": "Cost",
    "shipping_cost": "ShippingCost",
    "total_amount": "TotalAmount",
    "remark": "Remarks",
}
DUMP_TAG_ORDER = list(FIELD_TO_DUMP_TAG.values())

# Trailing tags the legacy dump always carries after the extracted fields -
# never sourced from entry.fields, always generated in _build_record_metadata.
METADATA_TAGS = ["UserID", "CreateDate", "UpdateDate"]

# Every one of the 90,321 real <DataM> records surveyed uses this exact
# value - not a single exception - so it's written unconditionally rather
# than guessed-at per record.
DEFAULT_USER_ID = "midemo1"

# The real dump's CreateDate format is NOT fixed across its whole history -
# the same production system used "20 August 2026 11:22:55" (24h, no
# weekday) for its August batches and "Monday, September 21, 2026
# 6:37:33 PM" (12h + weekday) for its September batches, confirming the
# legacy app's own date formatting changed over time (almost certainly a
# locale/culture setting on whatever machine generated the dump, not
# something this tool controls). Checked against the single most recent
# real record across all 90,321 surveyed (2026-09-30 21:03:56, USER - 20)
# to resolve which shape is current: it's the 12h+weekday style, confirming
# this is the right one to use, not a guess.
#
# That same check also showed neither the hour nor the day-of-month is
# zero-padded in any of the 16,277 single-digit-hour / 3,946 single-digit-
# day real examples checked ("9:03:56 PM", "September 1, 2026" - never
# "09:03:56 PM" or "September 01, 2026"). Python's %I/%d always zero-pad
# with no portable cross-platform way to turn that off (the no-pad strftime
# flag differs between Windows and the Linux container this runs on in
# production), so the string is built by hand below instead of via
# strftime for the day/hour components.
_IST = timezone(timedelta(hours=5, minutes=30))


def _format_dump_datetime(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local = dt.astimezone(_IST)
    hour12 = local.hour % 12 or 12
    ampm = "AM" if local.hour < 12 else "PM"
    return (
        f"{local.strftime('%A')}, {local.strftime('%B')} {local.day}, {local.year} "
        f"{hour12}:{local.minute:02d}:{local.second:02d} {ampm}"
    )


def _build_record_metadata(record: DocumentRecord, db_path: Path) -> dict[int, dict[str, str]]:
    """One pass computing UserID/CreateDate/UpdateDate per record, shared by
    both the XML file and the SQLite write below so the two can never
    disagree with each other.

    No timestamp here is invented. CreateDate is either a genuine
    per-record approval time (if this record was approved individually via
    the separate Approve & Submit flow before this submit) or the real,
    shared moment this whole document was submitted - never a fabricated,
    staggered sequence. It is also immutable once first written: a record
    resubmitted later (e.g. after an edit) keeps the CreateDate already on
    file, read back from submissions.db, rather than getting a new one.

    UpdateDate stays genuinely blank until a record is resubmitted with at
    least one field value actually different from what's already stored,
    at which point it's set to the real moment of that resubmission - it is
    never populated just because Submit was clicked again with no changes.
    """
    conn = _connect(db_path)
    try:
        cols = ["record_index", "CreateDate", "UpdateDate", *DUMP_TAG_ORDER]
        col_list = ", ".join(f'"{c}"' for c in cols)
        cur = conn.execute(
            f"SELECT {col_list} FROM submissions WHERE document_id = ?",
            (record.document_id,),
        )
        existing_by_record: dict[int, dict[str, str | None]] = {
            row[0]: dict(zip(cols[1:], row[1:])) for row in cur.fetchall()
        }
    finally:
        conn.close()

    shared_submit_time = _format_dump_datetime(datetime.now(timezone.utc))

    metadata: dict[int, dict[str, str]] = {}
    for entry in record.records:
        new_values = {
            tag: (entry.fields.get(f).value if entry.fields.get(f) else None)
            for f, tag in FIELD_TO_DUMP_TAG.items()
        }
        existing = existing_by_record.get(entry.record_index)

        if existing is None:
            create_date = _format_dump_datetime(entry.approved_at) if entry.approved_at else shared_submit_time
            update_date = ""
        else:
            create_date = existing.get("CreateDate") or shared_submit_time
            changed = any((existing.get(tag) or "") != (new_values.get(tag) or "") for tag in DUMP_TAG_ORDER)
            update_date = shared_submit_time if changed else (existing.get("UpdateDate") or "")

        metadata[entry.record_index] = {
            "UserID": DEFAULT_USER_ID,
            "CreateDate": create_date,
            "UpdateDate": update_date,
        }
    return metadata


def _record_to_datam_element(record: DocumentRecord, entry: RecordEntry, metadata: dict[str, str]) -> ET.Element:
    datam = ET.Element("DataM")
    image_name = ET.SubElement(datam, "ImageName")
    image_name.text = record.source_file.original_filename
    for our_field, tag in FIELD_TO_DUMP_TAG.items():
        field = entry.fields.get(our_field)
        el = ET.SubElement(datam, tag)
        # "N.A" (not "NA"/"N/A"/empty) is the real missing-value convention -
        # checked directly: 18,310 of 18,468 missing-value-shaped tokens
        # across the real ground truth use exactly this form; a truly blank
        # tag occurs in only 18 of 711,200 real field values. This also
        # means every field now always gets real open+close tags, since
        # ElementTree only self-closes a tag that has no text at all.
        el.text = field.value if field and field.value else "N.A"
    for tag in METADATA_TAGS:
        el = ET.SubElement(datam, tag)
        el.text = metadata[tag] or None
    return datam


def build_dump_xml(record: DocumentRecord, metadata_by_record: dict[int, dict[str, str]]) -> str:
    # Root element is <doc>, confirmed against all 25 real production
    # files - not <DataMRoot>, which earlier code here had guessed.
    root = ET.Element("doc")
    for entry in record.records:
        root.append(_record_to_datam_element(record, entry, metadata_by_record[entry.record_index]))
    ET.indent(root, space="  ")
    # ET's own xml_declaration=True emits single-quoted attributes
    # ('<?xml version=\'1.0\'...'); every real production file uses double
    # quotes, so the declaration is written out by hand instead.
    # short_empty_elements=False: the real dump always writes an explicit
    # open+close pair for every tag, even a genuinely empty one (e.g. an
    # UpdateDate that has never been resubmitted) - never a self-closing
    # "<Tag />". The 43 extracted fields never actually hit this (a
    # missing value already becomes the literal text "N.A", not emptiness
    # - see _record_to_datam_element), so this only changes the
    # metadata tags that can be genuinely blank.
    body = ET.tostring(root, encoding="unicode", short_empty_elements=False)
    return f'<?xml version="1.0" encoding="utf-8"?>\n{body}'


def write_dump_xml_file(
    record: DocumentRecord, submissions_dir: Path, metadata_by_record: dict[int, dict[str, str]]
) -> Path:
    submissions_dir.mkdir(parents=True, exist_ok=True)
    path = submissions_dir / f"{record.document_id}.xml"
    # utf-8-sig: every real production DataM.xml starts with a UTF-8 BOM.
    # Line endings are forced to \r\n explicitly (checked against raw file
    # bytes - real files use CRLF) rather than left to Python's platform
    # default, since that default is LF-only on the Linux container this
    # runs on in production, which would silently stop matching if left
    # unforced (it only happens to match by coincidence on Windows dev).
    text = build_dump_xml(record, metadata_by_record).replace("\n", "\r\n")
    path.write_text(text, encoding="utf-8-sig", newline="")
    return path


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    columns_sql = ", ".join(f'"{tag}" TEXT' for tag in DUMP_TAG_ORDER)
    metadata_sql = ", ".join(f'"{tag}" TEXT' for tag in METADATA_TAGS)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS submissions (
            document_id TEXT NOT NULL,
            record_index INTEGER NOT NULL,
            image_name TEXT,
            {columns_sql},
            {metadata_sql},
            submitted_at TEXT,
            PRIMARY KEY (document_id, record_index)
        )
        """
    )
    # A database written before UserID/CreateDate/UpdateDate existed won't
    # have these columns yet - add them in place rather than requiring the
    # file to be dropped and recreated. Existing rows/columns are never
    # touched by this, only ever added to.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(submissions)")}
    for tag in METADATA_TAGS:
        if tag not in existing_cols:
            conn.execute(f'ALTER TABLE submissions ADD COLUMN "{tag}" TEXT')
    conn.commit()
    return conn


def write_dump_xml_db(
    record: DocumentRecord, db_path: Path, metadata_by_record: dict[int, dict[str, str]]
) -> int:
    """Upserts one row per record. Returns the number of rows written."""
    conn = _connect(db_path)
    try:
        count = 0
        columns = ["document_id", "record_index", "image_name", *DUMP_TAG_ORDER, *METADATA_TAGS, "submitted_at"]
        placeholders = ", ".join("?" for _ in columns)
        column_list = ", ".join(f'"{c}"' for c in columns)
        for entry in record.records:
            field_values = {
                tag: (entry.fields.get(f).value if entry.fields.get(f) else None)
                for f, tag in FIELD_TO_DUMP_TAG.items()
            }
            meta = metadata_by_record[entry.record_index]
            row = [
                record.document_id,
                entry.record_index,
                record.source_file.original_filename,
                *[field_values[tag] for tag in DUMP_TAG_ORDER],
                *[meta[tag] for tag in METADATA_TAGS],
                datetime.now(timezone.utc).isoformat(),
            ]
            conn.execute(
                f"INSERT OR REPLACE INTO submissions ({column_list}) VALUES ({placeholders})",
                row,
            )
            count += 1
        conn.commit()
        return count
    finally:
        conn.close()


def submit_document(record: DocumentRecord, submissions_dir: Path, db_path: Path) -> tuple[Path, int]:
    """Writes both the flat XML file and the SQLite rows. Returns the XML
    file path and the number of record rows written to the database."""
    metadata_by_record = _build_record_metadata(record, db_path)
    xml_path = write_dump_xml_file(record, submissions_dir, metadata_by_record)
    row_count = write_dump_xml_db(record, db_path, metadata_by_record)
    return xml_path, row_count


def build_combined_dump_xml(records: list[DocumentRecord], metadata_by_document: dict[str, dict[int, dict]]) -> str:
    """Same element shape as build_dump_xml, but spanning every document
    in a batch under ONE shared <doc> root - this is what a batch submit
    (app/api/batches.py's submit_batch) uploads as the batch's single
    combined export, matching the real legacy dump's own convention of
    many <DataM> records in one file (see the module docstring)."""
    root = ET.Element("doc")
    for record in records:
        metadata_by_record = metadata_by_document[record.document_id]
        for entry in record.records:
            root.append(_record_to_datam_element(record, entry, metadata_by_record[entry.record_index]))
    ET.indent(root, space="  ")
    # short_empty_elements=False: see build_dump_xml's identical comment -
    # the real dump never self-closes an empty tag.
    body = ET.tostring(root, encoding="unicode", short_empty_elements=False)
    text = f'<?xml version="1.0" encoding="utf-8"?>\n{body}'
    return text.replace("\n", "\r\n")


def submit_batch_records(
    records: list[DocumentRecord], submissions_dir: Path, db_path: Path
) -> tuple[str, int]:
    """Submits every document in a batch via the SAME per-document logic
    submit_document already uses for each one individually (immutable
    CreateDate, UpdateDate only on genuine change, one upsert per record
    in submissions.db, and that document's own flat XML file on local
    disk unchanged) - this does not reimplement any of that. It only
    additionally combines every document's records into ONE XML string
    spanning the whole batch, for upload to Azure Blob Storage as a
    single combined export (see app/api/batches.py's submit_batch).

    Safe to call again for the same batch: every per-document write below
    is the same idempotent upsert submit_document already relies on, and
    the combined XML text is deterministically rebuilt from current
    approved data each time, not appended to - calling this twice with
    unchanged data produces byte-identical output.

    Returns the combined XML text (CRLF, ready to write/upload as bytes)
    and the total record count across every document."""
    metadata_by_document: dict[str, dict[int, dict]] = {}
    for record in records:
        metadata_by_record = _build_record_metadata(record, db_path)
        metadata_by_document[record.document_id] = metadata_by_record
        write_dump_xml_file(record, submissions_dir, metadata_by_record)
        write_dump_xml_db(record, db_path, metadata_by_record)

    combined_text = build_combined_dump_xml(records, metadata_by_document)
    total_records = sum(len(r.records) for r in records)
    return combined_text, total_records
