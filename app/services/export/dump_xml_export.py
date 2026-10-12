"""Submit export: writes each record in the SAME XML schema as the legacy
system's own data dump (<DataM><RecordNo>...</RecordNo>...</DataM>, one
<DataM> per record) - not the self-describing <Document>/<Record>/<Field>
schema in xml_export.py, which is a different, separate export format kept
for its own existing download link.

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


def _record_to_datam_element(record: DocumentRecord, entry: RecordEntry) -> ET.Element:
    datam = ET.Element("DataM")
    image_name = ET.SubElement(datam, "ImageName")
    image_name.text = record.source_file.original_filename
    for our_field, tag in FIELD_TO_DUMP_TAG.items():
        field = entry.fields.get(our_field)
        el = ET.SubElement(datam, tag)
        el.text = field.value if field and field.value is not None else ""
    submitted_by = ET.SubElement(datam, "SubmittedBy")
    submitted_by.text = entry.approved_by or record.audit.approved_by or ""
    submitted_at = ET.SubElement(datam, "SubmittedDate")
    submitted_at.text = (entry.approved_at or record.audit.updated_at).isoformat()
    return datam


def build_dump_xml(record: DocumentRecord) -> str:
    root = ET.Element("DataMRoot")
    for entry in record.records:
        root.append(_record_to_datam_element(record, entry))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def write_dump_xml_file(record: DocumentRecord, submissions_dir: Path) -> Path:
    submissions_dir.mkdir(parents=True, exist_ok=True)
    path = submissions_dir / f"{record.document_id}.xml"
    path.write_text(build_dump_xml(record), encoding="utf-8")
    return path


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    columns_sql = ", ".join(f'"{tag}" TEXT' for tag in DUMP_TAG_ORDER)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS submissions (
            document_id TEXT NOT NULL,
            record_index INTEGER NOT NULL,
            image_name TEXT,
            {columns_sql},
            submitted_by TEXT,
            submitted_at TEXT,
            PRIMARY KEY (document_id, record_index)
        )
        """
    )
    return conn


def write_dump_xml_db(record: DocumentRecord, db_path: Path) -> int:
    """Upserts one row per record. Returns the number of rows written."""
    conn = _connect(db_path)
    try:
        count = 0
        for entry in record.records:
            values = {tag: (entry.fields.get(f).value if entry.fields.get(f) else None) for f, tag in FIELD_TO_DUMP_TAG.items()}
            columns = ["document_id", "record_index", "image_name", *DUMP_TAG_ORDER, "submitted_by", "submitted_at"]
            placeholders = ", ".join("?" for _ in columns)
            column_list = ", ".join(f'"{c}"' for c in columns)
            row = [
                record.document_id,
                entry.record_index,
                record.source_file.original_filename,
                *[values[tag] for tag in DUMP_TAG_ORDER],
                entry.approved_by or record.audit.approved_by or None,
                (entry.approved_at or record.audit.updated_at).isoformat(),
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
    xml_path = write_dump_xml_file(record, submissions_dir)
    row_count = write_dump_xml_db(record, db_path)
    return xml_path, row_count
