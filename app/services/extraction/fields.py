"""The target field schema for the data-entry form.

This reproduces the EXACT field set of the existing legacy data-entry
application (see the reference screenshots) rather than an invented
schema - every name and order matches the legacy form so the frontend can
be a faithful modern recreation of it.

Two fields are not OCR-extracted at all:
- image_name: populated from the uploaded filename, not read from pixels.
- remark: derived AFTER extraction from which fields the source document
  explicitly marked N/A (see services/validation/remark.py) - never OCR'd
  directly.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.models.document import DataType


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    data_type: DataType
    required: bool = True
    derived: bool = False  # populated by business logic, not OCR extraction


FIELD_SPECS: list[FieldSpec] = [
    FieldSpec("image_name", "Image Name", DataType.TEXT, derived=True),
    FieldSpec("record_no", "Record No", DataType.ID_CODE),
    FieldSpec("customer_name", "Customer Name", DataType.NAME),
    FieldSpec("email_address", "Email Address", DataType.EMAIL),
    FieldSpec("res_address", "Res Address", DataType.ADDRESS),
    FieldSpec("city_1", "City 1", DataType.TEXT),
    FieldSpec("state_1", "State 1", DataType.ENUM),
    FieldSpec("zip_1", "Zip 1", DataType.TEXT),
    FieldSpec("phno_1", "Ph No 1", DataType.PHONE),
    FieldSpec("country_1", "Country 1", DataType.ENUM),
    FieldSpec("sex_1", "Sex 1", DataType.ENUM),
    FieldSpec("d_birth", "D. Birth", DataType.DATE),
    FieldSpec("height", "Height", DataType.NUMBER),
    FieldSpec("weight", "Weight", DataType.NUMBER),
    FieldSpec("blood_group", "Blood Group", DataType.ENUM),
    FieldSpec("billing_name", "Billing Name", DataType.NAME),
    FieldSpec("shipper_name", "Shipper Name", DataType.NAME),
    FieldSpec("city_2", "City 2", DataType.TEXT),
    FieldSpec("state_2", "State 2", DataType.ENUM),
    FieldSpec("zip_2", "Zip 2", DataType.TEXT),
    FieldSpec("country_2", "Country 2", DataType.ENUM),
    FieldSpec("phno_2", "Ph No 2", DataType.PHONE),
    FieldSpec("alcoholic", "Alcoholic", DataType.ENUM),
    FieldSpec("smoker", "Smoker", DataType.ENUM),
    FieldSpec("past_surg", "Past Surg", DataType.ENUM),
    FieldSpec("diabetic", "Diabetic", DataType.ENUM),
    FieldSpec("allergies", "Allergies", DataType.TEXT, required=False),
    FieldSpec("policy_no", "Policy No", DataType.ID_CODE),
    FieldSpec("d_b_life_assure", "D.B. Life Assure", DataType.DATE),
    FieldSpec("p_inst", "P. Inst.", DataType.CURRENCY),
    FieldSpec("name_p_holder", "Name P. Holder", DataType.NAME),
    FieldSpec("stm_name", "STM Name", DataType.TEXT),
    FieldSpec("stm_code", "STM Code", DataType.ID_CODE),
    FieldSpec("dob", "DOB", DataType.DATE),
    FieldSpec("sex_2", "Sex 2", DataType.ENUM),
    FieldSpec("card_name", "Card Name", DataType.ENUM),
    FieldSpec("medicine", "Medicine", DataType.ENUM),
    FieldSpec("dosage", "Dosage", DataType.TEXT),
    FieldSpec("tablets", "Tablets", DataType.NUMBER),
    FieldSpec("pill_rate", "Pill Rate", DataType.CURRENCY),
    FieldSpec("cost", "Cost", DataType.CURRENCY),
    FieldSpec("shipping_cost", "Shipping Cost", DataType.CURRENCY),
    FieldSpec("total_amount", "Total Amount", DataType.CURRENCY),
    FieldSpec("remark", "Remark", DataType.TEXT, required=False, derived=True),
]

FIELD_ORDER = [spec.name for spec in FIELD_SPECS]
FIELD_SPEC_BY_NAME = {spec.name: spec for spec in FIELD_SPECS}
OCR_FIELD_ORDER = [spec.name for spec in FIELD_SPECS if not spec.derived]
