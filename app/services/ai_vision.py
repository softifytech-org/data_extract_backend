"""Optional GPT Vision verification layer.

Not part of the default pipeline - triggered explicitly per record (see
POST /api/documents/{id}/records/{index}/ai-verify), never automatically
across a whole document. This exists to spend a small, fixed API budget
carefully: one image, one call, cached on the record so the same record is
never billed twice.

Why vision instead of a hand-built pixel-coordinate template: a vision
model looking at the actual crop inherently understands which value sits
under which label in 2D, which is exactly what our text-only extraction
(engine.py) lacks - it searches the record's OCR text for something
"shaped like" a name/date/amount with no true notion of field position,
which is the root cause of the Shipper Name / STM Name cross-assignment
bug. Hand-deriving pixel coordinates from a single reference image would
be brittle (other records have different text lengths, shifting columns)
and was explicitly not requested to be invented.

Source-fidelity is enforced the same way as the rest of the pipeline: the
model is instructed to return exactly what is visible, never to correct,
normalize, or guess - and this module never overwrites `value`, only
populates the separate `ai_value` / `ai_match` comparison fields.
"""
from __future__ import annotations

import base64
import json
import logging

import requests

from app.services.extraction.fields import FIELD_SPEC_BY_NAME, OCR_FIELD_ORDER

logger = logging.getLogger(__name__)

_API_URL = "https://api.openai.com/v1/chat/completions"
_TIMEOUT_SECONDS = 60

_SYSTEM_PROMPT = """You are reading ONE record cropped from a scanned medical insurance form. \
Extract exactly what is visible in the image for each requested field.

CRITICAL RULES - do not violate these:
1. Return the EXACT text as printed/typed in the image, character for character. \
Do not correct spelling, do not normalize dates, do not fix currency amounts, \
do not reformat phone numbers, do not capitalize or lowercase anything.
   Example: if the image shows "Septenber 7, 1996", return "Septenber 7, 1996" - \
NOT "September 7, 1996".
   Example: if the image shows "$2.o0.00", return "$2.o0.00" - NOT "$2.00".
2. If a field is not visible or you cannot read it with reasonable confidence, \
return null for that field's value - never guess or invent a value.
3. Pay close attention to WHICH label each piece of text sits next to or under - \
this record has multiple name-shaped fields (Customer Name, Billing Name, \
Shipper Name, Name_P_Holder, STM Name) and multiple date fields; match each \
value to its own correct field based on its position in the layout, not just \
"the next name/date found".
4. Return ONLY the fields listed below. Do not add fields, do not rename fields, \
do not reorder them.

Respond with a single JSON object shaped exactly like:
{"field_name": {"value": "exact text or null", "confidence": 0-100}, ...}
"""


class AIVisionError(RuntimeError):
    pass


def _build_field_list() -> str:
    lines = []
    for name in OCR_FIELD_ORDER:
        spec = FIELD_SPEC_BY_NAME[name]
        lines.append(f'- "{name}" (labeled "{spec.label}" on the form)')
    return "\n".join(lines)


def extract_fields_via_vision(image_bytes: bytes, *, api_key: str, model: str) -> dict[str, dict]:
    """Sends ONE image + prompt in ONE request. Raises AIVisionError on any
    failure (network, auth, malformed response) rather than silently
    returning partial/guessed data - the caller treats this as "AI
    verification unavailable this time", not as a source of truth to fall
    back on blindly."""
    if not api_key:
        raise AIVisionError("No OpenAI API key configured (AI_PROVIDER_API_KEY)")

    b64 = base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Fields to extract:\n" + _build_field_list()},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
                    },
                ],
            },
        ],
    }

    try:
        resp = requests.post(
            _API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise AIVisionError(f"OpenAI request failed: {exc}") from exc

    if resp.status_code != 200:
        raise AIVisionError(f"OpenAI API returned {resp.status_code}: {resp.text[:500]}")

    try:
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise AIVisionError(f"Could not parse OpenAI response: {exc}") from exc

    # Never trust the model to have respected "only the fields listed" -
    # drop anything outside the known schema rather than let it silently
    # introduce a field that doesn't exist.
    result: dict[str, dict] = {}
    for name in OCR_FIELD_ORDER:
        entry = parsed.get(name)
        if isinstance(entry, dict) and "value" in entry:
            result[name] = {
                "value": entry.get("value") or None,
                "confidence": entry.get("confidence"),
            }
    usage = body.get("usage", {})
    logger.info(
        "AI vision call: model=%s prompt_tokens=%s completion_tokens=%s",
        model, usage.get("prompt_tokens"), usage.get("completion_tokens"),
    )
    return result
