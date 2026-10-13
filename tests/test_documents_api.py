"""Document access tests for the open-access model - this application
has no authentication anywhere (explicit decision, see
app/api/batches.py's module docstring). A document created through a
batch must be reachable with NO credentials at all, same as one created
through the legacy single-document /upload endpoint - there is no
isolation between callers by design."""
from __future__ import annotations

TINY_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100


def _create_batch_document(client) -> str:
    resp = client.post("/api/batches", files=[("files", ("img.jpg", TINY_JPEG, "image/jpeg"))])
    assert resp.status_code == 200, resp.text
    batch = client.get(f"/api/batches/{resp.json()['batch_id']}").json()
    return batch["images"][0]["document_id"]


def test_batch_document_readable_with_no_credentials(client):
    document_id = _create_batch_document(client)
    resp = client.get(f"/api/documents/{document_id}")
    assert resp.status_code == 200


def test_batch_document_file_readable_with_no_credentials(client):
    document_id = _create_batch_document(client)
    resp = client.get(f"/api/documents/{document_id}/file")
    assert resp.status_code == 200
    assert resp.content == TINY_JPEG
