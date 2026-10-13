"""Tests for the FileStore abstraction (services/storage/blob_store.py).

LocalFileStore is exercised directly against a tmp_path - this is the
backend every other automated test uses (see conftest.py's client
fixture override), so this file is also where its own behavior is
verified in isolation. AzureBlobFileStore is exercised against a MOCKED
azure-storage-blob client - never a real network call - covering the
retry/fast-fail/naming logic; the real account was verified separately,
once, by hand (see the final report), not by this automated suite.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services.storage.blob_store import AzureBlobFileStore, LocalFileStore, StorageUnavailableError


# --- LocalFileStore ---

def test_local_store_upload_round_trip(test_settings):
    store = LocalFileStore(test_settings)
    key = store.save_upload(b"hello world", document_id="DOC-AAAAAAAAAAAA", extension=".jpg", batch_id="BATCH-1")
    assert store.upload_exists(key)
    assert store.read_upload(key) == b"hello world"


def test_local_store_open_as_local_path_yields_real_file(test_settings):
    store = LocalFileStore(test_settings)
    key = store.save_upload(b"abc", document_id="DOC-BBBBBBBBBBBB", extension=".jpg", batch_id=None)
    with store.open_upload_as_local_path(key) as path:
        assert path.exists()
        assert path.read_bytes() == b"abc"
    # local backend: nothing is deleted on context exit (it's the real file, not a copy)
    assert path.exists()


def test_local_store_missing_upload_raises_file_not_found(test_settings):
    store = LocalFileStore(test_settings)
    with pytest.raises(FileNotFoundError):
        store.read_upload("DOC-CCCCCCCCCCCC.jpg")
    with pytest.raises(FileNotFoundError):
        with store.open_upload_as_local_path("DOC-CCCCCCCCCCCC.jpg"):
            pass
    assert store.upload_exists("DOC-CCCCCCCCCCCC.jpg") is False


def test_local_store_delete_upload_is_safe_on_missing_key(test_settings):
    store = LocalFileStore(test_settings)
    store.delete_upload("does-not-exist.jpg")  # must not raise


def test_local_store_xml_round_trip(test_settings):
    store = LocalFileStore(test_settings)
    key = store.save_xml(b"<doc/>", batch_id="BATCH-2")
    assert store.read_xml(key) == b"<doc/>"


def test_local_store_resubmit_xml_overwrites_not_appends(test_settings):
    store = LocalFileStore(test_settings)
    key1 = store.save_xml(b"<doc><DataM>one</DataM></doc>", batch_id="BATCH-3")
    key2 = store.save_xml(b"<doc><DataM>two</DataM></doc>", batch_id="BATCH-3")
    assert key1 == key2
    assert store.read_xml(key2) == b"<doc><DataM>two</DataM></doc>"


def test_local_store_missing_xml_raises_file_not_found(test_settings):
    store = LocalFileStore(test_settings)
    with pytest.raises(FileNotFoundError):
        store.read_xml("BATCH-NOPE.xml")


# --- AzureBlobFileStore (mocked SDK - no network) ---

def _mock_settings():
    settings = MagicMock()
    settings.azure_storage_configured = True
    settings.azure_storage_connection_string = "DefaultEndpointsProtocol=https;AccountName=x;AccountKey=eQ==;EndpointSuffix=core.windows.net"
    settings.azure_upload_container = "uploaded-images"
    settings.azure_xml_container = "generated-xml"
    return settings


def test_azure_store_refuses_to_construct_when_not_configured():
    settings = _mock_settings()
    settings.azure_storage_configured = False
    with pytest.raises(StorageUnavailableError):
        AzureBlobFileStore(settings)


@patch("azure.storage.blob.BlobServiceClient.from_connection_string")
def test_azure_store_blob_name_is_batch_scoped(mock_from_conn):
    mock_client = MagicMock()
    mock_from_conn.return_value = mock_client
    store = AzureBlobFileStore(_mock_settings())

    key = store.save_upload(b"data", document_id="DOC-DDDDDDDDDDDD", extension=".png", batch_id="BATCH-42")
    assert key == "BATCH-42/DOC-DDDDDDDDDDDD.png"
    mock_client.get_blob_client.assert_called_with(container="uploaded-images", blob=key)


@patch("azure.storage.blob.BlobServiceClient.from_connection_string")
def test_azure_store_legacy_scope_when_no_batch(mock_from_conn):
    mock_from_conn.return_value = MagicMock()
    store = AzureBlobFileStore(_mock_settings())
    key = store.save_upload(b"data", document_id="DOC-EEEEEEEEEEEE", extension=".jpg", batch_id=None)
    assert key == "legacy/DOC-EEEEEEEEEEEE.jpg"


@patch("azure.storage.blob.BlobServiceClient.from_connection_string")
def test_azure_store_missing_blob_fails_fast_without_retry(mock_from_conn):
    """A genuinely missing blob must raise immediately, not after the
    full retry/backoff sequence - retrying a 404 can never succeed and
    would only add latency (this was a real bug caught and fixed by hand-
    verifying against the real account - see the final report)."""
    from azure.core.exceptions import ResourceNotFoundError

    mock_blob_client = MagicMock()
    mock_blob_client.download_blob.side_effect = ResourceNotFoundError("not found")
    mock_client = MagicMock()
    mock_client.get_blob_client.return_value = mock_blob_client
    mock_from_conn.return_value = mock_client

    store = AzureBlobFileStore(_mock_settings())
    with pytest.raises(FileNotFoundError):
        store.read_upload("legacy/DOC-FFFFFFFFFFFF.jpg")
    # fails on the first call - no retry sleep loop for a permanent 404
    assert mock_blob_client.download_blob.call_count == 1


@patch("azure.storage.blob.BlobServiceClient.from_connection_string")
def test_azure_store_retries_transient_failure_then_succeeds(mock_from_conn):
    mock_blob_client = MagicMock()
    mock_blob_client.upload_blob.side_effect = [TimeoutError("transient"), None]
    mock_client = MagicMock()
    mock_client.get_blob_client.return_value = mock_blob_client
    mock_from_conn.return_value = mock_client

    store = AzureBlobFileStore(_mock_settings())
    with patch("time.sleep"):  # don't actually wait during the test
        key = store.save_upload(b"data", document_id="DOC-111111111111", extension=".jpg", batch_id="BATCH-5")
    assert key == "BATCH-5/DOC-111111111111.jpg"
    assert mock_blob_client.upload_blob.call_count == 2


@patch("azure.storage.blob.BlobServiceClient.from_connection_string")
def test_azure_store_gives_up_after_max_retries(mock_from_conn):
    mock_blob_client = MagicMock()
    mock_blob_client.upload_blob.side_effect = TimeoutError("still down")
    mock_client = MagicMock()
    mock_client.get_blob_client.return_value = mock_blob_client
    mock_from_conn.return_value = mock_client

    store = AzureBlobFileStore(_mock_settings())
    with patch("time.sleep"):
        with pytest.raises(StorageUnavailableError):
            store.save_upload(b"data", document_id="DOC-222222222222", extension=".jpg", batch_id="BATCH-6")
