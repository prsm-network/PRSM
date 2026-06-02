"""
Unit tests for content upload CLI and API.

Tests cover:
- CLI calls correct endpoint
- CLI requires authentication
- CLI sends auth header
- API endpoint registers provenance
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from click.testing import CliRunner

from prsm.cli import main


@pytest.fixture
def cli_runner():
    """Provide a Click CLI test runner."""
    return CliRunner()


class TestContentUploadCli:
    """Tests for prsm storage upload CLI command."""

    def test_upload_calls_correct_endpoint(self, tmp_path, monkeypatch, cli_runner):
        """prsm storage upload calls /content/upload (sprint 832 F29 fix — NOT the
        phantom /api/v1/content/upload, whose router is unmounted on production
        daemons and 404s every operator)."""
        # Write a temp file to upload
        tmp_file = tmp_path / "test_upload.txt"
        tmp_file.write_text("test content")

        # Write mock credentials file
        cred_file = tmp_path / ".prsm" / "credentials.json"
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text(json.dumps({
            "access_token": "test-token",
            "user_id": "test-user",
            "api_url": "http://localhost:8000"
        }))
        monkeypatch.setattr("prsm.cli._CREDENTIALS_FILE", cred_file)

        # Mock httpx.post to capture the URL
        captured_url = []
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "cid": "QmTestCID123",
            "filename": "test_upload.txt",
            "size_bytes": 12,
            "content_hash": "abc123",
            "creator_id": "test-user",
            "royalty_rate": 0.01,
            "parent_cids": [],
            "provenance_registered": True,
            "access_url": "https://ipfs.io/ipfs/QmTestCID123"
        }

        def capture_post(*args, **kwargs):
            captured_url.append(args[0] if args else kwargs.get("url"))
            return mock_response

        with patch("httpx.post", side_effect=capture_post):
            result = cli_runner.invoke(main, ["storage", "upload", str(tmp_file)])

        # Assert captured URL ends with the working "/content/upload" path and
        # NEVER the phantom "/api/v1/content/upload" (F29 404 regression guard).
        assert len(captured_url) == 1
        assert captured_url[0].endswith("/content/upload")
        assert "/api/v1/content/upload" not in captured_url[0]

    def test_upload_no_auth_does_not_require_login(self, tmp_path, monkeypatch, cli_runner):
        """Sprint 832 — the inline /content/upload endpoint is intentionally
        unauthenticated (the legacy auth-gated router was unmounted), so the CLI
        no longer hard-fails on a missing JWT. With no creds AND no server it
        still exits 1, but via the connect-error path with an actionable
        "start the server" hint — NOT a "prsm login" gate."""
        # Create a real temp file (click.Path(exists=True) requires it to exist)
        tmp_file = tmp_path / "test_upload.txt"
        tmp_file.write_text("test content")

        # No credentials file present - point to non-existent file
        non_existent = tmp_path / ".prsm" / "credentials.json"
        monkeypatch.setattr("prsm.cli._CREDENTIALS_FILE", non_existent)

        # Invoke upload
        result = cli_runner.invoke(main, ["storage", "upload", str(tmp_file)])

        # Still exits 1 (no server to connect to), but pins the sprint-832
        # contract: login is no longer required; the actionable hint is to
        # start the node, not to log in.
        assert result.exit_code == 1
        assert "prsm node start" in result.output
        assert "prsm login" not in result.output

    def test_upload_sends_auth_header(self, tmp_path, monkeypatch, cli_runner):
        """prsm storage upload includes Authorization: Bearer in the request."""
        # Write a temp file to upload
        tmp_file = tmp_path / "test_upload.txt"
        tmp_file.write_text("test content")

        # Write mock credentials file
        cred_file = tmp_path / ".prsm" / "credentials.json"
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text(json.dumps({
            "access_token": "test-token",
            "user_id": "test-user",
            "api_url": "http://localhost:8000"
        }))
        monkeypatch.setattr("prsm.cli._CREDENTIALS_FILE", cred_file)

        # Mock httpx.post to capture headers
        captured_headers = []
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "cid": "QmTestCID123",
            "filename": "test_upload.txt",
            "size_bytes": 12,
            "content_hash": "abc123",
            "creator_id": "test-user",
            "royalty_rate": 0.01,
            "parent_cids": [],
            "provenance_registered": True,
            "access_url": "https://ipfs.io/ipfs/QmTestCID123"
        }

        def capture_post(*args, **kwargs):
            captured_headers.append(kwargs.get("headers", {}))
            return mock_response

        with patch("httpx.post", side_effect=capture_post):
            result = cli_runner.invoke(main, ["storage", "upload", str(tmp_file)])

        # Assert "Bearer test-token" in captured headers["Authorization"]
        assert len(captured_headers) == 1
        auth_header = captured_headers[0].get("Authorization", "")
        assert "Bearer test-token" in auth_header

    def test_upload_endpoint_registers_provenance(self, tmp_path):
        """POST /api/v1/content/upload calls ProvenanceQueries.upsert_provenance."""
        import asyncio
        from unittest.mock import AsyncMock

        # Mock the native ContentStore (v1.4.0+ path — IPFS was removed)
        # The endpoint now imports get_content_store and ContentHash from
        # prsm.storage inside the function body, so patch there.
        from prsm.storage import ContentHash

        deterministic_hash = ContentHash.from_data(b"test content")
        mock_store = MagicMock()
        mock_store.store_local = AsyncMock(return_value=deterministic_hash)

        # Mock ProvenanceQueries.upsert_provenance
        with patch("prsm.storage.get_content_store", return_value=mock_store), \
             patch("prsm.core.database.ProvenanceQueries.upsert_provenance", new_callable=AsyncMock) as mock_upsert:

            mock_upsert.return_value = True

            # Import the upload function directly to test without FastAPI multipart dependency
            from prsm.interface.api.content_api import upload_content_with_provenance
            from starlette.datastructures import UploadFile
            import io

            # Create a mock upload file
            mock_upload = UploadFile(
                filename="test.txt",
                file=io.BytesIO(b"test content")
            )

            # Call the endpoint directly
            async def run_test():
                result = await upload_content_with_provenance(
                    file=mock_upload,
                    description="Test file",
                    royalty_rate=0.01,
                    parent_cids="",
                    replicas=3,
                    current_user="test-user"
                )
                return result

            result = asyncio.run(run_test())

            # Assert result contains expected fields
            assert result.get("provenance_registered") == True
            assert result.get("cid") == deterministic_hash.hex()

            # Assert ProvenanceQueries.upsert_provenance called with creator_id == authenticated user
            mock_upsert.assert_called_once()
            call_args = mock_upsert.call_args[0][0]
            assert call_args.get("creator_id") == "test-user"
