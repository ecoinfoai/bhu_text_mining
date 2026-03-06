# tests/test_google_sheets.py
"""Tests for src/google_sheets.py."""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from forma.google_sheets import fetch_sheet_as_records


# ── fixtures ──────────────────────────────────────


@pytest.fixture
def credentials_file(tmp_path):
    """Write a fake OAuth2 credentials JSON."""
    creds = {
        "installed": {
            "client_id": "fake.apps.googleusercontent.com",
            "client_secret": "fake_secret",
            "redirect_uris": ["http://localhost"],
        }
    }
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps(creds))
    return str(path)


@pytest.fixture
def token_cache_dir(tmp_path):
    """Provide a temp dir for token caching."""
    d = tmp_path / "token_cache"
    d.mkdir()
    return str(d)


def _make_valid_creds():
    """Create a mock Credentials object that looks valid."""
    mock_creds = MagicMock()
    mock_creds.valid = True
    mock_creds.expired = False
    mock_creds.to_json.return_value = '{"token": "cached"}'
    return mock_creds


def _make_mock_gc(records=None):
    """Create a mock gspread client."""
    if records is None:
        records = []
    mock_gc = MagicMock()
    mock_gc.open_by_url.return_value.sheet1.get_all_records.return_value = (
        records
    )
    return mock_gc


# ──────────────────────────────────────────────────
# Group 1: credential file validation
# ──────────────────────────────────────────────────


class TestCredentialValidation:
    def test_missing_credentials_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="credentials file"):
            fetch_sheet_as_records(
                "https://docs.google.com/spreadsheets/d/abc",
                credentials_path=str(tmp_path / "nonexistent.json"),
                token_cache_dir=str(tmp_path / "cache"),
            )


# ──────────────────────────────────────────────────
# Group 2: successful fetch (fully mocked)
# ──────────────────────────────────────────────────


class TestFetchSuccess:
    def test_fetch_returns_records(self, credentials_file, token_cache_dir):
        expected = [
            {"student_id": "S001", "이름": "홍길동"},
            {"student_id": "S002", "이름": "김영희"},
        ]

        mock_creds = _make_valid_creds()
        mock_gc = _make_mock_gc(expected)

        # Pre-create a token file so the cached-token path is taken
        token_path = os.path.join(token_cache_dir, "token.json")
        with open(token_path, "w") as f:
            f.write("{}")

        with (
            patch(
                "google.oauth2.credentials.Credentials.from_authorized_user_file",
                return_value=mock_creds,
            ),
            patch("gspread.authorize", return_value=mock_gc),
        ):
            result = fetch_sheet_as_records(
                "https://docs.google.com/spreadsheets/d/abc",
                credentials_path=credentials_file,
                token_cache_dir=token_cache_dir,
            )

        assert result == expected
        mock_gc.open_by_url.assert_called_once()


# ──────────────────────────────────────────────────
# Group 3: token caching
# ──────────────────────────────────────────────────


class TestTokenCaching:
    def test_token_cached_after_auth(self, credentials_file, token_cache_dir):
        """After OAuth2 flow, token.json is written to cache dir."""
        mock_creds = _make_valid_creds()

        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = mock_creds

        mock_gc = _make_mock_gc()

        with (
            patch(
                "google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file",
                return_value=mock_flow,
            ),
            patch("gspread.authorize", return_value=mock_gc),
        ):
            fetch_sheet_as_records(
                "https://docs.google.com/spreadsheets/d/abc",
                credentials_path=credentials_file,
                token_cache_dir=token_cache_dir,
            )

        token_path = os.path.join(token_cache_dir, "token.json")
        assert os.path.exists(token_path)
        with open(token_path) as f:
            assert json.loads(f.read()) == {"token": "cached"}

    def test_expired_token_refreshed(self, credentials_file, token_cache_dir):
        """Expired token triggers refresh, not full re-auth."""
        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "refresh_me"
        mock_creds.to_json.return_value = '{"refreshed": true}'

        def do_refresh(request):
            mock_creds.valid = True

        mock_creds.refresh.side_effect = do_refresh

        mock_gc = _make_mock_gc()

        # Write an existing token file
        token_path = os.path.join(token_cache_dir, "token.json")
        with open(token_path, "w") as f:
            f.write('{"old": "token"}')

        with (
            patch(
                "google.oauth2.credentials.Credentials.from_authorized_user_file",
                return_value=mock_creds,
            ),
            patch("gspread.authorize", return_value=mock_gc),
        ):
            fetch_sheet_as_records(
                "https://docs.google.com/spreadsheets/d/abc",
                credentials_path=credentials_file,
                token_cache_dir=token_cache_dir,
            )

        mock_creds.refresh.assert_called_once()


# ──────────────────────────────────────────────────
# Group 4: network failure
# ──────────────────────────────────────────────────


class TestFetchFailure:
    def test_network_error_raises_runtime_error(
        self, credentials_file, token_cache_dir,
    ):
        mock_creds = _make_valid_creds()

        mock_gc = MagicMock()
        mock_gc.open_by_url.side_effect = Exception("Network unreachable")

        token_path = os.path.join(token_cache_dir, "token.json")
        with open(token_path, "w") as f:
            f.write("{}")

        with (
            patch(
                "google.oauth2.credentials.Credentials.from_authorized_user_file",
                return_value=mock_creds,
            ),
            patch("gspread.authorize", return_value=mock_gc),
        ):
            with pytest.raises(RuntimeError, match="Failed to fetch"):
                fetch_sheet_as_records(
                    "https://docs.google.com/spreadsheets/d/abc",
                    credentials_path=credentials_file,
                    token_cache_dir=token_cache_dir,
                )
