# src/google_sheets.py
"""Google Sheets OAuth2 integration for fetching form responses."""
from __future__ import annotations

import os
from pathlib import Path


def fetch_sheet_as_records(
    spreadsheet_url: str,
    credentials_path: str = "credentials.json",
    token_cache_dir: str = "~/.config/bhu-ocr",
) -> list[dict[str, str]]:
    """Fetch all rows from a Google Sheet as a list of dicts.

    Uses OAuth2 with token caching. On first run, opens a browser
    for authentication. Subsequent runs use the cached token.

    Args:
        spreadsheet_url: full Google Sheets URL.
        credentials_path: path to OAuth2 client credentials JSON.
        token_cache_dir: directory for caching the OAuth2 token.

    Returns:
        List of dicts, one per row, keyed by header names.

    Raises:
        FileNotFoundError: if *credentials_path* does not exist.
        RuntimeError: on network or API errors.
    """
    try:
        import gspread
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise ImportError(
            "google_sheets requires gspread and google-auth-oauthlib. "
            "Install with: pip install gspread google-auth-oauthlib"
        ) from exc

    if not os.path.exists(credentials_path):
        raise FileNotFoundError(
            f"OAuth2 credentials file not found: {credentials_path!r}"
        )

    cache_dir = Path(token_cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    token_path = cache_dir / "token.json"

    SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

    creds: Credentials | None = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if creds is None or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, SCOPES,
            )
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())
        os.chmod(token_path, 0o600)

    try:
        gc = gspread.authorize(creds)
        sheet = gc.open_by_url(spreadsheet_url)
        worksheet = sheet.sheet1
        return worksheet.get_all_records()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch Google Sheet: {exc}"
        ) from exc
