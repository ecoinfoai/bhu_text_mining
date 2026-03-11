"""Tests for get_smtp_config() in config.py.

Phase 3 (US1): Extract SMTP settings from forma.json configuration.
Maps JSON field names to SmtpConfig fields via _build_smtp_config().
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# get_smtp_config() -- happy path
# ---------------------------------------------------------------------------


class TestGetSmtpConfigHappyPath:
    """Tests for get_smtp_config() with valid input."""

    def test_minimal_smtp_section(self):
        """Minimal smtp section with server + sender_email returns SmtpConfig."""
        from forma.config import get_smtp_config
        from forma.delivery_send import SmtpConfig

        config = {
            "smtp": {
                "server": "smtp.gmail.com",
                "sender_email": "prof@univ.kr",
            }
        }
        cfg = get_smtp_config(config)
        assert isinstance(cfg, SmtpConfig)
        assert cfg.smtp_server == "smtp.gmail.com"
        assert cfg.sender_email == "prof@univ.kr"
        assert cfg.smtp_port == 587  # default
        assert cfg.use_tls is True  # default

    def test_full_smtp_section(self):
        """All JSON fields mapped correctly to SmtpConfig."""
        from forma.config import get_smtp_config

        config = {
            "smtp": {
                "server": "mail.example.com",
                "port": 465,
                "sender_email": "test@example.com",
                "sender_name": "Test Prof",
                "use_tls": False,
                "send_interval_sec": 2.0,
            }
        }
        cfg = get_smtp_config(config)
        assert cfg.smtp_server == "mail.example.com"
        assert cfg.smtp_port == 465
        assert cfg.sender_email == "test@example.com"
        assert cfg.sender_name == "Test Prof"
        assert cfg.use_tls is False
        assert cfg.send_interval_sec == 2.0

    def test_extra_keys_ignored(self):
        """Extra keys in smtp section are ignored."""
        from forma.config import get_smtp_config

        config = {
            "smtp": {
                "server": "smtp.x.com",
                "sender_email": "a@b.com",
                "unknown_field": "value",
            }
        }
        cfg = get_smtp_config(config)
        assert cfg.smtp_server == "smtp.x.com"


# ---------------------------------------------------------------------------
# get_smtp_config() -- error cases
# ---------------------------------------------------------------------------


class TestGetSmtpConfigErrors:
    """Tests for get_smtp_config() error handling."""

    def test_missing_smtp_section_raises_keyerror(self):
        """Config without 'smtp' key raises KeyError with Korean message."""
        from forma.config import get_smtp_config

        with pytest.raises(KeyError, match="smtp"):
            get_smtp_config({})

    def test_smtp_section_not_dict_raises_keyerror(self):
        """Config with smtp as non-dict (e.g., string) raises KeyError."""
        from forma.config import get_smtp_config

        with pytest.raises(KeyError, match="smtp"):
            get_smtp_config({"smtp": "not-a-dict"})

    def test_smtp_section_none_raises_keyerror(self):
        """Config with smtp=None raises KeyError."""
        from forma.config import get_smtp_config

        with pytest.raises(KeyError, match="smtp"):
            get_smtp_config({"smtp": None})

    def test_missing_server_in_smtp_raises_valueerror(self):
        """smtp section without 'server' key raises ValueError."""
        from forma.config import get_smtp_config

        with pytest.raises(ValueError, match="smtp_server"):
            get_smtp_config({"smtp": {"sender_email": "a@b.com"}})

    def test_missing_sender_email_raises_valueerror(self):
        """smtp section without 'sender_email' raises ValueError."""
        from forma.config import get_smtp_config

        with pytest.raises(ValueError, match="sender_email"):
            get_smtp_config({"smtp": {"server": "smtp.x.com"}})

    def test_invalid_port_raises_valueerror(self):
        """smtp section with invalid port raises ValueError."""
        from forma.config import get_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            get_smtp_config({
                "smtp": {
                    "server": "s",
                    "sender_email": "a@b.com",
                    "port": 0,
                }
            })

    def test_invalid_email_raises_valueerror(self):
        """smtp section with email without '@' raises ValueError."""
        from forma.config import get_smtp_config

        with pytest.raises(ValueError, match="sender_email"):
            get_smtp_config({
                "smtp": {
                    "server": "s",
                    "sender_email": "no-at-sign",
                }
            })


# ---------------------------------------------------------------------------
# JSON_FIELD_MAP constant
# ---------------------------------------------------------------------------


class TestJsonFieldMap:
    """Tests for JSON_FIELD_MAP exported from config.py."""

    def test_json_field_map_has_expected_keys(self):
        """JSON_FIELD_MAP contains all required mappings."""
        from forma.config import JSON_FIELD_MAP

        assert JSON_FIELD_MAP["server"] == "smtp_server"
        assert JSON_FIELD_MAP["port"] == "smtp_port"
        assert JSON_FIELD_MAP["sender_email"] == "sender_email"
        assert JSON_FIELD_MAP["sender_name"] == "sender_name"
        assert JSON_FIELD_MAP["use_tls"] == "use_tls"
        assert JSON_FIELD_MAP["send_interval_sec"] == "send_interval_sec"


# ---------------------------------------------------------------------------
# get_smtp_password()
# ---------------------------------------------------------------------------


class TestGetSmtpPassword:
    """Tests for get_smtp_password() in config.py."""

    def test_returns_password_when_present(self):
        """smtp.password field is returned as string."""
        from forma.config import get_smtp_password

        config = {
            "smtp": {
                "server": "smtp.gmail.com",
                "sender_email": "a@b.com",
                "password": "app-secret-16chars",
            }
        }
        assert get_smtp_password(config) == "app-secret-16chars"

    def test_returns_none_when_password_absent(self):
        """smtp section without password field returns None."""
        from forma.config import get_smtp_password

        config = {"smtp": {"server": "s", "sender_email": "a@b.com"}}
        assert get_smtp_password(config) is None

    def test_returns_none_when_no_smtp_section(self):
        """Config without smtp section returns None (no exception)."""
        from forma.config import get_smtp_password

        assert get_smtp_password({}) is None

    def test_password_is_string(self):
        """Numeric password value is converted to string."""
        from forma.config import get_smtp_password

        config = {"smtp": {"password": 12345678}}
        assert get_smtp_password(config) == "12345678"
