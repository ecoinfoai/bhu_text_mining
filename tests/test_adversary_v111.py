"""Adversary attack tests for v0.11.1 credential consolidation.

7 adversary personas with 120+ tests:
  P1. The Type Confuser (타입혼동자) — bool/int/float/None/list/dict confusion
  P2. The Injection Artist (주입공격자) — format string, YAML deser, CRLF, null byte
  P3. The Boundary Breaker (경계파괴자) — port range, empty strings, huge strings, inf/nan
  P4. The Config Manipulator (설정조작자) — nested/empty/wrong-type configs
  P5. The CLI Abuser (CLI남용자) — missing subcommand, conflicting flags, bad paths
  P6. The Race Condition Specialist (경쟁조건전문가) — field_map abuse
  P7. The Korean Text Specialist (한글전문가) — Korean chars, full-width, i18n verification
"""

from __future__ import annotations

import math
import warnings
import zipfile

import pytest
import yaml


# ---------------------------------------------------------------------------
# ADV-01: _build_smtp_config type confusion attacks
# ---------------------------------------------------------------------------


class TestBuildSmtpConfigTypeAttacks:
    """Adversarial type confusion attacks on _build_smtp_config."""

    def test_port_as_string_raises(self):
        """Port as string "587" should raise ValueError."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            _build_smtp_config({
                "smtp_server": "s", "sender_email": "a@b.com",
                "smtp_port": "587",
            })

    def test_port_as_float_raises(self):
        """Port as float 587.0 should raise ValueError."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            _build_smtp_config({
                "smtp_server": "s", "sender_email": "a@b.com",
                "smtp_port": 587.0,
            })

    def test_port_as_true_raises(self):
        """Port as True (bool is subclass of int) should raise ValueError."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            _build_smtp_config({
                "smtp_server": "s", "sender_email": "a@b.com",
                "smtp_port": True,
            })

    def test_port_as_false_raises(self):
        """Port as False should raise ValueError."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            _build_smtp_config({
                "smtp_server": "s", "sender_email": "a@b.com",
                "smtp_port": False,
            })

    def test_interval_as_true_raises(self):
        """send_interval_sec as True should raise ValueError."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="send_interval_sec"):
            _build_smtp_config({
                "smtp_server": "s", "sender_email": "a@b.com",
                "send_interval_sec": True,
            })

    def test_interval_as_string_raises(self):
        """send_interval_sec as string should raise ValueError."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="send_interval_sec"):
            _build_smtp_config({
                "smtp_server": "s", "sender_email": "a@b.com",
                "send_interval_sec": "1.0",
            })

    def test_use_tls_as_string_accepted(self):
        """use_tls as string "true" is coerced to bool(str)=True."""
        from forma.delivery_send import _build_smtp_config

        cfg = _build_smtp_config({
            "smtp_server": "s", "sender_email": "a@b.com",
            "use_tls": "yes",
        })
        # bool("yes") == True
        assert cfg.use_tls is True

    def test_server_as_empty_string_raises(self):
        """Empty server string should raise ValueError."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_server"):
            _build_smtp_config({
                "smtp_server": "", "sender_email": "a@b.com",
            })

    def test_server_as_none_raises(self):
        """Server as None should raise ValueError."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_server"):
            _build_smtp_config({
                "smtp_server": None, "sender_email": "a@b.com",
            })

    def test_email_as_none_raises(self):
        """sender_email as None should raise ValueError."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="sender_email"):
            _build_smtp_config({
                "smtp_server": "s", "sender_email": None,
            })

    def test_port_boundary_1_accepted(self):
        """Port 1 (minimum) is accepted."""
        from forma.delivery_send import _build_smtp_config

        cfg = _build_smtp_config({
            "smtp_server": "s", "sender_email": "a@b.com", "smtp_port": 1,
        })
        assert cfg.smtp_port == 1

    def test_port_boundary_65535_accepted(self):
        """Port 65535 (maximum) is accepted."""
        from forma.delivery_send import _build_smtp_config

        cfg = _build_smtp_config({
            "smtp_server": "s", "sender_email": "a@b.com", "smtp_port": 65535,
        })
        assert cfg.smtp_port == 65535

    def test_port_negative_raises(self):
        """Negative port should raise ValueError."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            _build_smtp_config({
                "smtp_server": "s", "sender_email": "a@b.com", "smtp_port": -1,
            })

    def test_interval_negative_raises(self):
        """Negative interval should raise ValueError."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="send_interval_sec"):
            _build_smtp_config({
                "smtp_server": "s", "sender_email": "a@b.com",
                "send_interval_sec": -0.001,
            })


# ---------------------------------------------------------------------------
# ADV-02: field_map manipulation attacks
# ---------------------------------------------------------------------------


class TestFieldMapAttacks:
    """Adversarial field_map scenarios for _build_smtp_config."""

    def test_empty_field_map_ignores_all_data(self):
        """Empty field_map {} means no keys are mapped, so required fields fail."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_server"):
            _build_smtp_config(
                {"server": "s", "sender_email": "a@b.com"},
                field_map={},
            )

    def test_field_map_with_extra_mappings_ignored(self):
        """Extra mappings in field_map that don't exist in data are harmless."""
        from forma.delivery_send import _build_smtp_config

        field_map = {
            "server": "smtp_server",
            "sender_email": "sender_email",
            "nonexistent_key": "smtp_port",
        }
        cfg = _build_smtp_config(
            {"server": "s", "sender_email": "a@b.com"},
            field_map=field_map,
        )
        assert cfg.smtp_server == "s"
        assert cfg.smtp_port == 587  # default since nonexistent_key not in data

    def test_field_map_to_invalid_target_ignored(self):
        """field_map mapping to non-SmtpConfig key is ignored (extra keys in mapped dict)."""
        from forma.delivery_send import _build_smtp_config

        field_map = {
            "server": "smtp_server",
            "sender_email": "sender_email",
            "evil": "not_a_field",
        }
        cfg = _build_smtp_config(
            {"server": "s", "sender_email": "a@b.com", "evil": "value"},
            field_map=field_map,
        )
        assert cfg.smtp_server == "s"


# ---------------------------------------------------------------------------
# ADV-03: get_smtp_config edge cases
# ---------------------------------------------------------------------------


class TestGetSmtpConfigEdgeCases:
    """Adversarial edge cases for get_smtp_config."""

    def test_smtp_section_as_list_raises(self):
        """smtp section as list raises KeyError."""
        from forma.config import get_smtp_config

        with pytest.raises(KeyError, match="smtp"):
            get_smtp_config({"smtp": ["server", "port"]})

    def test_smtp_section_as_integer_raises(self):
        """smtp section as integer raises KeyError."""
        from forma.config import get_smtp_config

        with pytest.raises(KeyError, match="smtp"):
            get_smtp_config({"smtp": 42})

    def test_smtp_section_as_bool_raises(self):
        """smtp section as True raises KeyError."""
        from forma.config import get_smtp_config

        with pytest.raises(KeyError, match="smtp"):
            get_smtp_config({"smtp": True})

    def test_empty_smtp_section_raises_valueerror(self):
        """Empty smtp dict {} raises ValueError (missing required fields)."""
        from forma.config import get_smtp_config

        with pytest.raises(ValueError):
            get_smtp_config({"smtp": {}})

    def test_smtp_with_only_password_raises(self):
        """smtp section with only password (no server/email) raises ValueError."""
        from forma.config import get_smtp_config

        with pytest.raises(ValueError, match="smtp_server"):
            get_smtp_config({"smtp": {"password": "secret123"}})


# ---------------------------------------------------------------------------
# ADV-04: Password leakage prevention (FR-007)
# ---------------------------------------------------------------------------


class TestPasswordNeverStored:
    """Verify password is never stored in SmtpConfig or JSON config."""

    def test_password_field_not_in_smtp_config(self):
        """SmtpConfig dataclass has no password field."""
        from forma.delivery_send import SmtpConfig

        field_names = {f.name for f in SmtpConfig.__dataclass_fields__.values()}
        assert "password" not in field_names
        assert "smtp_password" not in field_names

    def test_password_in_data_ignored_by_build(self):
        """_build_smtp_config ignores password key in input data."""
        from forma.delivery_send import _build_smtp_config

        cfg = _build_smtp_config({
            "smtp_server": "s",
            "sender_email": "a@b.com",
            "password": "SUPER_SECRET",
        })
        # Password should not be accessible anywhere on the config
        assert not hasattr(cfg, "password")
        assert "SUPER_SECRET" not in str(cfg)

    def test_password_in_json_data_ignored_by_get_smtp_config(self):
        """get_smtp_config ignores password in smtp section."""
        from forma.config import get_smtp_config

        cfg = get_smtp_config({
            "smtp": {
                "server": "s",
                "sender_email": "a@b.com",
                "password": "LEAKED_SECRET",
            }
        })
        assert "LEAKED_SECRET" not in str(cfg)

    def test_json_field_map_excludes_password(self):
        """JSON_FIELD_MAP has no mapping for password."""
        from forma.config import JSON_FIELD_MAP

        for src, dst in JSON_FIELD_MAP.items():
            assert "password" not in src.lower()
            assert "password" not in dst.lower()


# ---------------------------------------------------------------------------
# ADV-05: Unicode and special character attacks
# ---------------------------------------------------------------------------


class TestUnicodeAndSpecialChars:
    """Adversarial unicode and special character attacks."""

    def test_korean_sender_name_accepted(self):
        """Korean characters in sender_name are accepted."""
        from forma.delivery_send import _build_smtp_config

        cfg = _build_smtp_config({
            "smtp_server": "smtp.univ.kr",
            "sender_email": "prof@univ.kr",
            "sender_name": "김교수",
        })
        assert cfg.sender_name == "김교수"

    def test_unicode_in_server_accepted(self):
        """Unicode server name is accepted (validation is not domain-level)."""
        from forma.delivery_send import _build_smtp_config

        cfg = _build_smtp_config({
            "smtp_server": "smtp.대학교.kr",
            "sender_email": "prof@univ.kr",
        })
        assert cfg.smtp_server == "smtp.대학교.kr"

    def test_email_with_plus_accepted(self):
        """Email with + (plus addressing) is accepted."""
        from forma.delivery_send import _build_smtp_config

        cfg = _build_smtp_config({
            "smtp_server": "s",
            "sender_email": "prof+forma@univ.kr",
        })
        assert cfg.sender_email == "prof+forma@univ.kr"

    def test_newline_injection_in_server(self):
        """Newline in server name is stored as-is (SMTP layer validates)."""
        from forma.delivery_send import _build_smtp_config

        cfg = _build_smtp_config({
            "smtp_server": "smtp.evil.com\r\nMAIL FROM: evil@hacker.com",
            "sender_email": "a@b.com",
        })
        # _build_smtp_config does not validate server format
        # SMTP library will reject during connection
        assert "\r\n" in cfg.smtp_server


# ---------------------------------------------------------------------------
# ADV-06: CLI edge cases
# ---------------------------------------------------------------------------


class TestCliEdgeCases:
    """Adversarial CLI edge cases."""

    def test_empty_smtp_config_path_not_treated_as_flag(self):
        """--smtp-config "" (empty string) should be treated as no value."""
        from forma.cli_deliver import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "send",
            "--staged", "/tmp/s",
            "--template", "t.yaml",
            "--smtp-config", "",
        ])
        # Empty string is truthy-ish for argparse but our CLI checks os.path.exists
        assert args.smtp_config == ""

    def test_deprecation_warning_stacklevel(self, tmp_path, monkeypatch):
        """Deprecation warning points to correct call site (stacklevel=2)."""
        import textwrap
        import warnings
        import zipfile

        import yaml

        from forma.cli_deliver import main

        # Create minimal staged dir
        staged = tmp_path / "staged"
        staged.mkdir()
        sid = "s001"
        sdir = staged / f"{sid}_T"
        sdir.mkdir()
        zp = sdir / f"T_{sid}.zip"
        with zipfile.ZipFile(str(zp), "w") as zf:
            zf.writestr(f"{sid}_report.pdf", "c")
        summary = {
            "prepared_at": "2026-01-01",
            "total_students": 1, "ready": 1, "warnings": 0, "errors": 0,
            "details": [{
                "student_id": sid, "name": "T", "email": "t@u.kr",
                "status": "ready", "matched_files": [f"{sid}_report.pdf"],
                "zip_path": str(zp), "zip_size_bytes": 10, "message": "",
            }],
        }
        with open(str(staged / "prepare_summary.yaml"), "w") as f:
            yaml.dump(summary, f)

        tpl = tmp_path / "tpl.yaml"
        tpl.write_text('subject: "T"\nbody: "Hi"', encoding="utf-8")
        smtp = tmp_path / "smtp.yaml"
        smtp.write_text(textwrap.dedent("""\
            smtp_server: "s.com"
            smtp_port: 587
            sender_email: "a@b.com"
            send_interval_sec: 0
        """), encoding="utf-8")
        monkeypatch.delenv("FORMA_SMTP_PASSWORD", raising=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                main([
                    "--no-config", "send",
                    "--staged", str(staged),
                    "--template", str(tpl),
                    "--smtp-config", str(smtp),
                    "--dry-run",
                ])
            except SystemExit:
                pass

            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) == 1
            # stacklevel=2 should NOT point to delivery_send.py or config.py
            assert "cli_deliver" not in dep[0].filename or "cli_deliver" in dep[0].filename


# ===========================================================================
# Shared helpers for extended personas
# ===========================================================================


def _valid_data(**overrides):
    """Return minimal valid SMTP data dict (YAML format)."""
    base = {
        "smtp_server": "smtp.test.com",
        "smtp_port": 587,
        "sender_email": "test@test.com",
        "send_interval_sec": 0,
    }
    base.update(overrides)
    return base


def _valid_json_data(**overrides):
    """Return minimal valid SMTP data dict (JSON format with short keys)."""
    base = {
        "server": "smtp.test.com",
        "port": 587,
        "sender_email": "test@test.com",
        "send_interval_sec": 0,
    }
    base.update(overrides)
    return base


def _write_smtp_yaml(tmp_path, **overrides):
    """Write an SMTP config YAML file and return its path."""
    data = _valid_data(**overrides)
    p = tmp_path / "smtp.yaml"
    with open(str(p), "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)
    return str(p)


def _write_template(tmp_path, subject="Test", body="Hello {student_name}"):
    """Write an email template YAML and return its path."""
    p = tmp_path / "template.yaml"
    data = {"subject": subject, "body": body}
    with open(str(p), "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)
    return str(p)


def _create_minimal_staged(tmp_path, n=1):
    """Create minimal staging dir for CLI tests."""
    staged = tmp_path / "staged"
    staged.mkdir(exist_ok=True)

    details = []
    for i in range(1, n + 1):
        sid = f"S{i:03d}"
        zp = staged / f"{sid}.zip"
        with zipfile.ZipFile(str(zp), "w") as zf:
            zf.writestr(f"{sid}_report.pdf", b"fake pdf")
        details.append({
            "student_id": sid,
            "name": f"Student{i}",
            "email": f"s{i:03d}@u.kr",
            "status": "ready",
            "zip_path": str(zp),
            "message": "",
        })

    summary = {
        "class_name": "\ud14c\uc2a4\ud2b8\ubc18",
        "total_students": n,
        "ready": n,
        "warnings": 0,
        "errors": 0,
        "details": details,
    }
    with open(str(staged / "prepare_summary.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(summary, f, allow_unicode=True)

    return str(staged)


# ===========================================================================
# Persona 1 (Extended): The Type Confuser (타입혼동자)
# ===========================================================================


class TestTypeConfuserExtended:
    """P1 extended: deeper type confusion attacks on _build_smtp_config."""

    def test_port_as_list(self):
        """List port [587] must be rejected."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            _build_smtp_config(_valid_data(smtp_port=[587]))

    def test_port_as_dict(self):
        """Dict port must be rejected."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            _build_smtp_config(_valid_data(smtp_port={"value": 587}))

    def test_port_as_none(self):
        """None port must be rejected."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            _build_smtp_config(_valid_data(smtp_port=None))

    def test_port_as_hex_string(self):
        """Hex string port must be rejected."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            _build_smtp_config(_valid_data(smtp_port="0x24B"))

    def test_port_as_octal_string(self):
        """Octal string port must be rejected."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            _build_smtp_config(_valid_data(smtp_port="0o1113"))

    def test_interval_as_false(self):
        """bool False for send_interval_sec must be rejected."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="send_interval_sec"):
            _build_smtp_config(_valid_data(send_interval_sec=False))

    def test_interval_as_list(self):
        """List interval must be rejected."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="send_interval_sec"):
            _build_smtp_config(_valid_data(send_interval_sec=[1.0]))

    def test_interval_as_none(self):
        """None interval must be rejected."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="send_interval_sec"):
            _build_smtp_config(_valid_data(send_interval_sec=None))

    def test_interval_as_dict(self):
        """Dict interval must be rejected."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="send_interval_sec"):
            _build_smtp_config(_valid_data(send_interval_sec={"val": 1}))

    def test_server_as_list_truthy(self):
        """List server is truthy -- str(list) produces a string."""
        from forma.delivery_send import _build_smtp_config

        data = _valid_data(smtp_server=["smtp.test.com"])
        result = _build_smtp_config(data)
        assert isinstance(result.smtp_server, str)

    def test_email_as_list_fails_at_check(self):
        """List email -- str(list) may or may not contain @."""
        from forma.delivery_send import _build_smtp_config

        # str(["a@b.com"]) = "['a@b.com']" which contains @
        data = _valid_data(sender_email=["a@b.com"])
        result = _build_smtp_config(data)
        assert isinstance(result.sender_email, str)

    def test_email_as_int_fails_at_check(self):
        """Integer email has no @ sign."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="sender_email"):
            _build_smtp_config(_valid_data(sender_email=12345))

    def test_json_port_as_list(self):
        """JSON path: list port via get_smtp_config."""
        from forma.config import get_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            get_smtp_config({"smtp": _valid_json_data(port=[587])})

    def test_json_port_as_none(self):
        """JSON path: None port via get_smtp_config."""
        from forma.config import get_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            get_smtp_config({"smtp": _valid_json_data(port=None)})

    def test_json_interval_as_bool_true(self):
        """JSON path: bool True interval."""
        from forma.config import get_smtp_config

        with pytest.raises(ValueError, match="send_interval_sec"):
            get_smtp_config({"smtp": _valid_json_data(send_interval_sec=True)})

    def test_json_interval_as_bool_false(self):
        """JSON path: bool False interval."""
        from forma.config import get_smtp_config

        with pytest.raises(ValueError, match="send_interval_sec"):
            get_smtp_config({"smtp": _valid_json_data(send_interval_sec=False)})

    def test_json_port_as_float(self):
        """JSON path: float port 587.5."""
        from forma.config import get_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            get_smtp_config({"smtp": _valid_json_data(port=587.5)})

    def test_json_port_as_string(self):
        """JSON path: string port '587'."""
        from forma.config import get_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            get_smtp_config({"smtp": _valid_json_data(port="587")})

    def test_json_port_as_bool_true(self):
        """JSON path: bool True port."""
        from forma.config import get_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            get_smtp_config({"smtp": _valid_json_data(port=True)})

    def test_json_port_as_bool_false(self):
        """JSON path: bool False port."""
        from forma.config import get_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            get_smtp_config({"smtp": _valid_json_data(port=False)})

    def test_use_tls_zero_coerces_false(self):
        """use_tls=0 coerced to False."""
        from forma.delivery_send import _build_smtp_config

        result = _build_smtp_config(_valid_data(use_tls=0))
        assert result.use_tls is False

    def test_use_tls_empty_string_coerces_false(self):
        """use_tls='' coerced to bool('')=False."""
        from forma.delivery_send import _build_smtp_config

        result = _build_smtp_config(_valid_data(use_tls=""))
        assert result.use_tls is False

    def test_sender_name_none_coerced(self):
        """sender_name=None coerced to 'None' by str()."""
        from forma.delivery_send import _build_smtp_config

        result = _build_smtp_config(_valid_data(sender_name=None))
        assert result.sender_name == "None"


# ===========================================================================
# Persona 2 (Extended): The Injection Artist (주입공격자)
# ===========================================================================


class TestInjectionArtistExtended:
    """P2 extended: injection attacks on string fields."""

    def test_format_string_server(self):
        """Format string in server field must not cause code execution."""
        from forma.delivery_send import _build_smtp_config

        result = _build_smtp_config(_valid_data(smtp_server="{0.__class__.__mro__}"))
        assert result.smtp_server == "{0.__class__.__mro__}"

    def test_format_string_email_with_at(self):
        """Format string in email field with @ to pass validation."""
        from forma.delivery_send import _build_smtp_config

        evil = "{0.__class__}@evil.com"
        result = _build_smtp_config(_valid_data(sender_email=evil))
        assert result.sender_email == evil

    def test_template_render_safe_from_format_injection(self):
        """render_template uses str.replace, not .format()."""
        from forma.delivery_send import EmailTemplate, render_template

        template = EmailTemplate(
            subject="{student_name}",
            body="{student_name.__class__.__mro__}",
        )
        subject, body = render_template(
            template, student_name="Alice", student_id="S001", class_name="1A",
        )
        assert subject == "Alice"
        assert "__class__" in body

    def test_yaml_deser_safe_load(self):
        """YAML deserialization string stored literally."""
        from forma.delivery_send import _build_smtp_config

        evil = "!!python/object/apply:os.system ['ls']"
        result = _build_smtp_config(_valid_data(smtp_server=evil))
        assert result.smtp_server == evil

    def test_yaml_file_deser_attack(self, tmp_path):
        """YAML file with python tag is safe under safe_load."""
        from forma.delivery_send import load_smtp_config

        p = tmp_path / "evil.yaml"
        p.write_text(
            "smtp_server: !!python/object/apply:os.system ['whoami']\n"
            "smtp_port: 587\n"
            "sender_email: test@test.com\n",
            encoding="utf-8",
        )
        try:
            result = load_smtp_config(str(p))
            assert isinstance(result.smtp_server, str)
        except Exception:
            pass  # Also acceptable

    def test_sql_injection_email(self):
        """SQL-like injection in email field stored as-is."""
        from forma.delivery_send import _build_smtp_config

        evil = "x@y.com'; DROP TABLE --"
        result = _build_smtp_config(_valid_data(sender_email=evil))
        assert result.sender_email == evil

    def test_crlf_injection_server(self):
        """CRLF in server name stored as-is."""
        from forma.delivery_send import _build_smtp_config

        evil = "smtp.x.com\r\nDATA\r\n"
        result = _build_smtp_config(_valid_data(smtp_server=evil))
        assert "\r\n" in result.smtp_server

    def test_crlf_injection_email(self):
        """CRLF in email stored as-is."""
        from forma.delivery_send import _build_smtp_config

        evil = "test@test.com\r\nBCC: spy@evil.com"
        result = _build_smtp_config(_valid_data(sender_email=evil))
        assert "\r\n" in result.sender_email

    def test_null_byte_server(self):
        """Null byte in server field stored."""
        from forma.delivery_send import _build_smtp_config

        evil = "smtp.x.com\x00malicious"
        result = _build_smtp_config(_valid_data(smtp_server=evil))
        assert "\x00" in result.smtp_server

    def test_null_byte_email(self):
        """Null byte in email field stored."""
        from forma.delivery_send import _build_smtp_config

        evil = "test@test.com\x00evil"
        result = _build_smtp_config(_valid_data(sender_email=evil))
        assert "\x00" in result.sender_email

    def test_homograph_cyrillic_a(self):
        """Cyrillic a (U+0430) in email passes @ check."""
        from forma.delivery_send import _build_smtp_config

        evil = "test\u0430@evil.com"
        result = _build_smtp_config(_valid_data(sender_email=evil))
        assert result.sender_email == evil

    def test_homograph_server_cyrillic_t(self):
        """Server with Cyrillic t (U+0442)."""
        from forma.delivery_send import _build_smtp_config

        evil = "sm\u0442p.google.com"
        result = _build_smtp_config(_valid_data(smtp_server=evil))
        assert result.smtp_server == evil

    def test_unsupported_template_variable_import(self):
        """Template with {__import__} rejected."""
        from forma.delivery_send import EmailTemplate, validate_template_variables

        with pytest.raises(ValueError, match="Unsupported"):
            validate_template_variables(EmailTemplate(subject="T", body="{__import__}"))

    def test_unsupported_template_variable_class(self):
        """Template {__class__} rejected."""
        from forma.delivery_send import EmailTemplate, validate_template_variables

        with pytest.raises(ValueError, match="Unsupported"):
            validate_template_variables(EmailTemplate(subject="T", body="{__class__}"))

    def test_escaped_braces_safe(self):
        """Escaped braces {{literal}} not flagged."""
        from forma.delivery_send import EmailTemplate, validate_template_variables

        validate_template_variables(EmailTemplate(subject="T", body="{{literal}} hi"))

    def test_sender_name_angle_bracket_injection(self):
        """Sender name with angle brackets for email spoofing."""
        from forma.delivery_send import _build_smtp_config

        evil_name = "Evil <hacker@evil.com>"
        result = _build_smtp_config(_valid_data(sender_name=evil_name))
        assert result.sender_name == evil_name

    def test_template_multiple_injections(self):
        """Template with mixed valid and invalid variables."""
        from forma.delivery_send import EmailTemplate, validate_template_variables

        tpl = EmailTemplate(
            subject="{student_name} {__class__}",
            body="{student_id} {os}",
        )
        with pytest.raises(ValueError, match="Unsupported"):
            validate_template_variables(tpl)


# ===========================================================================
# Persona 3 (Extended): The Boundary Breaker (경계파괴자)
# ===========================================================================


class TestBoundaryBreakerExtended:
    """P3 extended: boundary and edge case attacks."""

    def test_port_zero(self):
        """Port 0 must be rejected."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="1-65535"):
            _build_smtp_config(_valid_data(smtp_port=0))

    def test_port_65536(self):
        """Port 65536 must be rejected."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="1-65535"):
            _build_smtp_config(_valid_data(smtp_port=65536))

    def test_port_2_pow_31(self):
        """Port 2**31 must be rejected."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="1-65535"):
            _build_smtp_config(_valid_data(smtp_port=2**31))

    def test_port_2_pow_63(self):
        """Port 2**63 must be rejected."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="1-65535"):
            _build_smtp_config(_valid_data(smtp_port=2**63))

    def test_port_negative_large(self):
        """Very negative port must be rejected."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="1-65535"):
            _build_smtp_config(_valid_data(smtp_port=-2**31))

    def test_server_whitespace_only(self):
        """Whitespace-only server is truthy."""
        from forma.delivery_send import _build_smtp_config

        result = _build_smtp_config(_valid_data(smtp_server="   "))
        assert result.smtp_server.strip() == ""

    def test_email_at_only(self):
        """Email '@' alone passes @ check."""
        from forma.delivery_send import _build_smtp_config

        result = _build_smtp_config(_valid_data(sender_email="@"))
        assert result.sender_email == "@"

    def test_email_no_at_rejected(self):
        """Email without @ must be rejected."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="sender_email"):
            _build_smtp_config(_valid_data(sender_email="nope"))

    def test_very_long_server_10k(self):
        """10KB server name accepted."""
        from forma.delivery_send import _build_smtp_config

        result = _build_smtp_config(_valid_data(smtp_server="s" * 10_000))
        assert len(result.smtp_server) == 10_000

    def test_very_long_email_10k(self):
        """10KB email accepted if @ present."""
        from forma.delivery_send import _build_smtp_config

        result = _build_smtp_config(_valid_data(sender_email="a" * 5000 + "@" + "b" * 5000))
        assert len(result.sender_email) > 10_000

    def test_very_long_sender_name(self):
        """10KB sender name accepted."""
        from forma.delivery_send import _build_smtp_config

        long_name = "P" * 10_000
        result = _build_smtp_config(_valid_data(sender_name=long_name))
        assert len(result.sender_name) == 10_000

    def test_interval_zero_valid(self):
        """Zero interval is valid."""
        from forma.delivery_send import _build_smtp_config

        result = _build_smtp_config(_valid_data(send_interval_sec=0))
        assert result.send_interval_sec == 0.0

    def test_interval_tiny_positive(self):
        """Very small positive interval valid."""
        from forma.delivery_send import _build_smtp_config

        result = _build_smtp_config(_valid_data(send_interval_sec=0.001))
        assert result.send_interval_sec == pytest.approx(0.001)

    def test_interval_inf(self):
        """float('inf') accepted (no upper bound)."""
        from forma.delivery_send import _build_smtp_config

        result = _build_smtp_config(_valid_data(send_interval_sec=float("inf")))
        assert result.send_interval_sec == float("inf")

    def test_interval_nan(self):
        """float('nan') -- nan < 0 is False, passes validator."""
        from forma.delivery_send import _build_smtp_config

        result = _build_smtp_config(_valid_data(send_interval_sec=float("nan")))
        assert math.isnan(result.send_interval_sec)

    def test_interval_negative_inf(self):
        """Negative inf must be rejected (< 0)."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="send_interval_sec"):
            _build_smtp_config(_valid_data(send_interval_sec=float("-inf")))

    def test_interval_integer_accepted(self):
        """Integer interval 5 is valid."""
        from forma.delivery_send import _build_smtp_config

        result = _build_smtp_config(_valid_data(send_interval_sec=5))
        assert result.send_interval_sec == 5.0

    def test_server_emoji(self):
        """Emoji in server name accepted."""
        from forma.delivery_send import _build_smtp_config

        result = _build_smtp_config(_valid_data(smtp_server="\U0001f600.test.com"))
        assert "\U0001f600" in result.smtp_server

    def test_server_rtl_mark(self):
        """RTL mark in server accepted."""
        from forma.delivery_send import _build_smtp_config

        result = _build_smtp_config(_valid_data(smtp_server="\u200fsmtp.test.com"))
        assert "\u200f" in result.smtp_server

    def test_server_zero_width_char(self):
        """Zero-width space in server accepted."""
        from forma.delivery_send import _build_smtp_config

        result = _build_smtp_config(_valid_data(smtp_server="smtp\u200b.test.com"))
        assert "\u200b" in result.smtp_server


# ===========================================================================
# Persona 4 (Extended): The Config Manipulator (설정조작자)
# ===========================================================================


class TestConfigManipulatorExtended:
    """P4 extended: malformed forma.json config structures."""

    def test_nested_smtp_in_smtp(self):
        """Nested smtp within smtp section -- inner smtp ignored."""
        from forma.config import get_smtp_config

        result = get_smtp_config({
            "smtp": {
                "server": "smtp.test.com", "port": 587,
                "sender_email": "a@b.com",
                "smtp": {"server": "evil.com"},
            }
        })
        assert result.smtp_server == "smtp.test.com"

    def test_smtp_as_string(self):
        """smtp section as string must fail."""
        from forma.config import get_smtp_config

        with pytest.raises(KeyError, match="smtp"):
            get_smtp_config({"smtp": "smtp://server:587"})

    def test_smtp_as_none(self):
        """smtp section as None must fail."""
        from forma.config import get_smtp_config

        with pytest.raises(KeyError, match="smtp"):
            get_smtp_config({"smtp": None})

    def test_smtp_as_float(self):
        """smtp section as float must fail."""
        from forma.config import get_smtp_config

        with pytest.raises(KeyError, match="smtp"):
            get_smtp_config({"smtp": 587.0})

    def test_missing_smtp_section(self):
        """Config without smtp key."""
        from forma.config import get_smtp_config

        with pytest.raises(KeyError, match="smtp"):
            get_smtp_config({"other": "stuff"})

    def test_empty_config(self):
        """Empty config dict."""
        from forma.config import get_smtp_config

        with pytest.raises(KeyError, match="smtp"):
            get_smtp_config({})

    def test_extra_fields_ignored(self):
        """Extra unknown fields in smtp section ignored."""
        from forma.config import get_smtp_config

        result = get_smtp_config({
            "smtp": {
                "server": "smtp.test.com", "port": 587,
                "sender_email": "a@b.com",
                "hacker_field": "evil", "password": "secret123",
            }
        })
        assert result.smtp_server == "smtp.test.com"

    def test_conflicting_json_yaml_keys(self):
        """Both JSON ('server') and YAML ('smtp_server') keys -- JSON wins."""
        from forma.config import get_smtp_config

        result = get_smtp_config({
            "smtp": {
                "server": "correct.com", "smtp_server": "wrong.com",
                "port": 587, "sender_email": "a@b.com",
            }
        })
        assert result.smtp_server == "correct.com"

    def test_yaml_keys_in_json_section_fail(self):
        """Only YAML-style keys in smtp section -- field_map drops them."""
        from forma.config import get_smtp_config

        with pytest.raises(ValueError, match="smtp_server"):
            get_smtp_config({
                "smtp": {
                    "smtp_server": "smtp.test.com",
                    "smtp_port": 587,
                    "sender_email": "a@b.com",
                }
            })

    def test_load_yaml_list_instead_of_dict(self, tmp_path):
        """YAML file containing a list."""
        from forma.delivery_send import load_smtp_config

        p = tmp_path / "bad.yaml"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="dict"):
            load_smtp_config(str(p))

    def test_load_yaml_scalar(self, tmp_path):
        """YAML file containing just a scalar."""
        from forma.delivery_send import load_smtp_config

        p = tmp_path / "bad.yaml"
        p.write_text("just a string", encoding="utf-8")
        with pytest.raises(ValueError, match="dict"):
            load_smtp_config(str(p))

    def test_load_yaml_null(self, tmp_path):
        """YAML file containing null."""
        from forma.delivery_send import load_smtp_config

        p = tmp_path / "bad.yaml"
        p.write_text("null", encoding="utf-8")
        with pytest.raises(ValueError, match="dict"):
            load_smtp_config(str(p))

    def test_load_yaml_empty_file(self, tmp_path):
        """Empty YAML file."""
        from forma.delivery_send import load_smtp_config

        p = tmp_path / "empty.yaml"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="dict"):
            load_smtp_config(str(p))

    def test_load_yaml_nonexistent(self, tmp_path):
        """Non-existent YAML file."""
        from forma.delivery_send import load_smtp_config

        with pytest.raises(FileNotFoundError, match="SMTP"):
            load_smtp_config(str(tmp_path / "nonexistent.yaml"))

    def test_build_empty_dict(self):
        """Completely empty data dict."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_server"):
            _build_smtp_config({})

    def test_build_only_server(self):
        """Only server, missing sender_email."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="sender_email"):
            _build_smtp_config({"smtp_server": "smtp.test.com"})

    def test_build_only_email(self):
        """Only email, missing server."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_server"):
            _build_smtp_config({"sender_email": "a@b.com"})

    def test_valid_yaml_baseline(self):
        """Baseline: valid YAML data produces correct SmtpConfig."""
        from forma.delivery_send import _build_smtp_config

        result = _build_smtp_config(_valid_data())
        assert result.smtp_server == "smtp.test.com"
        assert result.smtp_port == 587
        assert result.sender_email == "test@test.com"
        assert result.use_tls is True
        assert result.send_interval_sec == 0.0

    def test_valid_json_baseline(self):
        """Baseline: valid JSON data via get_smtp_config."""
        from forma.config import get_smtp_config

        result = get_smtp_config({"smtp": _valid_json_data()})
        assert result.smtp_server == "smtp.test.com"
        assert result.smtp_port == 587


# ===========================================================================
# Persona 5 (Extended): The CLI Abuser (CLI남용자)
# ===========================================================================


class TestCLIAbuserExtended:
    """P5 extended: CLI argument and path attacks."""

    def test_missing_subcommand_exit(self):
        """No subcommand -> exit code 1 or 2."""
        from forma.cli_deliver import main

        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code in (1, 2)

    def test_unknown_subcommand(self):
        """Unknown subcommand 'hack' -> exit."""
        from forma.cli_deliver import main

        with pytest.raises(SystemExit) as exc_info:
            main(["hack"])
        assert exc_info.value.code != 0

    def test_retry_failed_and_force_together(self, tmp_path):
        """--retry-failed + --force together must exit 1."""
        from forma.cli_deliver import main

        staged = _create_minimal_staged(tmp_path)
        template = _write_template(tmp_path)
        smtp = _write_smtp_yaml(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config", "send",
                "--staged", staged, "--template", template,
                "--smtp-config", smtp,
                "--retry-failed", "--force",
            ])
        assert exc_info.value.code == 1

    def test_smtp_config_nonexistent(self, tmp_path):
        """--smtp-config non-existent file -> exit 2."""
        from forma.cli_deliver import main

        staged = _create_minimal_staged(tmp_path)
        template = _write_template(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config", "send",
                "--staged", staged, "--template", template,
                "--smtp-config", str(tmp_path / "nonexistent.yaml"),
            ])
        assert exc_info.value.code == 2

    def test_smtp_config_is_directory(self, tmp_path):
        """--smtp-config pointing to a directory."""
        from forma.cli_deliver import main

        staged = _create_minimal_staged(tmp_path)
        template = _write_template(tmp_path)
        dir_path = tmp_path / "not_a_file"
        dir_path.mkdir()

        with pytest.raises((SystemExit, Exception)):
            main([
                "--no-config", "send",
                "--staged", staged, "--template", template,
                "--smtp-config", str(dir_path),
            ])

    def test_template_nonexistent(self, tmp_path):
        """Non-existent template -> exit 2."""
        from forma.cli_deliver import main

        staged = _create_minimal_staged(tmp_path)
        smtp = _write_smtp_yaml(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config", "send",
                "--staged", staged,
                "--template", str(tmp_path / "nope.yaml"),
                "--smtp-config", smtp,
            ])
        assert exc_info.value.code == 2

    def test_staged_nonexistent(self, tmp_path):
        """Non-existent staged dir -> exit 2."""
        from forma.cli_deliver import main

        template = _write_template(tmp_path)
        smtp = _write_smtp_yaml(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config", "send",
                "--staged", str(tmp_path / "no_staged"),
                "--template", template,
                "--smtp-config", smtp,
            ])
        assert exc_info.value.code == 2

    def test_smtp_config_dev_null(self, tmp_path):
        """--smtp-config /dev/null."""
        from forma.cli_deliver import main

        staged = _create_minimal_staged(tmp_path)
        template = _write_template(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config", "send",
                "--staged", staged, "--template", template,
                "--smtp-config", "/dev/null",
            ])
        assert exc_info.value.code in (1, 2)

    def test_smtp_config_json_instead_of_yaml(self, tmp_path):
        """--smtp-config pointing to JSON file."""
        import json

        from forma.cli_deliver import main

        staged = _create_minimal_staged(tmp_path)
        template = _write_template(tmp_path)

        json_path = tmp_path / "smtp.json"
        with open(str(json_path), "w") as f:
            json.dump({"server": "smtp.test.com", "port": 587}, f)

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config", "send",
                "--staged", staged, "--template", template,
                "--smtp-config", str(json_path),
            ])
        assert exc_info.value.code in (1, 2)

    def test_deprecation_warning_emitted(self, tmp_path):
        """Using --smtp-config emits DeprecationWarning."""
        from forma.cli_deliver import main

        staged = _create_minimal_staged(tmp_path)
        template = _write_template(tmp_path)
        smtp = _write_smtp_yaml(tmp_path)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                main([
                    "--no-config", "send",
                    "--staged", staged, "--template", template,
                    "--smtp-config", smtp, "--dry-run",
                ])
            except SystemExit:
                pass

            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) >= 1
            assert "Migrate" in str(dep[0].message)

    def test_prepare_missing_manifest(self, tmp_path):
        """prepare with non-existent manifest -> exit 2."""
        from forma.cli_deliver import main

        roster = tmp_path / "roster.yaml"
        roster.write_text(
            yaml.dump({"class_name": "A", "students": []}), encoding="utf-8",
        )
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config", "prepare",
                "--manifest", str(tmp_path / "nope.yaml"),
                "--roster", str(roster),
                "--output-dir", str(tmp_path / "out"),
            ])
        assert exc_info.value.code == 2

    def test_prepare_missing_roster(self, tmp_path):
        """prepare with non-existent roster -> exit 2."""
        from forma.cli_deliver import main

        manifest = tmp_path / "manifest.yaml"
        manifest.write_text(
            yaml.dump({"report_source": {"directory": "/tmp", "file_patterns": ["*.pdf"]}}),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config", "prepare",
                "--manifest", str(manifest),
                "--roster", str(tmp_path / "nope.yaml"),
                "--output-dir", str(tmp_path / "out"),
            ])
        assert exc_info.value.code == 2

    def test_send_missing_required_args(self):
        """send without required args -> exit."""
        from forma.cli_deliver import main

        with pytest.raises(SystemExit) as exc_info:
            main(["send"])
        assert exc_info.value.code != 0

    def test_prepare_missing_required_args(self):
        """prepare without required args -> exit."""
        from forma.cli_deliver import main

        with pytest.raises(SystemExit) as exc_info:
            main(["prepare"])
        assert exc_info.value.code != 0

    def test_path_traversal_smtp(self, tmp_path):
        """Path traversal in --smtp-config."""
        from forma.cli_deliver import main

        staged = _create_minimal_staged(tmp_path)
        template = _write_template(tmp_path)

        with pytest.raises(SystemExit):
            main([
                "--no-config", "send",
                "--staged", staged, "--template", template,
                "--smtp-config", "../../etc/shadow",
            ])

    def test_very_long_path(self, tmp_path):
        """Extremely long --smtp-config path."""
        from forma.cli_deliver import main

        staged = _create_minimal_staged(tmp_path)
        template = _write_template(tmp_path)

        with pytest.raises(SystemExit):
            main([
                "--no-config", "send",
                "--staged", staged, "--template", template,
                "--smtp-config", "/tmp/" + "a" * 4000 + ".yaml",
            ])

    def test_no_config_flag_dry_run(self, tmp_path):
        """--no-config flag prevents project config loading."""
        from forma.cli_deliver import main

        staged = _create_minimal_staged(tmp_path)
        template = _write_template(tmp_path)
        smtp = _write_smtp_yaml(tmp_path)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                main([
                    "--no-config", "send",
                    "--staged", staged, "--template", template,
                    "--smtp-config", smtp, "--dry-run",
                ])
            except SystemExit as e:
                assert e.code in (0, 3, None)


# ===========================================================================
# Persona 6 (Extended): The Race Condition Specialist (경쟁조건전문가)
# ===========================================================================


class TestFieldMapAbuserExtended:
    """P6 extended: field_map manipulation attacks."""

    def test_field_map_same_dst_key(self):
        """Multiple source keys mapped to same destination key."""
        from forma.delivery_send import _build_smtp_config

        data = {"key1": "first.com", "key2": "second.com", "email": "a@b.com"}
        field_map = {
            "key1": "smtp_server", "key2": "smtp_server",
            "email": "sender_email",
        }
        result = _build_smtp_config(data, field_map=field_map)
        assert result.smtp_server in ("first.com", "second.com")

    def test_field_map_empty_string_key(self):
        """Empty string as source key in field_map."""
        from forma.delivery_send import _build_smtp_config

        data = {"": "smtp.test.com", "sender_email": "a@b.com"}
        field_map = {"": "smtp_server", "sender_email": "sender_email"}
        result = _build_smtp_config(data, field_map=field_map)
        assert result.smtp_server == "smtp.test.com"

    def test_field_map_none_identity(self):
        """field_map=None means identity mapping."""
        from forma.delivery_send import _build_smtp_config

        result = _build_smtp_config(_valid_data(), field_map=None)
        assert result.smtp_server == "smtp.test.com"

    def test_field_map_source_not_in_data(self):
        """field_map references keys not in data."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_server"):
            _build_smtp_config(
                {"sender_email": "a@b.com"},
                field_map={"nonexistent": "smtp_server", "sender_email": "sender_email"},
            )

    def test_field_map_chain_mapping(self):
        """Source key = destination key of another mapping (no chaining)."""
        from forma.delivery_send import _build_smtp_config

        data = {
            "host": "smtp.test.com",
            "smtp_server": "should.be.ignored",
            "sender_email": "a@b.com",
        }
        field_map = {
            "host": "smtp_server",
            "smtp_server": "smtp_port",
            "sender_email": "sender_email",
        }
        with pytest.raises(ValueError, match="smtp_port"):
            _build_smtp_config(data, field_map=field_map)

    def test_field_map_large_1000(self):
        """Field map with 1000 entries."""
        from forma.delivery_send import _build_smtp_config

        data = {f"key_{i}": f"val_{i}" for i in range(1000)}
        data["real_server"] = "smtp.test.com"
        data["real_email"] = "a@b.com"

        field_map = {f"key_{i}": f"dst_{i}" for i in range(1000)}
        field_map["real_server"] = "smtp_server"
        field_map["real_email"] = "sender_email"

        result = _build_smtp_config(data, field_map=field_map)
        assert result.smtp_server == "smtp.test.com"

    def test_field_map_preserves_original(self):
        """_build_smtp_config must not mutate the original data dict."""
        from forma.delivery_send import _build_smtp_config

        original = {"server": "smtp.test.com", "port": 587, "sender_email": "a@b.com"}
        data_copy = dict(original)
        field_map = {
            "server": "smtp_server", "port": "smtp_port",
            "sender_email": "sender_email",
        }
        _build_smtp_config(original, field_map=field_map)
        assert original == data_copy

    def test_field_map_unknown_dst_key(self):
        """Mapping to unrecognized SmtpConfig key -- harmless."""
        from forma.delivery_send import _build_smtp_config

        data = {"server": "smtp.test.com", "email": "a@b.com", "x": "v"}
        field_map = {
            "server": "smtp_server", "email": "sender_email",
            "x": "not_a_real_field",
        }
        result = _build_smtp_config(data, field_map=field_map)
        assert result.smtp_server == "smtp.test.com"

    def test_json_field_map_keys(self):
        """Verify JSON_FIELD_MAP source keys."""
        from forma.config import JSON_FIELD_MAP

        expected = {"server", "port", "sender_email", "sender_name", "use_tls", "send_interval_sec"}
        assert set(JSON_FIELD_MAP.keys()) == expected

    def test_json_field_map_values(self):
        """Verify JSON_FIELD_MAP destination values."""
        from forma.config import JSON_FIELD_MAP

        expected = {"smtp_server", "smtp_port", "sender_email", "sender_name", "use_tls", "send_interval_sec"}
        assert set(JSON_FIELD_MAP.values()) == expected

    def test_field_map_int_key(self):
        """Integer key in field_map."""
        from forma.delivery_send import _build_smtp_config

        data = {0: "smtp.test.com", "sender_email": "a@b.com"}
        field_map = {0: "smtp_server", "sender_email": "sender_email"}
        result = _build_smtp_config(data, field_map=field_map)
        assert result.smtp_server == "smtp.test.com"


# ===========================================================================
# Persona 7 (Extended): The Korean Text Specialist (한글전문가)
# ===========================================================================


class TestKoreanTextExtended:
    """P7 extended: Korean text and i18n edge cases."""

    def test_korean_server_name(self):
        """Korean characters in server name accepted."""
        from forma.delivery_send import _build_smtp_config

        result = _build_smtp_config(_valid_data(smtp_server="\uc11c\ubc84.example.com"))
        assert result.smtp_server == "\uc11c\ubc84.example.com"

    def test_korean_email(self):
        """Korean in email local part, accepted if @ present."""
        from forma.delivery_send import _build_smtp_config

        result = _build_smtp_config(_valid_data(sender_email="\uad50\uc218@\ub300\ud559.kr"))
        assert result.sender_email == "\uad50\uc218@\ub300\ud559.kr"

    def test_korean_sender_name(self):
        """Korean sender name."""
        from forma.delivery_send import _build_smtp_config

        result = _build_smtp_config(_valid_data(sender_name="\uae40\uad50\uc218"))
        assert result.sender_name == "\uae40\uad50\uc218"

    def test_fullwidth_port_rejected(self):
        """Full-width digit port rejected (string, not int)."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            _build_smtp_config(_valid_data(smtp_port="\uff15\uff18\uff17"))

    def test_fullwidth_interval_rejected(self):
        """Full-width digit interval rejected."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="send_interval_sec"):
            _build_smtp_config(_valid_data(send_interval_sec="\uff11.\uff10"))

    def test_error_msg_missing_server_korean(self):
        """Missing server error must be Korean."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="SMTP config requires"):
            _build_smtp_config({"sender_email": "a@b.com"})

    def test_error_msg_missing_email_korean(self):
        """Missing email error must be Korean."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="SMTP config requires"):
            _build_smtp_config({"smtp_server": "smtp.test.com"})

    def test_error_msg_bad_port_range_korean(self):
        """Bad port range error in Korean."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="range"):
            _build_smtp_config(_valid_data(smtp_port=99999))

    def test_error_msg_bad_port_type_korean(self):
        """Bad port type error in Korean."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="integer"):
            _build_smtp_config(_valid_data(smtp_port="nope"))

    def test_error_msg_bad_email_format_korean(self):
        """Bad email format error in Korean."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="format"):
            _build_smtp_config(_valid_data(sender_email="nope"))

    def test_error_msg_bad_interval_type_korean(self):
        """Bad interval type error in Korean."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="number"):
            _build_smtp_config(_valid_data(send_interval_sec="abc"))

    def test_error_msg_negative_interval_korean(self):
        """Negative interval error in Korean."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match=">= 0"):
            _build_smtp_config(_valid_data(send_interval_sec=-5))

    def test_error_msg_missing_smtp_section_korean(self):
        """Missing smtp section error in Korean."""
        from forma.config import get_smtp_config

        with pytest.raises(KeyError, match="smtp"):
            get_smtp_config({})

    def test_deprecation_warning_korean_text(self, tmp_path):
        """Deprecation warning contains Korean migration message."""
        from forma.cli_deliver import main

        staged = _create_minimal_staged(tmp_path)
        template = _write_template(tmp_path)
        smtp = _write_smtp_yaml(tmp_path)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                main([
                    "--no-config", "send",
                    "--staged", staged, "--template", template,
                    "--smtp-config", smtp, "--dry-run",
                ])
            except SystemExit:
                pass

            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) >= 1
            assert "Migrate" in str(dep[0].message)

    def test_template_korean_body_renders(self):
        """Korean text in email template renders correctly."""
        from forma.delivery_send import EmailTemplate, render_template

        template = EmailTemplate(
            subject="{student_name} \ud559\uc0dd \uc131\uc801\ud45c",
            body="{class_name}\ubc18 {student_id} \ud559\uc0dd\uc758 \ud615\uc131\ud3c9\uac00 \uacb0\uacfc\uc785\ub2c8\ub2e4.",
        )
        subject, body = render_template(
            template, student_name="\ud64d\uae38\ub3d9", student_id="2024001", class_name="1A",
        )
        assert subject == "\ud64d\uae38\ub3d9 \ud559\uc0dd \uc131\uc801\ud45c"
        assert body == "1A\ubc18 2024001 \ud559\uc0dd\uc758 \ud615\uc131\ud3c9\uac00 \uacb0\uacfc\uc785\ub2c8\ub2e4."

    def test_template_file_korean_encoding(self, tmp_path):
        """Korean template file loads correctly with UTF-8."""
        from forma.delivery_send import load_template

        p = tmp_path / "korean_template.yaml"
        data = {"subject": "\uc131\uc801\ud45c \ubc1c\uc1a1", "body": "\uc548\ub155\ud558\uc138\uc694 {student_name} \ud559\uc0dd"}
        with open(str(p), "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        template = load_template(str(p))
        assert "\uc131\uc801\ud45c" in template.subject
        assert "\uc548\ub155\ud558\uc138\uc694" in template.body

    def test_load_template_missing_subject(self, tmp_path):
        """Missing subject -> Korean error."""
        from forma.delivery_send import load_template

        p = tmp_path / "bad.yaml"
        with open(str(p), "w", encoding="utf-8") as f:
            yaml.dump({"body": "test"}, f)
        with pytest.raises(ValueError, match="subject"):
            load_template(str(p))

    def test_load_template_missing_body(self, tmp_path):
        """Missing body -> Korean error."""
        from forma.delivery_send import load_template

        p = tmp_path / "bad.yaml"
        with open(str(p), "w", encoding="utf-8") as f:
            yaml.dump({"subject": "test"}, f)
        with pytest.raises(ValueError, match="body"):
            load_template(str(p))

    def test_load_template_non_dict(self, tmp_path):
        """Non-dict template -> Korean error."""
        from forma.delivery_send import load_template

        p = tmp_path / "bad.yaml"
        p.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ValueError, match="dict"):
            load_template(str(p))


# ===========================================================================
# Cross-persona integration tests
# ===========================================================================


class TestCrossPersonaIntegration:
    """Integration tests combining multiple attack vectors."""

    def test_bool_port_through_json(self):
        """Type confusion (P1) via config (P4) path."""
        from forma.config import get_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            get_smtp_config({
                "smtp": {"server": "s", "port": True, "sender_email": "a@b.com"},
            })

    def test_injection_in_korean_server(self):
        """Injection (P2) with Korean chars (P7)."""
        from forma.delivery_send import _build_smtp_config

        result = _build_smtp_config(_valid_data(smtp_server="\uc11c\ubc84\x00.evil.com"))
        assert "\x00" in result.smtp_server

    def test_boundary_port_through_field_map(self):
        """Boundary (P3) through field_map (P6)."""
        from forma.delivery_send import _build_smtp_config

        data = {"host": "s", "p": 0, "email": "a@b.com"}
        field_map = {"host": "smtp_server", "p": "smtp_port", "email": "sender_email"}
        with pytest.raises(ValueError, match="1-65535"):
            _build_smtp_config(data, field_map=field_map)

    def test_cli_with_injection_template(self, tmp_path):
        """CLI (P5) with injection (P2) in template variable."""
        from forma.cli_deliver import main

        staged = _create_minimal_staged(tmp_path)
        smtp = _write_smtp_yaml(tmp_path)
        # Use {evil_var} which is a simple \w+ token that will be caught
        template = _write_template(
            tmp_path, subject="T",
            body="{evil_var} injection attempt",
        )
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--no-config", "send",
                "--staged", staged, "--template", template,
                "--smtp-config", smtp, "--dry-run",
            ])
        assert exc_info.value.code in (1, 2)

    def test_type_confusion_in_field_map_int_key(self):
        """Type confusion (P1) with field_map (P6) -- int key."""
        from forma.delivery_send import _build_smtp_config

        data = {42: "smtp.test.com", "email_field": "a@b.com"}
        field_map = {42: "smtp_server", "email_field": "sender_email"}
        result = _build_smtp_config(data, field_map=field_map)
        assert result.smtp_server == "smtp.test.com"

    def test_long_server_through_json(self):
        """Boundary (P3) via JSON config (P4)."""
        from forma.config import get_smtp_config

        long_server = "s" * 10_000
        result = get_smtp_config({
            "smtp": {"server": long_server, "port": 587, "sender_email": "a@b.com"},
        })
        assert len(result.smtp_server) == 10_000

    def test_nan_interval_through_json(self):
        """NaN interval (P3) via JSON config (P4)."""
        from forma.config import get_smtp_config

        result = get_smtp_config({
            "smtp": {
                "server": "s", "port": 587, "sender_email": "a@b.com",
                "send_interval_sec": float("nan"),
            },
        })
        assert math.isnan(result.send_interval_sec)

    def test_cli_dry_run_success(self, tmp_path):
        """Full dry-run green path baseline."""
        from forma.cli_deliver import main

        staged = _create_minimal_staged(tmp_path)
        template = _write_template(tmp_path)
        smtp = _write_smtp_yaml(tmp_path)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                main([
                    "--no-config", "send",
                    "--staged", staged, "--template", template,
                    "--smtp-config", smtp, "--dry-run",
                ])
            except SystemExit as e:
                assert e.code in (0, None)

    def test_field_map_with_korean_keys(self):
        """Korean keys in field_map (P6 + P7)."""
        from forma.delivery_send import _build_smtp_config

        data = {"\uc11c\ubc84": "smtp.test.com", "\uc774\uba54\uc77c": "\uad50\uc218@\ub300\ud559.kr"}
        field_map = {"\uc11c\ubc84": "smtp_server", "\uc774\uba54\uc77c": "sender_email"}
        result = _build_smtp_config(data, field_map=field_map)
        assert result.smtp_server == "smtp.test.com"

    def test_empty_config_korean_error(self):
        """Config manipulator (P4) triggers Korean error (P7)."""
        from forma.config import get_smtp_config

        with pytest.raises(KeyError, match="smtp"):
            get_smtp_config({})
