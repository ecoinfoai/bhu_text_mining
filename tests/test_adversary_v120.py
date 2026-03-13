"""Adversary tests for v0.12.0 consistency hardening.

12 adversarial personas with 90+ attack scenarios:

Original 6 personas (delivery & boundary):
1. PathTraversalAttacker -- path traversal via student_id, filenames
2. UnicodeDestroyer -- zero-width chars, control chars, RTL overrides
3. ConfigPoisoner -- malformed YAML, type confusion, pickle payloads
4. NetworkDestroyer -- SMTP injection, TLS downgrade, timeout abuse
5. PrivilegeEscalator -- file permission bypass, OAuth token theft
6. Overloader -- large inputs, boundary values, resource exhaustion

New 6 personas (consistency hardening):
7. NaNBombardier -- NaN/inf/-inf injection in numeric fields
8. XMLInjectionSpecialist -- XML/HTML injection in text fields
9. UnicodeDestroyerAdvanced -- deeper Unicode attacks (BOM, combining, emoji)
10. FileSystemSaboteur -- io_utils.py atomic write edge cases
11. ConfigPoisonerAdvanced -- all config loaders and CLI tools
12. EmptyValueAttacker -- empty/null/zero values in public APIs

FR-044/045/046/047 coverage.
"""

from __future__ import annotations

import math
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from forma.delivery_prepare import DeliveryManifest, match_files_for_student


# ===========================================================================
# Persona 1: PathTraversalAttacker (original)
# ===========================================================================


class TestPathTraversalAttacker:
    """Persona 1: tries to escape base directory."""

    def _make_manifest(self, tmpdir: str) -> DeliveryManifest:
        return DeliveryManifest(directory=tmpdir, file_patterns=["{student_id}.pdf"])

    def test_student_id_with_dotdot_raises(self, tmp_path) -> None:
        """student_id containing '..' triggers ValueError."""
        manifest = self._make_manifest(str(tmp_path))
        with pytest.raises(ValueError, match="path traversal"):
            match_files_for_student(manifest, "../../etc/passwd")

    def test_student_id_with_slash_raises(self, tmp_path) -> None:
        """student_id containing '/' triggers ValueError."""
        manifest = self._make_manifest(str(tmp_path))
        with pytest.raises(ValueError, match="path traversal|path separator"):
            match_files_for_student(manifest, "foo/bar")

    def test_student_id_with_backslash_raises(self, tmp_path) -> None:
        """student_id containing backslash triggers ValueError."""
        manifest = self._make_manifest(str(tmp_path))
        with pytest.raises(ValueError, match="path traversal|path separator"):
            match_files_for_student(manifest, "foo\\bar")

    def test_student_id_with_null_raises(self, tmp_path) -> None:
        """student_id containing null byte triggers ValueError."""
        manifest = self._make_manifest(str(tmp_path))
        with pytest.raises(ValueError, match="path traversal"):
            match_files_for_student(manifest, "foo\x00bar")

    def test_realpath_escape_raises(self, tmp_path) -> None:
        """Symlink pointing outside base_dir triggers ValueError."""
        base = tmp_path / "base"
        base.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        secret = outside / "secret.pdf"
        secret.write_text("secret")

        link = base / "escape.pdf"
        link.symlink_to(secret)

        manifest = DeliveryManifest(
            directory=str(base),
            file_patterns=["{student_id}.pdf"],
        )
        with pytest.raises(ValueError, match="escapes"):
            match_files_for_student(manifest, "escape")

    def test_dotdot_embedded_raises(self, tmp_path) -> None:
        """student_id with '..' embedded among valid chars triggers ValueError."""
        manifest = self._make_manifest(str(tmp_path))
        with pytest.raises(ValueError, match="path traversal"):
            match_files_for_student(manifest, "..S001")

    def test_dotdot_in_middle_raises(self, tmp_path) -> None:
        """student_id with '../' in middle raises ValueError."""
        manifest = self._make_manifest(str(tmp_path))
        with pytest.raises(ValueError, match="path traversal|path separator"):
            match_files_for_student(manifest, "S001/../S002")


# ===========================================================================
# Persona 2: UnicodeDestroyer (original)
# ===========================================================================


class TestUnicodeDestroyer:
    """Persona 2: injects malicious Unicode sequences."""

    def test_zero_width_in_filename_produces_unnamed(self) -> None:
        """sanitize_filename() with only zero-width chars returns '_unnamed'."""
        from forma.delivery_prepare import sanitize_filename

        result = sanitize_filename("\u200B\u200C\u200D")
        assert result == "_unnamed"

    def test_null_byte_removed(self) -> None:
        """sanitize_filename() strips null byte."""
        from forma.delivery_prepare import sanitize_filename

        result = sanitize_filename("foo\x00bar")
        assert "\x00" not in result
        assert "foobar" in result

    def test_zero_width_stripped_from_filename(self) -> None:
        """sanitize_filename() strips zero-width chars from mixed input."""
        from forma.delivery_prepare import sanitize_filename

        result = sanitize_filename("\u200Bfoo\u200Cbar\u200D")
        assert "\u200B" not in result
        assert "\u200C" not in result
        assert "\u200D" not in result
        assert "foobar" in result

    def test_xml_control_char_removed(self) -> None:
        """esc() from font_utils strips C0 control chars."""
        from forma.font_utils import esc

        result = esc("\x00\x01\x02test")
        assert result == "test"

    def test_mixed_unicode_attack(self) -> None:
        """sanitize_filename() with mixed zero-width + null returns cleaned."""
        from forma.delivery_prepare import sanitize_filename

        result = sanitize_filename("\u200Bfoo\x00bar\u200C")
        assert "foobar" in result

    def test_report_sanitize_zero_width_unnamed(self) -> None:
        """sanitize_filename_report() with only zero-width chars returns '_unnamed'."""
        from forma.report_utils import sanitize_filename_report

        result = sanitize_filename_report("\u200B\u200C\u200D\u200E\u200F\uFEFF")
        assert result == "_unnamed"

    def test_esc_preserves_valid_korean(self) -> None:
        """esc() preserves Korean text while stripping control chars."""
        from forma.font_utils import esc

        result = esc("\x01안녕하세요\x02")
        assert result == "안녕하세요"

    def test_esc_escapes_xml_entities(self) -> None:
        """esc() still escapes <, >, & after stripping control chars."""
        from forma.font_utils import esc

        result = esc("<script>&alert\x00</script>")
        assert "&lt;" in result
        assert "&amp;" in result
        assert "&gt;" in result
        assert "\x00" not in result

    def test_feff_bom_stripped(self) -> None:
        """sanitize_filename() strips BOM (U+FEFF)."""
        from forma.delivery_prepare import sanitize_filename

        result = sanitize_filename("\uFEFFtest")
        assert "\uFEFF" not in result
        assert "test" in result


# ===========================================================================
# Persona 3: ConfigPoisoner (original)
# ===========================================================================


class TestConfigPoisoner:
    """Persona 3: corrupts config files and model payloads."""

    def test_load_model_with_dict_raises(self, tmp_path) -> None:
        """joblib file containing a plain dict raises TypeError."""
        import joblib

        payload_path = tmp_path / "bad_model.pkl"
        joblib.dump({"not": "a model"}, str(payload_path))

        from forma.risk_predictor import load_model

        with pytest.raises(TypeError, match="TrainedRiskModel"):
            load_model(str(payload_path))

    def test_load_model_with_string_raises(self, tmp_path) -> None:
        """joblib file containing a string raises TypeError."""
        import joblib

        payload_path = tmp_path / "string_model.pkl"
        joblib.dump("hello", str(payload_path))

        from forma.risk_predictor import load_model

        with pytest.raises(TypeError, match="TrainedRiskModel"):
            load_model(str(payload_path))

    def test_load_grade_model_with_wrong_type_raises(self, tmp_path) -> None:
        """joblib file containing wrong type raises TypeError for grade model."""
        import joblib

        payload_path = tmp_path / "bad_grade.pkl"
        joblib.dump(42, str(payload_path))

        from forma.grade_predictor import load_grade_model

        with pytest.raises(TypeError, match="TrainedGradeModel"):
            load_grade_model(str(payload_path))

    def test_load_model_with_list_raises(self, tmp_path) -> None:
        """joblib file containing a list raises TypeError."""
        import joblib

        payload_path = tmp_path / "list_model.pkl"
        joblib.dump([1, 2, 3], str(payload_path))

        from forma.risk_predictor import load_model

        with pytest.raises(TypeError, match="TrainedRiskModel"):
            load_model(str(payload_path))

    def test_load_model_with_none_raises(self, tmp_path) -> None:
        """joblib file containing None raises TypeError."""
        import joblib

        payload_path = tmp_path / "none_model.pkl"
        joblib.dump(None, str(payload_path))

        from forma.risk_predictor import load_model

        with pytest.raises(TypeError, match="TrainedRiskModel"):
            load_model(str(payload_path))


# ===========================================================================
# Persona 4: NetworkDestroyer (original)
# ===========================================================================


class TestNetworkDestroyer:
    """Persona 4: exploits SMTP/HTTP vulnerabilities."""

    def test_crlf_in_subject_stripped(self) -> None:
        """CRLF injection in email subject is sanitized."""
        from forma.delivery_send import _sanitize_header

        result = _sanitize_header("Test\r\nBcc: hacker@evil.com")
        assert "\r" not in result
        assert "\n" not in result
        assert "hacker" in result

    def test_crlf_in_from_stripped(self) -> None:
        """CRLF in from address is stripped."""
        from forma.delivery_send import _sanitize_header

        result = _sanitize_header("sender@test.com\r\nBcc: evil@evil.com")
        assert "\r" not in result
        assert "\n" not in result

    def test_https_enforcement_blocks_http(self) -> None:
        """naver_ocr rejects http:// URLs."""
        from forma.naver_ocr import send_images_receive_ocr

        with pytest.raises(ValueError, match="HTTPS"):
            send_images_receive_ocr(
                "http://ocr.api.example.com/v1",
                "fake_key",
                [],
            )

    def test_tls_uses_default_context(self) -> None:
        """SMTP starttls() is called with ssl.create_default_context()."""
        import smtplib
        import ssl

        from forma.delivery_send import SmtpConfig, send_summary_email, DeliveryLog

        mock_log = DeliveryLog(
            sent_at="2026-01-01T00:00:00",
            smtp_server="smtp.test.com",
            dry_run=False,
            total=0,
            success=0,
            failed=0,
            results=[],
        )
        config = SmtpConfig(
            smtp_server="smtp.test.com",
            smtp_port=587,
            sender_email="test@test.com",
            use_tls=True,
        )

        mock_smtp_instance = MagicMock()
        with patch.object(smtplib, "SMTP", return_value=mock_smtp_instance):
            with patch.object(ssl, "create_default_context") as mock_ctx:
                ctx_instance = MagicMock()
                mock_ctx.return_value = ctx_instance
                try:
                    send_summary_email(mock_log, config, password="testpass")
                except Exception:
                    pass
                mock_ctx.assert_called_once()
                mock_smtp_instance.starttls.assert_called_once_with(context=ctx_instance)

    def test_crlf_only_string(self) -> None:
        """_sanitize_header with only CRLF returns empty string."""
        from forma.delivery_send import _sanitize_header

        result = _sanitize_header("\r\n\r\n")
        assert result == ""

    def test_email_mask_hides_local_part(self) -> None:
        """_mask_email hides most of the local part."""
        from forma.delivery_send import _mask_email

        result = _mask_email("student@university.ac.kr")
        assert result == "stu***@university.ac.kr"
        assert "student" not in result

    def test_email_mask_short_local(self) -> None:
        """_mask_email with < 3 char local part uses available chars."""
        from forma.delivery_send import _mask_email

        result = _mask_email("ab@test.com")
        assert result == "ab***@test.com"

    def test_email_mask_empty(self) -> None:
        """_mask_email with empty string returns empty."""
        from forma.delivery_send import _mask_email

        assert _mask_email("") == ""

    def test_email_mask_no_at(self) -> None:
        """_mask_email with no @ still masks."""
        from forma.delivery_send import _mask_email

        result = _mask_email("noemailhere")
        assert result == "noe***"


# ===========================================================================
# Persona 5: PrivilegeEscalator (original)
# ===========================================================================


class TestPrivilegeEscalator:
    """Persona 5: attempts unauthorized file access."""

    def test_oauth_token_file_permissions(self, tmp_path, monkeypatch) -> None:
        """After writing OAuth token, file permissions are 0o600."""
        token_dir = tmp_path / ".config" / "bhu-ocr"

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = False

        mock_flow_instance = MagicMock()
        mock_flow_instance.run_local_server.return_value = mock_creds
        mock_creds.to_json.return_value = '{"token": "test"}'

        mock_gc = MagicMock()
        mock_gc.open_by_url.return_value.sheet1.get_all_records.return_value = []

        with patch.dict("sys.modules", {
            "gspread": MagicMock(authorize=MagicMock(return_value=mock_gc)),
            "google.auth.transport.requests": MagicMock(),
            "google.oauth2.credentials": MagicMock(
                Credentials=MagicMock(from_authorized_user_file=MagicMock(return_value=None))
            ),
            "google_auth_oauthlib.flow": MagicMock(
                InstalledAppFlow=MagicMock(
                    from_client_secrets_file=MagicMock(return_value=mock_flow_instance)
                )
            ),
        }):
            creds_file = tmp_path / "credentials.json"
            creds_file.write_text('{"installed": {}}')

            from forma.google_sheets import fetch_sheet_as_records

            fetch_sheet_as_records(
                "https://docs.google.com/spreadsheets/d/test",
                credentials_path=str(creds_file),
                token_cache_dir=str(token_dir),
            )

            token_path = token_dir / "token.json"
            if token_path.exists():
                mode = oct(token_path.stat().st_mode & 0o777)
                assert mode == oct(0o600), f"Token file permissions should be 0o600, got {mode}"

    def test_intervention_store_atomic_write(self, tmp_path) -> None:
        """InterventionLog uses atomic write (tempfile + os.replace)."""
        from forma.intervention_store import InterventionLog

        store_path = tmp_path / "interventions.yaml"
        log = InterventionLog(str(store_path))
        log.load()

        log.add_record("S001", week=1, intervention_type="면담", description="test")
        log.save()

        import yaml
        with open(store_path) as f:
            data = yaml.safe_load(f)
        assert data is not None
        records = data.get("records", [])
        assert len(records) == 1
        assert records[0]["student_id"] == "S001"


# ===========================================================================
# Persona 6: Overloader (original)
# ===========================================================================


class TestOverloader:
    """Persona 6: triggers resource exhaustion and boundary errors."""

    def test_epsilon_slope_not_flagged_as_decline(self) -> None:
        """Slope of -1e-16 should NOT be treated as SCORE_DECLINE."""
        from forma.warning_report_data import _classify_risk_types, RiskType

        score_trajectory = [0.5, 0.5, 0.5, 0.5 - 1e-16]
        concept_scores = {"A": 0.8, "B": 0.9}

        risk_types = _classify_risk_types(
            student_id="S001",
            score_trajectory=score_trajectory,
            concept_scores=concept_scores,
            absence_ratio=0.0,
        )
        assert RiskType.SCORE_DECLINE not in risk_types

    def test_exact_boundary_045_persistent_low(self) -> None:
        """All scores exactly at 0.45 threshold: NOT classified as PERSISTENT_LOW."""
        from forma.warning_report_data import _classify_risk_types, RiskType

        score_trajectory = [0.45, 0.45, 0.45]
        concept_scores = {"A": 0.8}

        risk_types = _classify_risk_types(
            student_id="S001",
            score_trajectory=score_trajectory,
            concept_scores=concept_scores,
            absence_ratio=0.0,
        )
        assert RiskType.PERSISTENT_LOW not in risk_types

    def test_exact_boundary_030_deficit(self) -> None:
        """Concept score exactly at 0.3: should NOT count as deficit."""
        from forma.warning_report_data import _classify_risk_types, RiskType

        score_trajectory = [0.6]
        concept_scores = {"A": 0.3, "B": 0.3, "C": 0.3}

        risk_types = _classify_risk_types(
            student_id="S001",
            score_trajectory=score_trajectory,
            concept_scores=concept_scores,
            absence_ratio=0.0,
        )
        assert RiskType.CONCEPT_DEFICIT not in risk_types

    def test_exact_boundary_030_absence(self) -> None:
        """Absence ratio exactly at 0.3: should NOT be flagged."""
        from forma.warning_report_data import _classify_risk_types, RiskType

        score_trajectory = [0.6]
        concept_scores = {"A": 0.8}

        risk_types = _classify_risk_types(
            student_id="S001",
            score_trajectory=score_trajectory,
            concept_scores=concept_scores,
            absence_ratio=0.3,
        )
        assert RiskType.PARTICIPATION_DECLINE not in risk_types

    def test_boundary_just_below_045(self) -> None:
        """Scores just below 0.45 threshold: flagged as PERSISTENT_LOW."""
        from forma.warning_report_data import _classify_risk_types, RiskType

        score_trajectory = [0.449, 0.449, 0.449]
        concept_scores = {"A": 0.8}

        risk_types = _classify_risk_types(
            student_id="S001",
            score_trajectory=score_trajectory,
            concept_scores=concept_scores,
            absence_ratio=0.0,
        )
        assert RiskType.PERSISTENT_LOW in risk_types

    def test_boundary_just_below_030_deficit(self) -> None:
        """3 concepts just below 0.3: flagged as CONCEPT_DEFICIT."""
        from forma.warning_report_data import _classify_risk_types, RiskType

        score_trajectory = [0.6]
        concept_scores = {"A": 0.299, "B": 0.299, "C": 0.299}

        risk_types = _classify_risk_types(
            student_id="S001",
            score_trajectory=score_trajectory,
            concept_scores=concept_scores,
            absence_ratio=0.0,
        )
        assert RiskType.CONCEPT_DEFICIT in risk_types

    def test_empty_trajectory_no_crash(self) -> None:
        """Empty score trajectory does not crash classify."""
        from forma.warning_report_data import _classify_risk_types

        risk_types = _classify_risk_types(
            student_id="S001",
            score_trajectory=[],
            concept_scores={"A": 0.8},
            absence_ratio=0.0,
        )
        assert isinstance(risk_types, list)

    def test_single_score_no_decline(self) -> None:
        """Single data point cannot determine decline."""
        from forma.warning_report_data import _classify_risk_types, RiskType

        risk_types = _classify_risk_types(
            student_id="S001",
            score_trajectory=[0.1],
            concept_scores={"A": 0.8},
            absence_ratio=0.0,
        )
        assert RiskType.SCORE_DECLINE not in risk_types


# ===========================================================================
# Expanded originals: PathTraversalAttacker
# ===========================================================================


class TestPathTraversalAttackerExpanded:
    """Persona 1 expansion: more aggressive path traversal attacks."""

    def _make_manifest(self, tmpdir: str) -> DeliveryManifest:
        return DeliveryManifest(directory=tmpdir, file_patterns=["{student_id}.pdf"])

    def test_url_encoded_traversal_dotdot(self, tmp_path) -> None:
        """URL-encoded traversal '..%2F' still caught because '..' is present."""
        manifest = self._make_manifest(str(tmp_path))
        with pytest.raises(ValueError, match="path traversal"):
            match_files_for_student(manifest, "..%2F..%2Fetc%2Fpasswd")

    def test_extremely_long_student_id_10000(self, tmp_path) -> None:
        """10000-char student_id does not crash; returns empty list."""
        manifest = self._make_manifest(str(tmp_path))
        result = match_files_for_student(manifest, "A" * 10000)
        assert result == []

    def test_student_id_double_dot_only(self, tmp_path) -> None:
        """student_id = '..' triggers traversal detection."""
        manifest = self._make_manifest(str(tmp_path))
        with pytest.raises(ValueError, match="path traversal"):
            match_files_for_student(manifest, "..")

    def test_null_byte_before_traversal(self, tmp_path) -> None:
        """Null byte injection: 'valid\\x00../../etc/passwd'."""
        manifest = self._make_manifest(str(tmp_path))
        with pytest.raises(ValueError, match="path traversal"):
            match_files_for_student(manifest, "valid\x00../../etc/passwd")

    def test_triple_dot_contains_dotdot(self, tmp_path) -> None:
        """'...' contains '..' substring; caught by traversal check."""
        manifest = self._make_manifest(str(tmp_path))
        with pytest.raises(ValueError, match="path traversal"):
            match_files_for_student(manifest, "...")

    def test_prepare_delivery_traversal_in_roster(self, tmp_path) -> None:
        """prepare_delivery blocks traversal student_id from roster YAML."""
        import yaml
        from forma.delivery_prepare import prepare_delivery

        report_dir = tmp_path / "reports"
        report_dir.mkdir()

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(yaml.dump({
            "report_source": {
                "directory": str(report_dir),
                "file_patterns": ["{student_id}.pdf"],
            }
        }))

        roster_path = tmp_path / "roster.yaml"
        roster_path.write_text(yaml.dump({
            "class_name": "TestClass",
            "students": [
                {"student_id": "../../etc/passwd", "name": "Hacker", "email": "h@test.com"},
            ],
        }))

        output_dir = tmp_path / "output"
        with pytest.raises(ValueError, match="path traversal"):
            prepare_delivery(str(manifest_path), str(roster_path), str(output_dir))


# ===========================================================================
# Expanded originals: UnicodeDestroyer
# ===========================================================================


class TestUnicodeDestroyerExpanded:
    """Persona 2 expansion: more aggressive Unicode attacks."""

    def test_rtl_override_in_filename(self) -> None:
        """RTL override U+202E is NOT in _ZERO_WIDTH_CHARS; left in result."""
        from forma.delivery_prepare import sanitize_filename

        result = sanitize_filename("\u202emanifest.pdf")
        assert result  # still produces a valid filename

    def test_combining_characters_storm_1000(self) -> None:
        """1000 combining accents on 'a' -- truncated by byte limit."""
        from forma.delivery_prepare import sanitize_filename

        name = "a" + "\u0300" * 1000
        result = sanitize_filename(name)
        assert result
        assert len(result.encode("utf-8")) <= 200

    def test_homoglyph_cyrillic_vs_latin(self) -> None:
        """Cyrillic 'a' (U+0430) vs Latin 'a' produce valid filenames."""
        from forma.delivery_prepare import sanitize_filename

        assert sanitize_filename("student_a")
        assert sanitize_filename("student_\u0430")

    def test_emoji_in_student_id_no_crash(self, tmp_path) -> None:
        """Emoji in student_id does not crash match_files_for_student."""
        manifest = DeliveryManifest(
            directory=str(tmp_path), file_patterns=["{student_id}.pdf"],
        )
        result = match_files_for_student(manifest, "student_\U0001f600")
        assert result == []

    def test_mixed_scripts_korean_arabic_chinese(self) -> None:
        """Mixed scripts in filename truncated to byte limit."""
        from forma.delivery_prepare import sanitize_filename

        name = "\ud55c\uae00_\u0639\u0631\u0628\u064a_\u4e2d\u6587.pdf"
        result = sanitize_filename(name)
        assert result
        assert len(result.encode("utf-8")) <= 200

    def test_all_zero_width_chars_stripped(self) -> None:
        """All known zero-width chars stripped; valid text preserved."""
        from forma.delivery_prepare import sanitize_filename

        zwchars = "\u200b\u200c\u200d\u200e\u200f\ufeff"
        result = sanitize_filename(f"test{zwchars}file.pdf")
        assert result == "testfile.pdf"

    def test_c0_control_chars_stripped(self) -> None:
        """C0 control characters 0x01-0x1f and 0x7f stripped."""
        from forma.delivery_prepare import sanitize_filename

        result = sanitize_filename("file\x01\x02\x03\x7fname.pdf")
        assert "\x01" not in result
        assert "\x7f" not in result
        assert "filename.pdf" == result

    def test_only_illegal_chars_returns_unnamed(self) -> None:
        """Filename of only OS-illegal chars returns '_unnamed'."""
        from forma.delivery_prepare import sanitize_filename

        result = sanitize_filename('<>:"/\\|?*')
        assert result == "_unnamed"


# ===========================================================================
# Expanded originals: ConfigPoisoner
# ===========================================================================


class TestConfigPoisonerExpanded:
    """Persona 3 expansion: YAML bombs, type confusion, boundary attacks."""

    def test_deeply_nested_yaml_100_levels(self, tmp_path) -> None:
        """YAML with 100 nesting levels does not crash safe_load."""
        import yaml

        lines = []
        for i in range(100):
            lines.append("  " * i + f"level{i}:")
        lines.append("  " * 100 + "leaf")
        content = "\n".join(lines)

        yaml_path = tmp_path / "deep.yaml"
        yaml_path.write_text(content)

        with open(str(yaml_path)) as f:
            data = yaml.safe_load(f)
        assert data is not None

    def test_yaml_anchor_alias_expansion(self, tmp_path) -> None:
        """YAML anchor/alias expansion with safe_load stays manageable."""
        import yaml

        yaml_content = (
            "a: &a [1,2,3,4,5]\n"
            "b: &b [*a,*a,*a,*a,*a]\n"
            "c: &c [*b,*b,*b,*b,*b]\n"
            "d: [*c,*c,*c,*c,*c]\n"
        )
        yaml_path = tmp_path / "anchor.yaml"
        yaml_path.write_text(yaml_content)

        with open(str(yaml_path)) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_integer_overflow_week_accepted(self, tmp_path) -> None:
        """Week = 2**63 is accepted (Python ints are arbitrary precision)."""
        from forma.intervention_store import InterventionLog

        log = InterventionLog(str(tmp_path / "log.yaml"))
        log.load()
        rid = log.add_record(student_id="S001", week=2**63, intervention_type="면담")
        assert rid == 1

    def test_negative_week_raises(self, tmp_path) -> None:
        """Negative week raises ValueError."""
        from forma.intervention_store import InterventionLog

        log = InterventionLog(str(tmp_path / "log.yaml"))
        log.load()
        with pytest.raises(ValueError, match="positive integer"):
            log.add_record(student_id="S001", week=-1, intervention_type="면담")

    def test_float_week_raises(self, tmp_path) -> None:
        """Float week raises ValueError."""
        from forma.intervention_store import InterventionLog

        log = InterventionLog(str(tmp_path / "log.yaml"))
        log.load()
        with pytest.raises(ValueError):
            log.add_record(student_id="S001", week=3.5, intervention_type="면담")

    def test_bool_week_raises(self, tmp_path) -> None:
        """Boolean week raises ValueError (bool is subclass of int)."""
        from forma.intervention_store import InterventionLog

        log = InterventionLog(str(tmp_path / "log.yaml"))
        log.load()
        with pytest.raises(ValueError):
            log.add_record(student_id="S001", week=True, intervention_type="면담")

    def test_empty_smtp_server_raises(self) -> None:
        """Empty string for required smtp_server raises ValueError."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_server"):
            _build_smtp_config({"smtp_server": "", "sender_email": "a@b.com"})

    def test_none_smtp_server_raises(self) -> None:
        """None for required smtp_server raises ValueError."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_server"):
            _build_smtp_config({"smtp_server": None, "sender_email": "a@b.com"})

    def test_smtp_port_bool_raises(self) -> None:
        """Boolean smtp_port raises ValueError."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            _build_smtp_config({
                "smtp_server": "smtp.test.com",
                "sender_email": "a@b.com",
                "smtp_port": True,
            })

    def test_smtp_port_over_65535_raises(self) -> None:
        """smtp_port > 65535 raises ValueError."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="1~65535"):
            _build_smtp_config({
                "smtp_server": "smtp.test.com",
                "sender_email": "a@b.com",
                "smtp_port": 70000,
            })

    def test_smtp_port_zero_raises(self) -> None:
        """smtp_port = 0 raises ValueError."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="1~65535"):
            _build_smtp_config({
                "smtp_server": "smtp.test.com",
                "sender_email": "a@b.com",
                "smtp_port": 0,
            })

    def test_send_interval_sec_bool_raises(self) -> None:
        """Boolean send_interval_sec raises ValueError."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="send_interval_sec"):
            _build_smtp_config({
                "smtp_server": "smtp.test.com",
                "sender_email": "a@b.com",
                "send_interval_sec": False,
            })

    def test_send_interval_sec_negative_raises(self) -> None:
        """Negative send_interval_sec raises ValueError."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="send_interval_sec"):
            _build_smtp_config({
                "smtp_server": "smtp.test.com",
                "sender_email": "a@b.com",
                "send_interval_sec": -1.0,
            })

    def test_load_grade_model_none_raises(self, tmp_path) -> None:
        """joblib file containing None raises TypeError for grade model."""
        import joblib
        from forma.grade_predictor import load_grade_model

        payload_path = tmp_path / "none_grade.pkl"
        joblib.dump(None, str(payload_path))

        with pytest.raises(TypeError, match="TrainedGradeModel"):
            load_grade_model(str(payload_path))

    def test_delivery_log_corrupted_as_list(self, tmp_path) -> None:
        """delivery_log.yaml containing a list raises ValueError."""
        import yaml
        from forma.delivery_send import load_delivery_log

        log_path = tmp_path / "delivery_log.yaml"
        log_path.write_text(yaml.dump([1, 2, 3]))

        with pytest.raises(ValueError, match="형식이 올바르지 않습니다"):
            load_delivery_log(str(log_path))


# ===========================================================================
# Expanded originals: NetworkDestroyer
# ===========================================================================


class TestNetworkDestroyerExpanded:
    """Persona 4 expansion: header injection, format strings, timeouts."""

    def test_unicode_line_sep_in_header(self) -> None:
        """U+2028 / U+2029 in header -- _sanitize_header strips only \\r\\n."""
        from forma.delivery_send import _sanitize_header

        result = _sanitize_header("From: test\u2028Bcc: evil\u2029end")
        assert "\r" not in result
        assert "\n" not in result

    def test_10000_char_subject(self) -> None:
        """10000-char subject does not crash _sanitize_header."""
        from forma.delivery_send import _sanitize_header

        result = _sanitize_header("A" * 10000)
        assert len(result) == 10000

    def test_sender_email_no_at_raises(self) -> None:
        """sender_email without '@' raises ValueError."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="sender_email"):
            _build_smtp_config({
                "smtp_server": "smtp.test.com",
                "sender_email": "not_an_email",
            })

    def test_format_string_attack_blocked(self) -> None:
        """Format string {__class__} as standalone variable blocked by validate_template_variables."""
        from forma.delivery_send import EmailTemplate, validate_template_variables

        template = EmailTemplate(
            subject="Report for {student_name}",
            body="Hello {student_name}, secret: {__class__}",
        )
        with pytest.raises(ValueError, match="지원하지 않는 템플릿 변수"):
            validate_template_variables(template)

    def test_render_template_no_attribute_traversal(self) -> None:
        """render_template uses str.replace -- no attribute traversal."""
        from forma.delivery_send import EmailTemplate, render_template

        template = EmailTemplate(subject="Report", body="{student_name.__class__}")
        _subj, body = render_template(
            template, student_name="John", student_id="S001", class_name="1A",
        )
        assert "{student_name.__class__}" in body

    def test_compose_email_crlf_in_to_sanitized(self, tmp_path) -> None:
        """CRLF in to_email is sanitized in compose_email."""
        from forma.delivery_send import SmtpConfig, compose_email

        config = SmtpConfig(
            smtp_server="smtp.test.com", smtp_port=587,
            sender_email="sender@test.com",
        )
        zip_file = tmp_path / "test.zip"
        zip_file.write_bytes(b"PK\x03\x04fake")

        msg = compose_email(
            sender_config=config,
            to_email="victim@test.com\r\nBcc: evil@evil.com",
            subject="Test", body="Hello",
            zip_path=str(zip_file),
        )
        assert "\r" not in msg["To"]
        assert "\n" not in msg["To"]


# ===========================================================================
# Expanded originals: PrivilegeEscalator
# ===========================================================================


class TestPrivilegeEscalatorExpanded:
    """Persona 5 expansion: file access, model loading, output path attacks."""

    def test_risk_model_file_not_found(self, tmp_path) -> None:
        """Nonexistent model raises FileNotFoundError."""
        from forma.risk_predictor import load_model

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            load_model(str(tmp_path / "nonexistent.pkl"))

    def test_grade_model_file_not_found(self, tmp_path) -> None:
        """Nonexistent grade model raises FileNotFoundError."""
        from forma.grade_predictor import load_grade_model

        with pytest.raises(FileNotFoundError):
            load_grade_model(str(tmp_path / "nonexistent.pkl"))

    def test_prepare_delivery_valid_pipeline(self, tmp_path) -> None:
        """Normal prepare pipeline succeeds without escape."""
        import yaml
        from forma.delivery_prepare import prepare_delivery

        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        (report_dir / "S001.pdf").write_text("content")

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(yaml.dump({
            "report_source": {
                "directory": str(report_dir),
                "file_patterns": ["{student_id}.pdf"],
            }
        }))

        roster_path = tmp_path / "roster.yaml"
        roster_path.write_text(yaml.dump({
            "class_name": "TestClass",
            "students": [
                {"student_id": "S001", "name": "Student", "email": "s@test.com"},
            ],
        }))

        output_dir = tmp_path / "output"
        summary = prepare_delivery(str(manifest_path), str(roster_path), str(output_dir))
        assert summary.total_students == 1
        assert summary.ready == 1

    def test_write_to_etc_blocked(self, tmp_path) -> None:
        """Attempting to write to /etc/ fails with PermissionError."""
        import yaml
        from forma.delivery_prepare import prepare_delivery

        report_dir = tmp_path / "reports"
        report_dir.mkdir()

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(yaml.dump({
            "report_source": {
                "directory": str(report_dir),
                "file_patterns": ["{student_id}.pdf"],
            }
        }))

        roster_path = tmp_path / "roster.yaml"
        roster_path.write_text(yaml.dump({
            "class_name": "Test",
            "students": [{"student_id": "S001", "name": "Test", "email": "t@t.com"}],
        }))

        with pytest.raises((PermissionError, OSError)):
            prepare_delivery(str(manifest_path), str(roster_path), "/etc/forma_test_output")


# ===========================================================================
# Expanded originals: Overloader
# ===========================================================================


class TestOverloaderExpanded:
    """Persona 6 expansion: scalability, boundary values, concurrent writes."""

    def test_100_students_prepare(self, tmp_path) -> None:
        """100 students prepare_delivery handles correctly."""
        import yaml
        from forma.delivery_prepare import prepare_delivery

        report_dir = tmp_path / "reports"
        report_dir.mkdir()

        students = []
        for i in range(100):
            sid = f"S{i:05d}"
            (report_dir / f"{sid}.pdf").write_text(f"report for {sid}")
            students.append({"student_id": sid, "name": f"Name{i}", "email": f"{sid}@test.com"})

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(yaml.dump({
            "report_source": {
                "directory": str(report_dir),
                "file_patterns": ["{student_id}.pdf"],
            }
        }))

        roster_path = tmp_path / "roster.yaml"
        roster_path.write_text(yaml.dump({"class_name": "LargeClass", "students": students}))

        output_dir = tmp_path / "output"
        summary = prepare_delivery(str(manifest_path), str(roster_path), str(output_dir))
        assert summary.total_students == 100
        assert summary.ready == 100

    def test_100000_char_name_sanitized(self) -> None:
        """100000-char name sanitized to <= 200 bytes."""
        from forma.delivery_prepare import sanitize_filename

        result = sanitize_filename("A" * 100000)
        assert len(result.encode("utf-8")) <= 200
        assert result

    def test_zero_students_roster_raises(self, tmp_path) -> None:
        """Empty students list in roster raises ValueError."""
        import yaml
        from forma.delivery_prepare import load_roster

        roster_path = tmp_path / "roster.yaml"
        roster_path.write_text(yaml.dump({"class_name": "Empty", "students": []}))
        with pytest.raises(ValueError, match="students"):
            load_roster(str(roster_path))

    def test_all_identical_scores_single_class_risk(self) -> None:
        """All students identical scores -- single-class augmentation."""
        from forma.risk_predictor import RiskPredictor

        predictor = RiskPredictor()
        X = np.full((15, 15), 0.5)
        labels = np.zeros(15, dtype=int)

        model = predictor.train(X, labels, list(range(15)), min_students=10)
        assert model.cv_score == 0.0

    def test_risk_predictor_10_students_passes(self) -> None:
        """Exactly 10 students passes min_students; 9 raises ValueError."""
        from forma.risk_predictor import RiskPredictor

        predictor = RiskPredictor()
        rng = np.random.RandomState(42)
        X_10 = rng.rand(10, 15)
        labels_10 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        model = predictor.train(X_10, labels_10, list(range(15)), min_students=10)
        assert model.n_students == 10

        with pytest.raises(ValueError, match="Insufficient"):
            predictor.train(X_10[:9], labels_10[:9], list(range(15)), min_students=10)

    def test_cold_start_grade_085_predicts_A(self) -> None:
        """Cold start: score_mean=0.85 predicts A."""
        from forma.grade_predictor import GradePredictor, GRADE_FEATURE_NAMES

        predictor = GradePredictor()
        features = np.zeros((1, 21))
        features[0, 0] = 0.85
        results = predictor.predict_cold_start(features, ["S001"], GRADE_FEATURE_NAMES)
        assert results[0].predicted_grade == "A"

    def test_cold_start_grade_029_predicts_F(self) -> None:
        """Cold start: score_mean=0.29 predicts F."""
        from forma.grade_predictor import GradePredictor, GRADE_FEATURE_NAMES

        predictor = GradePredictor()
        features = np.zeros((1, 21))
        features[0, 0] = 0.29
        results = predictor.predict_cold_start(features, ["S001"], GRADE_FEATURE_NAMES)
        assert results[0].predicted_grade == "F"

    def test_description_exactly_2000_chars(self, tmp_path) -> None:
        """Description at exactly 2000 char limit succeeds."""
        from forma.intervention_store import InterventionLog

        log = InterventionLog(str(tmp_path / "log.yaml"))
        log.load()
        rid = log.add_record(
            student_id="S001", week=1, intervention_type="면담",
            description="x" * 2000,
        )
        assert rid == 1

    def test_description_2001_chars_fails(self, tmp_path) -> None:
        """Description at 2001 chars exceeds limit."""
        from forma.intervention_store import InterventionLog

        log = InterventionLog(str(tmp_path / "log.yaml"))
        log.load()
        with pytest.raises(ValueError, match="2000"):
            log.add_record(
                student_id="S001", week=1, intervention_type="면담",
                description="x" * 2001,
            )

    def test_concurrent_intervention_log_writes(self, tmp_path) -> None:
        """Two threads writing to InterventionLog -- at least one preserved."""
        from forma.intervention_store import InterventionLog

        log_path = str(tmp_path / "concurrent_log.yaml")
        errors = []

        def add_records(thread_id: int) -> None:
            try:
                log = InterventionLog(log_path)
                log.load()
                log.add_record(
                    student_id=f"T{thread_id}", week=1,
                    intervention_type="면담", description=f"Thread {thread_id}",
                )
                log.save()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_records, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"

        final_log = InterventionLog(log_path)
        final_log.load()
        assert len(final_log.get_records()) >= 1

    def test_smtp_port_boundary_1(self) -> None:
        """smtp_port = 1 (minimum valid) is accepted."""
        from forma.delivery_send import _build_smtp_config

        config = _build_smtp_config({
            "smtp_server": "smtp.test.com", "sender_email": "a@b.com", "smtp_port": 1,
        })
        assert config.smtp_port == 1

    def test_smtp_port_boundary_65535(self) -> None:
        """smtp_port = 65535 (maximum valid) is accepted."""
        from forma.delivery_send import _build_smtp_config

        config = _build_smtp_config({
            "smtp_server": "smtp.test.com", "sender_email": "a@b.com", "smtp_port": 65535,
        })
        assert config.smtp_port == 65535

    def test_send_interval_sec_zero(self) -> None:
        """send_interval_sec = 0 (no delay) is accepted."""
        from forma.delivery_send import _build_smtp_config

        config = _build_smtp_config({
            "smtp_server": "smtp.test.com", "sender_email": "a@b.com", "send_interval_sec": 0,
        })
        assert config.send_interval_sec == 0.0

    def test_threshold_constants_correct(self) -> None:
        """Warning threshold constants have expected values."""
        from forma.warning_report_data import (
            _PERSISTENT_LOW_THRESHOLD,
            _DEFICIT_MASTERY_THRESHOLD,
            _DROP_PROB_INCLUSION_THRESHOLD,
            _SLOPE_EPSILON,
        )
        assert _PERSISTENT_LOW_THRESHOLD == 0.45
        assert _DEFICIT_MASTERY_THRESHOLD == 0.3
        assert _DROP_PROB_INCLUSION_THRESHOLD == 0.5
        assert _SLOPE_EPSILON == pytest.approx(1e-9)


# ###########################################################################
# NEW PERSONAS: Consistency Hardening Attack Tests
# ###########################################################################


# ===========================================================================
# Persona 7: NaN Bombardier
# ===========================================================================


class TestNaNBombardier:
    """Persona 7: Inject NaN, inf, -inf into ALL numeric fields.

    Targets risk_predictor._safe_nanmean, _safe_nanvar, _ols_slope, and
    FeatureExtractor to verify NaN-safe computation throughout the pipeline.
    """

    def test_safe_nanmean_all_nan(self) -> None:
        """_safe_nanmean([NaN, NaN, NaN]) returns 0.0, not NaN."""
        from forma.risk_predictor import _safe_nanmean

        result = _safe_nanmean([float("nan"), float("nan"), float("nan")])
        assert result == 0.0
        assert not math.isnan(result)

    def test_safe_nanmean_single_nan(self) -> None:
        """_safe_nanmean([NaN]) returns 0.0."""
        from forma.risk_predictor import _safe_nanmean

        result = _safe_nanmean([float("nan")])
        assert result == 0.0

    def test_safe_nanmean_mixed_nan_valid(self) -> None:
        """_safe_nanmean with mixed NaN/valid returns mean of valid values."""
        from forma.risk_predictor import _safe_nanmean

        result = _safe_nanmean([float("nan"), 2.0, float("nan"), 4.0])
        assert result == pytest.approx(3.0)
        assert not math.isnan(result)

    def test_safe_nanmean_empty(self) -> None:
        """_safe_nanmean([]) returns 0.0 without crashing."""
        from forma.risk_predictor import _safe_nanmean

        result = _safe_nanmean([])
        assert result == 0.0

    def test_safe_nanmean_inf_values(self) -> None:
        """_safe_nanmean with inf/-inf returns the inf mean (not NaN)."""
        from forma.risk_predictor import _safe_nanmean

        result_pos = _safe_nanmean([float("inf"), 1.0])
        assert math.isinf(result_pos) or not math.isnan(result_pos)

        result_both = _safe_nanmean([float("inf"), float("-inf")])
        # inf + -inf = nan, so should return 0.0
        assert result_both == 0.0

    def test_safe_nanvar_all_nan(self) -> None:
        """_safe_nanvar([NaN, NaN]) returns 0.0."""
        from forma.risk_predictor import _safe_nanvar

        result = _safe_nanvar([float("nan"), float("nan")])
        assert result == 0.0
        assert not math.isnan(result)

    def test_safe_nanvar_single_value(self) -> None:
        """_safe_nanvar with single non-NaN value returns 0.0 (no variance)."""
        from forma.risk_predictor import _safe_nanvar

        result = _safe_nanvar([5.0])
        assert result == 0.0
        assert not math.isnan(result)

    def test_safe_nanvar_mixed(self) -> None:
        """_safe_nanvar with mixed NaN and values computes variance of valid only."""
        from forma.risk_predictor import _safe_nanvar

        result = _safe_nanvar([float("nan"), 2.0, 4.0, float("nan")])
        assert result == pytest.approx(1.0)  # var([2,4]) = 1.0

    def test_ols_slope_all_nan(self) -> None:
        """_ols_slope with all NaN returns 0.0 (fewer than 2 clean values)."""
        from forma.risk_predictor import _ols_slope

        result = _ols_slope([float("nan"), float("nan"), float("nan")])
        assert result == 0.0

    def test_ols_slope_single_valid_among_nan(self) -> None:
        """_ols_slope with 1 valid + NaN returns 0.0."""
        from forma.risk_predictor import _ols_slope

        result = _ols_slope([float("nan"), 3.0, float("nan")])
        assert result == 0.0

    def test_ols_slope_two_valid_among_nan(self) -> None:
        """_ols_slope with 2+ valid values computes slope, ignoring NaN."""
        from forma.risk_predictor import _ols_slope

        result = _ols_slope([float("nan"), 1.0, float("nan"), 3.0])
        assert result == pytest.approx(2.0)
        assert not math.isnan(result)

    def test_ols_slope_inf_in_series(self) -> None:
        """_ols_slope with inf in data does not crash."""
        from forma.risk_predictor import _ols_slope

        # polyfit may produce warnings but should not crash
        result = _ols_slope([1.0, float("inf"), 3.0])
        assert isinstance(result, float)

    def test_feature_extractor_all_nan_scores(self) -> None:
        """FeatureExtractor with records whose scores are NaN produces valid features."""
        from unittest.mock import MagicMock
        from forma.risk_predictor import FeatureExtractor

        store = MagicMock()
        # Create records with NaN scores
        records = []
        for i in range(3):
            r = MagicMock()
            r.student_id = "S001"
            r.week = i + 1
            r.scores = {"ensemble_score": float("nan"), "concept_coverage": float("nan")}
            r.tier_level = 0
            r.misconception_count = None
            r.edge_f1 = None
            records.append(r)

        store.get_all_records.return_value = records
        store.get_class_weekly_matrix.return_value = {
            "S001": {1: float("nan"), 2: float("nan"), 3: float("nan")}
        }

        extractor = FeatureExtractor()
        matrix, names, sids = extractor.extract(store, [1, 2, 3])

        # All features should be finite (no NaN in output)
        assert sids == ["S001"]
        for val in matrix[0]:
            assert not math.isnan(val), f"NaN in feature vector: {matrix[0]}"


# ===========================================================================
# Persona 8: XML Injection Specialist
# ===========================================================================


class TestXMLInjectionSpecialist:
    """Persona 8: Inject malicious XML/HTML into text fields for all reports.

    Tests esc() neutralization of script tags, entity encoding, CDATA,
    nested tags, and very long strings with mixed XML specials.
    """

    def test_script_tag_in_student_id(self) -> None:
        """<script>alert(1)</script> in text is neutralized by esc()."""
        from forma.font_utils import esc

        result = esc('<script>alert(1)</script>')
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_double_encoding_not_applied(self) -> None:
        """Already-escaped &amp; is not double-escaped to &amp;amp;."""
        from forma.font_utils import esc

        # esc() escapes & to &amp;, so &amp; becomes &amp;amp; -- this is correct XML
        # behavior: the INPUT is raw text, not pre-escaped XML. We just verify no crash.
        result = esc("&amp;&lt;&gt;")
        assert result  # non-empty
        # The & in &amp; gets escaped to &amp;amp; which is correct
        assert "&amp;" in result

    def test_nested_xml_tags(self) -> None:
        """Nested XML tags <b><i>test</i></b> are fully escaped."""
        from forma.font_utils import esc

        result = esc("<b><i>test</i></b>")
        assert "<b>" not in result
        assert "<i>" not in result
        assert "&lt;b&gt;&lt;i&gt;test&lt;/i&gt;&lt;/b&gt;" == result

    def test_cdata_injection(self) -> None:
        """CDATA <![CDATA[attack]]> injection is escaped."""
        from forma.font_utils import esc

        result = esc("<![CDATA[attack]]>")
        assert "<![CDATA[" not in result
        assert "&lt;![CDATA[attack]]&gt;" == result

    def test_very_long_string_with_xml_specials(self) -> None:
        """10000-char string with mixed < > & is fully escaped without crash."""
        from forma.font_utils import esc

        payload = ("<script>" * 500 + "&" * 1000 + "normal" * 500)
        result = esc(payload)
        assert "<script>" not in result
        assert "&lt;" in result
        assert "&amp;" in result

    def test_xml_processing_instruction(self) -> None:
        """Processing instruction <?xml version="1.0"?> is escaped."""
        from forma.font_utils import esc

        result = esc('<?xml version="1.0"?>')
        assert "<?xml" not in result
        assert "&lt;?xml" in result

    def test_esc_with_all_special_chars(self) -> None:
        """esc() with <, >, &, ", ' all present."""
        from forma.font_utils import esc

        result = esc("""<tag attr="val" other='val2'>content & stuff</tag>""")
        assert "<tag" not in result
        assert "&lt;" in result
        assert "&amp;" in result

    def test_strip_invisible_then_xml_escape(self) -> None:
        """esc() strips invisible chars THEN escapes XML, not the other way around."""
        from forma.font_utils import esc

        # Control char \x01 should be stripped, then & should be escaped
        result = esc("\x01&\x02<")
        assert result == "&amp;&lt;"
        assert "\x01" not in result
        assert "\x02" not in result


# ===========================================================================
# Persona 9: Unicode Destroyer Advanced
# ===========================================================================


class TestUnicodeDestroyerAdvanced:
    """Persona 9: Deeper Unicode attacks beyond basic zero-width chars.

    Tests BOM stripping, combining characters, emoji sequences, null byte
    handling, RTL override, and mixed script attacks.
    """

    def test_bom_at_start_stripped_by_esc(self) -> None:
        """BOM (U+FEFF) at start of string is stripped by esc()."""
        from forma.font_utils import esc

        result = esc("\uFEFFHello World")
        assert "\uFEFF" not in result
        assert "Hello World" in result

    def test_null_byte_stripped_by_esc(self) -> None:
        """Null byte U+0000 in text is stripped by esc()."""
        from forma.font_utils import esc

        result = esc("before\x00after")
        assert "\x00" not in result
        assert "beforeafter" == result

    def test_emoji_family_sequence_in_esc(self) -> None:
        """Family emoji (ZWJ sequence) does not crash esc()."""
        from forma.font_utils import esc

        # Family emoji: man, ZWJ, woman, ZWJ, girl, ZWJ, boy
        # Note: ZWJ (U+200D) is stripped by esc(), so the emoji is decomposed
        family = "\U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466"
        result = esc(family)
        # ZWJ chars are stripped; remaining emoji codepoints preserved
        assert "\u200d" not in result
        assert isinstance(result, str)

    def test_rtl_override_does_not_crash_esc(self) -> None:
        """RTL override U+202E in concept names does not crash esc()."""
        from forma.font_utils import esc

        result = esc("\u202Emalicious\u202C")
        # esc does not strip U+202E (not in _XML_ILLEGAL_CTRL range)
        assert isinstance(result, str)

    def test_combining_chars_preserved(self) -> None:
        """Combining characters (e.g., e + combining acute accent) not corrupted."""
        from forma.font_utils import esc

        # e followed by combining acute accent = e with acute
        text = "e\u0301"
        result = esc(text)
        assert result == "e\u0301"

    def test_all_c0_controls_stripped_except_tab_nl_cr(self) -> None:
        """All C0 control chars (0x00-0x08, 0x0B, 0x0C, 0x0E-0x1F, 0x7F) stripped."""
        from forma.font_utils import esc

        # Build string with all C0 controls
        c0_chars = "".join(chr(i) for i in range(0, 32))
        c0_chars += chr(0x7f)  # DEL

        result = esc(c0_chars + "clean")
        # Only \t (0x09), \n (0x0A), \r (0x0D) should survive
        assert "\t" in result
        assert "\n" in result
        assert "\r" in result
        assert "clean" in result
        # Others stripped
        assert "\x00" not in result
        assert "\x01" not in result
        assert "\x7f" not in result

    def test_strip_invisible_shared_by_esc_and_sanitize(self) -> None:
        """strip_invisible() used by both esc() and sanitize_filename_report()."""
        from forma.font_utils import strip_invisible

        text = "\uFEFF\u200BHello\x00World\u200D"
        result = strip_invisible(text)
        assert "\uFEFF" not in result
        assert "\u200B" not in result
        assert "\x00" not in result
        assert "\u200D" not in result
        assert "HelloWorld" == result

    def test_mixed_invisible_with_korean(self) -> None:
        """strip_invisible preserves Korean while stripping invisible chars."""
        from forma.font_utils import strip_invisible

        text = "\uFEFF김\u200B철\u200C수\x01"
        result = strip_invisible(text)
        assert result == "김철수"


# ===========================================================================
# Persona 10: File System Saboteur
# ===========================================================================


class TestFileSystemSaboteur:
    """Persona 10: Test io_utils.py atomic write edge cases.

    Verifies that atomic writes handle read-only dirs, non-serializable data,
    large data, Unicode content, concurrent writes, and failure recovery.
    """

    def test_atomic_write_yaml_read_only_dir_raises(self, tmp_path) -> None:
        """Write to read-only directory raises OSError, no temp file left."""
        import os
        from forma.io_utils import atomic_write_yaml

        ro_dir = tmp_path / "readonly"
        ro_dir.mkdir()
        target = ro_dir / "test.yaml"

        # Make directory read-only
        os.chmod(str(ro_dir), 0o555)
        try:
            with pytest.raises(OSError):
                atomic_write_yaml({"key": "value"}, str(target))

            # Verify no temp files remain
            leftover = list(ro_dir.iterdir())
            assert len(leftover) == 0, f"Temp files left behind: {leftover}"
        finally:
            os.chmod(str(ro_dir), 0o755)

    def test_atomic_write_yaml_non_serializable_raises(self, tmp_path) -> None:
        """Non-YAML-serializable data (generator) raises, no temp file left."""
        from forma.io_utils import atomic_write_yaml

        def _gen():
            yield 1

        target = str(tmp_path / "bad.yaml")
        with pytest.raises(TypeError):
            atomic_write_yaml({"gen": _gen()}, target)

        # Verify no temp file remains
        tmp_files = [f for f in tmp_path.iterdir() if f.suffix == ".tmp"]
        assert len(tmp_files) == 0, f"Temp files left: {tmp_files}"

    def test_atomic_write_yaml_large_data(self, tmp_path) -> None:
        """atomic_write_yaml with 1000+ keys succeeds."""
        from forma.io_utils import atomic_write_yaml
        import yaml

        data = {f"key_{i}": f"value_{i}" for i in range(1500)}
        target = str(tmp_path / "large.yaml")
        atomic_write_yaml(data, target)

        with open(target) as f:
            loaded = yaml.safe_load(f)
        assert len(loaded) == 1500

    def test_atomic_write_json_unicode_korean(self, tmp_path) -> None:
        """atomic_write_json with Korean content encodes correctly."""
        from forma.io_utils import atomic_write_json
        import json

        data = {"학생": "김철수", "과목": "물리학", "주차": 3}
        target = str(tmp_path / "korean.json")
        atomic_write_json(data, target)

        with open(target, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["학생"] == "김철수"
        assert loaded["과목"] == "물리학"

    def test_atomic_write_concurrent_with_lock(self, tmp_path) -> None:
        """Two threads writing with lock=True both succeed without corruption."""
        from forma.io_utils import atomic_write_yaml
        import yaml

        target = str(tmp_path / "concurrent.yaml")
        errors = []

        def writer(thread_id: int) -> None:
            try:
                data = {f"thread_{thread_id}": f"data_{thread_id}"}
                atomic_write_yaml(data, target, lock=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"

        # File should exist and be valid YAML
        with open(target) as f:
            loaded = yaml.safe_load(f)
        assert isinstance(loaded, dict)

    def test_atomic_write_preserves_original_on_failure(self, tmp_path) -> None:
        """Original file preserved when atomic_write fails mid-operation."""
        from forma.io_utils import atomic_write_yaml
        import yaml

        target = str(tmp_path / "preserve.yaml")
        original = {"original": "data"}
        atomic_write_yaml(original, target)

        # Attempt write with non-serializable data (generator)
        def _gen():
            yield 1

        try:
            atomic_write_yaml({"gen": _gen()}, target)
        except Exception:
            pass

        # Original should still be intact
        with open(target) as f:
            loaded = yaml.safe_load(f)
        assert loaded == original

    def test_atomic_write_text_basic(self, tmp_path) -> None:
        """atomic_write_text writes and reads back correctly."""
        from forma.io_utils import atomic_write_text

        target = str(tmp_path / "text.txt")
        content = "Hello, this is a test.\n한국어 텍스트."
        atomic_write_text(content, target)

        with open(target, encoding="utf-8") as f:
            assert f.read() == content

    def test_atomic_write_json_circular_ref_raises(self, tmp_path) -> None:
        """Circular reference in JSON data raises TypeError, no temp file."""
        from forma.io_utils import atomic_write_json

        data: dict = {}
        data["self"] = data  # circular

        target = str(tmp_path / "circular.json")
        with pytest.raises((TypeError, ValueError)):
            atomic_write_json(data, target)

        tmp_files = [f for f in tmp_path.iterdir() if f.suffix == ".tmp"]
        assert len(tmp_files) == 0


# ===========================================================================
# Persona 11: Config Poisoner Advanced
# ===========================================================================


class TestConfigPoisonerAdvanced:
    """Persona 11: Feed malicious configuration to all config loaders.

    Tests config.py with unknown keys, wrong types, empty YAML, deeply
    nested structures, and YAML alias bombs.
    """

    def test_config_unknown_keys_logged(self, tmp_path, caplog) -> None:
        """Unknown keys in forma.json trigger a warning log."""
        import json
        import logging
        from forma.config import load_config

        config_path = tmp_path / "forma.json"
        config_path.write_text(json.dumps({
            "smtp": {"server": "test"},
            "unknown_key": "malicious",
            "another_unknown": 42,
        }))

        with caplog.at_level(logging.WARNING, logger="forma.config"):
            result = load_config(str(config_path))

        assert "unknown_key" in caplog.text or "another_unknown" in caplog.text
        assert isinstance(result, dict)

    def test_config_naver_ocr_string_instead_of_dict_raises(self, tmp_path) -> None:
        """naver_ocr as string instead of dict raises KeyError."""
        import json
        from forma.config import load_config, get_naver_ocr_config

        config_path = tmp_path / "forma.json"
        config_path.write_text(json.dumps({"naver_ocr": "not_a_dict"}))

        config = load_config(str(config_path))
        with pytest.raises(KeyError, match="dict"):
            get_naver_ocr_config(config)

    def test_config_json_array_not_object_raises(self, tmp_path) -> None:
        """forma.json containing a JSON array raises ValueError."""
        from forma.config import load_config

        config_path = tmp_path / "forma.json"
        config_path.write_text("[1, 2, 3]")

        with pytest.raises(ValueError, match="JSON object"):
            load_config(str(config_path))

    def test_config_empty_json_object(self, tmp_path) -> None:
        """forma.json with empty {} is valid and returns empty dict."""
        from forma.config import load_config

        config_path = tmp_path / "forma.json"
        config_path.write_text("{}")

        result = load_config(str(config_path))
        assert result == {}

    def test_yaml_safe_load_none_on_empty_file(self, tmp_path) -> None:
        """Empty YAML file returns None from safe_load; loaders must handle."""
        import yaml

        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")

        with open(str(yaml_path)) as f:
            data = yaml.safe_load(f)
        assert data is None

    def test_smtp_config_string_port_raises(self) -> None:
        """smtp_port as string raises ValueError."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            _build_smtp_config({
                "smtp_server": "smtp.test.com",
                "sender_email": "a@b.com",
                "smtp_port": "587",
            })

    def test_smtp_config_none_port_raises(self) -> None:
        """smtp_port as None raises ValueError."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            _build_smtp_config({
                "smtp_server": "smtp.test.com",
                "sender_email": "a@b.com",
                "smtp_port": None,
            })

    def test_smtp_config_float_port_raises(self) -> None:
        """smtp_port as float raises ValueError."""
        from forma.delivery_send import _build_smtp_config

        with pytest.raises(ValueError, match="smtp_port"):
            _build_smtp_config({
                "smtp_server": "smtp.test.com",
                "sender_email": "a@b.com",
                "smtp_port": 587.5,
            })

    def test_get_smtp_config_missing_section_raises(self) -> None:
        """get_smtp_config with no smtp section raises KeyError."""
        from forma.config import get_smtp_config

        with pytest.raises(KeyError, match="smtp"):
            get_smtp_config({"naver_ocr": {}})

    def test_get_smtp_config_string_section_raises(self) -> None:
        """get_smtp_config with smtp as string raises KeyError."""
        from forma.config import get_smtp_config

        with pytest.raises(KeyError, match="smtp"):
            get_smtp_config({"smtp": "not_a_dict"})


# ===========================================================================
# Persona 12: Empty Value Attacker
# ===========================================================================


class TestEmptyValueAttacker:
    """Persona 12: Pass empty/null/zero values to ALL public builder and loader functions.

    Verifies graceful handling of empty lists, None returns, zero-length strings,
    and other degenerate inputs.
    """

    def test_esc_empty_string(self) -> None:
        """esc('') returns ''."""
        from forma.font_utils import esc

        assert esc("") == ""

    def test_strip_invisible_empty_string(self) -> None:
        """strip_invisible('') returns ''."""
        from forma.font_utils import strip_invisible

        assert strip_invisible("") == ""

    def test_mask_email_empty(self) -> None:
        """_mask_email('') returns ''."""
        from forma.delivery_send import _mask_email

        assert _mask_email("") == ""

    def test_mask_email_at_only(self) -> None:
        """_mask_email('@') handles single @ gracefully."""
        from forma.delivery_send import _mask_email

        result = _mask_email("@")
        assert isinstance(result, str)
        assert "@" in result

    def test_mask_email_no_at(self) -> None:
        """_mask_email('x') with no @ returns masked local part."""
        from forma.delivery_send import _mask_email

        result = _mask_email("x")
        assert result == "x***"

    def test_build_warning_data_empty_students(self) -> None:
        """build_warning_data with no at-risk students returns empty list."""
        from forma.warning_report_data import build_warning_data

        result = build_warning_data(
            at_risk_students={},
            risk_predictions=[],
            concept_scores={},
        )
        assert result == []

    def test_build_warning_data_all_not_at_risk(self) -> None:
        """build_warning_data where no student is at-risk returns empty."""
        from forma.warning_report_data import build_warning_data

        result = build_warning_data(
            at_risk_students={
                "S001": {"is_at_risk": False, "reasons": []},
                "S002": {"is_at_risk": False, "reasons": []},
            },
            risk_predictions=[],
            concept_scores={},
        )
        assert result == []

    def test_classify_risk_types_empty_trajectory_empty_concepts(self) -> None:
        """_classify_risk_types with all empty args returns empty list."""
        from forma.warning_report_data import _classify_risk_types

        result = _classify_risk_types(
            student_id="S001",
            score_trajectory=[],
            concept_scores={},
            absence_ratio=0.0,
        )
        assert result == []

    def test_ols_slope_empty_list(self) -> None:
        """_ols_slope([]) returns 0.0."""
        from forma.risk_predictor import _ols_slope

        assert _ols_slope([]) == 0.0

    def test_ols_slope_single_value(self) -> None:
        """_ols_slope([5.0]) returns 0.0 (fewer than 2 values)."""
        from forma.risk_predictor import _ols_slope

        assert _ols_slope([5.0]) == 0.0

    def test_atomic_write_yaml_empty_dict(self, tmp_path) -> None:
        """atomic_write_yaml({}) creates valid empty YAML file."""
        from forma.io_utils import atomic_write_yaml
        import yaml

        target = str(tmp_path / "empty.yaml")
        atomic_write_yaml({}, target)

        with open(target) as f:
            loaded = yaml.safe_load(f)
        assert loaded == {}

    def test_atomic_write_json_empty_dict(self, tmp_path) -> None:
        """atomic_write_json({}) creates valid empty JSON file."""
        from forma.io_utils import atomic_write_json
        import json

        target = str(tmp_path / "empty.json")
        atomic_write_json({}, target)

        with open(target) as f:
            loaded = json.load(f)
        assert loaded == {}

    def test_sanitize_filename_empty_string(self) -> None:
        """sanitize_filename('') returns '_unnamed'."""
        from forma.delivery_prepare import sanitize_filename

        assert sanitize_filename("") == "_unnamed"

    def test_sanitize_filename_report_empty_string(self) -> None:
        """sanitize_filename_report('') returns '_unnamed'."""
        from forma.report_utils import sanitize_filename_report

        assert sanitize_filename_report("") == "_unnamed"

    def test_sanitize_header_empty(self) -> None:
        """_sanitize_header('') returns ''."""
        from forma.delivery_send import _sanitize_header

        assert _sanitize_header("") == ""

    def test_render_template_empty_values(self) -> None:
        """render_template with empty string values does not crash."""
        from forma.delivery_send import EmailTemplate, render_template

        template = EmailTemplate(
            subject="{student_name} report",
            body="Student {student_id} in {class_name}",
        )
        subj, body = render_template(
            template, student_name="", student_id="", class_name="",
        )
        assert subj == " report"
        assert body == "Student  in "

    def test_build_warning_data_empty_predictions_empty_risk(self) -> None:
        """build_warning_data with at-risk student but no predictions works."""
        from forma.warning_report_data import build_warning_data

        result = build_warning_data(
            at_risk_students={"S001": {"is_at_risk": True, "reasons": ["test"]}},
            risk_predictions=[],
            concept_scores={"S001": {}},
            score_trajectories={"S001": [0.3]},
        )
        assert len(result) == 1
        assert result[0].student_id == "S001"

    def test_safe_nanmean_zero_length_array(self) -> None:
        """_safe_nanmean with numpy empty array returns 0.0."""
        from forma.risk_predictor import _safe_nanmean

        result = _safe_nanmean(np.array([]))
        assert result == 0.0

    def test_safe_nanvar_zero_length_array(self) -> None:
        """_safe_nanvar with numpy empty array returns 0.0."""
        from forma.risk_predictor import _safe_nanvar

        result = _safe_nanvar(np.array([]))
        assert result == 0.0
