"""Delivery send module -- send emails with zip attachments via SMTP.

Handles email template rendering, SMTP connection, per-student email
composition and delivery, and delivery log management.

FR Coverage: FR-006 ~ FR-014, FR-017 ~ FR-019, FR-022.
"""

from __future__ import annotations

import email.message
import fcntl
import logging
import os
import ssl
import tempfile
from dataclasses import dataclass, field

import yaml

logger = logging.getLogger(__name__)

__all__ = [
    "EmailTemplate",
    "SmtpConfig",
    "DeliveryResult",
    "DeliveryLog",
    "load_template",
    "load_smtp_config",
    "_build_smtp_config",
    "validate_template_variables",
    "render_template",
    "compose_email",
    "send_emails",
    "save_delivery_log",
    "load_delivery_log",
    "print_delivery_summary",
    "send_summary_email",
]

# Supported template variables
SUPPORTED_VARIABLES = {"student_name", "student_id", "class_name"}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmailTemplate:
    """Email subject and body template.

    Args:
        subject: Email subject line.
        body: Email body text with variable placeholders.
    """

    subject: str
    body: str


@dataclass(frozen=True)
class SmtpConfig:
    """SMTP server connection settings (password excluded).

    Args:
        smtp_server: SMTP server hostname.
        smtp_port: SMTP server port (1-65535).
        sender_email: Sender email address.
        sender_name: Display name for sender.
        use_tls: Whether to use STARTTLS.
        send_interval_sec: Minimum seconds between sends.
    """

    smtp_server: str
    smtp_port: int
    sender_email: str
    sender_name: str = ""
    use_tls: bool = True
    send_interval_sec: float = 1.0


@dataclass
class DeliveryResult:
    """Per-student email delivery result.

    Args:
        student_id: Student identifier.
        email: Recipient email address.
        status: ``"success"`` or ``"failed"``.
        sent_at: ISO 8601 timestamp.
        attachment: Zip file name.
        size_bytes: Attachment size in bytes.
        error: Error message (empty string if success).
    """

    student_id: str
    email: str
    status: str
    sent_at: str
    attachment: str
    size_bytes: int
    error: str = ""


@dataclass
class DeliveryLog:
    """Complete delivery log for a send session.

    Args:
        sent_at: ISO 8601 timestamp of send start.
        smtp_server: SMTP server used.
        dry_run: Whether this was a dry-run.
        total: Total number of send targets.
        success: Number of successful sends.
        failed: Number of failed sends.
        results: Per-student delivery results.
    """

    sent_at: str
    smtp_server: str
    dry_run: bool
    total: int
    success: int
    failed: int
    results: list[DeliveryResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# YAML Loaders
# ---------------------------------------------------------------------------


def load_template(path: str) -> EmailTemplate:
    """Load an email template from a YAML file.

    Args:
        path: Path to the template YAML file.

    Returns:
        Parsed ``EmailTemplate``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required fields are missing or empty.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"템플릿 파일을 찾을 수 없습니다: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("템플릿 파일이 dict 형식이어야 합니다.")

    subject = data.get("subject")
    if not subject:
        raise ValueError("템플릿에 'subject' 필드가 필요합니다.")

    body = data.get("body")
    if not body:
        raise ValueError("템플릿에 'body' 필드가 필요합니다.")

    return EmailTemplate(subject=str(subject), body=str(body))


def _build_smtp_config(data: dict, field_map: dict | None = None) -> SmtpConfig:
    """Build an SmtpConfig from a raw dict with optional field name mapping.

    Args:
        data: Raw configuration dict.
        field_map: Optional mapping from source keys to SmtpConfig field names.
            When ``None``, source keys are assumed to match SmtpConfig fields
            (identity mapping, i.e. YAML format).

    Returns:
        Validated ``SmtpConfig``.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    if field_map is not None:
        mapped: dict = {}
        for src_key, dst_key in field_map.items():
            if src_key in data:
                mapped[dst_key] = data[src_key]
        data = mapped

    smtp_server = data.get("smtp_server")
    if not smtp_server:
        raise ValueError("SMTP 설정에 'smtp_server' 필드가 필요합니다.")

    sender_email = data.get("sender_email")
    if not sender_email:
        raise ValueError("SMTP 설정에 'sender_email' 필드가 필요합니다.")

    sender_email = str(sender_email)
    if "@" not in sender_email:
        raise ValueError(f"sender_email 형식이 올바르지 않습니다: {sender_email}")

    smtp_port = data.get("smtp_port", 587)
    if isinstance(smtp_port, bool) or not isinstance(smtp_port, int):
        raise ValueError(f"smtp_port는 유효한 정수여야 합니다: {smtp_port}")
    if smtp_port < 1 or smtp_port > 65535:
        raise ValueError(f"smtp_port는 1~65535 범위여야 합니다: {smtp_port}")

    sender_name = str(data.get("sender_name", ""))
    use_tls = data.get("use_tls", True)
    send_interval_sec = data.get("send_interval_sec", 1.0)

    if isinstance(send_interval_sec, bool) or not isinstance(send_interval_sec, (int, float)):
        raise ValueError(f"send_interval_sec는 숫자여야 합니다: {send_interval_sec}")
    if send_interval_sec < 0:
        raise ValueError(f"send_interval_sec는 0 이상이어야 합니다: {send_interval_sec}")

    return SmtpConfig(
        smtp_server=str(smtp_server),
        smtp_port=smtp_port,
        sender_email=sender_email,
        sender_name=sender_name,
        use_tls=bool(use_tls),
        send_interval_sec=float(send_interval_sec),
    )


def load_smtp_config(path: str) -> SmtpConfig:
    """Load SMTP configuration from a YAML file.

    Args:
        path: Path to the SMTP config YAML file.

    Returns:
        Parsed ``SmtpConfig``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required fields are missing or invalid.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"SMTP 설정 파일을 찾을 수 없습니다: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("SMTP 설정 파일이 dict 형식이어야 합니다.")

    return _build_smtp_config(data)


# ---------------------------------------------------------------------------
# Template validation and rendering
# ---------------------------------------------------------------------------


def validate_template_variables(template: EmailTemplate) -> None:
    """Validate that template uses only supported variables.

    Args:
        template: Email template to validate.

    Raises:
        ValueError: If unsupported variables are found.
    """
    import re

    pattern = re.compile(r"\{(\w+)\}")
    combined_text = template.subject + " " + template.body
    # Strip escaped braces ({{literal}}) before matching
    combined_text = combined_text.replace("{{", "").replace("}}", "")
    found_vars = set(pattern.findall(combined_text))
    unsupported = found_vars - SUPPORTED_VARIABLES
    if unsupported:
        raise ValueError(
            f"지원하지 않는 템플릿 변수: {', '.join(sorted(unsupported))}. "
            f"지원 변수: {', '.join(sorted(SUPPORTED_VARIABLES))}"
        )


def render_template(
    template: EmailTemplate,
    *,
    student_name: str,
    student_id: str,
    class_name: str,
) -> tuple[str, str]:
    """Render an email template with student-specific values.

    Args:
        template: Email template with placeholders.
        student_name: Student name.
        student_id: Student identifier.
        class_name: Class section name.

    Returns:
        Tuple of (rendered_subject, rendered_body).
    """
    values = {
        "student_name": student_name,
        "student_id": student_id,
        "class_name": class_name,
    }
    subject = template.subject
    body = template.body
    for key, val in values.items():
        subject = subject.replace("{" + key + "}", val)
        body = body.replace("{" + key + "}", val)
    return subject, body


# ---------------------------------------------------------------------------
# Email header sanitization (FR-006)
# ---------------------------------------------------------------------------


def _sanitize_header(value: str) -> str:
    """Strip CR and LF characters to prevent email header injection."""
    return value.replace('\r', '').replace('\n', '')


def _mask_email(email: str) -> str:
    """Mask an email address for PII protection (FR-030).

    Format: first 3 (or fewer) chars of local part + ``***`` + ``@domain``.
    If no ``@`` is present, masks as first 3 chars + ``***``.

    Args:
        email: Raw email address.

    Returns:
        Masked email string, or empty string if input is empty.
    """
    if not email:
        return ""
    if "@" in email:
        local, domain = email.split("@", 1)
        prefix = local[:3]
        return f"{prefix}***@{domain}"
    return f"{email[:3]}***"


# ---------------------------------------------------------------------------
# Email composition
# ---------------------------------------------------------------------------


def compose_email(
    *,
    sender_config: SmtpConfig,
    to_email: str,
    subject: str,
    body: str,
    zip_path: str,
) -> "email.message.Message":
    """Compose a MIMEMultipart email with zip attachment.

    Args:
        sender_config: SMTP configuration (for From header).
        to_email: Recipient email address.
        subject: Rendered email subject.
        body: Rendered email body (plain text).
        zip_path: Path to the zip file to attach.

    Returns:
        Composed ``MIMEMultipart`` message.
    """
    from email.mime.base import MIMEBase
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email import encoders

    msg = MIMEMultipart()

    # Set From with display name if provided (FR-006: sanitize headers)
    if sender_config.sender_name:
        msg["From"] = _sanitize_header(
            f"{sender_config.sender_name} <{sender_config.sender_email}>"
        )
    else:
        msg["From"] = _sanitize_header(sender_config.sender_email)

    msg["To"] = _sanitize_header(to_email)
    msg["Subject"] = _sanitize_header(subject)

    # Attach body as plain text
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # Attach zip file
    zip_filename = os.path.basename(zip_path)
    with open(zip_path, "rb") as f:
        part = MIMEBase("application", "zip")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f'attachment; filename="{zip_filename}"',
    )
    msg.attach(part)

    return msg


# ---------------------------------------------------------------------------
# Delivery log I/O
# ---------------------------------------------------------------------------


def save_delivery_log(log: DeliveryLog, path: str) -> None:
    """Save a delivery log to a YAML file.

    Args:
        log: Delivery log to save.
        path: Output file path.
    """
    results_list = []
    for r in log.results:
        results_list.append({
            "student_id": r.student_id,
            "email": r.email,
            "status": r.status,
            "sent_at": r.sent_at,
            "attachment": r.attachment,
            "size_bytes": r.size_bytes,
            "error": r.error,
        })

    data = {
        "sent_at": log.sent_at,
        "smtp_server": log.smtp_server,
        "dry_run": log.dry_run,
        "total": log.total,
        "success": log.success,
        "failed": log.failed,
        "results": results_list,
    }

    path_obj = os.path.abspath(path)
    dir_name = os.path.dirname(path_obj)
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
        os.replace(tmp_path, path_obj)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_delivery_log(path: str) -> DeliveryLog:
    """Load a delivery log from a YAML file.

    Args:
        path: Path to the delivery log YAML file.

    Returns:
        Parsed ``DeliveryLog``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"발송 로그 파일을 찾을 수 없습니다: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"delivery_log.yaml 형식이 올바르지 않습니다: {path}")

    _REQUIRED_KEYS = ("sent_at", "smtp_server", "total", "success", "failed")
    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(
            f"delivery_log.yaml에 필수 키가 없습니다: {', '.join(missing)} ({path})"
        )

    results = []
    for r in data.get("results", []):
        results.append(DeliveryResult(
            student_id=r["student_id"],
            email=r["email"],
            status=r["status"],
            sent_at=r["sent_at"],
            attachment=r["attachment"],
            size_bytes=r["size_bytes"],
            error=r.get("error", ""),
        ))

    return DeliveryLog(
        sent_at=data["sent_at"],
        smtp_server=data["smtp_server"],
        dry_run=data.get("dry_run", False),
        total=data["total"],
        success=data["success"],
        failed=data["failed"],
        results=results,
    )


# ---------------------------------------------------------------------------
# Email sending orchestrator
# ---------------------------------------------------------------------------


def send_emails(
    staging_dir: str,
    template_path: str,
    smtp_config_path: str,
    *,
    force: bool = False,
    dry_run: bool = False,
    retry_failed: bool = False,
    password: str | None = None,
    smtp_config: SmtpConfig | None = None,
) -> DeliveryLog:
    """Send emails to students based on prepare summary.

    Args:
        staging_dir: Path to the staging folder (from prepare step).
        template_path: Path to the email template YAML.
        smtp_config_path: Path to the SMTP config YAML.
        force: Allow re-send even if delivery_log exists with successes.
        dry_run: Preview only, no actual SMTP connection.
        retry_failed: Only retry previously failed deliveries.
        password: SMTP password (if None, reads from env var).
        smtp_config: Pre-built SMTP config. When provided, ``smtp_config_path``
            is ignored and no file load occurs.

    Returns:
        ``DeliveryLog`` with per-student results.

    Raises:
        ValueError: If password is missing or template has invalid variables.
        FileExistsError: If delivery_log exists with successes and force is False.
    """
    import smtplib
    import time
    from datetime import datetime, timezone

    template = load_template(template_path)
    if smtp_config is None:
        smtp_config = load_smtp_config(smtp_config_path)

    # Validate template variables before sending (FR-017)
    validate_template_variables(template)

    # Load prepare summary
    summary_path = os.path.join(staging_dir, "prepare_summary.yaml")
    with open(summary_path, encoding="utf-8") as f:
        summary_data = yaml.safe_load(f)

    if not isinstance(summary_data, dict):
        raise ValueError(f"prepare_summary.yaml 형식이 올바르지 않습니다: {summary_path}")

    # Check for existing delivery log (FR-022)
    log_path = os.path.join(staging_dir, "delivery_log.yaml")
    existing_log = None
    if os.path.exists(log_path):
        existing_log = load_delivery_log(log_path)
        if not dry_run and not retry_failed and not existing_log.dry_run:
            has_success = any(r.status == "success" for r in existing_log.results)
            if has_success and not force:
                raise ValueError(
                    "이미 발송 완료된 기록이 있습니다. --force로 강제 재발송하세요."
                )

    # Resolve password (FR-008)
    if not dry_run:
        if password is None:
            password = os.environ.get("FORMA_SMTP_PASSWORD")
        if not password:
            raise ValueError(
                "SMTP 비밀번호가 설정되지 않았습니다. "
                "FORMA_SMTP_PASSWORD 환경변수 또는 --password-from-stdin을 사용하세요."
            )

    # Determine send targets
    details = summary_data.get("details", [])
    class_name = summary_data.get("class_name", "")

    # Try to get class_name from summary or infer it
    if not class_name:
        # Load roster info if available in the summary
        class_name = ""

    # Filter targets based on status
    if retry_failed and existing_log:
        failed_ids = {r.student_id for r in existing_log.results if r.status == "failed"}
        targets = [d for d in details if d["student_id"] in failed_ids]
    else:
        targets = [d for d in details if d.get("status") in ("ready", "warning")]

    results: list[DeliveryResult] = []
    success_count = 0
    failed_count = 0

    # SMTP connection helper for reconnection (FR-023)
    _MAX_RETRIES = 3

    def _connect_smtp():
        conn = smtplib.SMTP(smtp_config.smtp_server, smtp_config.smtp_port, timeout=30)
        if smtp_config.use_tls:
            ctx = ssl.create_default_context()
            conn.starttls(context=ctx)
        conn.login(smtp_config.sender_email, password)
        return conn

    # SMTP connection (skip for dry run)
    smtp_conn = None
    if not dry_run:
        smtp_conn = _connect_smtp()

    try:
        for i, target in enumerate(targets):
            now = datetime.now(timezone.utc).isoformat()
            sid = target["student_id"]
            name = target.get("name", sid)
            email = target.get("email", "")
            zip_path = target.get("zip_path")

            if not zip_path or not os.path.exists(str(zip_path)):
                results.append(DeliveryResult(
                    student_id=sid, email=email, status="failed",
                    sent_at=now, attachment="", size_bytes=0,
                    error="zip 파일 없음",
                ))
                failed_count += 1
                continue

            zip_path = str(zip_path)
            zip_size = os.path.getsize(zip_path)

            # Render template
            subject, body = render_template(
                template,
                student_name=name,
                student_id=sid,
                class_name=class_name,
            )

            if dry_run:
                # Preview output (no actual sending)
                logger.info(
                    "[DRY-RUN] To: %s, Subject: %s, Attachment: %s",
                    _mask_email(email), subject, os.path.basename(zip_path),
                )
                results.append(DeliveryResult(
                    student_id=sid, email=email, status="success",
                    sent_at=now, attachment=os.path.basename(zip_path),
                    size_bytes=zip_size,
                ))
                success_count += 1
            else:
                try:
                    msg = compose_email(
                        sender_config=smtp_config,
                        to_email=email,
                        subject=subject,
                        body=body,
                        zip_path=zip_path,
                    )
                    # FR-023: Retry on SMTPServerDisconnected with reconnection
                    sent = False
                    last_error = None
                    for attempt in range(_MAX_RETRIES):
                        try:
                            smtp_conn.send_message(msg)
                            sent = True
                            break
                        except smtplib.SMTPServerDisconnected as disc_err:
                            last_error = disc_err
                            logger.warning(
                                "SMTP 연결 끊김 (시도 %d/%d): %s (%s)",
                                attempt + 1, _MAX_RETRIES, sid, disc_err,
                            )
                            try:
                                smtp_conn = _connect_smtp()
                            except Exception as reconn_err:
                                logger.warning("SMTP 재연결 실패: %s", reconn_err)
                                last_error = reconn_err
                    if sent:
                        results.append(DeliveryResult(
                            student_id=sid, email=email, status="success",
                            sent_at=now, attachment=os.path.basename(zip_path),
                            size_bytes=zip_size,
                        ))
                        success_count += 1
                    else:
                        logger.warning("발송 실패: %s (%s): %s", sid, _mask_email(email), last_error)
                        results.append(DeliveryResult(
                            student_id=sid, email=email, status="failed",
                            sent_at=now, attachment=os.path.basename(zip_path),
                            size_bytes=zip_size, error=str(last_error),
                        ))
                        failed_count += 1
                except Exception as e:
                    logger.warning("발송 실패: %s (%s): %s", sid, _mask_email(email), e)
                    results.append(DeliveryResult(
                        student_id=sid, email=email, status="failed",
                        sent_at=now, attachment=os.path.basename(zip_path),
                        size_bytes=zip_size, error=str(e),
                    ))
                    failed_count += 1

            # Rate limiting (FR-010): sleep between sends, not after last one
            if i < len(targets) - 1 and smtp_config.send_interval_sec > 0:
                time.sleep(smtp_config.send_interval_sec)
    finally:
        if smtp_conn is not None:
            try:
                smtp_conn.quit()
            except Exception:
                pass

    log = DeliveryLog(
        sent_at=datetime.now(timezone.utc).isoformat(),
        smtp_server=smtp_config.smtp_server,
        dry_run=dry_run,
        total=len(targets),
        success=success_count,
        failed=failed_count,
        results=results,
    )

    # Save delivery log
    save_delivery_log(log, log_path)

    return log


# ---------------------------------------------------------------------------
# Completion summary (FR-018, FR-019)
# ---------------------------------------------------------------------------


def print_delivery_summary(log: DeliveryLog) -> None:
    """Print delivery summary to console (FR-018).

    Args:
        log: Delivery log with results.
    """
    prefix = "[DRY-RUN] " if log.dry_run else ""
    print(
        f"{prefix}전체 {log.total}건 중 {log.success}건 성공, "
        f"{log.failed}건 실패"
    )

    if log.failed > 0:
        failed_results = [r for r in log.results if r.status == "failed"]
        for r in failed_results:
            print(f"  [FAILED] {r.student_id} ({_mask_email(r.email)}): {r.error}")


def send_summary_email(
    log: DeliveryLog,
    smtp_config: SmtpConfig,
    *,
    password: str,
) -> None:
    """Send a summary email to the professor (FR-019).

    Args:
        log: Delivery log with results.
        smtp_config: SMTP configuration.
        password: SMTP password.
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    prefix = "[DRY-RUN] " if log.dry_run else ""
    body_lines = [
        f"{prefix}발송 결과 요약",
        "",
        f"발송 시각: {log.sent_at}",
        f"SMTP 서버: {log.smtp_server}",
        f"전체: {log.total}건",
        f"성공: {log.success}건",
        f"실패: {log.failed}건",
    ]

    if log.failed > 0:
        body_lines.append("")
        body_lines.append("실패 목록:")
        for r in log.results:
            if r.status == "failed":
                body_lines.append(f"  - {r.student_id} ({_mask_email(r.email)}): {r.error}")

    body = "\n".join(body_lines)

    msg = MIMEMultipart()
    # FR-006: sanitize headers
    if smtp_config.sender_name:
        msg["From"] = _sanitize_header(
            f"{smtp_config.sender_name} <{smtp_config.sender_email}>"
        )
    else:
        msg["From"] = _sanitize_header(smtp_config.sender_email)
    msg["To"] = _sanitize_header(smtp_config.sender_email)
    msg["Subject"] = _sanitize_header(
        f"[forma] 발송 결과 요약 ({log.success}/{log.total})"
    )
    msg.attach(MIMEText(body, "plain", "utf-8"))

    conn = smtplib.SMTP(smtp_config.smtp_server, smtp_config.smtp_port, timeout=30)
    try:
        if smtp_config.use_tls:
            ctx = ssl.create_default_context()
            conn.starttls(context=ctx)
        conn.login(smtp_config.sender_email, password)
        conn.send_message(msg)
    finally:
        try:
            conn.quit()
        except Exception:
            pass
