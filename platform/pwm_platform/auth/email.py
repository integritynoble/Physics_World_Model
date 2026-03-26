"""Email sending utility for PWM Platform (Gmail SMTP via aiosmtplib)."""

from __future__ import annotations

import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import aiosmtplib

from pwm_platform.config import settings

logger = logging.getLogger(__name__)


async def send_email(to: str, subject: str, html_body: str) -> None:
    """Send an HTML email via Gmail SMTP (TLS on port 587).

    Raises RuntimeError if SMTP is not configured.
    Logs and re-raises on send failure.
    """
    if not settings.SMTP_USER or not settings.SMTP_PASSWORD:
        raise RuntimeError("SMTP not configured — set SMTP_USER and SMTP_PASSWORD in .env")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_FROM
    msg["To"] = to
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        await aiosmtplib.send(
            msg,
            hostname=settings.SMTP_HOST,
            port=settings.SMTP_PORT,
            username=settings.SMTP_USER,
            password=settings.SMTP_PASSWORD,
            start_tls=True,
        )
        logger.info("Email sent to %s: %s", to, subject)
    except Exception as exc:
        logger.error("Failed to send email to %s: %s", to, exc)
        raise


def _reset_email_html(reset_url: str, username: str) -> str:
    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:sans-serif;background:#f9fafb;margin:0;padding:40px 20px">
  <div style="max-width:480px;margin:0 auto;background:#fff;border-radius:12px;padding:40px;box-shadow:0 1px 4px rgba(0,0,0,.08)">
    <h1 style="color:#4f46e5;font-size:24px;margin:0 0 8px">Physics World Model</h1>
    <p style="color:#6b7280;font-size:14px;margin:0 0 32px">Password reset request</p>
    <p style="color:#111827;font-size:15px">Hi {username},</p>
    <p style="color:#374151;font-size:14px;line-height:1.6">
      We received a request to reset your password. Click the button below to choose a new one.
      This link expires in <strong>1 hour</strong>.
    </p>
    <div style="text-align:center;margin:32px 0">
      <a href="{reset_url}"
         style="background:#4f46e5;color:#fff;padding:12px 32px;border-radius:8px;
                text-decoration:none;font-size:15px;font-weight:600;display:inline-block">
        Reset password
      </a>
    </div>
    <p style="color:#6b7280;font-size:13px">
      If you didn't request this, you can safely ignore this email — your password won't change.
    </p>
    <hr style="border:none;border-top:1px solid #e5e7eb;margin:32px 0">
    <p style="color:#9ca3af;font-size:12px;text-align:center">
      pwm.platformai.org &nbsp;·&nbsp; Physics World Model
    </p>
  </div>
</body>
</html>
"""


async def send_password_reset_email(to: str, username: str, token: str) -> None:
    reset_url = f"https://pwm.platformai.org/reset-password?token={token}"
    subject = "Reset your PWM password"
    html = _reset_email_html(reset_url, username)
    await send_email(to, subject, html)
