"""
emailer.py
Builds and sends the result email. No Streamlit needed.

Settings come from .streamlit/secrets.toml ([email] table) or from environment variables:
    host, port, username, password, sender, security ("starttls" | "ssl" | "none")
    SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_SENDER, SMTP_SECURITY
"""
import html
import os
import re
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage
from email.utils import formataddr, make_msgid

import guidance

_EMAIL_RE = re.compile(r"^[^@\s,;<>]+@[^@\s,;<>]+\.[^@\s,;<>]{2,}$")


def is_valid_email(address: str) -> bool:
    return bool(address) and len(address) <= 254 and bool(_EMAIL_RE.match(address))


def load_config(secrets=None) -> dict:
    """Merge secrets.toml values with environment variables. Returns {} if email is not set up."""
    cfg = {}
    if secrets:
        cfg.update({k.lower(): v for k, v in dict(secrets).items()})
    for key in ("host", "port", "username", "password", "sender", "security"):
        env = os.environ.get(f"SMTP_{key.upper()}")
        if env:
            cfg[key] = env
    cfg["security"] = str(cfg.get("security", "starttls")).strip().lower()
    if cfg["security"] not in {"starttls", "ssl", "none"}:
        return {}
    for key in ("host", "username", "password", "sender"):
        if key in cfg:
            cfg[key] = str(cfg[key]).strip()
    # App passwords are commonly displayed in groups for readability; SMTP expects the characters only.
    cfg["password"] = "".join(cfg.get("password", "").split())
    cfg.setdefault("port", 465 if cfg["security"] == "ssl" else 587)
    cfg["sender"] = cfg.get("sender") or cfg.get("username")
    try:
        cfg["port"] = int(cfg["port"])
    except (TypeError, ValueError):
        return {}
    required = (cfg.get("host"), cfg.get("sender"), cfg.get("username"), cfg.get("password"))
    if not all(required) or any(str(value).startswith("your.") or "xxxx" in str(value).lower() for value in required):
        return {}
    return cfg


# ------------------------------------------------------------------------------------------------
# Message
# ------------------------------------------------------------------------------------------------
def _bold_html(text: str) -> str:
    """Escape text, then turn **bold** into <strong>."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(text))


def _plain(text: str) -> str:
    return text.replace("**", "")


def build_message(to, name, result, inputs, advice, sender, model_name):
    """result: output of risk_model.predict. inputs: {question: readable answer}. advice: guidance.build(...)."""
    when = datetime.now().strftime("%d %B %Y, %H:%M")
    greeting = f"Hello {name}," if name else "Hello,"
    accent = "#B26A00" if result["level"] == "higher" else "#1E7B45"
    background = "#FFF4E0" if result["level"] == "higher" else "#E8F6EE"

    # ---- plain text ----
    lines = [greeting, "", "Here is a copy of your diabetes risk screening.", f"Date: {when}", "",
             "RESULT", advice["headline"], _plain(advice["summary"]), "",
             "YOUR ANSWERS"]
    lines += [f"- {q}: {a}" for q, a in inputs.items()]
    for s in advice["sections"]:
        lines += ["", s["title"].upper()] + [f"- {_plain(i)}" for i in s["items"]]
    lines += ["", "IMPORTANT", guidance.DISCLAIMER, "",
              f"Model used: {model_name}. This message was sent automatically; please do not reply."]
    text = "\n".join(lines)

    # ---- HTML ----
    rows = "".join(
        f"<tr><td style='padding:6px 10px;border-bottom:1px solid #e5e5e5;color:#555'>{html.escape(q)}</td>"
        f"<td style='padding:6px 10px;border-bottom:1px solid #e5e5e5'><strong>{html.escape(str(a))}</strong></td></tr>"
        for q, a in inputs.items())
    sections = "".join(
        f"<h3 style='margin:24px 0 6px;font-size:16px'>{html.escape(s['title'])}</h3>"
        "<ul style='margin:0;padding-left:20px'>"
        + "".join(f"<li style='margin:6px 0'>{_bold_html(i)}</li>" for i in s["items"]) + "</ul>"
        for s in advice["sections"])
    body = f"""\
<html><body style="font-family:Arial,Helvetica,sans-serif;color:#222;line-height:1.5;max-width:640px;margin:auto;padding:16px">
<h2 style="margin-bottom:0">Your diabetes risk screening</h2>
<p style="color:#777;margin-top:2px">{html.escape(when)}</p>
<p>{html.escape(greeting)}</p>
<div style="background:{background};border-left:5px solid {accent};padding:14px 16px;border-radius:6px">
  <div style="font-size:18px;font-weight:bold;color:{accent}">{html.escape(advice['headline'])}</div>
  <p style="margin:8px 0 0">{_bold_html(advice['summary'])}</p>
</div>
<h3 style="margin:24px 0 6px;font-size:16px">Your answers</h3>
<table style="border-collapse:collapse;width:100%;font-size:14px">{rows}</table>
{sections}
<p style="margin-top:28px;padding:12px;background:#f3f3f3;border-radius:6px;font-size:13px"><strong>Important.</strong> {html.escape(guidance.DISCLAIMER)}</p>
<p style="color:#888;font-size:12px">Model used: {html.escape(model_name)}. This message was sent automatically; please do not reply.</p>
</body></html>"""

    msg = EmailMessage()
    msg["Subject"] = "Your diabetes risk screening result"
    msg["From"] = formataddr(("Diabetes Risk Screener", sender))
    msg["To"] = to
    msg["Message-ID"] = make_msgid()
    msg.set_content(text)
    msg.add_alternative(body, subtype="html")
    return msg


# ------------------------------------------------------------------------------------------------
# Sending
# ------------------------------------------------------------------------------------------------
def send(msg, cfg, timeout=20):
    """Send one message. Returns (True, "") or (False, reason in plain words)."""
    try:
        if cfg["security"] == "ssl":
            server = smtplib.SMTP_SSL(cfg["host"], cfg["port"], timeout=timeout, context=ssl.create_default_context())
        else:
            server = smtplib.SMTP(cfg["host"], cfg["port"], timeout=timeout)
        with server:
            server.ehlo()
            if cfg["security"] == "starttls":
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
            if cfg.get("username"):
                server.login(cfg["username"], cfg.get("password", ""))
            server.send_message(msg, from_addr=cfg["sender"])
        return True, ""
    except smtplib.SMTPAuthenticationError:
        return False, "The email server rejected the login. Check the username and app password in secrets.toml."
    except smtplib.SMTPRecipientsRefused:
        return False, "The email server did not accept that address."
    except (smtplib.SMTPException, OSError, TimeoutError) as e:
        return False, "Could not reach the email server. Check the mail settings and try again."
