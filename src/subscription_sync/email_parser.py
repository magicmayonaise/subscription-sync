"""Gmail API integration + Claude tool_use extraction.

Demonstrates:
- Gmail API with OAuth2 (credential refresh flow)
- Anthropic tool_use for structured LLM output (no JSON parsing heuristics)
- Schema-enforced extraction tool definition
- Batch email processing with error isolation
"""

from __future__ import annotations

import base64
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog
from anthropic import Anthropic
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from subscription_sync.config import Settings
from subscription_sync.models import EmailExtraction

logger = structlog.get_logger()

# Gmail API scopes (read-only — do not expand)
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_CARD_RE = re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b|\b\d{13,19}\b")

# Query to find billing/subscription emails (high recall, LLM handles precision)
BILLING_QUERY = (
    "subject:(receipt OR invoice OR billing OR subscription OR payment "
    "OR renewal OR charged OR confirmation OR monthly OR annual)"
)

# ═══════════════════════════════════════════════════════════════════════════
# Claude tool_use schema for structured extraction
# ═══════════════════════════════════════════════════════════════════════════

EXTRACTION_TOOL = {
    "name": "extract_subscription",
    "description": (
        "Extract subscription/billing information from an email. "
        "Call this tool with the structured data you find."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "service_name": {
                "type": "string",
                "description": "Name of the service/subscription (e.g., 'Netflix', 'Spotify')",
            },
            "amount": {
                "type": "number",
                "description": "Billing amount in USD. Null if not found.",
            },
            "billing_cycle": {
                "type": "string",
                "enum": ["monthly", "quarterly", "yearly", "unknown"],
                "description": "How often the subscription bills.",
            },
            "is_cancellation": {
                "type": "boolean",
                "description": "True if this email confirms a cancellation.",
            },
            "is_renewal": {
                "type": "boolean",
                "description": "True if this is a renewal/charge confirmation.",
            },
            "confidence": {
                "type": "number",
                "description": "Your confidence this is a subscription email (0.0 to 1.0).",
            },
            "billing_date": {
                "type": "string",
                "description": "The billing/charge date in YYYY-MM-DD format, if found.",
            },
        },
        "required": [
            "service_name",
            "billing_cycle",
            "is_cancellation",
            "is_renewal",
            "confidence",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Gmail Authentication
# ═══════════════════════════════════════════════════════════════════════════


def get_gmail_credentials(settings: Settings) -> Credentials:
    """Get or refresh Gmail OAuth2 credentials."""
    creds = None
    token_path = settings.gmail_token_path

    try:
        creds = Credentials.from_authorized_user_file(token_path, GMAIL_SCOPES)
    except (FileNotFoundError, ValueError):
        pass

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    elif not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            settings.gmail_credentials_path, GMAIL_SCOPES
        )
        creds = flow.run_local_server(port=0)

    # Save refreshed token with owner-only permissions
    _write_token_file(token_path, creds.to_json())

    return creds


def _write_token_file(path: str, contents: str) -> None:
    """Write OAuth token.json with mode 0o600."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, contents.encode("utf-8"))
    finally:
        os.close(fd)


def redact_sensitive_text(text: str) -> str:
    """Redact obvious email addresses and card numbers before sending to Claude."""
    redacted = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    return _CARD_RE.sub("[REDACTED_CARD]", redacted)


def authenticate_gmail(settings: Settings) -> None:
    """Run OAuth flow and save token. Used by `subsync auth` command."""
    creds = get_gmail_credentials(settings)
    logger.info("gmail_authenticated", valid=creds.valid)


# ═══════════════════════════════════════════════════════════════════════════
# Email Fetching
# ═══════════════════════════════════════════════════════════════════════════


def fetch_billing_emails(
    settings: Settings, lookback_days: int | None = None
) -> list[dict[str, Any]]:
    """Fetch billing-related emails from Gmail.

    Returns list of dicts with 'subject' and 'body' keys.
    """
    days = lookback_days or settings.lookback_days
    creds = get_gmail_credentials(settings)
    service = build("gmail", "v1", credentials=creds)

    after_date = datetime.now(tz=timezone.utc) - timedelta(days=days)
    query = f"{BILLING_QUERY} after:{after_date.strftime('%Y/%m/%d')}"

    logger.info("fetching_emails", query=query, lookback_days=days)

    results = (
        service.users()
        .messages()
        .list(userId="me", q=query, maxResults=50)
        .execute()
    )

    messages = results.get("messages", [])
    logger.info("emails_found", count=len(messages))

    emails = []
    for msg_ref in messages:
        try:
            msg = (
                service.users()
                .messages()
                .get(userId="me", id=msg_ref["id"], format="full")
                .execute()
            )
            email_data = _parse_email_message(msg)
            if email_data:
                emails.append(email_data)
        except Exception as e:
            logger.warning("email_fetch_error", msg_id=msg_ref["id"], error=str(e))

    return emails


def _parse_email_message(msg: dict) -> dict[str, str] | None:
    """Extract subject and body text from a Gmail message."""
    headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
    subject = headers.get("Subject", "")

    body = ""
    payload = msg.get("payload", {})

    # Try to get plain text body
    if payload.get("body", {}).get("data"):
        body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")
    elif payload.get("parts"):
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                body = base64.urlsafe_b64decode(part["body"]["data"]).decode(
                    "utf-8", errors="replace"
                )
                break

    if not subject and not body:
        return None

    # Truncate body to avoid token waste
    return {"subject": subject, "body": body[:3000]}


# ═══════════════════════════════════════════════════════════════════════════
# LLM Extraction via Claude tool_use
# ═══════════════════════════════════════════════════════════════════════════


def extract_subscription_from_email(
    email: dict[str, str], settings: Settings
) -> EmailExtraction | None:
    """Use Claude tool_use to extract structured subscription data.

    The tool_use pattern guarantees schema conformance — Claude must call
    the tool with valid JSON matching our input_schema, eliminating the
    need for fragile text→JSON parsing.
    """
    client = Anthropic(api_key=settings.anthropic_api_key.get_secret_value())

    subject = redact_sensitive_text(email["subject"])
    body = redact_sensitive_text(email["body"])

    prompt = f"""Analyze this email and extract subscription/billing information.

Subject: {subject}

Body:
{body}

If this is a subscription-related email (billing, receipt, renewal, cancellation),
call the extract_subscription tool with the details. If this is NOT a subscription
email (e.g., marketing, newsletter, promotion), still call the tool but set
confidence to 0.1 or lower."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=[EXTRACTION_TOOL],
            messages=[{"role": "user", "content": prompt}],
        )

        # Find the tool_use block in the response
        for block in response.content:
            if block.type == "tool_use" and block.name == "extract_subscription":
                extraction = EmailExtraction(
                    **block.input,
                    raw_subject=email["subject"],
                )
                logger.info(
                    "extraction_complete",
                    service=extraction.service_name,
                    confidence=extraction.confidence,
                )
                return extraction

        logger.debug("no_tool_use_in_response", subject=email["subject"])
        return None

    except Exception as e:
        logger.error("extraction_error", subject=email["subject"], error=str(e))
        return None
