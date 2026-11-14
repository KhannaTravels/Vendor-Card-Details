
"""CRM SSO token verification helpers for Vendor Card Scanner.

The CRM backend signs short-lived launch assertions. This module verifies them
without exposing the signing secret to browser JavaScript. Replay protection is
stored in the application's existing database via app.py's DB abstraction.
"""

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import time

from urllib.parse import urlsplit


class SSOError(ValueError):
    """Raised when a CRM SSO assertion is invalid or cannot be accepted."""


def _b64url(data: bytes) -> str:
    """Encode bytes using canonical Base64URL without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    """Decode Base64URL and reject non-canonical representations."""

    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+", value
    ):
        raise SSOError("Malformed SSO assertion")

    try:
        decoded = base64.urlsafe_b64decode(
            value + "=" * (-len(value) % 4)
        )
    except (ValueError, binascii.Error) as exc:
        raise SSOError("Malformed SSO assertion") from exc

    # Reject alternate encodings that decode to the same bytes.
    # This is important for strict signature verification.
    if _b64url(decoded) != value:
        raise SSOError("Non-canonical Base64URL encoding")

    return decoded


def configured_origins():
    """Return normalized exact origins allowed to embed this application."""

    raw = os.getenv("CRM_ALLOWED_ORIGINS", "")
    origins = set()

    for item in raw.split(","):
        item = item.strip().rstrip("/")

        if not item:
            continue

        parsed = urlsplit(item)

        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.path
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise RuntimeError(
                "CRM_ALLOWED_ORIGINS must contain HTTPS origins only, "
                "without credentials or paths"
            )

        origins.add(f"{parsed.scheme}://{parsed.netloc}")

    return origins


def request_origin(request):
    """Get an origin from the request URL without trusting forwarded headers."""

    scheme = request.scheme
    host = request.host

    return f"{scheme}://{host}".rstrip("/")


def url_origin(value: str) -> str:
    """Return the origin of an absolute URL, or an empty string if invalid."""

    try:
        parsed = urlsplit(value)

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ""

        return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")

    except ValueError:
        return ""


def verify_assertion(
    token: str,
    *,
    secret: str,
    issuer: str,
    audience: str,
    max_lifetime: int = 60,
    now: int | None = None,
) -> dict:
    """Verify compact payload.signature HMAC-SHA256 assertion.

    Claims: iss, aud, sub, name, email, permissions, iat, exp, jti.

    The caller must persist and consume jti exactly once before creating
    a session.
    """

    if not secret or len(secret.encode("utf-8")) < 32:
        raise SSOError(
            "SSO is not configured with a sufficiently strong secret"
        )

    if not token or len(token) > 8192 or token.count(".") != 1:
        raise SSOError("Malformed SSO assertion")

    encoded_payload, encoded_signature = token.split(".", 1)

    signing_input = encoded_payload.encode("ascii")

    supplied_signature = _b64url_decode(encoded_signature)

    expected_signature = hmac.new(
        secret.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()

    if not hmac.compare_digest(
        supplied_signature,
        expected_signature,
    ):
        raise SSOError("Invalid SSO signature")

    try:
        payload = json.loads(
            _b64url_decode(encoded_payload).decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SSOError("Malformed SSO claims") from exc

    if not isinstance(payload, dict):
        raise SSOError("Malformed SSO claims")

    current = int(time.time() if now is None else now)

    required = (
        "iss",
        "aud",
        "sub",
        "iat",
        "exp",
        "jti",
        "permissions",
    )

    if any(key not in payload for key in required):
        raise SSOError("Required SSO claims are missing")

    if payload["iss"] != issuer or payload["aud"] != audience:
        raise SSOError("SSO issuer or audience mismatch")

    if (
        not isinstance(payload["sub"], (str, int))
        or not str(payload["sub"]).strip()
    ):
        raise SSOError("Invalid CRM user identity")

    if not isinstance(payload["jti"], str) or not re.fullmatch(
        r"[A-Za-z0-9_-]{16,128}",
        payload["jti"],
    ):
        raise SSOError("Invalid SSO token identifier")

    if (
        not isinstance(payload["permissions"], list)
        or not all(
            isinstance(permission, str)
            for permission in payload["permissions"]
        )
    ):
        raise SSOError("Invalid SSO permissions")

    if "vendors.view" not in payload["permissions"]:
        raise SSOError("CRM user is not permitted to view vendors")

    try:
        issued_at = int(payload["iat"])
        expires_at = int(payload["exp"])
    except (TypeError, ValueError) as exc:
        raise SSOError("Invalid SSO timestamps") from exc

    if issued_at > current + 5 or expires_at <= current:
        raise SSOError(
            "SSO assertion has expired or is not yet valid"
        )

    if (
        expires_at <= issued_at
        or expires_at - issued_at > max_lifetime
    ):
        raise SSOError("SSO assertion lifetime is invalid")

    # Normalize identity values before placing them in the Flask session.
    return {
        "sub": str(payload["sub"]),
        "name": str(payload.get("name") or "CRM User")[:160],
        "email": str(payload.get("email") or "")[:254],
        "permissions": sorted(set(payload["permissions"])),
        "jti": payload["jti"],
        "exp": expires_at,
    }
