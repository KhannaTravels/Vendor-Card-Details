"""Standard-library tests for SSO assertion verification (no Flask dependency)."""
import base64
import hashlib
import hmac
import json
import secrets
import time
import unittest

import auth_sso

SECRET = "unit-test-secret-material-" + ("z" * 48)


def b64url(value):
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def assertion(**overrides):
    now = int(time.time())
    payload = {
        "iss": "khanna-travels-crm",
        "aud": "khanna-vendor-management",
        "sub": "user-42",
        "name": "Test User",
        "email": "test@example.test",
        "permissions": ["vendors.view"],
        "iat": now,
        "exp": now + 45,
        "jti": secrets.token_urlsafe(24),
    }
    payload.update(overrides)
    body = b64url(json.dumps(payload, separators=(",", ":")).encode())
    signature = hmac.new(SECRET.encode(), body.encode(), hashlib.sha256).digest()
    return body + "." + b64url(signature)


class AssertionTests(unittest.TestCase):
    def test_accepts_valid_assertion(self):
        identity = auth_sso.verify_assertion(
            assertion(), secret=SECRET, issuer="khanna-travels-crm",
            audience="khanna-vendor-management", max_lifetime=60,
        )
        self.assertEqual(identity["sub"], "user-42")
        self.assertIn("vendors.view", identity["permissions"])

    def test_rejects_invalid_signature(self):
        token = assertion()
        first, signature = token.split(".")
        replacement = "A" if signature[-1] != "A" else "B"
        with self.assertRaises(auth_sso.SSOError):
            auth_sso.verify_assertion(
                first + "." + signature[:-1] + replacement, secret=SECRET,
                issuer="khanna-travels-crm", audience="khanna-vendor-management",
            )

    def test_rejects_wrong_issuer_or_audience(self):
        with self.assertRaises(auth_sso.SSOError):
            auth_sso.verify_assertion(assertion(), secret=SECRET,
                issuer="wrong-issuer", audience="khanna-vendor-management")

    def test_rejects_expired_assertion(self):
        with self.assertRaises(auth_sso.SSOError):
            auth_sso.verify_assertion(assertion(exp=int(time.time()) - 1), secret=SECRET,
                issuer="khanna-travels-crm", audience="khanna-vendor-management")

    def test_rejects_missing_view_permission(self):
        with self.assertRaises(auth_sso.SSOError):
            auth_sso.verify_assertion(assertion(permissions=["vendors.scan"]), secret=SECRET,
                issuer="khanna-travels-crm", audience="khanna-vendor-management")

    def test_rejects_assertion_lifetime_above_limit(self):
        now = int(time.time())
        with self.assertRaises(auth_sso.SSOError):
            auth_sso.verify_assertion(assertion(iat=now, exp=now + 300), secret=SECRET,
                issuer="khanna-travels-crm", audience="khanna-vendor-management", max_lifetime=60)

    def test_rejects_weak_secret(self):
        with self.assertRaises(auth_sso.SSOError):
            auth_sso.verify_assertion(assertion(), secret="weak", issuer="khanna-travels-crm",
                audience="khanna-vendor-management")


if __name__ == "__main__":
    unittest.main(verbosity=2)
