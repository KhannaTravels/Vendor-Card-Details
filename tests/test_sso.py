
"""Focused Flask-side tests for the cross-origin CRM SSO proof of concept."""

import base64
import hashlib
import hmac
import json
import os
import secrets
import tempfile
import time
import unittest

TEST_DATA = tempfile.mkdtemp(prefix="vendor-sso-tests-")

os.environ["DATA_DIR"] = TEST_DATA
os.environ["APP_PASSWORD"] = "local-direct-test-password"
os.environ["SECRET_KEY"] = "local-test-secret-key-" + ("x" * 48)
os.environ["CRM_SSO_SECRET"] = "local-test-crm-sso-secret-" + ("y" * 48)
os.environ["CRM_ALLOWED_ORIGINS"] = "https://crm.example.test"
os.environ["CRM_SSO_ISSUER"] = "khanna-travels-crm"
os.environ["CRM_SSO_AUDIENCE"] = "khanna-vendor-management"

import app as vendor_app  # noqa: E402


VENDOR_ORIGIN = "https://vendor.example.test"


def vendor_headers():
    """Headers representing a same-origin request from the Flask application."""
    return {"Origin": VENDOR_ORIGIN}


def b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def make_assertion(*, permissions=None, expires_in=45, subject="crm-user-123"):
    now = int(time.time())

    payload = {
        "iss": "khanna-travels-crm",
        "aud": "khanna-vendor-management",
        "sub": subject,
        "name": "Operations Test User",
        "email": "operations@example.test",
        "permissions": permissions if permissions is not None else [
            "vendors.view",
            "vendors.scan",
            "vendors.create",
            "vendors.edit",
            "vendors.review",
            "vendors.import",
            "vendors.export",
            "vendors.delete",
            "vendors.settings",
            "vendors.sync",
            "vendors.backup",
        ],
        "iat": now,
        "exp": now + expires_in,
        "jti": secrets.token_urlsafe(24),
    }

    encoded = b64url(json.dumps(payload, separators=(",", ":")).encode())

    signature = hmac.new(
        os.environ["CRM_SSO_SECRET"].encode(),
        encoded.encode("ascii"),
        hashlib.sha256,
    ).digest()

    return encoded + "." + b64url(signature)


class VendorSSOTests(unittest.TestCase):

    def setUp(self):
        vendor_app.app.config.update(
            TESTING=True,
            WTF_CSRF_ENABLED=False,
        )
        self.client = vendor_app.app.test_client()

    def test_iframe_bootstrap_is_non_sensitive_and_framing_is_allowlisted(self):
        response = self.client.get("/auth/iframe")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Connecting to Vendor Management", response.data)

        self.assertIn(
            "frame-ancestors https://crm.example.test",
            response.headers["Content-Security-Policy"],
        )

    def test_direct_access_still_requires_basic_auth(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 401)
        self.assertIn("WWW-Authenticate", response.headers)

    def test_direct_basic_auth_fallback_still_works(self):
        credentials = base64.b64encode(
            b"vendor:local-direct-test-password"
        ).decode()

        response = self.client.get(
            "/",
            headers={"Authorization": "Basic " + credentials},
        )

        self.assertEqual(response.status_code, 200)

    def test_valid_sso_assertion_establishes_session(self):
        token = make_assertion()

        response = self.client.post(
            "/auth/crm/exchange",
            json={"assertion": token},
            headers=vendor_headers(),
            base_url=VENDOR_ORIGIN,
        )

        self.assertEqual(
            response.status_code,
            200,
            response.get_data(as_text=True),
        )
        self.assertTrue(response.json["ok"])

        page = self.client.get("/", base_url=VENDOR_ORIGIN)

        self.assertEqual(page.status_code, 200)
        self.assertIn(b"All vendors", page.data)

    def test_replayed_assertion_is_rejected(self):
        token = make_assertion()

        first = self.client.post(
            "/auth/crm/exchange",
            json={"assertion": token},
            headers=vendor_headers(),
            base_url=VENDOR_ORIGIN,
        )

        self.assertEqual(
            first.status_code,
            200,
            first.get_data(as_text=True),
        )

        self.client.post(
            "/auth/logout",
            base_url=VENDOR_ORIGIN,
            headers=vendor_headers(),
        )

        replay = self.client.post(
            "/auth/crm/exchange",
            json={"assertion": token},
            headers=vendor_headers(),
            base_url=VENDOR_ORIGIN,
        )

        self.assertEqual(replay.status_code, 401)

    def test_invalid_signature_is_rejected(self):
        token = make_assertion()
        token = token[:-1] + (
            "A" if token[-1] != "A" else "B"
        )

        response = self.client.post(
            "/auth/crm/exchange",
            json={"assertion": token},
            headers=vendor_headers(),
            base_url=VENDOR_ORIGIN,
        )

        self.assertEqual(response.status_code, 401)

    def test_missing_view_permission_is_rejected(self):
        token = make_assertion(permissions=["vendors.scan"])

        response = self.client.post(
            "/auth/crm/exchange",
            json={"assertion": token},
            headers=vendor_headers(),
            base_url=VENDOR_ORIGIN,
        )

        self.assertEqual(response.status_code, 401)

    def test_expired_assertion_is_rejected(self):
        token = make_assertion(expires_in=-10)

        response = self.client.post(
            "/auth/crm/exchange",
            json={"assertion": token},
            headers=vendor_headers(),
            base_url=VENDOR_ORIGIN,
        )

        self.assertEqual(response.status_code, 401)

    def test_crm_user_without_export_permission_cannot_export(self):
        token = make_assertion(permissions=["vendors.view"])

        response = self.client.post(
            "/auth/crm/exchange",
            json={"assertion": token},
            headers=vendor_headers(),
            base_url=VENDOR_ORIGIN,
        )

        self.assertEqual(
            response.status_code,
            200,
            response.get_data(as_text=True),
        )

        denied = self.client.get(
            "/export.xlsx",
            base_url=VENDOR_ORIGIN,
        )

        self.assertEqual(denied.status_code, 403)

    def test_exchange_rejects_missing_origin(self):
        """SSO exchange must reject requests without a verifiable Origin."""

        token = make_assertion()

        response = self.client.post(
            "/auth/crm/exchange",
            json={"assertion": token},
            base_url=VENDOR_ORIGIN,
        )

        self.assertEqual(response.status_code, 403)

    def test_missing_sso_origin_is_not_allowed_to_embed(self):
        old = os.environ.get("CRM_ALLOWED_ORIGINS")

        os.environ["CRM_ALLOWED_ORIGINS"] = ""

        try:
            response = self.client.get("/auth/iframe")

            self.assertIn(
                "frame-ancestors 'none'",
                response.headers["Content-Security-Policy"],
            )
        finally:
            if old is not None:
                os.environ["CRM_ALLOWED_ORIGINS"] = old


if __name__ == "__main__":
    unittest.main(verbosity=2)
