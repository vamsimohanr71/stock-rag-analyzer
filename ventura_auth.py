"""
Ventura Securities (EaseAPI) authentication flow, per their official docs:

  1. User is sent to https://easeapi.venturasecurities.com/auth/v1/login
     with our app_key (and optional state param).
  2. User logs in on Ventura's page.
  3. Ventura redirects to OUR PRE-REGISTERED redirect URL (configured on
     Ventura's developer portal, NOT passed as a query param here) with a
     request_token in the query string. Valid for only 10 minutes.
  4. We POST the request_token + SHA256(app_key + secret_key) (lowercase
     hex, sent as the "data" field) to the token endpoint.
  5. Ventura returns client_id, auth_token, auth_expiry, refresh_token,
     refresh_expiry.
"""
import hashlib

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings

VENTURA_LOGIN_URL = "https://easeapi.venturasecurities.com/auth/v1/login"
VENTURA_TOKEN_URL = "https://easeapi.venturasecurities.com/login/v1/authorization/token"


def get_login_url(state: str = "") -> str:
    """
    Step 1: build the URL to send the user to for Ventura login.
    The redirect target after login is whatever's registered for this
    app_key on Ventura's developer portal - it is NOT passed here.
    state is optional and gets echoed back on the redirect if provided -
    useful for tracking, not required.
    """
    url = f"{VENTURA_LOGIN_URL}?app_key={settings.VENTURA_APP_KEY}"
    if state:
        url += f"&state={state}"
    return url


def _make_checksum() -> str:
    """SHA256(app_key + secret_key), lowercase hex - sent as the "data" field."""
    raw = f"{settings.VENTURA_APP_KEY}{settings.VENTURA_SECRET_KEY}"
    return hashlib.sha256(raw.encode()).hexdigest().lower()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def exchange_request_token(request_token: str) -> dict:
    """
    Step 4->5: exchange a request_token (received on the registered
    redirect URL as a query param after login) for real auth tokens.
    Note: request_token is only valid for 10 minutes from issuance - if
    this fails with an auth/expiry error, the user needs to log in again
    via get_login_url() rather than retrying the same token.
    """
    headers = {
        "x-app-key": settings.VENTURA_APP_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "request_token": request_token,
        "data": _make_checksum(),
    }
    resp = requests.post(VENTURA_TOKEN_URL, headers=headers, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return {
        "client_id": data.get("client_id"),
        "auth_token": data.get("auth_token"),
        "auth_expiry": data.get("auth_expiry"),
        "refresh_token": data.get("refresh_token"),
        "refresh_expiry": data.get("refresh_expiry"),
    }


VENTURA_TOTP_URL = "https://easeapi.venturasecurities.com/login/v1/authorization/totp"


def _get_mac_address() -> str:
    """
    Best-effort local MAC address for the x-mac-address header Ventura's
    TOTP endpoint requires. This works on a local machine but is NOT
    reliable in cloud/container environments (Streamlit Community Cloud
    included) - containers don't have a stable, pre-registered MAC, so
    this login method is only recommended for local runs. If Ventura
    rejects the login due to an unrecognized MAC, that's this limitation,
    not a bug - use the OAuth redirect flow (get_login_url /
    exchange_request_token above) for the deployed app instead.
    """
    import uuid
    mac = uuid.getnode()
    return ":".join(f"{(mac >> ele) & 0xff:02x}" for ele in range(40, -8, -8))


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
def login_with_totp(client_id: str, pin: str, totp: str) -> dict:
    """
    Alternative to the OAuth redirect flow - logs in directly via Ventura
    PIN + authenticator TOTP code, no browser redirect needed.

    SECURITY NOTE: unlike the OAuth flow (where the PIN is only ever typed
    on Ventura's own login page), this sends the PIN through our own app's
    code. Recommended for local/personal use only - avoid using this login
    method on a publicly deployed instance of this app, since the
    credential-entry surface becomes our code instead of Ventura's domain.

    Also subject to Ventura's documented constraints:
    - Max 1 request/second per app_key
    - Requires TOTP-based auth enabled in the Ventura account (My Profile
      -> Authenticator)
    - NOT available for "partner" app types - only individual/personal apps
    - Requires the x-mac-address header, which is unreliable in cloud/
      container environments (see _get_mac_address docstring above)
    """
    headers = {
        "x-app-key": settings.VENTURA_APP_KEY,
        "x-client-id": client_id,
        "x-mac-address": _get_mac_address(),
        "Content-Type": "application/json",
    }
    payload = {
        "password": pin,
        "data": _make_checksum(),
        "totp": totp,
    }
    resp = requests.post(VENTURA_TOTP_URL, headers=headers, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return {
        "client_id": data.get("client_id"),
        "auth_token": data.get("auth_token"),
        "auth_expiry": data.get("auth_expiry"),
        "refresh_token": data.get("refresh_token"),
        "refresh_expiry": data.get("refresh_expiry"),
    }