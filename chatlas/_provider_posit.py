from __future__ import annotations

import json
import os
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, Optional

import requests
from platformdirs import user_cache_dir

DEVICE_AUTHORIZE_URL = "https://login.posit.cloud/oauth/device/authorize"
TOKEN_URL = "https://login.posit.cloud/oauth/token"
OAUTH_CLIENT_ID = "rstudio-ide"
OAUTH_SCOPE = "prism"


def is_interactive() -> bool:
    return sys.stdin.isatty()


def token_from_response(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "access_token": data["access_token"],
        "refresh_token": data.get("refresh_token"),
        "expires_at": time.time() + data.get("expires_in", 3600),
    }


class PositCredentials:
    """
    Manages a cached OAuth (RFC 8628 device-flow) access token for the Posit
    AI gateway, refreshing or re-authenticating as needed.
    """

    def __init__(self, cache_path: Optional[Path] = None):
        self._cache_path = cache_path or (
            Path(user_cache_dir("chatlas")) / "posit-credentials.json"
        )

    def get_token(self) -> str:
        cached = self._read_cache()

        if cached is not None and cached.get("expires_at", 0) - time.time() > 60:
            return cached["access_token"]

        if cached is not None and cached.get("refresh_token"):
            try:
                token = self._refresh(cached["refresh_token"])
                self._write_cache(token)
                return token["access_token"]
            except requests.HTTPError:
                pass

        token = self._device_flow()
        self._write_cache(token)
        return token["access_token"]

    def _read_cache(self) -> Optional[dict[str, Any]]:
        if not self._cache_path.exists():
            return None
        try:
            return json.loads(self._cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _write_cache(self, token: dict[str, Any]) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(token))
        os.chmod(self._cache_path, 0o600)

    def _refresh(self, refresh_token: str) -> dict[str, Any]:
        resp = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": OAUTH_CLIENT_ID,
            },
        )
        resp.raise_for_status()
        return token_from_response(resp.json())

    def _device_flow(self) -> dict[str, Any]:
        resp = requests.post(
            DEVICE_AUTHORIZE_URL,
            data={"client_id": OAUTH_CLIENT_ID, "scope": OAUTH_SCOPE},
        )
        resp.raise_for_status()
        device = resp.json()

        verification_uri = (
            device.get("verification_uri_complete") or device["verification_uri"]
        )
        print(f"Visit {verification_uri} and enter code {device['user_code']}")
        if is_interactive():
            try:
                webbrowser.open(verification_uri)
            except Exception:
                pass

        interval = device.get("interval", 5)
        deadline = time.time() + device.get("expires_in", 600)

        while time.time() < deadline:
            time.sleep(interval)

            resp = requests.post(
                TOKEN_URL,
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device["device_code"],
                    "client_id": OAUTH_CLIENT_ID,
                },
            )
            body = resp.json()

            if resp.status_code == 200:
                return token_from_response(body)

            error = body.get("error")
            if error == "slow_down":
                interval += 5
            elif error == "authorization_pending":
                continue
            elif error == "expired_token":
                raise TimeoutError(
                    "Posit AI device-flow login expired before it was confirmed."
                )
            elif error == "access_denied":
                raise PermissionError("Posit AI device-flow login was denied.")
            else:
                resp.raise_for_status()

        raise TimeoutError("Timed out waiting for Posit AI device-flow login.")
