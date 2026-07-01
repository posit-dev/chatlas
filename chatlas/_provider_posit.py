from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import httpx
import requests
from platformdirs import user_cache_dir

from ._provider import ModelInfo
from ._provider_anthropic import AnthropicProvider, StructuredOutputMode
from ._provider_openai_completions import OpenAICompletionsProvider

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


def _safe_json(response: Any) -> Optional[dict[str, Any]]:
    try:
        return response.json()
    except ValueError:
        return None


def _posit_error_message(
    status_code: int, body: Optional[dict[str, Any]]
) -> Optional[str]:
    if status_code == 402:
        return "Your Posit AI credits are depleted."

    if body is None:
        return None

    error = body.get("error")
    error_type = body.get("error_type")
    if error_type is None and isinstance(error, dict):
        error_type = error.get("error_type")

    if error_type == "prism_account_not_found":
        return (
            "You must finish setting up your Posit AI account before using the "
            "API. Visit https://posit.ai/ to accept the service agreement."
        )

    return None


class PositAuth(httpx.Auth):
    """
    Turns a bearer-token callable into `Authorization: Bearer` headers on
    every request, overriding any `x-api-key` header a client set by default.
    """

    def __init__(self, credentials: Callable[[], str]):
        self._credentials = credentials

    def _apply(self, request: httpx.Request, token: str) -> None:
        request.headers.pop("x-api-key", None)
        request.headers["Authorization"] = f"Bearer {token}"

    def sync_auth_flow(self, request: httpx.Request):
        token = self._credentials()
        self._apply(request, token)
        yield request

    async def async_auth_flow(self, request: httpx.Request):
        token = await asyncio.to_thread(self._credentials)
        self._apply(request, token)
        yield request


def _make_posit_error_hook(error_cls: type[Exception]):
    def hook(response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        response.read()
        message = _posit_error_message(response.status_code, _safe_json(response))
        if message is not None:
            raise error_cls(message)

    return hook


def _make_posit_error_hook_async(error_cls: type[Exception]):
    async def hook(response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        await response.aread()
        message = _posit_error_message(response.status_code, _safe_json(response))
        if message is not None:
            raise error_cls(message)

    return hook


def list_models_posit(
    base_url: str = "https://gateway.posit.ai",
    credentials: Optional[Callable[[], str]] = None,
) -> list[ModelInfo]:
    token_provider = credentials or PositCredentials().get_token

    resp = requests.get(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {token_provider()}"},
    )

    if not resp.ok:
        message = _posit_error_message(resp.status_code, _safe_json(resp))
        if message is not None:
            raise RuntimeError(message)
        resp.raise_for_status()

    data = resp.json()

    res: list[ModelInfo] = []
    for m in data.get("chat", []):
        info: ModelInfo = {"id": m["id"], "name": m.get("display_name", m["id"])}
        res.append(info)

    return res


class PositAnthropicProvider(AnthropicProvider):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        credentials: Callable[[], str],
        max_tokens: int = 4096,
        cache: Literal["5m", "1h", "none"] = "5m",
        structured_output_mode: StructuredOutputMode = "auto",
        name: str = "Posit",
    ):
        super().__init__(
            model=model,
            api_key="not-used",
            max_tokens=max_tokens,
            cache=cache,
            structured_output_mode=structured_output_mode,
            name=name,
        )

        from anthropic import Anthropic, AnthropicError, AsyncAnthropic

        self._gateway_base_url = base_url.rstrip("/")
        self._credentials = credentials

        auth = PositAuth(credentials)
        flavor_base_url = f"{self._gateway_base_url}/anthropic/v1"

        self._client = Anthropic(
            api_key="not-used",
            base_url=flavor_base_url,
            http_client=httpx.Client(
                auth=auth,
                event_hooks={"response": [_make_posit_error_hook(AnthropicError)]},
            ),
        )
        self._async_client = AsyncAnthropic(
            api_key="not-used",
            base_url=flavor_base_url,
            http_client=httpx.AsyncClient(
                auth=auth,
                event_hooks={
                    "response": [_make_posit_error_hook_async(AnthropicError)]
                },
            ),
        )

    def list_models(self) -> list[ModelInfo]:
        return list_models_posit(self._gateway_base_url, self._credentials)


class PositOpenAIProvider(OpenAICompletionsProvider):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        credentials: Callable[[], str],
        name: str = "Posit",
    ):
        super().__init__(model=model, api_key="not-used", name=name)

        from openai import AsyncOpenAI, OpenAI, OpenAIError

        self._gateway_base_url = base_url.rstrip("/")
        self._credentials = credentials

        auth = PositAuth(credentials)
        flavor_base_url = f"{self._gateway_base_url}/openai/v1"

        self._client = OpenAI(
            api_key="not-used",
            base_url=flavor_base_url,
            http_client=httpx.Client(
                auth=auth,
                event_hooks={"response": [_make_posit_error_hook(OpenAIError)]},
            ),
        )
        self._async_client = AsyncOpenAI(
            api_key="not-used",
            base_url=flavor_base_url,
            http_client=httpx.AsyncClient(
                auth=auth,
                event_hooks={"response": [_make_posit_error_hook_async(OpenAIError)]},
            ),
        )

    def list_models(self) -> list[ModelInfo]:
        return list_models_posit(self._gateway_base_url, self._credentials)
