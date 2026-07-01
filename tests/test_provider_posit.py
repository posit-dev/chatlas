import json
import time
from pathlib import Path
from typing import Any, Optional

import pytest
import requests

from chatlas._provider_posit import PositCredentials


class _FakeResponse:
    def __init__(self, status_code: int, json_data: dict[str, Any]):
        self.status_code = status_code
        self._json_data = json_data

    def json(self) -> dict[str, Any]:
        return self._json_data

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


def test_get_token_returns_cached_token_when_not_expired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_path = tmp_path / "posit-credentials.json"
    cache_path.write_text(
        json.dumps(
            {
                "access_token": "cached-token",
                "refresh_token": "cached-refresh",
                "expires_at": time.time() + 3600,
            }
        )
    )

    def fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("should not make a network call for a fresh token")

    monkeypatch.setattr("chatlas._provider_posit.requests.post", fail_if_called)

    creds = PositCredentials(cache_path=cache_path)
    assert creds.get_token() == "cached-token"


def test_get_token_refreshes_expired_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_path = tmp_path / "posit-credentials.json"
    cache_path.write_text(
        json.dumps(
            {
                "access_token": "old-token",
                "refresh_token": "old-refresh",
                "expires_at": time.time() - 10,
            }
        )
    )

    calls: list[dict[str, Any]] = []

    def fake_post(url: str, data: Optional[dict[str, Any]] = None, **kwargs: Any):
        calls.append({"url": url, "data": data})
        return _FakeResponse(
            200,
            {
                "access_token": "new-token",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            },
        )

    monkeypatch.setattr("chatlas._provider_posit.requests.post", fake_post)

    creds = PositCredentials(cache_path=cache_path)
    assert creds.get_token() == "new-token"

    assert len(calls) == 1
    assert calls[0]["data"]["grant_type"] == "refresh_token"
    assert calls[0]["data"]["refresh_token"] == "old-refresh"

    cached = json.loads(cache_path.read_text())
    assert cached["access_token"] == "new-token"


def test_get_token_falls_back_to_device_flow_when_refresh_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_path = tmp_path / "posit-credentials.json"
    cache_path.write_text(
        json.dumps(
            {
                "access_token": "old-token",
                "refresh_token": "old-refresh",
                "expires_at": time.time() - 10,
            }
        )
    )

    device_response = _FakeResponse(
        200,
        {
            "device_code": "device-abc",
            "user_code": "ABCD-1234",
            "verification_uri": "https://login.posit.cloud/device",
            "interval": 1,
            "expires_in": 60,
        },
    )
    poll_response = _FakeResponse(
        200,
        {"access_token": "fresh-token", "refresh_token": "fresh-refresh", "expires_in": 3600},
    )

    def fake_post(url: str, data: Optional[dict[str, Any]] = None, **kwargs: Any):
        if data and data.get("grant_type") == "refresh_token":
            raise requests.HTTPError("refresh failed")
        if url.endswith("/device/authorize"):
            return device_response
        return poll_response

    monkeypatch.setattr("chatlas._provider_posit.requests.post", fake_post)
    monkeypatch.setattr("chatlas._provider_posit.time.sleep", lambda _: None)
    monkeypatch.setattr("chatlas._provider_posit.is_interactive", lambda: False)

    creds = PositCredentials(cache_path=cache_path)
    assert creds.get_token() == "fresh-token"


def test_get_token_runs_device_flow_when_no_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_path = tmp_path / "posit-credentials.json"

    device_response = _FakeResponse(
        200,
        {
            "device_code": "device-abc",
            "user_code": "ABCD-1234",
            "verification_uri": "https://login.posit.cloud/device",
            "interval": 1,
            "expires_in": 60,
        },
    )
    poll_response = _FakeResponse(
        200, {"access_token": "new-token", "refresh_token": "new-refresh", "expires_in": 3600}
    )
    calls = {"count": 0}

    def fake_post(url: str, data: Optional[dict[str, Any]] = None, **kwargs: Any):
        calls["count"] += 1
        if url.endswith("/device/authorize"):
            return device_response
        return poll_response

    monkeypatch.setattr("chatlas._provider_posit.requests.post", fake_post)
    monkeypatch.setattr("chatlas._provider_posit.time.sleep", lambda _: None)
    monkeypatch.setattr("chatlas._provider_posit.is_interactive", lambda: False)

    creds = PositCredentials(cache_path=cache_path)
    assert creds.get_token() == "new-token"
    assert calls["count"] == 2  # one authorize call, one poll call

    assert cache_path.exists()
    assert (cache_path.stat().st_mode & 0o777) == 0o600
    cached = json.loads(cache_path.read_text())
    assert cached["access_token"] == "new-token"


def test_device_flow_handles_pending_and_slow_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_path = tmp_path / "posit-credentials.json"

    device_response = _FakeResponse(
        200,
        {
            "device_code": "device-abc",
            "user_code": "ABCD-1234",
            "verification_uri": "https://login.posit.cloud/device",
            "interval": 1,
            "expires_in": 60,
        },
    )
    responses = [
        device_response,
        _FakeResponse(400, {"error": "authorization_pending"}),
        _FakeResponse(400, {"error": "slow_down"}),
        _FakeResponse(200, {"access_token": "final-token", "expires_in": 3600}),
    ]

    def fake_post(url: str, data: Optional[dict[str, Any]] = None, **kwargs: Any):
        return responses.pop(0)

    sleep_calls: list[float] = []
    monkeypatch.setattr("chatlas._provider_posit.requests.post", fake_post)
    monkeypatch.setattr("chatlas._provider_posit.time.sleep", sleep_calls.append)
    monkeypatch.setattr("chatlas._provider_posit.is_interactive", lambda: False)

    creds = PositCredentials(cache_path=cache_path)
    assert creds.get_token() == "final-token"
    assert sleep_calls == [1, 1, 6]  # interval starts at 1s, +5s after slow_down
