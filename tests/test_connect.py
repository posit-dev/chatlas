from types import SimpleNamespace

import pytest
from chatlas import ChatAnthropic, ChatOpenAI
from chatlas._connect import (
    CONNECT_VIEWER_TOKEN_HEADER,
    _connect_viewer_token,
    _is_connect_gateway_url,
    connect_viewer_headers,
)

GATEWAY_URL = "https://connect.example.com/__gateway__/anthropic/guid/v1"


@pytest.fixture
def connect_env(monkeypatch):
    monkeypatch.setenv("CONNECT_SERVER", "https://connect.example.com/")
    monkeypatch.setenv("RSTUDIO_PRODUCT", "CONNECT")


@pytest.fixture
def viewer_token(monkeypatch):
    monkeypatch.setattr(
        "chatlas._connect._connect_viewer_token", lambda: "token"
    )


def test_viewer_token_forwarded_only_to_connect_gateway(connect_env, viewer_token):
    headers = connect_viewer_headers(GATEWAY_URL)
    assert headers == {CONNECT_VIEWER_TOKEN_HEADER: "token"}

    assert connect_viewer_headers("https://api.anthropic.com/v1") == {}
    assert connect_viewer_headers("https://evil.example.com/__gateway__/anthropic/guid/v1") == {}
    # A host that only shares a string prefix with the server is not the server.
    assert connect_viewer_headers("https://connect.example.com.evil.com/__gateway__/x") == {}


def test_viewer_token_never_sent_off_connect(monkeypatch, viewer_token):
    monkeypatch.delenv("CONNECT_SERVER", raising=False)
    assert connect_viewer_headers(GATEWAY_URL) == {}
    assert not _is_connect_gateway_url(GATEWAY_URL)


def test_gateway_url_matching(connect_env):
    assert _is_connect_gateway_url(GATEWAY_URL)
    # Case-insensitive scheme/host
    assert _is_connect_gateway_url("HTTPS://CONNECT.EXAMPLE.COM/__gateway__/x")
    # Path must be under /__gateway__/
    assert not _is_connect_gateway_url("https://connect.example.com/content/123")
    # Port must match
    assert not _is_connect_gateway_url("https://connect.example.com:8443/__gateway__/x")
    # An explicit default port is equivalent to an omitted one
    assert _is_connect_gateway_url("https://connect.example.com:443/__gateway__/x")


def test_session_token_not_read_when_not_on_connect(monkeypatch):
    monkeypatch.setenv("CONNECT_SERVER", "https://connect.example.com")
    monkeypatch.delenv("RSTUDIO_PRODUCT", raising=False)
    assert _connect_viewer_token() is None


def test_viewer_token_read_from_shiny_session(connect_env, monkeypatch):
    shiny = pytest.importorskip("shiny.session")

    session = SimpleNamespace(
        http_conn=SimpleNamespace(
            headers={"posit-connect-user-session-token": "shiny-token"}
        )
    )
    monkeypatch.setattr(shiny, "get_current_session", lambda: session)
    assert _connect_viewer_token() == "shiny-token"

    # Header containers that preserve original casing are also handled
    session.http_conn.headers = {CONNECT_VIEWER_TOKEN_HEADER: "shiny-token"}
    assert _connect_viewer_token() == "shiny-token"


def test_no_token_without_session(connect_env, monkeypatch):
    shiny = pytest.importorskip("shiny.session")
    monkeypatch.setattr(shiny, "get_current_session", lambda: None)
    assert _connect_viewer_token() is None
    assert connect_viewer_headers(GATEWAY_URL) == {}


def test_chat_perform_args_include_viewer_header(connect_env, viewer_token):
    chat = ChatAnthropic(
        api_key="fake-key",
        kwargs={"base_url": GATEWAY_URL},
    )
    kwargs = chat.provider._chat_perform_args(
        stream=False, turns=[], tools={}, data_model=None, kwargs=None
    )
    assert kwargs["extra_headers"][CONNECT_VIEWER_TOKEN_HEADER] == "token"

    with pytest.warns(UserWarning, match="Responses API"):
        chat = ChatOpenAI(
            api_key="fake-key",
            base_url="https://connect.example.com/__gateway__/openai/guid/v1",
        )
    kwargs = chat.provider._chat_perform_args(
        stream=False, turns=[], tools={}, data_model=None, kwargs=None
    )
    assert kwargs["extra_headers"][CONNECT_VIEWER_TOKEN_HEADER] == "token"


def test_chat_perform_args_no_header_off_gateway(connect_env, viewer_token):
    chat = ChatAnthropic(api_key="fake-key")
    kwargs = chat.provider._chat_perform_args(
        stream=False, turns=[], tools={}, data_model=None, kwargs=None
    )
    assert CONNECT_VIEWER_TOKEN_HEADER not in (kwargs.get("extra_headers") or {})
