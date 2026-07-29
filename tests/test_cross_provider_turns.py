"""Turns from one provider must be sendable to another.

Built-in tool content (web search / fetch) and citations are provider-native
annotations: the provider that produced them stashes its own raw payload in
`extra` and replays it verbatim on the next request. That payload is meaningless
to a different provider, so each provider replays only what it produced and
drops the rest -- otherwise `Chat.set_turns()` across providers (documented in
docs/reference/Turn.qmd) raises, or silently sends an invalid payload.
"""

import pytest
from chatlas._provider_anthropic import AnthropicProvider
from chatlas._provider_google import GoogleProvider
from chatlas._provider_openai import OpenAIProvider
from chatlas._turn import AssistantTurn, Turn, UserTurn
from chatlas.types import (
    ContentCitation,
    ContentText,
    ContentToolRequestFetch,
    ContentToolRequestSearch,
    ContentToolResponseFetch,
    ContentToolResponseSearch,
    WebSource,
)


def google_grounded_turns() -> list[Turn]:
    source = WebSource(url="https://a.com", title="A")
    return [
        UserTurn("When was ggplot2 1.0.0 released?"),
        AssistantTurn(
            [
                ContentToolRequestSearch(
                    query="ggplot2 1.0.0", extra={"web_search_queries": ["ggplot2"]}
                ),
                ContentToolResponseSearch(
                    sources=[source], extra={"grounding_metadata": {}}
                ),
                ContentText(text="2014-05-21"),
                ContentCitation(
                    source=source,
                    grounded_span="2014-05-21",
                    extra={"grounding_support": {}},
                ),
                ContentToolRequestFetch(
                    url="https://a.com", extra={"url_metadata": {}}
                ),
                ContentToolResponseFetch(
                    url="https://a.com", status="success", extra={"url_metadata": {}}
                ),
            ]
        ),
    ]


def openai_grounded_turns() -> list[Turn]:
    source = WebSource(url="https://a.com", title="A")
    return [
        UserTurn("When was ggplot2 1.0.0 released?"),
        AssistantTurn(
            [
                ContentToolRequestSearch(
                    query="ggplot2 1.0.0",
                    extra={
                        "type": "web_search_call",
                        "id": "ws_1",
                        "status": "completed",
                    },
                ),
                ContentText(text="2014-05-21"),
                ContentCitation(
                    source=source,
                    grounded_span="2014-05-21",
                    extra={"type": "url_citation", "url": "https://a.com"},
                ),
            ]
        ),
    ]


def anthropic_grounded_turns() -> list[Turn]:
    source = WebSource(url="https://a.com", title="A")
    return [
        UserTurn("When was ggplot2 1.0.0 released?"),
        AssistantTurn(
            [
                ContentToolRequestSearch(
                    query="ggplot2 1.0.0",
                    extra={
                        "type": "server_tool_use",
                        "id": "srvtoolu_1",
                        "name": "web_search",
                        "input": {"query": "ggplot2 1.0.0"},
                    },
                ),
                ContentToolResponseSearch(
                    sources=[source],
                    extra={
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_1",
                        "content": [],
                    },
                ),
                ContentText(text="2014-05-21"),
                ContentCitation(
                    source=source,
                    grounded_span="2014-05-21",
                    extra={"type": "web_search_result_location"},
                ),
            ]
        ),
    ]


def anthropic_provider():
    return AnthropicProvider(model="claude-sonnet-4-5", api_key="dummy", kwargs=None)


def google_provider():
    return GoogleProvider(
        model="gemini-2.5-flash", api_key="dummy", name="Google/Gemini", kwargs=None
    )


def openai_provider():
    return OpenAIProvider(model="gpt-4o", api_key="dummy", kwargs=None)


ALL_TURNS = {
    "google": google_grounded_turns,
    "openai": openai_grounded_turns,
    "anthropic": anthropic_grounded_turns,
}


@pytest.mark.parametrize("source_provider", sorted(ALL_TURNS))
def test_anthropic_accepts_turns_from_any_provider(source_provider: str):
    turns = ALL_TURNS[source_provider]()
    messages = anthropic_provider()._as_message_params(turns)

    blocks = [b for m in messages for b in m["content"]]
    assert any(b["type"] == "text" for b in blocks)
    # Anthropic block params always carry a "type"; a foreign payload wouldn't.
    assert all("type" in b for b in blocks)
    native = [
        b for b in blocks if b["type"] in ("server_tool_use", "web_search_tool_result")
    ]
    assert bool(native) == (source_provider == "anthropic")


@pytest.mark.parametrize("source_provider", sorted(ALL_TURNS))
def test_openai_accepts_turns_from_any_provider(source_provider: str):
    turns = ALL_TURNS[source_provider]()
    inputs = openai_provider()._turns_as_inputs(turns)

    native = [x for x in inputs if x.get("type") == "web_search_call"]
    assert bool(native) == (source_provider == "openai")


@pytest.mark.parametrize("source_provider", sorted(ALL_TURNS))
def test_google_accepts_turns_from_any_provider(source_provider: str):
    turns = ALL_TURNS[source_provider]()
    contents = google_provider()._google_contents(turns)

    parts = [p for c in contents for p in (c.parts or [])]
    assert any(p.text for p in parts)


def test_anthropic_replays_its_own_search_payload_verbatim():
    """Same-provider replay keeps server-side search context in history."""
    turns = anthropic_grounded_turns()
    messages = anthropic_provider()._as_message_params(turns)

    blocks = [b for m in messages for b in m["content"]]
    server_use = next(b for b in blocks if b["type"] == "server_tool_use")
    assert server_use["id"] == "srvtoolu_1"
    assert server_use["name"] == "web_search"
    result = next(b for b in blocks if b["type"] == "web_search_tool_result")
    assert result["tool_use_id"] == "srvtoolu_1"


def test_openai_replays_its_own_search_payload_verbatim():
    turns = openai_grounded_turns()
    inputs = openai_provider()._turns_as_inputs(turns)

    call = next(x for x in inputs if x.get("type") == "web_search_call")
    assert call["id"] == "ws_1"


def test_providers_drop_annotations_with_no_payload():
    """Content whose `extra` was never populated can't be replayed anywhere."""
    turns: list[Turn] = [
        UserTurn("hi"),
        AssistantTurn(
            [
                ContentToolRequestSearch(query="q"),
                ContentToolResponseSearch(sources=[WebSource(url="https://a.com")]),
                ContentText(text="answer"),
                ContentCitation(source=WebSource(url="https://a.com")),
                ContentToolRequestFetch(url="https://a.com"),
                ContentToolResponseFetch(url="https://a.com", status="success"),
            ]
        ),
    ]

    messages = anthropic_provider()._as_message_params(turns)
    blocks = [b for m in messages for b in m["content"]]
    assert all("type" in b for b in blocks)

    inputs = openai_provider()._turns_as_inputs(turns)
    assert all(x for x in inputs)

    google_provider()._google_contents(turns)
