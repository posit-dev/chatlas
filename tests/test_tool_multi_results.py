"""One tool call must produce exactly one tool result on the wire.

A tool may emit several results for a single call by yielding more than once;
each one is echoed, passed to `.on_tool_result()` callbacks, and included in
`content="all"` streams as it arrives. Providers, however, key their wire-level
result on the request id and reject more than one result per id -- Anthropic
with "each tool_use must have a single result". So results answering the same
request are combined into one before being sent (chatlas/_content_expand.py),
with image/PDF parts unrolled into request-scoped content alongside it.
"""

from typing import Any, cast

import pytest
from chatlas import ChatOpenAI
from chatlas._chat import ToolFailureWarning
from chatlas._content import ContentToolRequest, ToolInfo
from chatlas._provider_anthropic import AnthropicProvider
from chatlas._provider_google import GoogleProvider
from chatlas._provider_openai import OpenAIProvider
from chatlas._provider_openai_completions import OpenAICompletionsProvider
from chatlas._turn import (
    AssistantTurn,
    Turn,
    UserTurn,
    normalize_turns_for_provider,
)
from chatlas.types import ContentImageInline, ContentJson, ContentToolResult

CALL_ID = "call_abc123"
IMAGE = ContentImageInline(data="aGVsbG8=", image_content_type="image/png")


def tool_turns(func: Any, name: str) -> list[Turn]:
    """Register `func` as a tool, invoke it, and return the resulting turns.

    Goes through `Chat._invoke_tool` so the results are stamped with a single
    shared request, exactly as they are in a real conversation.
    """
    chat = ChatOpenAI(api_key="dummy")
    chat.register_tool(func, name=name)
    request = ContentToolRequest(
        id=CALL_ID,
        name=name,
        arguments={},
        tool=ToolInfo.from_tool(chat._tools[name]),
    )
    results = list(chat._invoke_tool(request))
    return normalize_turns_for_provider(
        [UserTurn("go"), AssistantTurn([request]), UserTurn(results)]
    )


def anthropic_blocks(turns: list[Turn]) -> list[dict[str, Any]]:
    """The content blocks of the final (tool result) user message."""
    messages = AnthropicProvider(
        model="claude-sonnet-4-5", api_key="dummy", kwargs=None
    )._as_message_params(turns)
    return [cast(dict[str, Any], b) for b in messages[-1]["content"]]


def multi_text():
    """Yields two text parts."""
    yield "part one"
    yield "part two"


def text_and_image():
    """Yields a text part and an image part."""
    yield "here is the plot"
    yield ContentToolResult(value=IMAGE, model_format="as_is")


def test_yielded_text_parts_become_one_result():
    blocks = anthropic_blocks(tool_turns(multi_text, "multi_text"))
    results = [b for b in blocks if b["type"] == "tool_result"]

    assert len(results) == 1
    assert results[0]["tool_use_id"] == CALL_ID
    assert results[0]["content"] == "part one\npart two"


def test_yielded_image_part_is_unrolled_beside_the_result():
    blocks = anthropic_blocks(tool_turns(text_and_image, "text_and_image"))
    types = [b["type"] for b in blocks]

    assert types.count("tool_result") == 1
    assert types.count("image") == 1
    # The single result forward-points at the content that follows it.
    assert f'<tool-contents call-id="{CALL_ID}">' in blocks[0]["content"]


def test_yielding_parts_matches_returning_a_list():
    """Both spellings of "one call, several contents" reach the wire alike."""

    def returns_parts():
        """Returns a text part and an image part as one list."""
        return ["here is the plot", IMAGE]

    assert anthropic_blocks(tool_turns(text_and_image, "t")) == anthropic_blocks(
        tool_turns(returns_parts, "t")
    )


def test_error_after_partial_output_becomes_an_error_result():
    """A part that fails invalidates the call; earlier parts are dropped."""

    def flaky():
        """Yields a part, then fails."""
        yield "part one"
        raise RuntimeError("boom")

    with pytest.warns(ToolFailureWarning):
        turns = tool_turns(flaky, "flaky")

    results = [b for b in anthropic_blocks(turns) if b["type"] == "tool_result"]

    assert len(results) == 1
    assert results[0]["is_error"] is True
    assert results[0]["content"] == "Tool call failed with error: 'boom'"


def test_anthropic_emits_single_tool_result_block():
    turns = tool_turns(text_and_image, "text_and_image")
    messages = AnthropicProvider(
        model="claude-sonnet-4-5", api_key="dummy", kwargs=None
    )._as_message_params(turns)

    blocks = [
        b
        for m in messages
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]

    assert len(blocks) == 1
    assert blocks[0]["tool_use_id"] == CALL_ID

    images = [
        b
        for m in messages
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "image"
    ]
    assert len(images) == 1


def test_openai_responses_emits_single_function_call_output():
    turns = tool_turns(text_and_image, "text_and_image")
    inputs = OpenAIProvider(
        model="gpt-4o", api_key="dummy", kwargs=None
    )._turns_as_inputs(turns)

    outputs = [
        cast(dict[str, Any], i)
        for i in inputs
        if isinstance(i, dict) and i.get("type") == "function_call_output"
    ]

    assert len(outputs) == 1
    assert outputs[0]["call_id"] == CALL_ID
    # This API takes a string result, so the parts are unrolled into following
    # content items that the result points at. Nothing is lost.
    assert "here is the plot" in repr(inputs)


def test_openai_completions_emits_single_tool_message():
    turns = tool_turns(text_and_image, "text_and_image")
    messages = OpenAICompletionsProvider(
        model="gpt-4o", api_key="dummy", kwargs=None
    )._turns_as_inputs(turns)

    tool_messages = [
        cast(dict[str, Any], m)
        for m in messages
        if isinstance(m, dict) and m.get("role") == "tool"
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == CALL_ID
    assert "here is the plot" in repr(messages)


def test_google_emits_single_function_response():
    turns = tool_turns(text_and_image, "text_and_image")
    contents = GoogleProvider(
        model="gemini-2.5-flash", api_key="dummy", name="Google/Gemini", kwargs=None
    )._google_contents(turns)

    responses = [
        p for c in contents for p in (c.parts or []) if p.function_response is not None
    ]

    assert len(responses) == 1
    assert responses[0].function_response.name == "text_and_image"


def _anthropic_result_ids(turns: list[Turn]) -> list[str]:
    messages = AnthropicProvider(
        model="claude-sonnet-4-5", api_key="dummy", kwargs=None
    )._as_message_params(turns)
    return [
        cast(dict[str, Any], b)["tool_use_id"]
        for m in messages
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]


def _openai_result_ids(turns: list[Turn]) -> list[str]:
    inputs = OpenAIProvider(model="gpt-4o", api_key="dummy", kwargs=None)._turns_as_inputs(
        turns
    )
    return [
        cast(dict[str, Any], i)["call_id"]
        for i in inputs
        if isinstance(i, dict) and i.get("type") == "function_call_output"
    ]


def _openai_completions_result_ids(turns: list[Turn]) -> list[str]:
    messages = OpenAICompletionsProvider(
        model="gpt-4o", api_key="dummy", kwargs=None
    )._turns_as_inputs(turns)
    return [
        cast(dict[str, Any], m)["tool_call_id"]
        for m in messages
        if isinstance(m, dict) and m.get("role") == "tool"
    ]


def _google_result_ids(turns: list[Turn]) -> list[str]:
    contents = GoogleProvider(
        model="gemini-2.5-flash", api_key="dummy", name="Google/Gemini", kwargs=None
    )._google_contents(turns)
    return [
        p.function_response.id
        for c in contents
        for p in (c.parts or [])
        if p.function_response is not None and p.function_response.id is not None
    ]


@pytest.mark.parametrize(
    "result_ids",
    [
        _anthropic_result_ids,
        _openai_result_ids,
        _openai_completions_result_ids,
        _google_result_ids,
    ],
    ids=["anthropic", "openai", "openai_completions", "google"],
)
def test_distinct_requests_are_not_merged(result_ids):
    """Combining must key on the request id, not lump all results together."""
    req_a = ContentToolRequest(id="call_a", name="repl", arguments={})
    req_b = ContentToolRequest(id="call_b", name="repl", arguments={})
    turns = normalize_turns_for_provider(
        [
            UserTurn("go"),
            AssistantTurn([req_a, req_b]),
            UserTurn(
                [
                    ContentToolResult(value="a1", request=req_a),
                    ContentToolResult(value="a2", request=req_a),
                    ContentToolResult(value="b1", request=req_b),
                ]
            ),
        ]
    )

    assert sorted(result_ids(turns)) == ["call_a", "call_b"]


def test_non_text_parts_are_rendered_before_joining():
    """Without an image in the group, the result must be a single string."""

    def text_and_dict():
        """Yields a text part and a dict part."""
        yield "hi"
        yield {"a": 1}

    blocks = anthropic_blocks(tool_turns(text_and_dict, "text_and_dict"))
    results = [b for b in blocks if b["type"] == "tool_result"]

    assert len(results) == 1
    assert results[0]["content"] == 'hi\n{"a":1}'


def test_each_part_keeps_its_own_model_format():
    def formatted_parts():
        """Yields parts with differing model_format settings."""
        yield ContentToolResult(value={"a": 1}, model_format="str")
        yield ContentToolResult(value={"b": 2}, model_format="json")

    blocks = anthropic_blocks(tool_turns(formatted_parts, "formatted_parts"))
    results = [b for b in blocks if b["type"] == "tool_result"]

    assert results[0]["content"] == '{\'a\': 1}\n{"b":2}'


def test_extra_is_not_carried_into_the_combined_result():
    """`extra` is per-part display metadata; callbacks already received it."""

    def parts_with_extra():
        """Yields two parts carrying display metadata."""
        yield ContentToolResult(value="one", extra={"index": 0})
        yield ContentToolResult(value="two", extra={"index": 1})

    turns = tool_turns(parts_with_extra, "parts_with_extra")
    results = [c for c in turns[-1].contents if isinstance(c, ContentToolResult)]

    assert len(results) == 1
    assert results[0].extra is None


def test_non_image_content_part_is_rendered_before_joining():
    """A non-image/PDF Content part (e.g. ContentJson) isn't unrollable, so it
    must be rendered into the joined string rather than left in a raw list."""

    def text_and_json():
        """Yields a text part and a ContentJson part."""
        yield "hi"
        yield ContentToolResult(value=ContentJson(value={"a": 1}), model_format="as_is")

    blocks = anthropic_blocks(tool_turns(text_and_json, "text_and_json"))
    results = [b for b in blocks if b["type"] == "tool_result"]

    assert len(results) == 1
    assert isinstance(results[0]["content"], str)


def test_bare_image_part_is_unrolled_regardless_of_model_format():
    """An image/PDF part must be unrolled even without an explicit as_is."""

    def text_and_bare_image():
        """Yields a text part and a bare image part (default model_format)."""
        yield "here is the plot"
        yield IMAGE

    blocks = anthropic_blocks(tool_turns(text_and_bare_image, "text_and_bare_image"))
    types = [b["type"] for b in blocks]

    assert types.count("tool_result") == 1
    assert types.count("image") == 1


def test_image_nested_in_a_list_part_is_unrolled():
    """An image nested inside one part's own list value must still unroll,
    not be stringified when combined with a sibling part."""

    def progress_and_chart():
        """Yields a plain string, then a list containing text and an image."""
        yield "progress"
        yield ["chart", IMAGE]

    blocks = anthropic_blocks(tool_turns(progress_and_chart, "progress_and_chart"))
    types = [b["type"] for b in blocks]

    assert types.count("tool_result") == 1
    assert types.count("image") == 1
    assert "progress" in repr(blocks)
    assert "chart" in repr(blocks)
