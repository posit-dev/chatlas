"""Tests for turn content expansion with images and PDFs in tool results."""

import base64
from typing import Any, cast

from chatlas import UserTurn
from chatlas._content import (
    ContentImageInline,
    ContentImageRemote,
    ContentPDF,
    ContentText,
    ContentToolRequest,
    ContentToolResult,
    ToolInfo,
)
from chatlas._provider_anthropic import AnthropicProvider
from chatlas._turn import AssistantTurn


def test_expand_turn_no_tool_results():
    """Test that turns without tool results are unchanged."""
    turn = UserTurn([ContentText(text="Hello")])

    assert len(turn.contents) == 1
    assert isinstance(turn.contents[0], ContentText)
    assert turn.contents[0].text == "Hello"


def test_expand_turn_tool_result_without_images():
    """Test that tool results with regular values are unchanged."""
    request = ContentToolRequest(
        id="call_123",
        name="test_tool",
        arguments={},
        tool=ToolInfo(name="test_tool", description="", parameters={}),
    )

    result = ContentToolResult(value="test result", request=request)
    turn = UserTurn([result])

    assert len(turn.contents) == 1
    assert isinstance(turn.contents[0], ContentToolResult)
    assert turn.contents[0].value == "test result"


def test_expand_turn_tool_result_with_single_image():
    """Test expansion of tool result containing a single image."""
    request = ContentToolRequest(
        id="call_456",
        name="get_image",
        arguments={},
        tool=ToolInfo(name="get_image", description="", parameters={}),
    )

    # Create a simple image
    image = ContentImageInline(
        data=base64.b64encode(b"fake image data").decode("utf-8"),
        image_content_type="image/png",
    )

    result = ContentToolResult(value=image, request=request)
    turn = UserTurn([result])

    # Should have 4 items: result with placeholder, open tag, image, close tag
    assert len(turn.contents) == 4

    # First item: modified tool result with placeholder
    assert isinstance(turn.contents[0], ContentToolResult)
    assert 'See <tool-content call-id="call_456"> below.' == turn.contents[0].value

    # Second item: opening XML tag
    assert isinstance(turn.contents[1], ContentText)
    assert turn.contents[1].text == '<tool-content call-id="call_456">'

    # Third item: the actual image
    assert isinstance(turn.contents[2], ContentImageInline)
    assert turn.contents[2] is image

    # Fourth item: closing XML tag
    assert isinstance(turn.contents[3], ContentText)
    assert turn.contents[3].text == "</tool-content>"


def test_expand_turn_tool_result_with_remote_image():
    """Test expansion of tool result containing a remote image URL."""
    request = ContentToolRequest(
        id="call_789",
        name="fetch_image",
        arguments={},
        tool=ToolInfo(name="fetch_image", description="", parameters={}),
    )

    image = ContentImageRemote(url="https://example.com/image.png")

    result = ContentToolResult(value=image, request=request)
    turn = UserTurn([result])

    assert len(turn.contents) == 4
    assert isinstance(turn.contents[0], ContentToolResult)
    assert isinstance(turn.contents[2], ContentImageRemote)
    assert turn.contents[2].url == "https://example.com/image.png"


def test_expand_turn_tool_result_with_pdf():
    """Test expansion of tool result containing a PDF."""
    request = ContentToolRequest(
        id="call_pdf",
        name="get_pdf",
        arguments={},
        tool=ToolInfo(name="get_pdf", description="", parameters={}),
    )

    pdf = ContentPDF(data=b"fake pdf data", filename="test.pdf")

    result = ContentToolResult(value=pdf, request=request)
    turn = UserTurn([result])

    assert len(turn.contents) == 4
    assert isinstance(turn.contents[0], ContentToolResult)
    assert 'See <tool-content call-id="call_pdf"> below.' == turn.contents[0].value
    assert isinstance(turn.contents[2], ContentPDF)
    assert turn.contents[2].filename == "test.pdf"


def test_expand_turn_tool_result_with_list_of_images():
    """Test expansion of tool result containing a list of images."""
    request = ContentToolRequest(
        id="call_multi",
        name="get_images",
        arguments={},
        tool=ToolInfo(name="get_images", description="", parameters={}),
    )

    image1 = ContentImageInline(
        data=base64.b64encode(b"image 1").decode("utf-8"),
        image_content_type="image/png",
    )
    image2 = ContentImageInline(
        data=base64.b64encode(b"image 2").decode("utf-8"),
        image_content_type="image/jpeg",
    )

    result = ContentToolResult(value=[image1, image2], request=request)
    turn = UserTurn([result])

    # Should have: result, open wrapper, (open tag, image1, close tag),
    # (open tag, image2, close tag), close wrapper = 9 items
    assert len(turn.contents) == 9

    # First item: modified tool result
    assert isinstance(turn.contents[0], ContentToolResult)
    assert 'See <tool-contents call-id="call_multi"> below.' == turn.contents[0].value

    # Second item: opening wrapper tag
    assert isinstance(turn.contents[1], ContentText)
    assert turn.contents[1].text == '<tool-contents call-id="call_multi">'

    # Items for first image
    assert isinstance(turn.contents[2], ContentText)
    assert turn.contents[2].text == "<tool-content>"
    assert isinstance(turn.contents[3], ContentImageInline)
    assert turn.contents[3] is image1
    assert isinstance(turn.contents[4], ContentText)
    assert turn.contents[4].text == "</tool-content>"

    # Items for second image
    assert isinstance(turn.contents[5], ContentText)
    assert turn.contents[5].text == "<tool-content>"
    assert isinstance(turn.contents[6], ContentImageInline)
    assert turn.contents[6] is image2
    assert isinstance(turn.contents[7], ContentText)
    assert turn.contents[7].text == "</tool-content>"

    # Final closing wrapper
    assert isinstance(turn.contents[8], ContentText)
    assert turn.contents[8].text == "</tool-contents>"


def test_expand_turn_multiple_tool_results():
    """Test turn with multiple tool results, some needing expansion."""
    request1 = ContentToolRequest(
        id="call_1",
        name="tool1",
        arguments={},
        tool=ToolInfo(name="tool1", description="", parameters={}),
    )
    request2 = ContentToolRequest(
        id="call_2",
        name="tool2",
        arguments={},
        tool=ToolInfo(name="tool2", description="", parameters={}),
    )

    result1 = ContentToolResult(value="plain text", request=request1)

    image = ContentImageInline(
        data=base64.b64encode(b"image").decode("utf-8"),
        image_content_type="image/png",
    )
    result2 = ContentToolResult(value=image, request=request2)

    turn = UserTurn([result1, result2])

    # First result unchanged (1 item)
    # Second result expanded (4 items)
    # Total: 5 items
    assert len(turn.contents) == 5

    # First result should be unchanged and come first
    assert isinstance(turn.contents[0], ContentToolResult)
    assert turn.contents[0].value == "plain text"

    # Second result should be expanded
    assert isinstance(turn.contents[1], ContentToolResult)
    assert 'See <tool-content call-id="call_2"> below.' == turn.contents[1].value
    assert isinstance(turn.contents[3], ContentImageInline)


def test_expand_turn_preserves_other_content():
    """Test that non-tool-result content is preserved."""
    request = ContentToolRequest(
        id="call_x",
        name="toolx",
        arguments={},
        tool=ToolInfo(name="toolx", description="", parameters={}),
    )

    text1 = "Before"
    image = ContentImageInline(
        data=base64.b64encode(b"img").decode("utf-8"),
        image_content_type="image/png",
    )
    result = ContentToolResult(value=image, request=request)
    text2 = ContentText(text="After")

    turn = UserTurn([text1, result, text2])

    # Tool result is expanded to 4 items (result, open tag, image, close tag)
    # Only the ContentToolResult itself is reordered to the front
    # Total: 6 items
    assert len(turn.contents) == 6

    assert isinstance(turn.contents[0], ContentToolResult)
    assert 'See <tool-content call-id="call_x"> below.' == turn.contents[0].value
    assert isinstance(turn.contents[1], ContentText)
    assert turn.contents[1].text == "Before"
    assert isinstance(turn.contents[2], ContentText)
    assert turn.contents[2].text == '<tool-content call-id="call_x">'
    assert isinstance(turn.contents[3], ContentImageInline)
    assert isinstance(turn.contents[4], ContentText)
    assert turn.contents[4].text == "</tool-content>"
    assert isinstance(turn.contents[5], ContentText)
    assert turn.contents[5].text == "After"


def test_expand_turn_empty_contents():
    """Test that turns with empty contents are handled gracefully."""
    turn = UserTurn([])

    assert len(turn.contents) == 0


def tool_request(id: str = "call_1", name: str = "repl") -> ContentToolRequest:
    return ContentToolRequest(
        id=id,
        name=name,
        arguments={},
        tool=ToolInfo(name=name, description="", parameters={}),
    )


def test_merge_joins_multiple_text_results():
    """Several text parts for one call arrive as one newline-joined result."""
    request = tool_request()
    turn = UserTurn(
        [
            ContentToolResult(value="first", request=request),
            ContentToolResult(value="second", request=request),
        ]
    )

    assert len(turn.contents) == 1
    assert isinstance(turn.contents[0], ContentToolResult)
    assert turn.contents[0].value == "first\nsecond"


def test_merge_leaves_single_result_untouched():
    request = tool_request()
    result = ContentToolResult(value="only", request=request)

    turn = UserTurn([result])

    assert len(turn.contents) == 1
    assert turn.contents[0] is result


def test_merge_keeps_distinct_requests_separate():
    req_a = tool_request(id="call_a")
    req_b = tool_request(id="call_b")
    turn = UserTurn(
        [
            ContentToolResult(value="a1", request=req_a),
            ContentToolResult(value="b1", request=req_b),
            ContentToolResult(value="a2", request=req_a),
        ]
    )

    results = [c for c in turn.contents if isinstance(c, ContentToolResult)]
    assert len(results) == 2
    # The merged result keeps the position of the group's first result.
    assert results[0].id == "call_a"
    assert results[0].value == "a1\na2"
    assert results[1].id == "call_b"
    assert results[1].value == "b1"


def test_merge_propagates_error_from_any_part():
    """A failure anywhere in the group must not be masked by sibling output."""
    request = tool_request()
    turn = UserTurn(
        [
            ContentToolResult(value="partial output", request=request),
            ContentToolResult(value=None, error=RuntimeError("boom"), request=request),
        ]
    )

    assert len(turn.contents) == 1
    result = turn.contents[0]
    assert isinstance(result, ContentToolResult)
    assert result.error is not None
    assert "boom" in str(result.error)


def test_merge_mixed_text_and_image_is_expanded():
    """Text parts plus an image unroll into pointer + separate contents."""
    request = tool_request()
    image = ContentImageInline(
        data=base64.b64encode(b"png").decode("utf-8"),
        image_content_type="image/png",
    )
    turn = UserTurn(
        [
            ContentToolResult(value="stdout", request=request),
            ContentToolResult(value=image, request=request),
        ]
    )

    results = [c for c in turn.contents if isinstance(c, ContentToolResult)]
    assert len(results) == 1
    assert 'See <tool-contents call-id="call_1"> below.' == results[0].value

    # Neither the text nor the image is dropped.
    assert any(isinstance(c, ContentText) and c.text == "stdout" for c in turn.contents)
    assert any(isinstance(c, ContentImageInline) for c in turn.contents)


def test_merge_keeps_non_result_content_in_relative_order():
    request = tool_request()
    turn = UserTurn(
        [
            ContentText(text="Before"),
            ContentToolResult(value="one", request=request),
            ContentToolResult(value="two", request=request),
            ContentText(text="After"),
        ]
    )

    assert len(turn.contents) == 3
    assert isinstance(turn.contents[0], ContentToolResult)
    assert turn.contents[0].value == "one\ntwo"
    assert isinstance(turn.contents[1], ContentText)
    assert turn.contents[1].text == "Before"
    assert isinstance(turn.contents[2], ContentText)
    assert turn.contents[2].text == "After"


def test_merge_ignores_results_without_a_request():
    """A result with no request has no id to group on, so it passes through."""
    turn = UserTurn(
        [
            ContentToolResult(value="orphan a"),
            ContentToolResult(value="orphan b"),
        ]
    )

    results = [c for c in turn.contents if isinstance(c, ContentToolResult)]
    assert len(results) == 2


def test_expanded_content_does_not_push_a_later_result_out_of_position():
    """Anthropic requires every tool_result at the start of the user message."""
    request_a = tool_request(id="call_a", name="plot")
    request_b = tool_request(id="call_b", name="lookup")

    image = ContentImageInline(
        data=base64.b64encode(b"img").decode("utf-8"),
        image_content_type="image/png",
    )

    turn = UserTurn(
        [
            ContentToolResult(value=image, request=request_a),
            ContentToolResult(value="text b", request=request_b),
        ]
    )

    results = [c for c in turn.contents if isinstance(c, ContentToolResult)]
    assert len(results) == 2
    assert turn.contents[:2] == results


def test_tool_results_precede_user_authored_content():
    request = tool_request(id="call_x", name="plot")

    image = ContentImageInline(
        data=base64.b64encode(b"img").decode("utf-8"),
        image_content_type="image/png",
    )

    turn = UserTurn(
        [
            ContentText(text="Before"),
            ContentToolResult(value=image, request=request),
            ContentText(text="After"),
        ]
    )

    assert isinstance(turn.contents[0], ContentToolResult)
    # Non-result content keeps its own relative order behind the results.
    texts = [c.text for c in turn.contents if isinstance(c, ContentText)]
    assert texts.index("Before") < texts.index("After")


def test_ordering_is_untouched_when_there_are_no_tool_results():
    turn = UserTurn([ContentText(text="one"), ContentText(text="two")])

    assert [c.text for c in turn.contents if isinstance(c, ContentText)] == [
        "one",
        "two",
    ]


def test_anthropic_puts_every_tool_result_block_first():
    request_a = tool_request(id="call_a", name="plot")
    request_b = tool_request(id="call_b", name="lookup")

    image = ContentImageInline(
        data=base64.b64encode(b"img").decode("utf-8"),
        image_content_type="image/png",
    )

    turns = [
        UserTurn("go"),
        AssistantTurn([request_a, request_b]),
        UserTurn(
            [
                ContentToolResult(value=image, request=request_a),
                ContentToolResult(value="text b", request=request_b),
            ]
        ),
    ]

    messages = AnthropicProvider(
        model="claude-sonnet-4-5", api_key="dummy", kwargs=None
    )._as_message_params(turns)

    types = [
        cast(dict[str, Any], b).get("type")
        for b in messages[-1]["content"]
        if isinstance(b, dict)
    ]

    assert types[:2] == ["tool_result", "tool_result"]
    assert "tool_result" not in types[2:]


def test_merge_expand_and_reorder_together():
    """One turn exercising all three passes: a call whose parts get combined
    and expanded, a plain call, and interstitial user-authored content."""
    request_a = tool_request(id="call_a", name="plot")
    request_b = tool_request(id="call_b", name="lookup")

    image = ContentImageInline(
        data=base64.b64encode(b"img").decode("utf-8"),
        image_content_type="image/png",
    )

    turns = [
        UserTurn("go"),
        AssistantTurn([request_a, request_b]),
        UserTurn(
            [
                ContentText(text="Before"),
                ContentToolResult(value="here is the plot", request=request_a),
                ContentToolResult(value=image, request=request_a),
                ContentText(text="Between"),
                ContentToolResult(value="text b", request=request_b),
                ContentText(text="After"),
            ]
        ),
    ]

    messages = AnthropicProvider(
        model="claude-sonnet-4-5", api_key="dummy", kwargs=None
    )._as_message_params(turns)

    blocks = [
        cast(dict[str, Any], b)
        for b in messages[-1]["content"]
        if isinstance(b, dict)
    ]
    types = [b["type"] for b in blocks]

    # Both tool_result blocks come first, before any other block.
    assert types[:2] == ["tool_result", "tool_result"]
    assert "tool_result" not in types[2:]
    assert {b["tool_use_id"] for b in blocks if b["type"] == "tool_result"} == {
        "call_a",
        "call_b",
    }

    # The image from the merged+expanded call survives as a real block.
    assert types.count("image") == 1

    # The merged call's text and the interstitial content are all present,
    # and the non-result content keeps its own relative order.
    texts = [b["text"] for b in blocks if b["type"] == "text"]
    assert "here is the plot" in texts
    assert texts.index("Before") < texts.index("Between") < texts.index("After")
