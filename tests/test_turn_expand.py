"""Tests for turn content expansion with images and PDFs in tool results."""

import base64

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


def test_expand_turn_tool_result_with_mixed_list():
    """Test that lists with non-Content items are not expanded."""
    request = ContentToolRequest(
        id="call_mixed",
        name="mixed_tool",
        arguments={},
        tool=ToolInfo(name="mixed_tool", description="", parameters={}),
    )

    # Mixed list: image and string
    image = ContentImageInline(
        data=base64.b64encode(b"image").decode("utf-8"),
        image_content_type="image/png",
    )
    result = ContentToolResult(value=[image, "not content"], request=request)
    turn = UserTurn([result])

    # Should not be expanded since the list contains non-Content items
    assert len(turn.contents) == 1
    assert isinstance(turn.contents[0], ContentToolResult)
    assert isinstance(turn.contents[0].value, list)


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

    text1 = ContentText(text="Before")
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

    assert isinstance(turn.contents[0], ContentText)
    assert turn.contents[0].text == "Before"
    assert isinstance(turn.contents[1], ContentToolResult)
    assert 'See <tool-content call-id="call_x"> below.' == turn.contents[1].value
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
