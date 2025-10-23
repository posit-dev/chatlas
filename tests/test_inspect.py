import pytest

pytest.importorskip("inspect_ai")

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ContentData, ContentImage
from inspect_ai.tool import ContentText as IContentText
from inspect_ai.tool import ToolCall

from chatlas._content import (
    ContentImageRemote,
    ContentJson,
    ContentText,
    ContentToolRequest,
    ContentToolResult,
)
from chatlas._inspect import content_to_chatlas, content_to_inspect, turn_as_messages
from chatlas._turn import Turn


def test_inspect_helpers_require_inspect_ai():
    with pytest.raises(ImportError, match="requires the optional dependency"):
        turn_as_messages(Turn("user", "test"))

    with pytest.raises(ImportError, match="requires the optional dependency"):
        from chatlas._content import ContentText

        content_to_inspect(ContentText(text="test"))

    with pytest.raises(ImportError, match="requires the optional dependency"):
        content_to_chatlas("test")


def test_turn_as_messages_system():
    pytest.importorskip("inspect_ai")

    turn = Turn("system", "You are a helpful assistant.")
    messages = turn_as_messages(turn)

    assert len(messages) == 1
    assert isinstance(messages[0], ChatMessageSystem)
    assert messages[0].content == "You are a helpful assistant."


def test_turn_as_messages_user():
    pytest.importorskip("inspect_ai")

    turn = Turn("user", "Hello!")
    messages = turn_as_messages(turn)

    assert len(messages) == 1
    assert isinstance(messages[0], ChatMessageUser)


def test_turn_as_messages_assistant():
    pytest.importorskip("inspect_ai")

    turn = Turn("assistant", "Hi there!")
    messages = turn_as_messages(turn, model="gpt-5-nano-2025-08-07")

    assert len(messages) == 1
    assert isinstance(messages[0], ChatMessageAssistant)
    assert messages[0].model == "gpt-5-nano-2025-08-07"


def test_content_to_inspect_text():
    pytest.importorskip("inspect_ai")

    content = ContentText(text="Hello world")
    inspect_content = content_to_inspect(content)

    assert isinstance(inspect_content, IContentText)
    assert inspect_content.text == "Hello world"


def test_content_to_inspect_image_remote():
    pytest.importorskip("inspect_ai")

    content = ContentImageRemote(url="https://example.com/image.jpg", detail="high")
    inspect_content = content_to_inspect(content)

    assert isinstance(inspect_content, ContentImage)
    assert inspect_content.image == "https://example.com/image.jpg"
    assert inspect_content.detail == "high"


def test_content_to_inspect_json():
    pytest.importorskip("inspect_ai")

    content = ContentJson(value={"key": "value", "number": 42})
    inspect_content = content_to_inspect(content)

    assert isinstance(inspect_content, ContentData)
    assert inspect_content.data == {"key": "value", "number": 42}


def test_content_to_chatlas_string():
    pytest.importorskip("inspect_ai")

    result = content_to_chatlas("Hello world")

    assert isinstance(result, ContentText)
    assert result.text == "Hello world"


def test_content_to_chatlas_text():
    pytest.importorskip("inspect_ai")

    inspect_content = IContentText(text="Test message")
    result = content_to_chatlas(inspect_content)

    assert isinstance(result, ContentText)
    assert result.text == "Test message"


def test_content_to_chatlas_image_url():
    pytest.importorskip("inspect_ai")

    inspect_content = ContentImage(image="https://example.com/test.jpg", detail="low")
    result = content_to_chatlas(inspect_content)

    assert isinstance(result, ContentImageRemote)
    assert result.url == "https://example.com/test.jpg"
    assert result.detail == "low"


def test_content_to_chatlas_json():
    pytest.importorskip("inspect_ai")

    inspect_content = ContentData(data={"test": "data", "value": 123})
    result = content_to_chatlas(inspect_content)

    assert isinstance(result, ContentJson)
    assert result.value == {"test": "data", "value": 123}


def test_turn_as_messages_assistant_with_tool_requests():
    pytest.importorskip("inspect_ai")

    tool_request1 = ContentToolRequest(
        id="call_123",
        name="get_weather",
        arguments={"city": "San Francisco", "units": "F"},
    )
    tool_request2 = ContentToolRequest(
        id="call_456", name="get_time", arguments={"timezone": "PST"}
    )
    text_content = ContentText(text="Let me check the weather and time for you.")

    turn = Turn("assistant", [text_content, tool_request1, tool_request2])
    messages = turn_as_messages(turn, model="gpt-5-nano-2025-08-07")

    assert len(messages) == 1
    assert isinstance(messages[0], ChatMessageAssistant)
    assert messages[0].model == "gpt-5-nano-2025-08-07"

    assert len(messages[0].content) == 1
    assert isinstance(messages[0].content[0], IContentText)
    assert messages[0].content[0].text == "Let me check the weather and time for you."

    assert len(messages[0].tool_calls) == 2

    assert isinstance(messages[0].tool_calls[0], ToolCall)
    assert messages[0].tool_calls[0].id == "call_123"
    assert messages[0].tool_calls[0].function == "get_weather"
    assert messages[0].tool_calls[0].arguments == {
        "city": "San Francisco",
        "units": "F",
    }

    assert isinstance(messages[0].tool_calls[1], ToolCall)
    assert messages[0].tool_calls[1].id == "call_456"
    assert messages[0].tool_calls[1].function == "get_time"
    assert messages[0].tool_calls[1].arguments == {"timezone": "PST"}


def test_turn_as_messages_user_with_tool_results():
    pytest.importorskip("inspect_ai")

    tool_request1 = ContentToolRequest(
        id="call_123", name="get_weather", arguments={"city": "San Francisco"}
    )
    tool_request2 = ContentToolRequest(
        id="call_456", name="get_time", arguments={"timezone": "PST"}
    )

    tool_result1 = ContentToolResult(value="Sunny, 75°F", request=tool_request1)
    tool_result2 = ContentToolResult(value="11:30 PST", request=tool_request2)
    text_content = ContentText(text="Here are the results:")

    turn = Turn("user", [tool_result1, tool_result2, text_content])
    messages = turn_as_messages(turn)

    assert len(messages) == 3

    assert isinstance(messages[0], ChatMessageTool)
    assert messages[0].tool_call_id == "call_123"
    assert messages[0].content == "Sunny, 75°F"
    assert messages[0].function == "get_weather"

    assert isinstance(messages[1], ChatMessageTool)
    assert messages[1].tool_call_id == "call_456"
    assert messages[1].content == "11:30 PST"
    assert messages[1].function == "get_time"

    assert isinstance(messages[2], ChatMessageUser)
    assert len(messages[2].content) == 1

    assert isinstance(messages[2].content[0], IContentText)
    assert messages[2].content[0].text == "Here are the results:"


def test_turn_as_messages_user_with_only_tool_results():
    pytest.importorskip("inspect_ai")

    tool_request = ContentToolRequest(
        id="call_789", name="calculate", arguments={"x": 5, "y": 3}
    )
    tool_result = ContentToolResult(value=8, request=tool_request)

    turn = Turn("user", [tool_result])
    messages = turn_as_messages(turn)

    assert len(messages) == 1
    assert isinstance(messages[0], ChatMessageTool)
    assert messages[0].tool_call_id == "call_789"
    assert messages[0].content == "8"
    assert messages[0].function == "calculate"


def test_turn_as_messages_assistant_with_only_tool_requests():
    pytest.importorskip("inspect_ai")

    tool_request = ContentToolRequest(
        id="call_999", name="search", arguments={"query": "python tutorials"}
    )

    turn = Turn("assistant", [tool_request])
    messages = turn_as_messages(turn, model="gpt-5-nano-2025-08-07")

    assert len(messages) == 1
    assert isinstance(messages[0], ChatMessageAssistant)
    assert messages[0].model == "gpt-5-nano-2025-08-07"

    assert len(messages[0].content) == 0
    assert len(messages[0].tool_calls) == 1

    assert isinstance(messages[0].tool_calls[0], ToolCall)
    assert messages[0].tool_calls[0].id == "call_999"
    assert messages[0].tool_calls[0].function == "search"
    assert messages[0].tool_calls[0].arguments == {"query": "python tutorials"}
