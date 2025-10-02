import pytest

from chatlas._content import (
    ContentToolRequest,
    ContentText,
    ContentImageRemote,
    ContentJson,
)
from chatlas._inspect import (
    content_to_inspect,
    content_to_chatlas,
    turn_as_messages,
)
from chatlas._turn import Turn

from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    ChatMessageAssistant,
)
from inspect_ai.tool import (
    ContentText as IContentText,
    ContentImage,
    ContentData,
)


def test_inspect_helpers_require_inspect_ai():
    from chatlas._inspect import INSPECT_AVAILABLE

    # If inspect_ai is actually not available, test the error
    if not INSPECT_AVAILABLE:
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
    messages = turn_as_messages(turn, model="gpt-4")

    assert len(messages) == 1
    assert isinstance(messages[0], ChatMessageAssistant)
    assert messages[0].model == "gpt-4"


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


def test_content_to_inspect_tool_request_raises():
    pytest.importorskip("inspect_ai")

    content = ContentToolRequest(id="call_123", name="test_tool", arguments={})

    with pytest.raises(ValueError, match="cannot be directly translated"):
        content_to_inspect(content)
