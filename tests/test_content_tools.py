from typing import Union

import pytest

from chatlas import ChatOpenAI
from chatlas.types import ContentToolRequest, ContentToolResult


def test_register_tool():
    chat = ChatOpenAI()

    # -------------------------

    def add(x: int, y: int) -> int:
        return x + y

    chat.register_tool(add)

    assert len(chat._tools) == 1
    tool = chat._tools["add"]
    assert tool.name == "add"
    assert tool.func == add
    assert tool.schema["function"]["name"] == "add"
    assert "description" in tool.schema["function"]
    assert tool.schema["function"]["description"] == ""
    assert "parameters" in tool.schema["function"]
    assert tool.schema["function"]["parameters"] == {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
        },
        "required": ["x", "y"],
    }


def test_register_tool_with_complex_parameters():
    chat = ChatOpenAI()

    def foo(
        x: list[tuple[str, float, bool]],
        y: Union[int, None] = None,
        z: Union[dict[str, str], None] = None,
    ):
        """Dummy tool for testing parameter JSON schema."""
        pass

    chat.register_tool(foo)

    assert len(chat._tools) == 1
    tool = chat._tools["foo"]
    assert tool.name == "foo"
    assert tool.func == foo
    assert tool.schema["function"]["name"] == "foo"
    assert "description" in tool.schema["function"]
    assert (
        tool.schema["function"]["description"]
        == "Dummy tool for testing parameter JSON schema."
    )
    assert "parameters" in tool.schema["function"]
    assert tool.schema["function"]["parameters"] == {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "x": {
                "type": "array",
                "items": {
                    "type": "array",
                    "maxItems": 3,
                    "minItems": 3,
                    "prefixItems": [
                        {"type": "string"},
                        {"type": "number"},
                        {"type": "boolean"},
                    ],
                },
            },
            "y": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "null"},
                ],
            },
            "z": {
                "anyOf": [
                    {
                        "additionalProperties": {
                            "type": "string",
                        },
                        "type": "object",
                    },
                    {
                        "type": "null",
                    },
                ],
            },
        },
        "required": ["x", "y", "z"],
    }


def test_invoke_tool_returns_tool_result():
    chat = ChatOpenAI()

    def tool():
        return 1

    chat.register_tool(tool)

    res = chat._invoke_tool_request(
        ContentToolRequest(id="x", name="tool", arguments={})
    )
    assert isinstance(res, ContentToolResult)
    assert res.id == "x"
    assert res.error is None
    assert res.result.assistant == "1"

    res = chat._invoke_tool_request(
        ContentToolRequest(id="x", name="tool", arguments={"x": 1})
    )
    assert isinstance(res, ContentToolResult)
    assert res.id == "x"
    assert res.error is not None
    assert "got an unexpected keyword argument" in res.error
    assert res.result is None

    res = chat._invoke_tool_request(
        ContentToolRequest(id="x", name="foo", arguments={"x": 1})
    )
    assert isinstance(res, ContentToolResult)
    assert res.id == "x"
    assert res.error == "Unknown tool"
    assert res.result is None


@pytest.mark.asyncio
async def test_invoke_tool_returns_tool_result_async():
    chat = ChatOpenAI()

    async def tool():
        return 1

    chat.register_tool(tool)

    res = await chat._invoke_tool_request_async(
        ContentToolRequest(id="x", name="tool", arguments={})
    )
    assert isinstance(res, ContentToolResult)
    assert res.id == "x"
    assert res.error is None
    assert res.result.assistant == "1"

    res = await chat._invoke_tool_request_async(
        ContentToolRequest(id="x", name="tool", arguments={"x": 1})
    )
    assert isinstance(res, ContentToolResult)
    assert res.id == "x"
    assert res.error is not None
    assert "got an unexpected keyword argument" in res.error
    assert res.result is None

    res = await chat._invoke_tool_request_async(
        ContentToolRequest(id="x", name="foo", arguments={"x": 1})
    )
    assert isinstance(res, ContentToolResult)
    assert res.id == "x"
    assert res.error == "Unknown tool"
    assert res.result is None
