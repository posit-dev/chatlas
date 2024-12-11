from typing import Union

import pytest
from chatlas import ChatOpenAI
from chatlas.types import ContentToolResult


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

    res = chat._invoke_tool(tool, {}, id_="x")
    assert isinstance(res, ContentToolResult)
    assert res.id == "x"
    assert res.error is None
    assert res.value == 1

    res = chat._invoke_tool(tool, {"x": 1}, id_="x")
    assert isinstance(res, ContentToolResult)
    assert res.id == "x"
    assert res.error is not None
    assert "got an unexpected keyword argument" in res.error
    assert res.value is None

    res = chat._invoke_tool(None, {"x": 1}, id_="x")
    assert isinstance(res, ContentToolResult)
    assert res.id == "x"
    assert res.error == "Unknown tool"
    assert res.value is None


@pytest.mark.asyncio
async def test_invoke_tool_returns_tool_result_async():
    chat = ChatOpenAI()

    async def tool():
        return 1

    res = await chat._invoke_tool_async(tool, {}, id_="x")
    assert isinstance(res, ContentToolResult)
    assert res.id == "x"
    assert res.error is None
    assert res.value == 1

    res = await chat._invoke_tool_async(tool, {"x": 1}, id_="x")
    assert isinstance(res, ContentToolResult)
    assert res.id == "x"
    assert res.error is not None
    assert "got an unexpected keyword argument" in res.error
    assert res.value is None

    res = await chat._invoke_tool_async(None, {"x": 1}, id_="x")
    assert isinstance(res, ContentToolResult)
    assert res.id == "x"
    assert res.error == "Unknown tool"
    assert res.value is None
