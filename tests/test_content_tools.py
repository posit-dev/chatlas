import sys

import pytest
from chatlas import ChatOpenAI, Tool
from chatlas.types import ContentToolResult


def test_register_tool():
    chat = ChatOpenAI()

    # -------------------------

    def add(x: int, y: int) -> int:
        return x + y

    chat.register_tool(add)

    assert len(chat.tools) == 1
    tool = chat.tools["add"]
    assert tool.name == "add"
    assert tool.func == add
    assert tool.schema["function"]["name"] == "add"
    assert tool.schema["function"]["description"] == ""
    assert tool.schema["function"]["parameters"] == {
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
        },
        "required": ["x", "y"],
    }

    # -------------------------

    chat.register_tool(
        Tool(
            add,
            name="add2",
            description="Add two numbers.",
            parameter_descriptions={
                "x": "The first number.",
                "y": "The second number.",
            },
        )
    )

    assert len(chat.tools) == 2
    tool = chat.tools["add2"]
    assert tool.name == "add2"
    assert tool.func == add
    assert tool.schema["function"]["name"] == "add2"
    assert tool.schema["function"]["description"] == "Add two numbers."
    assert tool.schema["function"]["parameters"] == {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "The first number."},
            "y": {"type": "integer", "description": "The second number."},
        },
        "required": ["x", "y"],
    }


@pytest.mark.skipif(sys.version_info <= (3, 9), reason="requires Python 3.10 or higher")
def test_register_tool_with_complex_parameters():
    chat = ChatOpenAI()

    def foo(
        x: list[tuple[str, float, bool]],
        y: int | None = None,
        z: dict[str, str] | None = None,
    ):
        """Dummy tool for testing parameter JSON schema."""
        pass

    chat.register_tool(foo)

    assert len(chat.tools) == 1
    tool = chat.tools["foo"]
    assert tool.name == "foo"
    assert tool.func == foo
    assert tool.schema["function"]["name"] == "foo"
    assert (
        tool.schema["function"]["description"]
        == "Dummy tool for testing parameter JSON schema."
    )
    assert tool.schema["function"]["parameters"] == {
        "type": "object",
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
