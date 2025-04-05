from typing import Any, Optional, Union

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

    def new_tool_request(
        name: Optional[str] = "tool",
        args: Optional[dict[str, Any]] = None,
    ):
        return ContentToolRequest(
            id="id",
            name=name,
            arguments=args or {},
        )

    res = chat._invoke_tool(new_tool_request())
    assert isinstance(res, ContentToolResult)
    assert res.id == "id"
    assert res.error is None
    assert res.value == 1

    res = chat._invoke_tool(new_tool_request(args={"x": 1}))
    assert isinstance(res, ContentToolResult)
    assert res.id == "id"
    assert res.error is not None
    assert "got an unexpected keyword argument" in str(res.error)
    assert res.value is None

    res = chat._invoke_tool(
        new_tool_request(
            name="foo",
            args={"x": 1},
        )
    )
    assert isinstance(res, ContentToolResult)
    assert res.id == "id"
    assert str(res.error) == "Unknown tool: foo"
    assert res.value is None


@pytest.mark.asyncio
async def test_invoke_tool_returns_tool_result_async():
    chat = ChatOpenAI()

    async def tool():
        return 1

    chat.register_tool(tool)

    def new_tool_request(
        name: Optional[str] = "tool",
        args: Optional[dict[str, Any]] = None,
    ):
        return ContentToolRequest(
            id="id",
            name=name,
            arguments=args or {},
        )

    res = await chat._invoke_tool_async(new_tool_request())
    assert isinstance(res, ContentToolResult)
    assert res.id == "id"
    assert res.error is None
    assert res.value == 1

    res = await chat._invoke_tool_async(new_tool_request(args={"x": 1}))
    assert isinstance(res, ContentToolResult)
    assert res.id == "id"
    assert res.error is not None
    assert "got an unexpected keyword argument" in str(res.error)
    assert res.value is None

    res = await chat._invoke_tool_async(new_tool_request(name="foo", args={"x": 1}))
    assert isinstance(res, ContentToolResult)
    assert res.id == "id"
    assert str(res.error) == "Unknown tool: foo"
    assert res.value is None


def test_tool_custom_result():
    chat = ChatOpenAI()

    class CustomResult(ContentToolResult):
        pass

    def custom_tool():
        return CustomResult(value=1, extra={"foo": "bar"})

    def custom_tool_err():
        return CustomResult(
            value=None,
            error=Exception("foo"),
            extra={"foo": "bar"},
        )

    chat.register_tool(custom_tool)
    chat.register_tool(custom_tool_err)

    req = ContentToolRequest(
        id="id",
        name="custom_tool",
        arguments={},
    )
    req_err = ContentToolRequest(
        id="id",
        name="custom_tool_err",
        arguments={},
    )

    res = chat._invoke_tool(req)
    assert isinstance(res, CustomResult)
    assert res.id == "id"
    assert res.error is None
    assert res.value == 1
    assert res.extra == {"foo": "bar"}
    assert res.request == req

    res_err = chat._invoke_tool(req_err)
    assert isinstance(res_err, CustomResult)
    assert res_err.id == "id"
    assert res_err.error is not None
    assert str(res_err.error) == "foo"
    assert res_err.value is None
    assert res_err.extra == {"foo": "bar"}
    assert res_err.request == req_err


@pytest.mark.asyncio
async def test_tool_custom_result_async():
    chat = ChatOpenAI()

    class CustomResult(ContentToolResult):
        pass

    async def custom_tool():
        return CustomResult(value=1, extra={"foo": "bar"})

    async def custom_tool_err():
        return CustomResult(
            value=None,
            error=Exception("foo"),
            extra={"foo": "bar"},
        )

    chat.register_tool(custom_tool)
    chat.register_tool(custom_tool_err)

    req = ContentToolRequest(
        id="id",
        name="custom_tool",
        arguments={},
    )

    req_err = ContentToolRequest(
        id="id",
        name="custom_tool_err",
        arguments={},
    )

    res = await chat._invoke_tool_async(req)
    assert isinstance(res, CustomResult)
    assert res.id == "id"
    assert res.error is None
    assert res.value == 1
    assert res.extra == {"foo": "bar"}
    assert res.request == req

    res_err = await chat._invoke_tool_async(req_err)
    assert isinstance(res_err, CustomResult)
    assert res_err.id == "id"
    assert res_err.error is not None
    assert str(res_err.error) == "foo"
    assert res_err.value is None
    assert res_err.extra == {"foo": "bar"}
    assert res_err.request == req_err
