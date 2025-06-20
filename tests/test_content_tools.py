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


@pytest.mark.filterwarnings("ignore")
def test_invoke_tool_returns_tool_result():
    chat = ChatOpenAI()

    def tool():
        return 1

    chat.register_tool(tool)

    def new_tool_request(
        name: str = "tool",
        args: Optional[dict[str, Any]] = None,
    ):
        return ContentToolRequest(
            id="id",
            name=name,
            arguments=args or {},
        )

    req1 = new_tool_request()
    results = list(chat._invoke_tool(req1))
    assert len(results) == 1
    res = results[0]
    assert isinstance(res, ContentToolResult)
    assert res.id == "id"
    assert res.error is None
    assert res.value == 1
    assert res.request == req1
    assert res.id == req1.id
    assert res.name == req1.name
    assert res.arguments == req1.arguments

    req2 = new_tool_request(args={"x": 1})
    results = list(chat._invoke_tool(req2))
    assert len(results) == 1
    res = results[0]
    assert isinstance(res, ContentToolResult)
    assert res.id == "id"
    assert res.error is not None
    assert "got an unexpected keyword argument" in str(res.error)
    assert res.value is None
    assert res.request == req2
    assert res.id == req2.id
    assert res.name == req2.name
    assert res.arguments == req2.arguments

    req3 = new_tool_request(
        name="foo",
        args={"x": 1},
    )
    results = list(chat._invoke_tool(req3))
    assert len(results) == 1
    res = results[0]
    assert isinstance(res, ContentToolResult)
    assert res.id == "id"
    assert "Unknown tool" in str(res.error)
    assert res.value is None
    assert res.request == req3
    assert res.id == req3.id
    assert res.name == req3.name
    assert res.arguments == req3.arguments


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore")
async def test_invoke_tool_returns_tool_result_async():
    chat = ChatOpenAI()

    async def tool():
        return 1

    chat.register_tool(tool)

    def new_tool_request(
        name: str = "tool",
        args: Optional[dict[str, Any]] = None,
    ):
        return ContentToolRequest(
            id="id",
            name=name,
            arguments=args or {},
        )

    req1 = new_tool_request()
    results = []
    async for result in chat._invoke_tool_async(req1):
        results.append(result)
    assert len(results) == 1
    res = results[0]
    assert isinstance(res, ContentToolResult)
    assert res.id == "id"
    assert res.error is None
    assert res.value == 1
    assert res.request == req1
    assert res.id == req1.id
    assert res.name == req1.name
    assert res.arguments == req1.arguments

    req2 = new_tool_request(args={"x": 1})
    results = []
    async for result in chat._invoke_tool_async(req2):
        results.append(result)
    assert len(results) == 1
    res = results[0]
    assert isinstance(res, ContentToolResult)
    assert res.id == "id"
    assert res.error is not None
    assert "got an unexpected keyword argument" in str(res.error)
    assert res.value is None
    assert res.request == req2
    assert res.id == req2.id
    assert res.name == req2.name
    assert res.arguments == req2.arguments

    req3 = new_tool_request(
        name="foo",
        args={"x": 1},
    )
    results = []
    async for result in chat._invoke_tool_async(req3):
        results.append(result)
    assert len(results) == 1
    res = results[0]
    assert isinstance(res, ContentToolResult)
    assert res.id == "id"
    assert "Unknown tool" in str(res.error)
    assert res.value is None
    assert res.request == req3
    assert res.id == req3.id
    assert res.name == req3.name
    assert res.arguments == req3.arguments


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

    results = list(chat._invoke_tool(req))
    assert len(results) == 1
    res = results[0]
    assert isinstance(res, CustomResult)
    assert res.id == "id"
    assert res.error is None
    assert res.value == 1
    assert res.extra == {"foo": "bar"}
    assert res.request == req
    assert res.id == req.id
    assert res.name == req.name
    assert res.arguments == req.arguments

    results_err = list(chat._invoke_tool(req_err))
    assert len(results_err) == 1
    res_err = results_err[0]
    assert isinstance(res_err, CustomResult)
    assert res_err.id == "id"
    assert res_err.error is not None
    assert str(res_err.error) == "foo"
    assert res_err.value is None
    assert res_err.extra == {"foo": "bar"}
    assert res_err.request == req_err
    assert res_err.id == req_err.id
    assert res_err.name == req_err.name
    assert res_err.arguments == req_err.arguments


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

    results = []
    async for result in chat._invoke_tool_async(req):
        results.append(result)
    assert len(results) == 1
    res = results[0]
    assert isinstance(res, CustomResult)
    assert res.id == "id"
    assert res.error is None
    assert res.value == 1
    assert res.extra == {"foo": "bar"}
    assert res.request == req
    assert res.id == req.id
    assert res.name == req.name
    assert res.arguments == req.arguments

    results_err = []
    async for result in chat._invoke_tool_async(req_err):
        results_err.append(result)
    assert len(results_err) == 1
    res_err = results_err[0]
    assert isinstance(res_err, CustomResult)
    assert res_err.id == "id"
    assert res_err.error is not None
    assert str(res_err.error) == "foo"
    assert res_err.value is None
    assert res_err.extra == {"foo": "bar"}
    assert res_err.request == req_err
    assert res_err.id == req_err.id
    assert res_err.name == req_err.name
    assert res_err.arguments == req_err.arguments
