from contextlib import asynccontextmanager
from pathlib import Path
import sys
from time import sleep
from typing import Callable

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from chatlas import ChatOpenAI, ChatBedrockAnthropic
from chatlas._tools import Tool
import asyncio
import threading
import subprocess

@asynccontextmanager
async def sse_mcp_server(script_path: str):
    process = subprocess.Popen(
        args=[sys.executable, script_path],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
    )

    await asyncio.sleep(1.0)

    # Wait for server to be accessible
    async with httpx.AsyncClient() as client:
        while True:
            try:
                await client.get("http://localhost:8000")
                break
            except httpx.ConnectError:
                await asyncio.sleep(0.1)

    try:
        yield
    finally:
        process.kill()
        process.wait()

def get_resource(resource_name: str) -> str:
    return str(Path(__file__).parent / "resources" / resource_name)

# @asynccontextmanager
# async def sse_mcp_server(tools: list[tuple[Callable, str, str]]):
#     mcp = FastMCP("test", host="localhost", port=8000)
    
#     for tool in tools:
#         mcp.add_tool(tool[0], name=tool[1], description=tool[2])
    
#     def run_mcp():
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         loop.run_until_complete(mcp.run_sse_async())
    
#     # Start MCP server in background thread
#     thread = threading.Thread(target=run_mcp, daemon=True)
#     thread.start()
    
#     # Wait for server to be accessible
#     async with httpx.AsyncClient() as client:
#         while True:
#             try:
#                 await client.get("http://localhost:8000")
#                 break
#             except httpx.ConnectError:
#                 await asyncio.sleep(0.1)
    
#     try:
#         yield
#     finally:
#         thread.join(timeout=1.0)


@pytest.mark.asyncio
async def test_register_sse_mcp_server():
    chat = ChatBedrockAnthropic(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        aws_region="us-east-1",
    )

    async with sse_mcp_server(get_resource("sse_mcp_server_add.py")):
        await chat.register_sse_mcp_server_async("test", "http://localhost:8000/sse",)

        assert "test" in chat._mcp_sessions
        assert len(chat._tools) == 1
        tool = chat._tools["add"]
        assert tool.name == "add"
        assert tool.schema["function"]["name"] == "add"
        assert tool.schema["function"]["description"] == "Add two numbers."
        assert tool.schema["function"]["parameters"] == {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
            },
            "required": ["x", "y"],
        }

        await chat.close_mcp_sessions()


@pytest.mark.asyncio
async def test_register_stdio_mcp_server():
    chat = ChatBedrockAnthropic(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        aws_region="us-east-1",
    )

    await chat.register_stdio_mcp_server_async(
        name="test",
        command=sys.executable,
        args=[get_resource("stdio_mcp_server_subtract_multiply.py")],
        exclude_tools=["subtract"],
    )

    assert "test" in chat._mcp_sessions
    assert len(chat._tools) == 1
    tool = chat._tools["multiply"]
    assert tool.name == "multiply"
    assert tool.schema["function"]["name"] == "multiply"
    assert tool.schema["function"]["description"] == "Multiply two numbers."
    assert tool.schema["function"]["parameters"] == {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    }

    await chat.close_mcp_sessions()

@pytest.mark.asyncio
async def test_register_multiple_mcp_servers():
    chat = ChatBedrockAnthropic(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        aws_region="us-east-1",
    )

    await chat.register_stdio_mcp_server_async(
        name="stdio_test",
        command=sys.executable,
        args=[get_resource("stdio_mcp_server_subtract_multiply.py")],
        include_tools=["subtract"],
    )
    
    async with sse_mcp_server(get_resource("sse_mcp_server_add.py")):
        await chat.register_sse_mcp_server_async("sse_test", "http://localhost:8000/sse")

        expected_tools = {
            "add": Tool(
                func=lambda x: x,
                name="add",
                description="Add two numbers.",
                parameters={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                    },
                    "required": ["x", "y"],
                },
            ),
            "subtract": Tool(
                func=lambda x: x,
                name="subtract",
                description="Subtract two numbers.",
                parameters={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "y": {"type": "integer"},
                        "z": {"type": "integer"},
                    },
                    "required": ["y", "z"],
                },
            )
        }

        assert len(chat._tools) == len(expected_tools)
        for tool_name, expected_tool in expected_tools.items():
            tool = chat._tools[tool_name]
            assert tool.name == expected_tool.name
            assert tool.schema["function"]["name"] == expected_tool.schema["function"]["name"]
            assert tool.schema["function"]["description"] == expected_tool.schema["function"]["description"]
            assert tool.schema["function"]["parameters"] == expected_tool.schema["function"]["parameters"]

        await chat.close_mcp_sessions()

@pytest.mark.asyncio
async def test_call_sse_mcp_tool():
    chat = ChatBedrockAnthropic(
        system_prompt="Be very terse, not even punctuation.",
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        aws_region="us-east-1",
    )
    
    async with sse_mcp_server(get_resource("sse_mcp_server_current_date.py")):
        await chat.register_sse_mcp_server_async("test", "http://localhost:8000/sse")

        response = await chat.chat_async(
            "What's the current date in YMD format?", stream=True
        )
        assert "2024-01-01" in await response.get_content()

        with pytest.raises(Exception, match="async tools in a synchronous chat"):
            str(chat.chat("Great. Do it again.", stream=True))
        
        await chat.close_mcp_sessions()

@pytest.mark.asyncio
async def test_call_stdio_mcp_tool():
    chat = ChatBedrockAnthropic(
        system_prompt="Be very terse, not even punctuation.",
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        aws_region="us-east-1",
    )

    await chat.register_stdio_mcp_server_async(
        name="stdio_test",
        command=sys.executable,
        args=[get_resource("stdio_mcp_server_current_date.py")],
    )

    response = await chat.chat_async(
        "What's the current date in YMD format?", stream=True
    )

    assert "2024-01-01" in await response.get_content()

    with pytest.raises(Exception, match="async tools in a synchronous chat"):
        str(chat.chat("Great. Do it again.", stream=True))
    
    await chat.close_mcp_sessions()
