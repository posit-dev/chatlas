import asyncio
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from chatlas import ChatOpenAI
from chatlas._tools import Tool

try:
    import mcp  # noqa: F401
except ImportError:
    pytest.skip("Skipping MCP tests", allow_module_level=True)

# Directory where MCP server implementations are located
MCP_SERVER_DIR = Path(__file__).parent / "mcp_servers"

# Allow port to be set via environment variable
# (MCP server implementations should listen to this environment variable)
ENV_VARS = os.environ.copy()
ENV_VARS["MCP_PORT"] = "8081"
SERVER_URL = f"http://localhost:{ENV_VARS['MCP_PORT']}/mcp"


@asynccontextmanager
async def sse_mcp_server(server_file: str):
    full_path = str(MCP_SERVER_DIR / server_file)

    process = subprocess.Popen(
        args=[sys.executable, full_path],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        env=ENV_VARS,
    )

    # Throw if the process fails to start
    if process.returncode is not None:
        raise RuntimeError(f"Failed to start MCP server: {process.returncode}")

    async with httpx.AsyncClient() as client:
        timeout = 10  # seconds
        start_time = asyncio.get_event_loop().time()
        while True:
            try:
                await client.get(f"http://localhost:{ENV_VARS['MCP_PORT']}")
                break
            except httpx.ConnectError:
                if asyncio.get_event_loop().time() - start_time > timeout:
                    process.kill()
                    process.wait()
                    raise TimeoutError("Failed to connect to MCP server")
                await asyncio.sleep(0.1)

    try:
        yield
    finally:
        process.kill()
        process.wait()


@pytest.mark.asyncio
async def test_register_sse_mcp_server():
    chat = ChatOpenAI()

    async with sse_mcp_server("sse_add.py"):
        cleanup = await chat.register_mcp_tools_http_stream(
            name="test",
            url=SERVER_URL,
        )

        assert "test" in chat._mcp_exit_stacks
        assert len(chat._tools) == 1
        tool = chat._tools["add"]
        assert tool.name == "add"
        func = tool.schema["function"]
        assert func["name"] == "add"
        assert func.get("description") == "Add two numbers."
        assert func.get("parameters") == {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
            },
            "required": ["x", "y"],
        }

        await cleanup()


@pytest.mark.asyncio
async def test_register_stdio_mcp_server():
    chat = ChatOpenAI()

    cleanup = await chat.register_mcp_tools_stdio(
        name="test",
        command=sys.executable,
        args=[str(MCP_SERVER_DIR / "stdio_subtract_multiply.py")],
        exclude_tools=["subtract"],
    )

    assert "test" in chat._mcp_exit_stacks
    assert len(chat._tools) == 1
    tool = chat._tools["multiply"]
    assert tool.name == "multiply"
    func = tool.schema["function"]
    assert func["name"] == "multiply"
    assert func.get("description") == "Multiply two numbers."
    assert func.get("parameters") == {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    }

    await cleanup()


@pytest.mark.asyncio
async def test_register_multiple_mcp_servers():
    chat = ChatOpenAI()

    await chat.register_mcp_tools_stdio(
        name="stdio_test",
        command=sys.executable,
        args=[str(MCP_SERVER_DIR / "stdio_subtract_multiply.py")],
        include_tools=["subtract"],
    )

    async with sse_mcp_server("sse_add.py"):
        cleanup = await chat.register_mcp_tools_http_stream(
            name="sse_test",
            url=SERVER_URL,
        )

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
            ),
        }

        assert len(chat._tools) == len(expected_tools)
        for tool_name, expected_tool in expected_tools.items():
            tool = chat._tools[tool_name]
            assert tool.name == expected_tool.name
            func = tool.schema["function"]
            assert func["name"] == expected_tool.schema["function"]["name"]
            assert func.get("description") == expected_tool.schema["function"].get(
                "description", "N/A"
            )
            assert func.get("parameters") == expected_tool.schema["function"].get(
                "parameters", "N/A"
            )

        await cleanup()


@pytest.mark.asyncio
async def test_call_sse_mcp_tool():
    chat = ChatOpenAI(system_prompt="Be very terse, not even punctuation.")

    async with sse_mcp_server("sse_current_date.py"):
        cleanup = await chat.register_mcp_tools_http_stream(
            name="test",
            url=SERVER_URL,
        )

        response = await chat.chat_async(
            "What's the current date in YMD format?", stream=True
        )
        assert "2024-01-01" in await response.get_content()

        with pytest.raises(Exception, match="async tools in a synchronous chat"):
            str(chat.chat("Great. Do it again.", stream=True))

        await cleanup()


@pytest.mark.asyncio
async def test_call_stdio_mcp_tool():
    chat = ChatOpenAI(system_prompt="Be very terse, not even punctuation.")

    cleanup = await chat.register_mcp_tools_stdio(
        name="stdio_test",
        command=sys.executable,
        args=[str(MCP_SERVER_DIR / "stdio_current_date.py")],
    )

    response = await chat.chat_async(
        "What's the current date in YMD format?", stream=True
    )

    assert "2024-01-01" in await response.get_content()

    with pytest.raises(Exception, match="async tools in a synchronous chat"):
        str(chat.chat("Great. Do it again.", stream=True))

    await cleanup()
