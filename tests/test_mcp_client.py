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
    pytest.skip("MCP package not available", allow_module_level=True)

# Directory where MCP server implementations are located
MCP_SERVER_DIR = Path(__file__).parent / "mcp_servers"

# Allow port to be set via environment variable
# (MCP server implementations should listen to this environment variable)
ENV_VARS = os.environ.copy()
ENV_VARS["MCP_PORT"] = "8081"
SERVER_URL = f"http://localhost:{ENV_VARS['MCP_PORT']}/mcp"


@asynccontextmanager
async def http_mcp_server(server_file: str):
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
async def test_register_http_mcp_server():
    chat = ChatOpenAI()

    async with http_mcp_server("http_add.py"):
        await chat.register_mcp_tools_http_stream_async(
            url=SERVER_URL,
        )

        # Name derived from server info (the server name is "test")
        assert "test" in chat._mcp_manager._mcp_sessions
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

        await chat.cleanup_mcp_tools()


@pytest.mark.asyncio
async def test_register_stdio_mcp_server():
    chat = ChatOpenAI()

    await chat.register_mcp_tools_stdio_async(
        command=sys.executable,
        args=[str(MCP_SERVER_DIR / "stdio_subtract_multiply.py")],
        exclude_tools=["subtract"],
    )

    # Name derived from server info (the server name is "test")
    assert "test" in chat._mcp_manager._mcp_sessions
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

    await chat.cleanup_mcp_tools()


@pytest.mark.asyncio
async def test_register_multiple_mcp_servers():
    chat = ChatOpenAI()

    await chat.register_mcp_tools_stdio_async(
        name="stdio_test",
        command=sys.executable,
        args=[str(MCP_SERVER_DIR / "stdio_subtract_multiply.py")],
        include_tools=["subtract"],
    )

    async with http_mcp_server("http_add.py"):
        await chat.register_mcp_tools_http_stream_async(
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

        await chat.cleanup_mcp_tools()


@pytest.mark.asyncio
async def test_call_http_mcp_tool():
    chat = ChatOpenAI(system_prompt="Be very terse, not even punctuation.")

    async with http_mcp_server("http_current_date.py"):
        await chat.register_mcp_tools_http_stream_async(
            name="test",
            url=SERVER_URL,
        )

        response = await chat.chat_async(
            "What's the current date in YMD format?", stream=True
        )
        assert "2024-01-01" in await response.get_content()

        with pytest.raises(Exception, match="async tools in a synchronous chat"):
            str(chat.chat("Great. Do it again.", stream=True))

        await chat.cleanup_mcp_tools()


@pytest.mark.asyncio
async def test_call_stdio_mcp_tool():
    chat = ChatOpenAI(system_prompt="Be very terse, not even punctuation.")

    await chat.register_mcp_tools_stdio_async(
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

    await chat.cleanup_mcp_tools()


class TestMCPErrorHandling:
    """Test error handling for MCP functionality."""

    @pytest.mark.asyncio
    async def test_register_duplicate_session_name_error(self):
        """Test that registering with duplicate session name raises error."""
        chat = ChatOpenAI()

        await chat.register_mcp_tools_stdio_async(
            name="test",
            command=sys.executable,
            args=[str(MCP_SERVER_DIR / "stdio_subtract_multiply.py")],
        )

        with pytest.raises(RuntimeError, match="Failed to register tools"):
            await chat.register_mcp_tools_stdio_async(
                name="test",  # Same name
                command=sys.executable,
                args=[str(MCP_SERVER_DIR / "stdio_current_date.py")],
            )

        await chat.cleanup_mcp_tools()

    @pytest.mark.asyncio
    async def test_register_invalid_command_error(self):
        """Test error handling for invalid MCP server command."""
        chat = ChatOpenAI()

        with pytest.raises(RuntimeError, match="Failed to register tools"):
            await chat.register_mcp_tools_stdio_async(
                command="nonexistent_command",
                args=["invalid", "args"],
            )

    @pytest.mark.asyncio
    async def test_register_both_include_exclude_error(self):
        """Test error when both include_tools and exclude_tools are specified."""
        chat = ChatOpenAI()

        with pytest.raises(RuntimeError, match="Failed to register tools"):
            await chat.register_mcp_tools_stdio_async(
                command=sys.executable,
                args=[str(MCP_SERVER_DIR / "stdio_subtract_multiply.py")],
                include_tools=["subtract"],
                exclude_tools=["multiply"],
            )

    @pytest.mark.asyncio
    async def test_register_overlapping_tool_names_error(self):
        """Test error when MCP tools overlap with existing tool names."""
        chat = ChatOpenAI()

        # Register a regular tool first
        def add(x: int, y: int) -> int:
            return x + y

        chat.register_tool(add)

        # Try to register MCP server with overlapping tool name
        async with http_mcp_server("http_add.py"):
            with pytest.raises(
                ValueError, match="following tools are already registered: {'add'}"
            ):
                await chat.register_mcp_tools_http_stream_async(
                    url=SERVER_URL,
                )

    @pytest.mark.asyncio
    async def test_register_overlapping_tools_with_namespace(self):
        """Test that namespace prevents tool name collisions."""
        chat = ChatOpenAI()

        # Register a regular tool first
        def add(x: int, y: int) -> int:
            return x + y

        chat.register_tool(add)

        # Register MCP server with namespace - should work
        async with http_mcp_server("http_add.py"):
            await chat.register_mcp_tools_http_stream_async(
                url=SERVER_URL, namespace="mcp"
            )

            # Should have both tools with different names
            tools = chat.get_tools()
            tool_names = {tool.name for tool in tools}
            assert tool_names == {"add", "mcp.add"}

            await chat.cleanup_mcp_tools()

    @pytest.mark.asyncio
    async def test_invalid_url_error(self):
        """Test error handling for invalid MCP server URL."""
        chat = ChatOpenAI()

        with pytest.raises(RuntimeError, match="Failed to register tools"):
            await chat.register_mcp_tools_http_stream_async(
                url="http://localhost:99999/invalid",
            )


class TestMCPCleanup:
    """Test MCP cleanup functionality."""

    @pytest.mark.asyncio
    async def test_cleanup_specific_session(self):
        """Test cleaning up a specific MCP session."""
        chat = ChatOpenAI()

        # Register two sessions
        await chat.register_mcp_tools_stdio_async(
            name="session1",
            command=sys.executable,
            args=[str(MCP_SERVER_DIR / "stdio_subtract_multiply.py")],
            include_tools=["subtract"],
        )

        await chat.register_mcp_tools_stdio_async(
            name="session2",
            command=sys.executable,
            args=[str(MCP_SERVER_DIR / "stdio_subtract_multiply.py")],
            include_tools=["multiply"],
        )

        assert len(chat._mcp_manager._mcp_sessions) == 2
        assert len(chat.get_tools()) == 2

        # Clean up only session1
        await chat.cleanup_mcp_tools("session1")

        assert len(chat._mcp_manager._mcp_sessions) == 1
        assert "session1" not in chat._mcp_manager._mcp_sessions
        assert "session2" in chat._mcp_manager._mcp_sessions

        # Only multiply tool should remain
        tools = chat.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "multiply"

        await chat.cleanup_mcp_tools("session2")

    @pytest.mark.asyncio
    async def test_cleanup_all_sessions(self):
        """Test cleaning up all MCP sessions."""
        chat = ChatOpenAI()

        # Register two sessions
        await chat.register_mcp_tools_stdio_async(
            name="session1",
            command=sys.executable,
            args=[str(MCP_SERVER_DIR / "stdio_subtract_multiply.py")],
            include_tools=["subtract"],
        )

        await chat.register_mcp_tools_stdio_async(
            name="session2",
            command=sys.executable,
            args=[str(MCP_SERVER_DIR / "stdio_subtract_multiply.py")],
            include_tools=["multiply"],
        )

        assert len(chat._mcp_manager._mcp_sessions) == 2
        assert len(chat.get_tools()) == 2

        # Clean up all sessions
        await chat.cleanup_mcp_tools()

        assert len(chat._mcp_manager._mcp_sessions) == 0
        assert len(chat.get_tools()) == 0
        assert len(chat._mcp_manager._mcp_sessions) == 0

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_session_warning(self):
        """Test warning when cleaning up non-existent session."""
        chat = ChatOpenAI()

        with pytest.warns(
            UserWarning, match="Cannot close MCP session named 'nonexistent'"
        ):
            await chat.cleanup_mcp_tools("nonexistent")

    # TODO: maybe support this?
    # @pytest.mark.asyncio
    # async def test_cleanup_callback_returned_by_register(self):
    #    """Test that cleanup callback returned by register methods works."""
    #    chat = ChatOpenAI()
    #
    #    cleanup = await chat.register_mcp_tools_stdio_async(
    #        name="test",
    #        command=sys.executable,
    #        args=[str(MCP_SERVER_DIR / "stdio_subtract_multiply.py")],
    #    )
    #
    #    assert len(chat._mcp_sessions) == 1
    #    assert len(chat.get_tools()) == 2
    #
    #    await cleanup()
    #
    #    assert len(chat._mcp_sessions) == 0
    #    assert len(chat.get_tools()) == 0


class TestMCPToolFiltering:
    """Test MCP tool filtering with include_tools and exclude_tools."""

    @pytest.mark.asyncio
    async def test_include_tools_filter(self):
        """Test including only specific tools."""
        chat = ChatOpenAI()

        await chat.register_mcp_tools_stdio_async(
            command=sys.executable,
            args=[str(MCP_SERVER_DIR / "stdio_subtract_multiply.py")],
            include_tools=["subtract"],
        )

        tools = chat.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "subtract"

        await chat.cleanup_mcp_tools()

    @pytest.mark.asyncio
    async def test_exclude_tools_filter(self):
        """Test excluding specific tools."""
        chat = ChatOpenAI()

        await chat.register_mcp_tools_stdio_async(
            command=sys.executable,
            args=[str(MCP_SERVER_DIR / "stdio_subtract_multiply.py")],
            exclude_tools=["subtract"],
        )

        tools = chat.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "multiply"

        await chat.cleanup_mcp_tools()

    @pytest.mark.asyncio
    async def test_include_nonexistent_tool(self):
        """Test including a tool that doesn't exist."""
        chat = ChatOpenAI()

        with pytest.warns(UserWarning, match="did not match"):
            await chat.register_mcp_tools_stdio_async(
                command=sys.executable,
                args=[str(MCP_SERVER_DIR / "stdio_subtract_multiply.py")],
                include_tools=["nonexistent"],
            )

        # Should result in no tools being registered
        tools = chat.get_tools()
        assert len(tools) == 0

        await chat.cleanup_mcp_tools()

    @pytest.mark.asyncio
    async def test_exclude_all_tools(self):
        """Test excluding all available tools."""
        chat = ChatOpenAI()

        await chat.register_mcp_tools_stdio_async(
            command=sys.executable,
            args=[str(MCP_SERVER_DIR / "stdio_subtract_multiply.py")],
            exclude_tools=["subtract", "multiply"],
        )

        # Should result in no tools being registered
        tools = chat.get_tools()
        assert len(tools) == 0

        await chat.cleanup_mcp_tools()


class TestMCPTransportKwargs:
    """Test MCP transport keyword arguments."""

    @pytest.mark.asyncio
    async def test_stdio_transport_kwargs(self):
        """Test passing transport_kwargs to stdio client."""
        chat = ChatOpenAI()

        # Test with env override (should still work)
        await chat.register_mcp_tools_stdio_async(
            command=sys.executable,
            args=[str(MCP_SERVER_DIR / "stdio_subtract_multiply.py")],
            transport_kwargs={"env": {"TEST_VAR": "test_value"}},
        )

        # Should successfully register tools
        tools = chat.get_tools()
        assert len(tools) == 2

        await chat.cleanup_mcp_tools()

    @pytest.mark.asyncio
    async def test_http_transport_kwargs(self):
        """Test passing transport_kwargs to HTTP client."""
        chat = ChatOpenAI()

        async with http_mcp_server("http_add.py"):
            await chat.register_mcp_tools_http_stream_async(
                url=SERVER_URL,
                transport_kwargs={"timeout": 30.0},
            )

            # Should successfully register tools
            tools = chat.get_tools()
            assert len(tools) == 1

            await chat.cleanup_mcp_tools()
