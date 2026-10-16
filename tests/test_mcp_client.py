import asyncio
import contextlib
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
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


@contextlib.contextmanager
def sync_http_mcp_server(server_file: str):
    """Synchronous variant of http_mcp_server for testing the sync API."""
    full_path = str(MCP_SERVER_DIR / server_file)
    process = subprocess.Popen(
        args=[sys.executable, full_path],
        stdout=subprocess.DEVNULL,
        env=ENV_VARS,
    )
    try:
        start = time.time()
        while True:
            try:
                urllib.request.urlopen(
                    f"http://localhost:{ENV_VARS['MCP_PORT']}", timeout=1
                )
                break
            except urllib.error.HTTPError:
                # Any HTTP response (even 404) means the server is up
                break
            except urllib.error.URLError:
                if time.time() - start > 10:
                    process.kill()
                    process.wait()
                    raise TimeoutError("Failed to connect to MCP server")
                time.sleep(0.1)
        yield
    finally:
        process.kill()
        process.wait()


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
async def test_mcp_negotiates_modern_protocol_http():
    chat = ChatOpenAI()

    async with http_mcp_server("http_add.py"):
        await chat.register_mcp_tools_http_stream_async(url=SERVER_URL)

        info = chat._mcp_manager._mcp_sessions["test"]
        assert info.session is not None
        assert info.session.protocol_version == "2026-07-28"

        tool = chat._tools["add"]
        assert isinstance(tool, Tool)
        res = await tool.func(x=1, y=2)
        assert res.value == "3"

        await chat.cleanup_mcp_tools()


@pytest.mark.asyncio
async def test_mcp_negotiates_modern_protocol_stdio():
    chat = ChatOpenAI()

    await chat.register_mcp_tools_stdio_async(
        command=sys.executable,
        args=[str(MCP_SERVER_DIR / "stdio_subtract_multiply.py")],
    )

    info = chat._mcp_manager._mcp_sessions["test"]
    assert info.session is not None
    assert info.session.protocol_version == "2026-07-28"

    tool = chat._tools["multiply"]
    assert isinstance(tool, Tool)
    res = await tool.func(a=2, b=3)
    assert res.value == "6"

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


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_call_http_mcp_tool():
    chat = ChatOpenAI(
        model="gpt-5.4", system_prompt="Be very terse, not even punctuation."
    )

    async with http_mcp_server("http_current_date.py"):
        await chat.register_mcp_tools_http_stream_async(
            name="test",
            url=SERVER_URL,
        )

        response = await chat.chat_async(
            "What's the current date in YYYY-MM-DD format?", stream=True
        )
        assert "2024-01-01" in await response.get_content()

        with pytest.raises(Exception, match="async tools in a synchronous chat"):
            str(chat.chat("Great. Do it again.", stream=True))

        await chat.cleanup_mcp_tools()


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_call_stdio_mcp_tool():
    chat = ChatOpenAI(
        model="gpt-5.4", system_prompt="Be very terse, not even punctuation."
    )

    await chat.register_mcp_tools_stdio_async(
        name="stdio_test",
        command=sys.executable,
        args=[str(MCP_SERVER_DIR / "stdio_current_date.py")],
    )

    response = await chat.chat_async(
        "What's the current date in YYYY-MM-DD format?", stream=True
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

            # The namespaced tool should still call the server's own tool name
            res = await chat._tools["mcp.add"].func(x=1, y=2)
            assert res.value == "3"

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
        import httpx

        chat = ChatOpenAI()

        async with http_mcp_server("http_add.py"):
            # The streamable_http_client accepts an http_client param for custom configuration
            async with httpx.AsyncClient(timeout=30.0) as client:
                await chat.register_mcp_tools_http_stream_async(
                    url=SERVER_URL,
                    transport_kwargs={"http_client": client},
                )

                # Should successfully register tools
                tools = chat.get_tools()
                assert len(tools) == 1

                await chat.cleanup_mcp_tools()


class TestStatelessRegistration:
    """Manager-level tests for stateless (per-call) MCP registrations."""

    @pytest.mark.asyncio
    async def test_stateless_http_registration(self):
        from chatlas._mcp_manager import HTTPSessionInfo, MCPSessionManager

        manager = MCPSessionManager()
        async with http_mcp_server("http_add.py"):
            info = HTTPSessionInfo(name="", url=SERVER_URL)
            await manager.register_stateless_tools(info)

            assert info.name == "test"
            assert info.protocol_version == "2026-07-28"
            assert info.discover_result is not None
            assert not manager.has_open_sessions
            assert manager.has_registrations

            res = await info.tools["add"].func(x=1, y=2)
            assert res.value == "3"

            closed = await manager.close_sessions()
            assert [s.name for s in closed] == ["test"]
            assert not manager.has_registrations

    @pytest.mark.asyncio
    async def test_stateless_stdio_registration_sync_tools(self):
        from chatlas._mcp_manager import MCPSessionManager, STDIOSessionInfo

        manager = MCPSessionManager()
        info = STDIOSessionInfo(
            name="",
            command=sys.executable,
            args=[str(MCP_SERVER_DIR / "stdio_subtract_multiply.py")],
        )
        await manager.register_stateless_tools(info, sync=True)

        assert info.protocol_version == "2026-07-28"
        tool = info.tools["multiply"]
        assert not tool._is_async
        await manager.close_sessions()

    @pytest.mark.asyncio
    async def test_persistent_sessions_still_work(self):
        # Regression: the existing background-session path is untouched
        from chatlas._mcp_manager import MCPSessionManager

        manager = MCPSessionManager()
        async with http_mcp_server("http_add.py"):
            info = await manager.register_http_stream_tools(
                url=SERVER_URL,
                name=None,
                include_tools=(),
                exclude_tools=(),
                namespace=None,
                transport_kwargs={},
            )
            assert manager.has_open_sessions
            res = await info.tools["add"].func(x=2, y=2)
            assert res.value == "4"
            await manager.close_sessions()

    @pytest.mark.asyncio
    async def test_stateless_registration_legacy_server_raises(self):
        from chatlas._mcp_manager import (
            HTTPSessionInfo,
            LegacyMCPServerError,
            MCPSessionManager,
        )

        manager = MCPSessionManager()
        async with http_mcp_server("http_legacy_stub.py"):
            info = HTTPSessionInfo(name="", url=SERVER_URL)
            with pytest.raises(LegacyMCPServerError, match="persistent session"):
                await manager.register_stateless_tools(info)
            assert not manager.has_registrations


class TestRegisterMCPToolsAsync:
    @pytest.mark.asyncio
    async def test_http_stateless(self):
        chat = ChatOpenAI()
        async with http_mcp_server("http_add.py"):
            await chat.register_mcp_tools_async(url=SERVER_URL)

            assert not chat._mcp_manager.has_open_sessions
            assert chat._mcp_manager.has_registrations
            info = chat._mcp_manager._registrations["test"]
            assert info.protocol_version == "2026-07-28"

            res = await chat._tools["add"].func(x=1, y=2)
            assert res.value == "3"

            await chat.cleanup_mcp_tools()
            assert "add" not in chat._tools
            assert not chat._mcp_manager.has_registrations

    @pytest.mark.asyncio
    async def test_http_legacy_falls_back_to_persistent(self):
        chat = ChatOpenAI()
        async with http_mcp_server("http_legacy_stub.py"):
            await chat.register_mcp_tools_async(url=SERVER_URL)

            assert chat._mcp_manager.has_open_sessions
            info = chat._mcp_manager._mcp_sessions["legacy_stub"]
            assert info.session.protocol_version == "2025-11-25"

            res = await chat._tools["echo"].func(text="hi")
            assert res.value == "hi"
            await chat.cleanup_mcp_tools()

    @pytest.mark.asyncio
    async def test_stdio_persistent_by_default(self):
        chat = ChatOpenAI()
        await chat.register_mcp_tools_async(
            command=sys.executable,
            args=[str(MCP_SERVER_DIR / "stdio_subtract_multiply.py")],
        )
        assert chat._mcp_manager.has_open_sessions
        res = await chat._tools["multiply"].func(a=2, b=3)
        assert res.value == "6"
        await chat.cleanup_mcp_tools()

    @pytest.mark.asyncio
    async def test_stdio_per_call(self):
        chat = ChatOpenAI()
        await chat.register_mcp_tools_async(
            command=sys.executable,
            args=[str(MCP_SERVER_DIR / "stdio_subtract_multiply.py")],
            persistent=False,
        )
        assert not chat._mcp_manager.has_open_sessions
        res = await chat._tools["multiply"].func(a=2, b=3)
        assert res.value == "6"
        await chat.cleanup_mcp_tools()

    @pytest.mark.asyncio
    async def test_namespace_and_include(self):
        chat = ChatOpenAI()
        await chat.register_mcp_tools_async(
            command=sys.executable,
            args=[str(MCP_SERVER_DIR / "stdio_subtract_multiply.py")],
            persistent=False,
            namespace="ns",
            include_tools=["multiply"],
        )
        assert set(chat._tools) == {"ns.multiply"}
        res = await chat._tools["ns.multiply"].func(a=2, b=3)
        assert res.value == "6"
        await chat.cleanup_mcp_tools()

    @pytest.mark.asyncio
    async def test_overlapping_tools_rolls_back(self):
        chat = ChatOpenAI()

        def add(x: int, y: int) -> int:
            return x + y

        chat.register_tool(add)
        async with http_mcp_server("http_add.py"):
            with pytest.raises(ValueError, match="already registered"):
                await chat.register_mcp_tools_async(url=SERVER_URL)
            assert not chat._mcp_manager.has_registrations

    @pytest.mark.asyncio
    async def test_url_and_command_mutually_exclusive(self):
        chat = ChatOpenAI()
        with pytest.raises(ValueError, match="Exactly one"):
            await chat.register_mcp_tools_async()


class TestRegisterMCPToolsSync:
    def test_http_stateless(self):
        chat = ChatOpenAI()
        with sync_http_mcp_server("http_add.py"):
            chat.register_mcp_tools(url=SERVER_URL)

            assert not chat._mcp_manager.has_open_sessions
            assert chat._mcp_manager.has_registrations

            tool = chat._tools["add"]
            assert not tool._is_async
            res = tool.func(x=1, y=2)
            assert res.value == "3"

    def test_stdio_spawn_per_call(self):
        chat = ChatOpenAI()
        chat.register_mcp_tools(
            command=sys.executable,
            args=[str(MCP_SERVER_DIR / "stdio_subtract_multiply.py")],
        )
        assert not chat._mcp_manager.has_open_sessions
        res = chat._tools["multiply"].func(a=2, b=3)
        assert res.value == "6"

    def test_legacy_server_errors(self):
        chat = ChatOpenAI()
        with sync_http_mcp_server("http_legacy_stub.py"):
            with pytest.raises(RuntimeError, match="register_mcp_tools_async"):
                chat.register_mcp_tools(url=SERVER_URL)
        assert not chat._mcp_manager.has_registrations

    @pytest.mark.asyncio
    async def test_registration_in_running_loop_errors(self):
        chat = ChatOpenAI()
        with pytest.raises(RuntimeError, match="running event loop"):
            chat.register_mcp_tools(url="http://localhost:1/mcp")

    def test_url_and_command_mutually_exclusive(self):
        chat = ChatOpenAI()
        with pytest.raises(ValueError, match="Exactly one"):
            chat.register_mcp_tools()
