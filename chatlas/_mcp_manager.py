from __future__ import annotations

import asyncio
import warnings
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

from ._tools import Tool

if TYPE_CHECKING:
    from mcp import ClientSession as MCPClientSession


@dataclass
class SessionInfo:
    name: str
    tools: dict[str, Tool] = field(default_factory=dict)
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    shutdown_event: asyncio.Event = field(default_factory=asyncio.Event)
    exit_stack: AsyncExitStack = field(default_factory=AsyncExitStack)
    task: asyncio.Task | None = None
    error: asyncio.CancelledError | Exception | None = None


@dataclass
class HTTPSessionInfo(SessionInfo):
    url: str = ""
    transport_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class STDIOSessionInfo(SessionInfo):
    command: str = ""
    args: list[str] = field(default_factory=list)
    transport_kwargs: dict[str, Any] = field(default_factory=dict)


class MCPSessionManager:
    """Manages MCP (Model Context Protocol) server connections and tools."""

    def __init__(self):
        self._mcp_sessions: dict[str, SessionInfo] = {}

    async def register_http_stream_tools(
        self,
        *,
        name: str,
        url: str,
        include_tools: Sequence[str],
        exclude_tools: Sequence[str],
        namespace: str | None,
        transport_kwargs: dict[str, Any],
    ):
        if name in self._mcp_sessions:
            raise ValueError(f"MCP Session {name} already exists.")

        _ = try_import_mcp()

        session_info = HTTPSessionInfo(
            name=name,
            url=url,
            transport_kwargs=transport_kwargs or {},
        )

        # Launch background task that runs until MCP session is *shutdown*
        # N.B. this is needed since mcp sessions must be opened and closed in the same task
        asyncio.create_task(
            self.open_mcp_session(
                session_info=session_info,
                include_tools=include_tools,
                exclude_tools=exclude_tools,
                namespace=namespace,
            )
        )

        # Wait for a ready event from the background task (signals that tools are registered)
        await session_info.ready_event.wait()

        if session_info.error:
            raise RuntimeError(
                f"Failed to register tools from MCP server '{name}' at URL '{url}'"
            ) from session_info.error

        return session_info

    async def register_stdio_tools(
        self,
        *,
        name: str,
        command: str,
        args: list[str],
        include_tools: Sequence[str],
        exclude_tools: Sequence[str],
        namespace: str | None,
        transport_kwargs: dict[str, Any],
    ):
        if name in self._mcp_sessions:
            raise ValueError(f"MCP Session {name} already exists.")

        _ = try_import_mcp()

        # Launch background task that runs until MCP session is *shutdown*
        session_info = STDIOSessionInfo(
            name=name,
            command=command,
            args=args,
            transport_kwargs=transport_kwargs or {},
        )

        # Launch a background task to initialize the MCP server
        # N.B. this is needed since mcp sessions must be opened and closed in the same task
        asyncio.create_task(
            self.open_mcp_session(
                session_info=session_info,
                include_tools=include_tools,
                exclude_tools=exclude_tools,
                namespace=namespace,
            )
        )

        # Wait for a ready event from the background task (signals that tools are registered)
        await session_info.ready_event.wait()

        if session_info.error:
            raise RuntimeError(
                f"Failed to register tools from MCP server '{name}' with command '{command} {args}'"
            ) from session_info.error

        return session_info

    async def open_mcp_session(
        self,
        session_info: "SessionInfo",
        include_tools: Sequence[str],
        exclude_tools: Sequence[str],
        namespace: str | None,
    ):
        session_info.task = asyncio.current_task()
        self._mcp_sessions[session_info.name] = session_info

        try:
            # Establish the MCP session
            if isinstance(session_info, HTTPSessionInfo):
                session = await self.open_http_session(session_info)
            elif isinstance(session_info, STDIOSessionInfo):
                session = await self.open_stdio_session(session_info)
            else:
                raise TypeError(
                    f"Unsupported session type: {type(session_info).__name__}. "
                    "Expected HTTPMCPSessionInfo or STDIOMCPSessionInfo."
                )

            # Request the available MCP tools, filter/namespace them,
            # and convert into to our Tool class
            session_info.tools = await self.request_tools(
                session=session,
                include_tools=include_tools,
                exclude_tools=exclude_tools,
                namespace=namespace,
            )

        except (asyncio.CancelledError, Exception) as err:
            # Remember error so we can handle in the main task
            session_info.error = err
            # And also close the exit stack in case the connection was opened
            try:
                await session_info.exit_stack.aclose()
            except Exception:
                pass
            return
        finally:
            # Whether successful or not, set ready state to prevent deadlock
            session_info.ready_event.set()

        # If successful, wait for shutdown signal
        await session_info.shutdown_event.wait()

        # On shutdown close connection to MCP server
        # This is why we're using a background task in the 1st place...
        # we must close in the same task that opened the session
        await session_info.exit_stack.aclose()

    async def open_http_session(self, session_info: "HTTPSessionInfo"):
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        read, write, _ = await session_info.exit_stack.enter_async_context(
            streamablehttp_client(
                session_info.url,
                **session_info.transport_kwargs,
            )
        )
        session = await session_info.exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        return session

    async def open_stdio_session(self, session_info: "STDIOSessionInfo"):
        # Try to initialize the MCP session (and cleanup if it fails)
        mcp = try_import_mcp()
        from mcp.client.stdio import stdio_client

        command = session_info.command
        args = session_info.args

        server_params = mcp.StdioServerParameters(
            command=command,
            args=args,
            **session_info.transport_kwargs,
        )

        transport = await session_info.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        session = await session_info.exit_stack.enter_async_context(
            mcp.ClientSession(*transport)
        )
        await session.initialize()
        return session

    async def close_sessions(self, names: Optional[Sequence[str]] = None):
        if names is None:
            names = list(self._mcp_sessions.keys())

        if isinstance(names, str):
            names = [names]

        closed_sessions: list[SessionInfo] = []
        for x in names:
            if x not in self._mcp_sessions:
                warnings.warn(
                    f"No MCP session found with name '{x}'. Skipping cleanup.",
                    stacklevel=2,
                )
                continue
            session = self._mcp_sessions[x]
            closed_sessions.append(session)
            # Signal shutdown and wait for the task to finish (i.e., the session to close)
            session.shutdown_event.set()
            if session.task is not None:
                await session.task
            del self._mcp_sessions[x]

        return closed_sessions

    async def request_tools(
        self,
        session: "MCPClientSession",
        include_tools: Sequence[str],
        exclude_tools: Sequence[str],
        namespace: str | None,
    ):
        if include_tools and exclude_tools:
            raise ValueError("Cannot specify both include_tools and exclude_tools.")

        # Request the MCP tools available
        response = await session.list_tools()
        mcp_tools = response.tools
        tool_names = set(x.name for x in mcp_tools)

        # Warn if tools are mis-specified
        include = set(include_tools or [])
        missing_include = include.difference(tool_names)
        if missing_include:
            warnings.warn(
                f"Specified include_tools {missing_include} did not match any tools from the MCP server. "
                f"The tools available are: {tool_names}",
                stacklevel=2,
            )
        exclude = set(exclude_tools or [])
        missing_exclude = exclude.difference(tool_names)
        if missing_exclude:
            warnings.warn(
                f"Specified exclude_tools {missing_exclude} did not match any tools from the MCP server. "
                f"The tools available are: {tool_names}",
                stacklevel=2,
            )

        # Filter the tool names
        if include:
            tool_names = include.intersection(tool_names)
        if exclude:
            tool_names = tool_names.difference(exclude)

        # Apply namespace and convert to chatlas.Tool instances
        res: dict[str, Tool] = {}
        for tool in mcp_tools:
            if tool.name not in tool_names:
                continue
            if namespace:
                tool.name = f"{namespace}.{tool.name}"
            res[tool.name] = Tool.from_mcp(session=session, mcp_tool=tool)

        return res


def try_import_mcp():
    try:
        import mcp

        return mcp
    except ImportError:
        raise ImportError(
            "The `mcp` package is required to connect to MCP servers. "
            "Install it with `pip install mcp`."
        )
