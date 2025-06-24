from __future__ import annotations

import asyncio
import warnings
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

from ._tools import Tool

if TYPE_CHECKING:
    from mcp import ClientSession


@dataclass
class SessionInfo(ABC):
    # Input parameters
    name: str
    include_tools: Sequence[str] = field(default_factory=list)
    exclude_tools: Sequence[str] = field(default_factory=list)
    namespace: str | None = None

    # Primary derived attributes
    session: ClientSession | None = None
    tools: dict[str, Tool] = field(default_factory=dict)

    # Background task management
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    shutdown_event: asyncio.Event = field(default_factory=asyncio.Event)
    exit_stack: AsyncExitStack = field(default_factory=AsyncExitStack)
    task: asyncio.Task | None = None
    error: asyncio.CancelledError | Exception | None = None

    @abstractmethod
    async def open_session(self) -> None: ...

    async def close_session(self) -> None:
        await self.exit_stack.aclose()

    async def request_tools(self) -> None:
        if self.session is None:
            raise ValueError("Session must be opened before requesting tools.")

        if self.include_tools and self.exclude_tools:
            raise ValueError("Cannot specify both include_tools and exclude_tools.")

        # Request the MCP tools available
        response = await self.session.list_tools()
        tool_names = set(x.name for x in response.tools)

        # Warn if tools are mis-specified
        include = set(self.include_tools or [])
        missing_include = include.difference(tool_names)
        if missing_include:
            warnings.warn(
                f"Specified include_tools {missing_include} did not match any tools from the MCP server. "
                f"The tools available are: {tool_names}",
                stacklevel=2,
            )
        exclude = set(self.exclude_tools or [])
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
        self_tools: dict[str, Tool] = {}
        for tool in response.tools:
            if tool.name not in tool_names:
                continue
            if self.namespace:
                tool.name = f"{self.namespace}.{tool.name}"
            self_tools[tool.name] = Tool.from_mcp(
                session=self.session,
                mcp_tool=tool,
            )

        # Store the tools
        self.tools = self_tools


@dataclass
class HTTPSessionInfo(SessionInfo):
    url: str = ""
    transport_kwargs: dict[str, Any] = field(default_factory=dict)

    async def open_session(self):
        mcp = try_import_mcp()
        from mcp.client.streamable_http import streamablehttp_client

        read, write, _ = await self.exit_stack.enter_async_context(
            streamablehttp_client(
                self.url,
                **self.transport_kwargs,
            )
        )
        session = await self.exit_stack.enter_async_context(
            mcp.ClientSession(read, write)
        )
        server = await session.initialize()
        self.session = session
        if not self.name:
            self.name = server.serverInfo.name or "mcp"


@dataclass
class STDIOSessionInfo(SessionInfo):
    command: str = ""
    args: list[str] = field(default_factory=list)
    transport_kwargs: dict[str, Any] = field(default_factory=dict)

    async def open_session(self):
        mcp = try_import_mcp()
        from mcp.client.stdio import stdio_client

        server_params = mcp.StdioServerParameters(
            command=self.command,
            args=self.args,
            **self.transport_kwargs,
        )

        transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        session = await self.exit_stack.enter_async_context(
            mcp.ClientSession(*transport)
        )
        server = await session.initialize()
        self.session = session
        if not self.name:
            self.name = server.serverInfo.name or "mcp"


class MCPSessionManager:
    """Manages MCP (Model Context Protocol) server connections and tools."""

    def __init__(self):
        self._mcp_sessions: dict[str, SessionInfo] = {}

    async def register_http_stream_tools(
        self,
        *,
        url: str,
        name: str | None,
        include_tools: Sequence[str],
        exclude_tools: Sequence[str],
        namespace: str | None,
        transport_kwargs: dict[str, Any],
    ):
        session_info = HTTPSessionInfo(
            name=name or "",
            url=url,
            include_tools=include_tools,
            exclude_tools=exclude_tools,
            namespace=namespace,
            transport_kwargs=transport_kwargs or {},
        )

        # Launch background task that runs until MCP session is *shutdown*
        # N.B. this is needed since mcp sessions must be opened and closed in the same task
        asyncio.create_task(self.open_session(session_info))

        # Wait for a ready event from the task (signals that tools are registered)
        await session_info.ready_event.wait()

        # An error might have been caught in the background task
        if session_info.error:
            raise RuntimeError(
                f"Failed to register tools from MCP server '{name}' at URL '{url}'"
            ) from session_info.error

        return session_info

    async def register_stdio_tools(
        self,
        *,
        command: str,
        args: list[str],
        name: str | None,
        include_tools: Sequence[str],
        exclude_tools: Sequence[str],
        namespace: str | None,
        transport_kwargs: dict[str, Any],
    ):
        session_info = STDIOSessionInfo(
            name=name or "",
            command=command,
            args=args,
            include_tools=include_tools,
            exclude_tools=exclude_tools,
            namespace=namespace,
            transport_kwargs=transport_kwargs or {},
        )

        # Launch a background task to initialize the MCP server
        # N.B. this is needed since mcp sessions must be opened and closed in the same task
        asyncio.create_task(self.open_session(session_info))

        # Wait for a ready event from the task (signals that tools are registered)
        await session_info.ready_event.wait()

        # An error might have been caught in the background task
        if session_info.error:
            raise RuntimeError(
                f"Failed to register tools from MCP server '{name}' with command '{command} {args}'"
            ) from session_info.error

        return session_info

    async def open_session(self, session_info: "SessionInfo"):
        session_info.task = asyncio.current_task()

        try:
            # Open the MCP session
            await session_info.open_session()
            # Request the tools
            await session_info.request_tools()
            # Make sure session can be added to the manager
            self.add_session(session_info)
        except (asyncio.CancelledError, Exception) as err:
            # Keep the error so we can handle in the main task
            session_info.error = err
            # Make sure the session is closed
            try:
                await session_info.close_session()
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
        await session_info.close_session()

    async def close_sessions(self, names: Optional[Sequence[str]] = None):
        if names is None:
            names = list(self._mcp_sessions.keys())

        if isinstance(names, str):
            names = [names]

        closed_sessions: list[SessionInfo] = []
        for x in names:
            session = await self.close_background_session(x)
            if session is None:
                continue
            closed_sessions.append(session)

        return closed_sessions

    async def close_background_session(self, name: str) -> SessionInfo | None:
        session = self.remove_session(name)
        if session is None:
            return None

        # Signal shutdown and wait for the task to finish
        session.shutdown_event.set()
        if session.task is not None:
            await session.task

        return session

    def add_session(self, session_info: SessionInfo) -> None:
        name = session_info.name
        if name in self._mcp_sessions:
            raise ValueError(f"Already connected to an MCP server named: '{name}'.")
        self._mcp_sessions[name] = session_info

    def remove_session(self, name: str) -> SessionInfo | None:
        if name not in self._mcp_sessions:
            warnings.warn(
                f"Cannot close MCP session named '{name}' since it was not found.",
                stacklevel=2,
            )
            return None
        session = self._mcp_sessions[name]
        del self._mcp_sessions[name]
        return session


def try_import_mcp():
    try:
        import mcp

        return mcp
    except ImportError:
        raise ImportError(
            "The `mcp` package is required to connect to MCP servers. "
            "Install it with `pip install mcp`."
        )
