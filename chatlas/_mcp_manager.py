from __future__ import annotations

import asyncio
import warnings
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

from ._tools import Tool, ToolBuiltIn

if TYPE_CHECKING:
    from mcp import Client, ClientSession

    from ._tools import MCPClientFactory


class LegacyMCPServerError(RuntimeError):
    """Raised when an MCP server only speaks a handshake-era (<=2025-11-25)
    protocol revision, which requires a persistent session."""


@dataclass
class SessionInfo(ABC):
    # Input parameters
    name: str
    include_tools: Sequence[str] = field(default_factory=list)
    exclude_tools: Sequence[str] = field(default_factory=list)
    namespace: str | None = None

    # Primary derived attributes
    client: Client | None = None
    session: ClientSession | None = None
    tools: dict[str, Tool | ToolBuiltIn] = field(default_factory=dict)

    # Stateless registration state (modern, per-request protocol revisions)
    protocol_version: str | None = None
    discover_result: Any | None = None

    # Background task management
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    shutdown_event: asyncio.Event = field(default_factory=asyncio.Event)
    exit_stack: AsyncExitStack = field(default_factory=AsyncExitStack)
    task: asyncio.Task | None = None
    error: asyncio.CancelledError | Exception | None = None

    @abstractmethod
    def make_transport(self) -> Any:
        """Create a fresh (un-entered) transport for a new client connection."""
        ...

    async def open_session(self) -> None:
        await self._connect(self.make_transport())

    async def register_stateless(self, *, sync: bool = False) -> None:
        """Negotiate era and list tools with a short-lived client.

        Nothing is held open on return. Tools call the server through
        ephemeral per-call clients. Raises `LegacyMCPServerError` if the
        server only speaks a handshake-era protocol revision.
        """
        mcp = try_import_mcp()
        async with AsyncExitStack() as stack:
            client = await stack.enter_async_context(
                mcp.Client(self.make_transport(), mode="auto")
            )
            if not self.name:
                server_info = client.server_info
                self.name = (server_info.name if server_info else None) or "mcp"
            self.protocol_version = client.session.protocol_version
            self.discover_result = client.session.discover_result
            await self.request_tools(
                client, client_factory=self.client_factory(), sync=sync
            )
        if self.discover_result is None:
            raise LegacyMCPServerError(
                f"MCP server '{self.name}' only speaks the legacy MCP protocol "
                f"(revision {self.protocol_version}), which requires a "
                "persistent session."
            )

    def client_factory(self) -> "MCPClientFactory":
        """A factory of fresh clients pinned to the negotiated modern revision."""

        def factory() -> "Client":
            mcp = try_import_mcp()
            if self.protocol_version is None or self.discover_result is None:
                raise RuntimeError(
                    "MCP server registration is incomplete; cannot create a client."
                )
            return mcp.Client(
                self.make_transport(),
                mode=self.protocol_version,
                prior_discover=self.discover_result,
            )

        return factory

    async def close_session(self) -> None:
        await self.exit_stack.aclose()

    async def _connect(self, transport: Any) -> None:
        mcp = try_import_mcp()
        # mode="auto" probes `server/discover` to negotiate a 2026-07-28+
        # (stateless) protocol revision, falling back to the legacy
        # `initialize` handshake for older servers.
        client = await self.exit_stack.enter_async_context(
            mcp.Client(transport, mode="auto")
        )
        self.client = client
        self.session = client.session
        if not self.name:
            server_info = client.server_info
            self.name = (server_info.name if server_info else None) or "mcp"

    async def request_tools(
        self,
        client: "Client | None" = None,
        *,
        client_factory: "MCPClientFactory | None" = None,
        sync: bool = False,
    ) -> None:
        client = client or self.client
        if client is None:
            raise ValueError("Session must be opened before requesting tools.")

        if self.include_tools and self.exclude_tools:
            raise ValueError("Cannot specify both include_tools and exclude_tools.")

        # Request the MCP tools available
        response = await client.list_tools()
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
        self_tools: dict[str, Tool | ToolBuiltIn] = {}
        for tool in response.tools:
            if tool.name not in tool_names:
                continue
            # Only the chatlas-side name is namespaced: the server must still be
            # called with its own tool name, so don't mutate the MCP tool object
            name = f"{self.namespace}.{tool.name}" if self.namespace else tool.name
            self_tools[name] = Tool.from_mcp(
                session=client if client_factory is None else None,
                mcp_tool=tool,
                name=name,
                client_factory=client_factory,
                sync=sync,
            )

        # Store the tools
        self.tools = self_tools


@dataclass
class HTTPSessionInfo(SessionInfo):
    url: str = ""
    transport_kwargs: dict[str, Any] = field(default_factory=dict)

    def make_transport(self):
        try_import_mcp()
        from mcp.client.streamable_http import streamable_http_client

        return streamable_http_client(self.url, **self.transport_kwargs)


@dataclass
class STDIOSessionInfo(SessionInfo):
    command: str = ""
    args: list[str] = field(default_factory=list)
    transport_kwargs: dict[str, Any] = field(default_factory=dict)

    def make_transport(self):
        mcp = try_import_mcp()
        from mcp.client.stdio import stdio_client

        server_params = mcp.StdioServerParameters(
            command=self.command,
            args=self.args,
            **self.transport_kwargs,
        )
        return stdio_client(server_params)


class MCPSessionManager:
    """Manages MCP (Model Context Protocol) server connections and tools."""

    def __init__(self):
        # Held (persistent) sessions: legacy servers and persistent stdio
        self._mcp_sessions: dict[str, SessionInfo] = {}
        # Stateless registrations: nothing held open; tools call per-request
        self._registrations: dict[str, SessionInfo] = {}

    @property
    def has_open_sessions(self) -> bool:
        """Whether any MCP server sessions with held resources exist."""
        return bool(self._mcp_sessions)

    @property
    def has_registrations(self) -> bool:
        """Whether any MCP servers (persistent or stateless) are registered."""
        return bool(self._mcp_sessions or self._registrations)

    async def register_stateless_tools(
        self, session_info: "SessionInfo", *, sync: bool = False
    ) -> "SessionInfo":
        try:
            await session_info.register_stateless(sync=sync)
        except LegacyMCPServerError:
            raise
        # Only Exception is wrapped: unlike the background-task session path,
        # this runs inline in the caller's task, so CancelledError must
        # propagate (e.g. for asyncio.wait_for() timeouts to work).
        except Exception as err:
            raise RuntimeError(
                f"Failed to register tools from MCP server '{session_info.name}'"
            ) from err
        self.add_registration(session_info)
        return session_info

    def add_registration(self, session_info: "SessionInfo") -> None:
        name = session_info.name
        if name in self._mcp_sessions or name in self._registrations:
            raise ValueError(f"Already connected to an MCP server named: '{name}'.")
        self._registrations[name] = session_info

    def remove_registration(self, name: str) -> "SessionInfo | None":
        if name not in self._registrations:
            warnings.warn(
                f"Cannot remove MCP registration named '{name}' since it was not found.",
                stacklevel=2,
            )
            return None
        return self._registrations.pop(name)

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

        # If successful, wait for shutdown signal. This task may also be
        # cancelled without a clean shutdown (e.g. the caller errors before
        # calling cleanup, or the event loop is closing) -- close_session()
        # must still run in that case, since it's the only thing that can
        # close the underlying subprocess/transport, and it must run in this
        # same task (that's the whole reason this is a background task).
        # Leaving it unclosed leaks the subprocess and can deadlock the event
        # loop's shutdown_asyncgens() pass trying to finalize it elsewhere.
        try:
            await session_info.shutdown_event.wait()
        finally:
            await session_info.close_session()

    async def close_sessions(self, names: Optional[Sequence[str]] = None):
        if names is None:
            names = [*self._mcp_sessions.keys(), *self._registrations.keys()]

        if isinstance(names, str):
            names = [names]

        closed_sessions: list[SessionInfo] = []
        for x in names:
            if x in self._registrations:
                removed = self.remove_registration(x)
                if removed is not None:
                    closed_sessions.append(removed)
                continue
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
        if name in self._mcp_sessions or name in self._registrations:
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
