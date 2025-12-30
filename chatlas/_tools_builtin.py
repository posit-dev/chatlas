"""
Built-in provider tools for web search and fetch.

These classes provide a provider-agnostic way to configure built-in tools
like web search and URL fetching. Each provider translates these configurations
into their specific API format.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, Optional, overload

from ._tools import ToolBuiltIn

if TYPE_CHECKING:
    from anthropic.types import WebSearchTool20250305Param
    from anthropic.types.beta import BetaWebFetchTool20250910Param
    from anthropic.types.beta.beta_citations_config_param import (
        BetaCitationsConfigParam,
    )
    from anthropic.types.cache_control_ephemeral_param import (
        CacheControlEphemeralParam,
    )
    from google.genai.types import (
        GoogleSearch,
        Interval,
        PhishBlockThreshold,
        UrlContext,
    )
    from openai.types.responses import WebSearchToolParam

    from ._typing_extensions import TypedDict

    class UserLocation(TypedDict, total=False):
        """User location for localizing search results."""

        country: str
        """Two-letter ISO country code (e.g., 'US', 'GB')."""
        city: str
        """City name."""
        region: str
        """Region/state name."""
        timezone: str
        """IANA timezone (e.g., 'America/New_York')."""


__all__ = (
    "tool_web_search",
    "tool_web_fetch",
    "ToolWebSearch",
    "ToolWebFetch",
)


def _warn_unsupported(param_name: str, provider: str) -> None:
    """Warn that a parameter is not supported by a provider."""
    warnings.warn(
        f"{param_name} is not supported by {provider} and will be ignored.",
        UserWarning,
        stacklevel=3,
    )


class ToolWebSearch(ToolBuiltIn):
    """
    A provider-agnostic web search tool configuration.

    This class stores configuration for web search functionality. Each provider
    translates this configuration into their specific API format.

    Parameters
    ----------
    allowed_domains
        Restrict searches to specific domains (e.g., ['nytimes.com', 'bbc.com']).
        Not all providers support this parameter.
    blocked_domains
        Exclude specific domains from searches.
        Only supported by Claude. Cannot be used with allowed_domains.
    user_location
        Location information to localize search results.
    max_uses
        Maximum number of searches allowed per request.
        Only supported by Claude.
    """

    def __init__(
        self,
        *,
        allowed_domains: Optional[list[str]] = None,
        blocked_domains: Optional[list[str]] = None,
        user_location: "Optional[UserLocation]" = None,
        max_uses: Optional[int] = None,
    ):
        if allowed_domains and blocked_domains:
            raise ValueError(
                "Cannot specify both allowed_domains and blocked_domains. "
                "Use one or the other."
            )

        self.allowed_domains = allowed_domains
        self.blocked_domains = blocked_domains
        self.user_location = user_location
        self.max_uses = max_uses

        # Initialize ToolBuiltIn with placeholder definition
        # (providers will use get_definition() to get the actual definition)
        super().__init__(name="web_search", definition={})

    @overload
    def get_definition(
        self,
        provider_name: Literal["openai"],
        *,
        search_context_size: "Literal['low', 'medium', 'high'] | None" = None,
    ) -> "WebSearchToolParam": ...

    @overload
    def get_definition(
        self,
        provider_name: Literal["anthropic"],
        *,
        cache_control: "CacheControlEphemeralParam | None" = None,
    ) -> "WebSearchTool20250305Param": ...

    @overload
    def get_definition(
        self,
        provider_name: Literal["google"],
        *,
        blocking_confidence: "PhishBlockThreshold | None" = None,
        time_range_filter: "Interval | None" = None,
    ) -> "GoogleSearch": ...

    def get_definition(
        self,
        provider_name: Literal["openai", "anthropic", "google"],
        *,
        # OpenAI-specific
        search_context_size: "Literal['low', 'medium', 'high'] | None" = None,
        # Anthropic-specific
        cache_control: "CacheControlEphemeralParam | None" = None,
        # Google-specific
        blocking_confidence: "PhishBlockThreshold | None" = None,
        time_range_filter: "Interval | None" = None,
    ) -> "WebSearchToolParam | WebSearchTool20250305Param | GoogleSearch":
        """
        Get the provider-specific tool definition.

        Parameters
        ----------
        provider_name
            The name of the provider ('openai', 'anthropic', or 'google').
        search_context_size
            OpenAI only. Amount of search context: 'low', 'medium', or 'high'.
        cache_control
            Anthropic only. Cache control settings for prompt caching.
        blocking_confidence
            Google only. Threshold for blocking phishing sites.
        time_range_filter
            Google only. Filter search results to a specific time range.

        Returns
        -------
        :
            The provider-specific tool definition.
        """
        if provider_name == "openai":
            if self.blocked_domains:
                _warn_unsupported("blocked_domains", "OpenAI")
            if self.max_uses is not None:
                _warn_unsupported("max_uses", "OpenAI")
            return self._openai_definition(
                allowed_domains=self.allowed_domains,
                user_location=self.user_location,
                search_context_size=search_context_size,
            )
        elif provider_name == "anthropic":
            return self._anthropic_definition(
                allowed_domains=self.allowed_domains,
                blocked_domains=self.blocked_domains,
                user_location=self.user_location,
                max_uses=self.max_uses,
                cache_control=cache_control,
            )
        elif provider_name == "google":
            if self.allowed_domains:
                _warn_unsupported("allowed_domains", "Google")
            if self.user_location:
                _warn_unsupported("user_location", "Google")
            if self.max_uses is not None:
                _warn_unsupported("max_uses", "Google")
            return self._google_definition(
                blocked_domains=self.blocked_domains,
                blocking_confidence=blocking_confidence,
                time_range_filter=time_range_filter,
            )
        else:
            raise ValueError(
                f"Web search is not supported for provider '{provider_name}'. "
                "Supported providers: openai, anthropic, google."
            )

    @staticmethod
    def _openai_definition(
        *,
        allowed_domains: Optional[list[str]],
        user_location: "Optional[UserLocation]",
        search_context_size: "Literal['low', 'medium', 'high'] | None" = None,
    ) -> "WebSearchToolParam":
        """Generate OpenAI web search tool definition."""
        # https://platform.openai.com/docs/guides/tools-web-search
        definition: WebSearchToolParam = {"type": "web_search"}

        if search_context_size is not None:
            definition["search_context_size"] = search_context_size

        if allowed_domains:
            domains = [
                d.removeprefix("https://").removeprefix("http://")
                for d in allowed_domains
            ]
            definition["filters"] = {"allowed_domains": domains}

        if user_location:
            definition["user_location"] = {
                "type": "approximate",
                **user_location,
            }

        return definition

    @staticmethod
    def _anthropic_definition(
        *,
        allowed_domains: Optional[list[str]],
        blocked_domains: Optional[list[str]],
        user_location: "Optional[UserLocation]",
        max_uses: Optional[int],
        cache_control: "CacheControlEphemeralParam | None" = None,
    ) -> "WebSearchTool20250305Param":
        """Generate Anthropic/Claude web search tool definition."""
        # https://docs.claude.com/en/docs/agents-and-tools/tool-use/web-search-tool
        definition: WebSearchTool20250305Param = {
            "name": "web_search",
            "type": "web_search_20250305",
        }

        if max_uses is not None:
            definition["max_uses"] = max_uses
        if allowed_domains:
            definition["allowed_domains"] = allowed_domains
        if blocked_domains:
            definition["blocked_domains"] = blocked_domains
        if user_location:
            definition["user_location"] = {
                "type": "approximate",
                **user_location,
            }
        if cache_control is not None:
            definition["cache_control"] = cache_control

        return definition

    @staticmethod
    def _google_definition(
        *,
        blocked_domains: Optional[list[str]],
        blocking_confidence: "PhishBlockThreshold | None" = None,
        time_range_filter: "Interval | None" = None,
    ) -> "GoogleSearch":
        """Generate Google/Gemini web search tool definition."""
        # https://ai.google.dev/gemini-api/docs/google-search
        from google.genai.types import GoogleSearch

        return GoogleSearch(
            exclude_domains=blocked_domains,
            blocking_confidence=blocking_confidence,
            time_range_filter=time_range_filter,
        )


class ToolWebFetch(ToolBuiltIn):
    """
    A provider-agnostic URL fetch tool configuration.

    This class stores configuration for URL fetching functionality. Each provider
    translates this configuration into their specific API format.

    Note: Currently supported by Claude and Google. OpenAI does not have a built-in
    URL fetch tool.

    Parameters
    ----------
    allowed_domains
        Restrict fetches to specific domains.
        Only supported by Claude.
    blocked_domains
        Exclude specific domains from fetches.
        Only supported by Claude. Cannot be used with allowed_domains.
    max_uses
        Maximum number of fetches allowed per request.
        Only supported by Claude.
    """

    def __init__(
        self,
        *,
        allowed_domains: Optional[list[str]] = None,
        blocked_domains: Optional[list[str]] = None,
        max_uses: Optional[int] = None,
    ):
        if allowed_domains and blocked_domains:
            raise ValueError(
                "Cannot specify both allowed_domains and blocked_domains. "
                "Use one or the other."
            )

        # Store configuration
        self.allowed_domains = allowed_domains
        self.blocked_domains = blocked_domains
        self.max_uses = max_uses

        # Initialize ToolBuiltIn with placeholder definition
        super().__init__(name="web_fetch", definition={})

    @overload
    def get_definition(
        self,
        provider_name: Literal["anthropic"],
        *,
        cache_control: "CacheControlEphemeralParam | None" = None,
        citations: "BetaCitationsConfigParam | None" = None,
        max_content_tokens: Optional[int] = None,
    ) -> "BetaWebFetchTool20250910Param": ...

    @overload
    def get_definition(
        self,
        provider_name: Literal["google"],
    ) -> "UrlContext": ...

    def get_definition(
        self,
        provider_name: Literal["anthropic", "google"],
        *,
        # Anthropic-specific
        cache_control: "CacheControlEphemeralParam | None" = None,
        citations: "BetaCitationsConfigParam | None" = None,
        max_content_tokens: Optional[int] = None,
    ) -> "BetaWebFetchTool20250910Param | UrlContext":
        """
        Get the provider-specific tool definition.

        Parameters
        ----------
        provider_name
            The name of the provider ('anthropic' or 'google').
        cache_control
            Anthropic only. Cache control settings for prompt caching.
        citations
            Anthropic only. Configuration for inline citations.
        max_content_tokens
            Anthropic only. Maximum content tokens to fetch.

        Returns
        -------
        :
            The provider-specific tool definition.
        """
        if provider_name == "anthropic":
            return self._anthropic_definition(
                allowed_domains=self.allowed_domains,
                blocked_domains=self.blocked_domains,
                max_uses=self.max_uses,
                cache_control=cache_control,
                citations=citations,
                max_content_tokens=max_content_tokens,
            )
        elif provider_name == "google":
            if self.allowed_domains:
                _warn_unsupported("allowed_domains", "Google")
            if self.blocked_domains:
                _warn_unsupported("blocked_domains", "Google")
            if self.max_uses is not None:
                _warn_unsupported("max_uses", "Google")
            return self._google_definition()
        else:
            raise ValueError(
                f"Web fetch is not supported for provider '{provider_name}'. "
                "Supported providers: anthropic, google."
            )

    @staticmethod
    def _anthropic_definition(
        *,
        allowed_domains: Optional[list[str]],
        blocked_domains: Optional[list[str]],
        max_uses: Optional[int],
        cache_control: "CacheControlEphemeralParam | None" = None,
        citations: "BetaCitationsConfigParam | None" = None,
        max_content_tokens: Optional[int] = None,
    ) -> "BetaWebFetchTool20250910Param":
        """Generate Anthropic/Claude web fetch tool definition."""
        # https://docs.claude.com/en/docs/agents-and-tools/tool-use/web-fetch-tool
        definition: BetaWebFetchTool20250910Param = {
            "name": "web_fetch",
            "type": "web_fetch_20250910",
        }

        if max_uses is not None:
            definition["max_uses"] = max_uses
        if allowed_domains:
            definition["allowed_domains"] = allowed_domains
        if blocked_domains:
            definition["blocked_domains"] = blocked_domains
        if cache_control is not None:
            definition["cache_control"] = cache_control
        if citations is not None:
            definition["citations"] = citations
        if max_content_tokens is not None:
            definition["max_content_tokens"] = max_content_tokens

        return definition

    @staticmethod
    def _google_definition() -> "UrlContext":
        """Generate Google/Gemini URL fetch tool definition."""
        # https://ai.google.dev/gemini-api/docs/url-context
        from google.genai.types import UrlContext

        return UrlContext()


def tool_web_search(
    *,
    allowed_domains: Optional[list[str]] = None,
    blocked_domains: Optional[list[str]] = None,
    user_location: "Optional[UserLocation]" = None,
    max_uses: Optional[int] = None,
) -> ToolWebSearch:
    """
    Create a web search tool for use with chat models.

    This function creates a provider-agnostic web search tool that can be
    registered with any supported chat provider. The tool allows the model
    to search the web for up-to-date information.

    Supported providers: OpenAI, Claude (Anthropic), Google (Gemini)

    Prerequisites
    -------------
    - **OpenAI**: Web search is available by default.
    - **Claude**: Web search must be enabled in the Anthropic Console by your
      organization administrator. It costs extra ($10 per 1,000 searches at
      time of writing).
    - **Google**: Web search (grounding) is available by default with Gemini.

    Parameters
    ----------
    allowed_domains
        Restrict searches to specific domains (e.g., `['nytimes.com', 'bbc.com']`).
        Supported by OpenAI and Claude. Cannot be used with `blocked_domains`.
    blocked_domains
        Exclude specific domains from searches.
        Supported by Claude and Google. Cannot be used with `allowed_domains`.
    user_location
        Location information to localize search results. A dictionary with
        optional keys: `country` (2-letter ISO code), `city`, `region`, and
        `timezone` (IANA timezone like 'America/New_York').
        Supported by OpenAI and Claude.
    max_uses
        Maximum number of searches allowed per request.
        Only supported by Claude.

    Returns
    -------
    ToolWebSearch
        A web search tool that can be registered with `chat.register_tool()`.

    Examples
    --------
    ```python
    from chatlas import ChatOpenAI, tool_web_search

    # Basic usage
    chat = ChatOpenAI()
    chat.register_tool(tool_web_search())
    chat.chat("What are the top news stories today?")

    # With domain restrictions
    chat = ChatOpenAI()
    chat.register_tool(tool_web_search(allowed_domains=["nytimes.com", "bbc.com"]))
    chat.chat("What's happening in the economy?")

    # With location for localized results
    chat = ChatOpenAI()
    chat.register_tool(
        tool_web_search(
            user_location={
                "country": "US",
                "city": "San Francisco",
                "timezone": "America/Los_Angeles",
            }
        )
    )
    chat.chat("What's the weather forecast?")
    ```

    Note
    ----
    Not all parameters are supported by all providers:

    - `allowed_domains`: OpenAI, Claude
    - `blocked_domains`: Claude, Google
    - `user_location`: OpenAI, Claude
    - `max_uses`: Claude only

    Unsupported parameters are silently ignored by providers that don't support
    them.
    """
    return ToolWebSearch(
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
        user_location=user_location,
        max_uses=max_uses,
    )


def tool_web_fetch(
    *,
    allowed_domains: Optional[list[str]] = None,
    blocked_domains: Optional[list[str]] = None,
    max_uses: Optional[int] = None,
) -> ToolWebFetch:
    """
    Create a URL fetch tool for use with chat models.

    This function creates a provider-agnostic URL fetch tool that can be
    registered with supported chat providers. The tool allows the model to
    fetch and analyze content from web URLs.

    Supported providers: Claude (Anthropic), Google (Gemini)

    Prerequisites
    -------------
    - **Claude**: The web fetch tool requires the beta header
      `anthropic-beta: web-fetch-2025-09-10`. Pass this via the `kwargs`
      parameter's `default_headers` option (see examples below).
    - **Google**: URL context is available by default with Gemini.

    Parameters
    ----------
    allowed_domains
        Restrict fetches to specific domains.
        Only supported by Claude.
    blocked_domains
        Exclude specific domains from fetches.
        Only supported by Claude. Cannot be used with `allowed_domains`.
    max_uses
        Maximum number of fetches allowed per request.
        Only supported by Claude.

    Returns
    -------
    ToolWebFetch
        A URL fetch tool that can be registered with `chat.register_tool()`.

    Examples
    --------
    ```python
    from chatlas import ChatAnthropic, tool_web_fetch

    # Basic usage with Claude (requires beta header)
    chat = ChatAnthropic(
        kwargs={"default_headers": {"anthropic-beta": "web-fetch-2025-09-10"}}
    )
    chat.register_tool(tool_web_fetch())
    chat.chat("Summarize the content at https://en.wikipedia.org/wiki/Python")

    # With domain restrictions
    chat = ChatAnthropic(
        kwargs={"default_headers": {"anthropic-beta": "web-fetch-2025-09-10"}}
    )
    chat.register_tool(tool_web_fetch(allowed_domains=["wikipedia.org", "python.org"]))
    chat.chat("Summarize the content at https://en.wikipedia.org/wiki/Guido_van_Rossum")
    ```

    Note
    ----
    For Claude, the model can only fetch URLs that appear in the conversation
    context (user messages or previous tool results). For security reasons,
    Claude cannot dynamically construct URLs to fetch.

    Using with OpenAI (and other providers)
    ---------------------------------------
    OpenAI does not have a built-in URL fetch tool. For OpenAI and other
    providers without native fetch support, use the MCP Fetch server from
    the Model Context Protocol project:
    https://github.com/modelcontextprotocol/servers/tree/main/src/fetch

    ```python
    import asyncio
    from chatlas import ChatOpenAI


    async def main():
        chat = ChatOpenAI()
        await chat.register_mcp_tools_stdio_async(
            command="npx",
            args=["-y", "@anthropic-ai/mcp-fetch"],
        )
        await chat.chat_async("Summarize the content at https://www.python.org")
        await chat.cleanup_mcp_tools()


    asyncio.run(main())
    ```

    This approach works with any provider, making it useful for consistent
    behavior across different LLM backends.
    """
    return ToolWebFetch(
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
        max_uses=max_uses,
    )
