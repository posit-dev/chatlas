"""Tests for built-in web search and fetch tools."""

import pytest

from chatlas import (
    ChatAnthropic,
    ChatGoogle,
    ChatOpenAI,
    tool_web_fetch,
    tool_web_search,
)
from chatlas._tools_builtin import ToolWebFetch, ToolWebSearch


class TestToolWebSearchConfiguration:
    """Test ToolWebSearch configuration and validation."""

    def test_basic_configuration(self):
        """Test creating a basic web search tool."""
        tool = tool_web_search()
        assert isinstance(tool, ToolWebSearch)
        assert tool.name == "web_search"
        assert tool.allowed_domains is None
        assert tool.blocked_domains is None
        assert tool.user_location is None
        assert tool.max_uses is None

    def test_with_allowed_domains(self):
        """Test web search tool with allowed domains."""
        tool = tool_web_search(allowed_domains=["nytimes.com", "bbc.com"])
        assert tool.allowed_domains == ["nytimes.com", "bbc.com"]
        assert tool.blocked_domains is None

    def test_with_blocked_domains(self):
        """Test web search tool with blocked domains."""
        tool = tool_web_search(blocked_domains=["spam.com"])
        assert tool.blocked_domains == ["spam.com"]
        assert tool.allowed_domains is None

    def test_cannot_use_both_domain_filters(self):
        """Test that both allowed and blocked domains cannot be specified."""
        with pytest.raises(
            ValueError, match="Cannot specify both allowed_domains and blocked_domains"
        ):
            tool_web_search(
                allowed_domains=["good.com"], blocked_domains=["bad.com"]
            )

    def test_with_user_location(self):
        """Test web search tool with user location."""
        location = {
            "country": "US",
            "city": "San Francisco",
            "region": "California",
            "timezone": "America/Los_Angeles",
        }
        tool = tool_web_search(user_location=location)
        assert tool.user_location == location

    def test_with_max_uses(self):
        """Test web search tool with max uses."""
        tool = tool_web_search(max_uses=5)
        assert tool.max_uses == 5


class TestToolWebFetchConfiguration:
    """Test ToolWebFetch configuration and validation."""

    def test_basic_configuration(self):
        """Test creating a basic web fetch tool."""
        tool = tool_web_fetch()
        assert isinstance(tool, ToolWebFetch)
        assert tool.name == "web_fetch"
        assert tool.allowed_domains is None
        assert tool.blocked_domains is None
        assert tool.max_uses is None

    def test_with_allowed_domains(self):
        """Test web fetch tool with allowed domains."""
        tool = tool_web_fetch(allowed_domains=["example.com"])
        assert tool.allowed_domains == ["example.com"]

    def test_cannot_use_both_domain_filters(self):
        """Test that both allowed and blocked domains cannot be specified."""
        with pytest.raises(
            ValueError, match="Cannot specify both allowed_domains and blocked_domains"
        ):
            tool_web_fetch(
                allowed_domains=["good.com"], blocked_domains=["bad.com"]
            )


class TestToolWebSearchProviderDefinitions:
    """Test that provider-specific definitions are generated correctly."""

    def test_openai_basic_definition(self):
        """Test OpenAI definition for basic web search."""
        tool = tool_web_search()
        definition = tool.get_definition("openai")
        assert definition["type"] == "web_search"

    def test_openai_with_allowed_domains(self):
        """Test OpenAI definition with allowed domains."""
        tool = tool_web_search(allowed_domains=["https://nytimes.com", "bbc.com"])
        definition = tool.get_definition("openai")
        assert definition["type"] == "web_search"
        # Should strip http/https prefixes
        assert definition["filters"]["allowed_domains"] == ["nytimes.com", "bbc.com"]

    def test_openai_with_user_location(self):
        """Test OpenAI definition with user location."""
        tool = tool_web_search(user_location={"country": "US", "city": "NYC"})
        definition = tool.get_definition("openai")
        assert definition["user_location"]["type"] == "approximate"
        assert definition["user_location"]["country"] == "US"
        assert definition["user_location"]["city"] == "NYC"

    def test_anthropic_basic_definition(self):
        """Test Anthropic definition for basic web search."""
        tool = tool_web_search()
        definition = tool.get_definition("anthropic")
        assert definition["name"] == "web_search"
        assert definition["type"] == "web_search_20250305"

    def test_anthropic_with_all_options(self):
        """Test Anthropic definition with all options."""
        tool = tool_web_search(
            allowed_domains=["nytimes.com"],
            user_location={"country": "US"},
            max_uses=10,
        )
        definition = tool.get_definition("anthropic")
        assert definition["allowed_domains"] == ["nytimes.com"]
        assert definition["user_location"] == {"type": "approximate", "country": "US"}
        assert definition["max_uses"] == 10

    def test_anthropic_with_blocked_domains(self):
        """Test Anthropic definition with blocked domains."""
        tool = tool_web_search(blocked_domains=["spam.com"])
        definition = tool.get_definition("anthropic")
        assert definition["blocked_domains"] == ["spam.com"]

    def test_google_definition(self):
        """Test Google definition for web search."""
        from google.genai.types import GoogleSearch

        tool = tool_web_search()
        definition = tool.get_definition("google")
        assert isinstance(definition, GoogleSearch)

    def test_google_with_blocked_domains(self):
        """Test Google definition with blocked domains."""
        tool = tool_web_search(blocked_domains=["spam.com", "ads.com"])
        definition = tool.get_definition("google")
        assert definition.exclude_domains == ["spam.com", "ads.com"]

    def test_unsupported_provider(self):
        """Test that unsupported provider raises error."""
        tool = tool_web_search()
        with pytest.raises(ValueError, match="Web search is not supported"):
            tool.get_definition("unsupported_provider")

    def test_openai_warns_on_unsupported_params(self):
        """Test that OpenAI warns about unsupported parameters."""
        tool = tool_web_search(blocked_domains=["spam.com"], max_uses=5)
        with pytest.warns(UserWarning, match="blocked_domains is not supported by OpenAI"):
            tool.get_definition("openai")

    def test_google_warns_on_unsupported_params(self):
        """Test that Google warns about unsupported parameters."""
        tool = tool_web_search(
            allowed_domains=["example.com"],
            user_location={"country": "US"},
            max_uses=5,
        )
        with pytest.warns(UserWarning) as record:
            tool.get_definition("google")
        messages = [str(w.message) for w in record]
        assert any("allowed_domains" in m for m in messages)
        assert any("user_location" in m for m in messages)
        assert any("max_uses" in m for m in messages)

    def test_openai_with_search_context_size(self):
        """Test that OpenAI definition accepts search_context_size parameter."""
        tool = tool_web_search()
        definition = tool.get_definition("openai", search_context_size="high")
        assert definition["search_context_size"] == "high"

    def test_anthropic_with_cache_control(self):
        """Test that Anthropic definition accepts cache_control parameter."""
        tool = tool_web_search()
        definition = tool.get_definition(
            "anthropic", cache_control={"type": "ephemeral"}
        )
        assert definition["cache_control"] == {"type": "ephemeral"}


class TestToolWebFetchProviderDefinitions:
    """Test that provider-specific fetch definitions are generated correctly."""

    def test_anthropic_basic_definition(self):
        """Test Anthropic definition for basic web fetch."""
        tool = tool_web_fetch()
        definition = tool.get_definition("anthropic")
        assert definition["name"] == "web_fetch"
        assert definition["type"] == "web_fetch_20250910"

    def test_anthropic_with_all_options(self):
        """Test Anthropic definition with all options."""
        tool = tool_web_fetch(
            allowed_domains=["example.com"],
            max_uses=5,
        )
        definition = tool.get_definition("anthropic")
        assert definition["allowed_domains"] == ["example.com"]
        assert definition["max_uses"] == 5

    def test_google_definition(self):
        """Test Google definition for web fetch."""
        from google.genai.types import UrlContext

        tool = tool_web_fetch()
        definition = tool.get_definition("google")
        assert isinstance(definition, UrlContext)

    def test_openai_not_supported(self):
        """Test that OpenAI raises error for web fetch."""
        tool = tool_web_fetch()
        with pytest.raises(ValueError, match="Web fetch is not supported"):
            tool.get_definition("openai")

    def test_google_warns_on_unsupported_params(self):
        """Test that Google warns about unsupported parameters."""
        tool = tool_web_fetch(
            allowed_domains=["example.com"],
            blocked_domains=None,
            max_uses=5,
        )
        # Reset to test blocked_domains separately
        tool2 = tool_web_fetch(blocked_domains=["spam.com"])

        with pytest.warns(UserWarning) as record:
            tool.get_definition("google")
        messages = [str(w.message) for w in record]
        assert any("allowed_domains" in m for m in messages)
        assert any("max_uses" in m for m in messages)

        with pytest.warns(UserWarning, match="blocked_domains is not supported by Google"):
            tool2.get_definition("google")

    def test_anthropic_with_extra_params(self):
        """Test that Anthropic definition accepts provider-specific parameters."""
        tool = tool_web_fetch()
        definition = tool.get_definition(
            "anthropic",
            cache_control={"type": "ephemeral"},
            max_content_tokens=1000,
        )
        assert definition["cache_control"] == {"type": "ephemeral"}
        assert definition["max_content_tokens"] == 1000


class TestChatRegistration:
    """Test registering web search tools with chat instances."""

    def test_register_web_search_openai(self):
        """Test registering web search tool with OpenAI chat."""
        chat = ChatOpenAI()
        tool = tool_web_search()
        chat.register_tool(tool)

        tools = chat.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "web_search"
        assert isinstance(tools[0], ToolWebSearch)

    def test_register_web_search_anthropic(self):
        """Test registering web search tool with Anthropic chat."""
        chat = ChatAnthropic()
        tool = tool_web_search()
        chat.register_tool(tool)

        tools = chat.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "web_search"

    def test_register_web_search_google(self):
        """Test registering web search tool with Google chat."""
        chat = ChatGoogle()
        tool = tool_web_search()
        chat.register_tool(tool)

        tools = chat.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "web_search"

    def test_register_web_fetch_anthropic(self):
        """Test registering web fetch tool with Anthropic chat."""
        chat = ChatAnthropic()
        tool = tool_web_fetch()
        chat.register_tool(tool)

        tools = chat.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "web_fetch"

    def test_register_multiple_builtin_tools(self):
        """Test registering multiple built-in tools."""
        chat = ChatAnthropic()
        chat.register_tool(tool_web_search())
        chat.register_tool(tool_web_fetch())

        tools = chat.get_tools()
        assert len(tools) == 2
        tool_names = {t.name for t in tools}
        assert tool_names == {"web_search", "web_fetch"}

    def test_register_builtin_and_regular_tools(self):
        """Test registering both built-in and regular tools."""
        chat = ChatOpenAI()

        def add(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        chat.register_tool(tool_web_search())
        chat.register_tool(add)

        tools = chat.get_tools()
        assert len(tools) == 2
        tool_names = {t.name for t in tools}
        assert tool_names == {"web_search", "add"}
