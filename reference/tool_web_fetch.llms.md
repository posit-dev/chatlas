# tool_web_fetch

``` python
tool_web_fetch(allowed_domains=None, blocked_domains=None, max_uses=None)
```

Create a URL fetch tool for use with chat models.

This function creates a provider-agnostic URL fetch tool that can be registered with supported chat providers. The tool allows the model to fetch and analyze content from web URLs.

Supported providers: Claude (Anthropic), Google (Gemini)

## Prerequisites

- **Claude**: The web fetch tool requires the beta header `anthropic-beta: web-fetch-2025-09-10`. Pass this via the `kwargs` parameter’s `default_headers` option (see examples below).
- **Google**: URL context is available by default with Gemini.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| allowed_domains | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[list](https://docs.python.org/3/library/stdtypes.html#list)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\]\] | Restrict fetches to specific domains. Only supported by Claude. | `None` |
| blocked_domains | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[list](https://docs.python.org/3/library/stdtypes.html#list)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\]\] | Exclude specific domains from fetches. Only supported by Claude. Cannot be used with `allowed_domains`. | `None` |
| max_uses | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[int](https://docs.python.org/3/library/functions.html#int)\] | Maximum number of fetches allowed per request. Only supported by Claude. | `None` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | `ToolWebFetch` | A URL fetch tool that can be registered with `chat.register_tool()`. |

## Examples

``` python
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

## Note

For Claude, the model can only fetch URLs that appear in the conversation context (user messages or previous tool results). For security reasons, Claude cannot dynamically construct URLs to fetch.

## Using with OpenAI (and other providers)

OpenAI does not have a built-in URL fetch tool. For OpenAI and other providers without native fetch support, use the MCP Fetch server from the Model Context Protocol project: https://github.com/modelcontextprotocol/servers/tree/main/src/fetch

``` python
import asyncio
from chatlas import ChatOpenAI


async def main():
    chat = ChatOpenAI()
    await chat.register_mcp_tools_stdio_async(
        command="uvx",
        args=["mcp-server-fetch"],
    )
    await chat.chat_async("Summarize the content at https://www.python.org")
    await chat.cleanup_mcp_tools()


asyncio.run(main())
```

This approach works with any provider, making it useful for consistent behavior across different LLM backends.
