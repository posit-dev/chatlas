# tool_web_search

``` python
tool_web_search(
    allowed_domains=None,
    blocked_domains=None,
    user_location=None,
    max_uses=None,
)
```

Create a web search tool for use with chat models.

This function creates a provider-agnostic web search tool that can be registered with any supported chat provider. The tool allows the model to search the web for up-to-date information.

Supported providers: OpenAI, Claude (Anthropic), Google (Gemini)

## Prerequisites

- **OpenAI**: Web search is available by default.
- **Claude**: Web search must be enabled in the Anthropic Console by your organization administrator. It costs extra (\$10 per 1,000 searches at time of writing).
- **Google**: Web search (grounding) is available by default with Gemini.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| allowed_domains | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[list](https://docs.python.org/3/library/stdtypes.html#list)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\]\] | Restrict searches to specific domains (e.g., `['nytimes.com', 'bbc.com']`). Supported by OpenAI and Claude. Cannot be used with `blocked_domains`. | `None` |
| blocked_domains | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[list](https://docs.python.org/3/library/stdtypes.html#list)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\]\] | Exclude specific domains from searches. Supported by Claude and Google. Cannot be used with `allowed_domains`. | `None` |
| user_location | 'Optional\[UserLocation\]' | Location information to localize search results. A dictionary with optional keys: `country` (2-letter ISO code), `city`, `region`, and `timezone` (IANA timezone like ‘America/New_York’). Supported by OpenAI and Claude. | `None` |
| max_uses | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[int](https://docs.python.org/3/library/functions.html#int)\] | Maximum number of searches allowed per request. Only supported by Claude. | `None` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | `ToolWebSearch` | A web search tool that can be registered with `chat.register_tool()`. |

## Examples

``` python
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

## Note

Not all parameters are supported by all providers:

- `allowed_domains`: OpenAI, Claude
- `blocked_domains`: Claude, Google
- `user_location`: OpenAI, Claude
- `max_uses`: Claude only

Unsupported parameters are silently ignored by providers that don’t support them.
