# ChatAuto

``` python
ChatAuto(
    provider_model=None,
    *,
    system_prompt=None,
    provider=DEPRECATED,
    model=DEPRECATED,
    **kwargs,
)
```

Chat with any provider.

This is a generic interface to all the other `Chat*()` functions, allowing you to pick the provider (and model) with a simple string.

## Prerequisites

> **NOTE:**
>
> Follow the instructions for the specific provider to obtain an API key.

> **NOTE:**
>
> Follow the instructions for the specific provider to install the required Python packages.

## Examples

`ChatAuto()` makes it easy to switch between different chat providers and models.

``` python
import pandas as pd
from chatlas import ChatAuto

# Default provider (OpenAI) & model
chat = ChatAuto()
print(chat.provider.name)
print(chat.provider.model)

# Different provider (Anthropic) & default model
chat = ChatAuto("anthropic")

# List models available through the provider
models = chat.list_models()
print(pd.DataFrame(models))

# Choose specific provider/model (Claude Sonnet 4)
chat = ChatAuto("anthropic/claude-sonnet-4-0")
```

The default provider/model can also be controlled through an environment variable:

``` bash
export CHATLAS_CHAT_PROVIDER_MODEL="anthropic/claude-sonnet-4-0"
```

``` python
from chatlas import ChatAuto

chat = ChatAuto()
print(chat.provider.name)   # anthropic
print(chat.provider.model)  # claude-sonnet-4-0
```

For application-specific configurations, consider defining your own environment variables:

``` bash
export MYAPP_PROVIDER_MODEL="google/gemini-2.5-flash"
```

And passing them to `ChatAuto()` as an alternative way to configure the provider/model:

``` python
import os
from chatlas import ChatAuto

chat = ChatAuto(os.getenv("MYAPP_PROVIDER_MODEL"))
print(chat.provider.name)   # google
print(chat.provider.model)  # gemini-2.5-flash
```

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| provider_model | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The name of the provider and model to use in the format `"{provider}/{model}"`. Providers are strings formatted in kebab-case, e.g. to use `ChatBedrockAnthropic` set `provider="bedrock-anthropic"`, and models are the provider-specific model names, e.g. `"claude-sonnet-4-6"`. The `/{model}` portion may also be omitted, in which case, the default model for that provider will be used. If no value is provided, the `CHATLAS_CHAT_PROVIDER_MODEL` environment variable will be consulted for a fallback value. If this variable is also not set, a default value of `"openai"` is used. | `None` |
| system_prompt | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A system prompt to set the behavior of the assistant. | `None` |
| provider | `AutoProviders` \| [DEPRECATED_TYPE](https://posit-dev.github.io/chatlas/reference/types.MISSING_TYPE.html#chatlas.types.MISSING_TYPE) | Deprecated; use `provider_model` instead. | `DEPRECATED` |
| model | [str](https://docs.python.org/3/library/stdtypes.html#str) \| [DEPRECATED_TYPE](https://posit-dev.github.io/chatlas/reference/types.MISSING_TYPE.html#chatlas.types.MISSING_TYPE) | Deprecated; use `provider_model` instead. | `DEPRECATED` |
| \*\*kwargs |  | Additional keyword arguments to pass to the `Chat` constructor. See the documentation for each provider for more details on the available options. These arguments can also be provided via the `CHATLAS_CHAT_ARGS` environment variable as a JSON string. When any additional arguments are provided to `ChatAuto()`, the env var is ignored. Note that `system_prompt` and `turns` can’t be set via environment variables. They must be provided/set directly to/on `ChatAuto()`. | `{}` |

## Note

If you want to work with a specific provider, but don’t know what models are available (or the exact model name), use `ChatAuto('provider_name').list_models()` to list available models. Another option is to use the provider more directly (e.g., `ChatAnthropic()`). There, the `model` parameter may have type hints for available models.

## Returns

| Name | Type | Description |
|----|----|----|
|  | [Chat](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat) | A chat instance using the specified provider. |

## Raises

| Name | Type | Description |
|----|----|----|
|  | [ValueError](https://docs.python.org/3/library/exceptions.html#ValueError) | If no valid provider is specified either through parameters or environment variables. |
