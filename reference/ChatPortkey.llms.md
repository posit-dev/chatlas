# ChatPortkey

``` python
ChatPortkey(
    system_prompt=None,
    model=None,
    api_key=None,
    virtual_key=None,
    base_url='https://api.portkey.ai/v1',
    kwargs=None,
)
```

Chat with a model hosted on PortkeyAI

[PortkeyAI](https://portkey.ai/docs/product/ai-gateway/universal-api) provides an interface (AI Gateway) to connect through its Universal API to a variety of LLMs providers with a single endpoint.

## Prerequisites

> **NOTE:**
>
> Follow the instructions at <https://portkey.ai/docs/introduction/make-your-first-request> to get started making requests to PortkeyAI. You will need to set the `PORTKEY_API_KEY` environment variable to your Portkey API key, and optionally the `PORTKEY_VIRTUAL_KEY` environment variable to your virtual key.

## Examples

``` python
import os
from chatlas import ChatPortkey

chat = ChatPortkey(api_key=os.getenv("PORTKEY_API_KEY"))
chat.chat("What is the capital of France?")
```

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| system_prompt | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A system prompt to set the behavior of the assistant. | `None` |
| model | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The model to use for the chat. The default, None, will pick a reasonable default, and warn you about it. We strongly recommend explicitly choosing a model for all but the most casual use. | `None` |
| api_key | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The API key to use for authentication. You generally should not supply this directly, but instead set the `PORTKEY_API_KEY` environment variable. | `None` |
| virtual_key | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | An (optional) virtual identifier, storing the LLM provider’s API key. See [documentation](https://portkey.ai/docs/product/ai-gateway/virtual-keys). You generally should not supply this directly, but instead set the `PORTKEY_VIRTUAL_KEY` environment variable. | `None` |
| base_url | [str](https://docs.python.org/3/library/stdtypes.html#str) | The base URL for the Portkey API. The default is suitable for most users. | `'https://api.portkey.ai/v1'` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\['ChatClientArgs'\] | Additional arguments to pass to the OpenAIProvider, such as headers or other client configuration options. | `None` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | [Chat](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat) | A chat object that retains the state of the conversation. |

## Notes

This function is a lightweight wrapper around [`ChatOpenAI`](https://posit-dev.github.io/chatlas/reference/ChatOpenAI.html#chatlas.ChatOpenAI) with the defaults tweaked for PortkeyAI.
