# ChatOpenRouter

``` python
ChatOpenRouter(
    system_prompt=None,
    model=None,
    api_key=None,
    base_url='https://openrouter.ai/api/v1',
    seed=MISSING,
    kwargs=None,
)
```

Chat with one of the many models hosted on OpenRouter.

OpenRouter provides access to a wide variety of language models from different providers through a unified API. Support for features depends on the underlying model that you use.

## Prerequisites

> **NOTE:**
>
> Sign up at <https://openrouter.ai> to get an API key.

## Examples

``` python
import os
from chatlas import ChatOpenRouter

chat = ChatOpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))
chat.chat("What is the capital of France?")
```

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| system_prompt | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A system prompt to set the behavior of the assistant. | `None` |
| model | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The model to use for the chat. The default, None, will pick a reasonable default, and warn you about it. We strongly recommend explicitly choosing a model for all but the most casual use. See <https://openrouter.ai/models> for available models. | `None` |
| api_key | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The API key to use for authentication. You generally should not supply this directly, but instead set the `OPENROUTER_API_KEY` environment variable. | `None` |
| base_url | [str](https://docs.python.org/3/library/stdtypes.html#str) | The base URL to the endpoint; the default uses OpenRouter’s API. | `'https://openrouter.ai/api/v1'` |
| seed | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[int](https://docs.python.org/3/library/functions.html#int)\] \| [MISSING_TYPE](https://posit-dev.github.io/chatlas/reference/types.MISSING_TYPE.html#chatlas.types.MISSING_TYPE) | Optional integer seed that the model uses to try and make output more reproducible. | `MISSING` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\['ChatClientArgs'\] | Additional arguments to pass to the `openai.OpenAI()` client constructor. | `None` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | [Chat](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat) | A chat object that retains the state of the conversation. |

## Note

This function is a lightweight wrapper around [`ChatOpenAI`](https://posit-dev.github.io/chatlas/reference/ChatOpenAI.html#chatlas.ChatOpenAI) with the defaults tweaked for OpenRouter.

## Note

Pasting an API key into a chat constructor (e.g., `ChatOpenRouter(api_key="...")`) is the simplest way to get started, and is fine for interactive use, but is problematic for code that may be shared with others.

Instead, consider using environment variables or a configuration file to manage your credentials. One popular way to manage credentials is to use a `.env` file to store your credentials, and then use the `python-dotenv` package to load them into your environment.

``` shell
pip install python-dotenv
```

``` shell
# .env
OPENROUTER_API_KEY=...
```

``` python
from chatlas import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()
chat = ChatOpenRouter()
chat.console()
```

Another, more general, solution is to load your environment variables into the shell before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

``` shell
export OPENROUTER_API_KEY=...
```
