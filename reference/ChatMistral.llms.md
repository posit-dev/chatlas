# ChatMistral

``` python
ChatMistral(
    system_prompt=None,
    model=None,
    api_key=None,
    base_url='https://api.mistral.ai/v1/',
    seed=MISSING,
    kwargs=None,
)
```

Chat with a model hosted on Mistral’s La Plateforme.

Mistral AI provides high-performance language models through their API platform.

## Prerequisites

> **NOTE:**
>
> Get your API key from https://console.mistral.ai/api-keys.

## Examples

``` python
import os
from chatlas import ChatMistral

chat = ChatMistral(api_key=os.getenv("MISTRAL_API_KEY"))
chat.chat("Tell me three jokes about statisticians")
```

## Known limitations

- Tool calling may be unstable.
- Images require a model that supports vision.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| system_prompt | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A system prompt to set the behavior of the assistant. | `None` |
| model | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The model to use for the chat. The default, None, will pick a reasonable default, and warn you about it. We strongly recommend explicitly choosing a model for all but the most casual use. | `None` |
| api_key | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The API key to use for authentication. You generally should not supply this directly, but instead set the `MISTRAL_API_KEY` environment variable. | `None` |
| base_url | [str](https://docs.python.org/3/library/stdtypes.html#str) | The base URL to the endpoint; the default uses Mistral AI. | `'https://api.mistral.ai/v1/'` |
| seed | [int](https://docs.python.org/3/library/functions.html#int) \| None \| [MISSING_TYPE](https://posit-dev.github.io/chatlas/reference/types.MISSING_TYPE.html#chatlas.types.MISSING_TYPE) | Optional integer seed that Mistral uses to try and make output more reproducible. | `MISSING` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\['ChatClientArgs'\] | Additional arguments to pass to the `openai.OpenAI()` client constructor (Mistral uses OpenAI-compatible API). | `None` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | [Chat](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat) | A chat object that retains the state of the conversation. |

## Note

Pasting an API key into a chat constructor (e.g., `ChatMistral(api_key="...")`) is the simplest way to get started, and is fine for interactive use, but is problematic for code that may be shared with others.

Instead, consider using environment variables or a configuration file to manage your credentials. One popular way to manage credentials is to use a `.env` file to store your credentials, and then use the `python-dotenv` package to load them into your environment.

``` shell
pip install python-dotenv
```

``` shell
# .env
MISTRAL_API_KEY=...
```

``` python
from chatlas import ChatMistral
from dotenv import load_dotenv

load_dotenv()
chat = ChatMistral()
chat.console()
```

Another, more general, solution is to load your environment variables into the shell before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

``` shell
export MISTRAL_API_KEY=...
```
