# ChatCloudflare

``` python
ChatCloudflare(
    account=None,
    system_prompt=None,
    model=None,
    api_key=None,
    seed=MISSING,
    kwargs=None,
)
```

Chat with a model hosted on Cloudflare Workers AI.

Cloudflare Workers AI hosts a variety of open-source AI models.

## Prerequisites

> **NOTE:**
>
> To use the Cloudflare API, you must have an Account ID and an Access Token, which you can obtain by following the instructions at <https://developers.cloudflare.com/workers-ai/get-started/rest-api/>.

## Examples

``` python
import os
from chatlas import ChatCloudflare

chat = ChatCloudflare(
    api_key=os.getenv("CLOUDFLARE_API_KEY"),
    account=os.getenv("CLOUDFLARE_ACCOUNT_ID"),
)
chat.chat("What is the capital of France?")
```

## Known limitations

- Tool calling does not appear to work.
- Images don’t appear to work.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| account | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The Cloudflare account ID. You generally should not supply this directly, but instead set the `CLOUDFLARE_ACCOUNT_ID` environment variable. | `None` |
| system_prompt | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A system prompt to set the behavior of the assistant. | `None` |
| model | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The model to use for the chat. The default, None, will pick a reasonable default, and warn you about it. We strongly recommend explicitly choosing a model for all but the most casual use. | `None` |
| api_key | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The API key to use for authentication. You generally should not supply this directly, but instead set the `CLOUDFLARE_API_KEY` environment variable. | `None` |
| seed | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[int](https://docs.python.org/3/library/functions.html#int)\] \| [MISSING_TYPE](https://posit-dev.github.io/chatlas/reference/types.MISSING_TYPE.html#chatlas.types.MISSING_TYPE) | Optional integer seed that ChatGPT uses to try and make output more reproducible. | `MISSING` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\['ChatClientArgs'\] | Additional arguments to pass to the `openai.OpenAI()` client constructor. | `None` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | [Chat](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat) | A chat object that retains the state of the conversation. |

## Note

This function is a lightweight wrapper around [`ChatOpenAI`](https://posit-dev.github.io/chatlas/reference/ChatOpenAI.html#chatlas.ChatOpenAI) with the defaults tweaked for Cloudflare.

## Note

Pasting credentials into a chat constructor (e.g., `ChatCloudflare(api_key="...", account="...")`) is the simplest way to get started, and is fine for interactive use, but is problematic for code that may be shared with others.

Instead, consider using environment variables or a configuration file to manage your credentials. One popular way to manage credentials is to use a `.env` file to store your credentials, and then use the `python-dotenv` package to load them into your environment.

``` shell
pip install python-dotenv
```

``` shell
# .env
CLOUDFLARE_API_KEY=...
CLOUDFLARE_ACCOUNT_ID=...
```

``` python
from chatlas import ChatCloudflare
from dotenv import load_dotenv

load_dotenv()
chat = ChatCloudflare()
chat.console()
```

Another, more general, solution is to load your environment variables into the shell before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

``` shell
export CLOUDFLARE_API_KEY=...
export CLOUDFLARE_ACCOUNT_ID=...
```
