# ChatGoogle

``` python
ChatGoogle(
    system_prompt=None,
    model=None,
    reasoning=None,
    api_key=None,
    kwargs=None,
)
```

Chat with a Google Gemini model.

## Prerequisites

> **NOTE:**
>
> To use Google’s models (i.e., Gemini), you’ll need to sign up for an account and [get an API key](https://ai.google.dev/gemini-api/docs/get-started/tutorial?lang=python).

> **NOTE:**
>
> `ChatGoogle` requires the `google-genai` package: `pip install "chatlas[google]"`.

## Examples

``` python
import os
from chatlas import ChatGoogle

chat = ChatGoogle(api_key=os.getenv("GOOGLE_API_KEY"))
chat.chat("What is the capital of France?")
```

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| system_prompt | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A system prompt to set the behavior of the assistant. | `None` |
| model | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The model to use for the chat. The default, None, will pick a reasonable default, and warn you about it. We strongly recommend explicitly choosing a model for all but the most casual use. | `None` |
| reasoning | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\['int \| ThinkingConfigDict'\] | If provided, enables reasoning (a.k.a. “thoughts”) in the model’s responses. This can be an integer number of tokens to use for reasoning, or a full `ThinkingConfigDict` to customize the reasoning behavior. | `None` |
| api_key | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The API key to use for authentication. You generally should not supply this directly, but instead set the `GOOGLE_API_KEY` environment variable. | `None` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\['ChatClientArgs'\] | Additional arguments to pass to the `genai.Client` constructor. | `None` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | [Chat](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat) | A Chat object. |

## Note

Pasting an API key into a chat constructor (e.g., `ChatGoogle(api_key="...")`) is the simplest way to get started, and is fine for interactive use, but is problematic for code that may be shared with others.

Instead, consider using environment variables or a configuration file to manage your credentials. One popular way to manage credentials is to use a `.env` file to store your credentials, and then use the `python-dotenv` package to load them into your environment.

``` shell
pip install python-dotenv
```

``` shell
# .env
GOOGLE_API_KEY=...
```

``` python
from chatlas import ChatGoogle
from dotenv import load_dotenv

load_dotenv()
chat = ChatGoogle()
chat.console()
```

Another, more general, solution is to load your environment variables into the shell before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

``` shell
export GOOGLE_API_KEY=...
```
