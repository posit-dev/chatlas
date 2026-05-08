# ChatOpenAI

``` python
ChatOpenAI(
    system_prompt=None,
    model=None,
    base_url='https://api.openai.com/v1',
    reasoning=None,
    service_tier=None,
    api_key=None,
    kwargs=None,
)
```

Chat with an OpenAI model using the responses API.

[OpenAI](https://openai.com/) provides a number of chat-based models, mostly under the [ChatGPT](https://chat.openai.com/) brand.

## Prerequisites

> **NOTE:**
>
> Note that a ChatGPT Plus membership does not give you the ability to call models via the API. You will need to go to the [developer platform](https://platform.openai.com) to sign up (and pay for) a developer account that will give you an API key that you can use with this package.

## Examples

``` python
import os
from chatlas import ChatOpenAI

chat = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chat.chat("What is the capital of France?")
```

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| system_prompt | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A system prompt to set the behavior of the assistant. | `None` |
| model | 'Optional\[ResponsesModel \| str\]' | The model to use for the chat. The default, None, will pick a reasonable default, and warn you about it. We strongly recommend explicitly choosing a model for all but the most casual use. | `None` |
| base_url | [str](https://docs.python.org/3/library/stdtypes.html#str) | The base URL to the endpoint; the default uses OpenAI. | `'https://api.openai.com/v1'` |
| reasoning | 'Optional\[ReasoningEffort \| Reasoning\]' | The reasoning effort to use (for reasoning-capable models like the o and gpt-5 series). | `None` |
| service_tier | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['auto', 'default', 'flex', 'scale', 'priority'\]\] | Request a specific service tier. Options: - `"auto"` (default): uses the service tier configured in Project settings. - `"default"`: standard pricing and performance. - `"flex"`: slower and cheaper. - `"scale"`: batch-like pricing for high-volume use. - `"priority"`: faster and more expensive. | `None` |
| api_key | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The API key to use for authentication. You generally should not supply this directly, but instead set the `OPENAI_API_KEY` environment variable. | `None` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\['ChatClientArgs'\] | Additional arguments to pass to the `openai.OpenAI()` client constructor. | `None` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | [Chat](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat) | A chat object that retains the state of the conversation. |

## Note

Pasting an API key into a chat constructor (e.g., `ChatOpenAI(api_key="...")`) is the simplest way to get started, and is fine for interactive use, but is problematic for code that may be shared with others.

Instead, consider using environment variables or a configuration file to manage your credentials. One popular way to manage credentials is to use a `.env` file to store your credentials, and then use the `python-dotenv` package to load them into your environment.

``` shell
pip install python-dotenv
```

``` shell
# .env
OPENAI_API_KEY=...
```

``` python
from chatlas import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
chat = ChatOpenAI()
chat.console()
```

Another, more general, solution is to load your environment variables into the shell before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

``` shell
export OPENAI_API_KEY=...
```

## Note

The responses API does not support the `seed` parameter. If you need reproducible output, use `ChatOpenAICompletions` instead.
