# ChatPosit

``` python
ChatPosit(
    system_prompt=None,
    base_url='https://gateway.posit.ai',
    model=None,
    credentials=None,
    cache='5m',
)
```

Chat with a model hosted by Posit AI.

[Posit AI](https://posit.ai) provides access to a curated set of models for Posit subscribers. The gateway exposes two API flavors: Claude models are served via the Anthropic Messages API and all other models are served via an OpenAI-compatible API. `ChatPosit()` automatically picks the appropriate flavor based on the model name.

## Prerequisites

> **NOTE:**
>
> Claude models require the `anthropic` package: `pip install "chatlas[posit]"`. OpenAI-compatible models work with the base `chatlas` install.

> **NOTE:**
>
> By default, `ChatPosit()` authenticates with an OAuth device flow against `login.posit.cloud`: the first time you use it, you’ll be prompted to visit a URL and enter a code. The resulting tokens are cached on disk and refreshed automatically, so you should only need to do this once per machine.

## Examples

``` python
from chatlas import ChatPosit

chat = ChatPosit()
chat.chat("Tell me three jokes about statisticians")
```

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| system_prompt | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A system prompt to set the behavior of the assistant. | `None` |
| base_url | [str](https://docs.python.org/3/library/stdtypes.html#str) | The base URL of the Posit AI gateway. | `'https://gateway.posit.ai'` |
| model | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The model to use for the chat. The default, None, will pick a reasonable default, and warn you about it. We strongly recommend explicitly choosing a model for all but the most casual use. | `None` |
| credentials | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)\[\[\], [str](https://docs.python.org/3/library/stdtypes.html#str)\]\] | A zero-argument function that returns a bearer token string. If omitted, `ChatPosit()` manages the OAuth device-flow login and its on-disk token cache automatically. | `None` |
| cache | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['5m', '1h', 'none'\] | How long to cache inputs? Defaults to “5m” (five minutes). Only applies when a Claude model is selected. See `ChatAnthropic` for details. | `'5m'` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | [Chat](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat) | A chat object that retains the state of the conversation. Note that the concrete provider (and hence some response-object typing precision) depends on whether a Claude model or another model was selected. |
