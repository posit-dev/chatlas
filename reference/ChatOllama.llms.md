# ChatOllama

``` python
ChatOllama(
    model=None,
    *,
    system_prompt=None,
    base_url='http://localhost:11434',
    seed=MISSING,
    kwargs=None,
)
```

Chat with a local Ollama model.

[Ollama](https://ollama.com) makes it easy to run a wide-variety of open-source models locally, making it a great choice for privacy and security.

## Prerequisites

> **NOTE:**
>
> `ChatOllama` requires the [ollama](https://ollama.com/download) executable to be installed and running on your machine.

> **NOTE:**
>
> Once ollama is running locally, download a model from the command line (e.g. `ollama pull llama3.2`).

## Examples

``` python
from chatlas import ChatOllama

chat = ChatOllama(model="llama3.2")
chat.chat("What is the capital of France?")
```

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| model | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The model to use for the chat. If `None`, a list of locally installed models will be printed. | `None` |
| system_prompt | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A system prompt to set the behavior of the assistant. | `None` |
| base_url | [str](https://docs.python.org/3/library/stdtypes.html#str) | The base URL to the endpoint; the default uses ollama’s API. | `'http://localhost:11434'` |
| seed | [int](https://docs.python.org/3/library/functions.html#int) \| None \| [MISSING_TYPE](https://posit-dev.github.io/chatlas/reference/types.MISSING_TYPE.html#chatlas.types.MISSING_TYPE) | Optional integer seed that helps to make output more reproducible. | `MISSING` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\['ChatClientArgs'\] | Additional arguments to pass to the `openai.OpenAI()` client constructor. | `None` |

## Note

This function is a lightweight wrapper around `ChatOpenAICompletions` with the defaults tweaked for ollama.

## Limitations

`ChatOllama` currently doesn’t work with streaming tools, and tool calling more generally doesn’t seem to work very well with currently available models.
