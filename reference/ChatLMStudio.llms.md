# ChatLMStudio

``` python
ChatLMStudio(
    model=None,
    *,
    system_prompt=None,
    base_url='http://localhost:1234',
    api_key=None,
    seed=MISSING,
    kwargs=None,
)
```

Chat with a local LM Studio model.

[LM Studio](https://lmstudio.ai) makes it easy to run a wide variety of open-source models locally on Mac, Windows, and Linux. It is particularly notable for its excellent support for Apple’s MLX inference engine, making it a compelling choice for local inference on Apple Silicon.

## Prerequisites

> **NOTE:**
>
> `ChatLMStudio` requires [LM Studio](https://lmstudio.ai/download) to be installed and running on your machine with at least one model loaded.

> **NOTE:**
>
> Open LM Studio, load a model from the Discover tab, then start the local server from the Developer tab.

## Examples

``` python
from chatlas import ChatLMStudio

chat = ChatLMStudio(model="zai-org/glm-4.7-flash")
chat.chat("What is the capital of France?")
```

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| model | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The model to use for the chat. If `None`, a list of locally loaded models will be printed. | `None` |
| system_prompt | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A system prompt to set the behavior of the assistant. | `None` |
| base_url | [str](https://docs.python.org/3/library/stdtypes.html#str) | The base URL to the endpoint. The default uses the LM Studio local server. You can also set the `LMSTUDIO_BASE_URL` environment variable to override this default. | `'http://localhost:1234'` |
| api_key | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | An optional API key. LM Studio doesn’t require credentials for local usage. If you’re accessing an LM Studio instance behind a reverse proxy or secured endpoint that enforces bearer-token authentication, you can set the `LMSTUDIO_API_KEY` environment variable or provide a value here. | `None` |
| seed | [int](https://docs.python.org/3/library/functions.html#int) \| None \| [MISSING_TYPE](https://posit-dev.github.io/chatlas/reference/types.MISSING_TYPE.html#chatlas.types.MISSING_TYPE) | Optional integer seed that helps to make output more reproducible. | `MISSING` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\['ChatClientArgs'\] | Additional arguments to pass to the `openai.OpenAI()` client constructor. | `None` |

## Note

This function is a lightweight wrapper around `ChatOpenAICompletions` with the defaults tweaked for LM Studio.
