# ChatVertex

``` python
ChatVertex(
    model=None,
    project=None,
    location=None,
    api_key=None,
    system_prompt=None,
    kwargs=None,
)
```

Chat with a Google Vertex AI model.

## Prerequisites

> **NOTE:**
>
> `ChatGoogle` requires the `google-genai` package: `pip install "chatlas[vertex]"`.

> **NOTE:**
>
> To use Google’s models (i.e., Vertex AI), you’ll need to sign up for an account with [Vertex AI](https://cloud.google.com/vertex-ai), then specify the appropriate model, project, and location.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| model | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The model to use for the chat. The default, None, will pick a reasonable default, and warn you about it. We strongly recommend explicitly choosing a model for all but the most casual use. | `None` |
| project | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The Google Cloud project ID (e.g., “your-project-id”). If not provided, the GOOGLE_CLOUD_PROJECT environment variable will be used. | `None` |
| location | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The Google Cloud location (e.g., “us-central1”). If not provided, the GOOGLE_CLOUD_LOCATION environment variable will be used. | `None` |
| system_prompt | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A system prompt to set the behavior of the assistant. | `None` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | [Chat](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat) | A Chat object. |

## Examples

``` python
import os
from chatlas import ChatVertex

chat = ChatVertex(
    project="your-project-id",
    location="us-central1",
)
chat.chat("What is the capital of France?")
```
