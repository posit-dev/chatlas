# ChatAzureOpenAI

``` python
ChatAzureOpenAI(
    endpoint,
    deployment_id,
    api_version,
    api_key=None,
    system_prompt=None,
    reasoning=None,
    service_tier=None,
    kwargs=None,
)
```

Chat with a model hosted on Azure OpenAI.

The [Azure OpenAI server](https://azure.microsoft.com/en-us/products/ai-services/openai-service) hosts a number of open source models as well as proprietary models from OpenAI.

## Examples

``` python
import os
from chatlas import ChatAzureOpenAI

chat = ChatAzureOpenAI(
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_id="REPLACE_WITH_YOUR_DEPLOYMENT_ID",
    api_version="YYYY-MM-DD",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

chat.chat("What is the capital of France?")
```

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| endpoint | [str](https://docs.python.org/3/library/stdtypes.html#str) | Azure OpenAI endpoint url with protocol and hostname, i.e. `https://{your-resource-name}.openai.azure.com`. Defaults to using the value of the `AZURE_OPENAI_ENDPOINT` environment variable. | *required* |
| deployment_id | [str](https://docs.python.org/3/library/stdtypes.html#str) | Deployment id for the model you want to use. | *required* |
| api_version | [str](https://docs.python.org/3/library/stdtypes.html#str) | The API version to use. | *required* |
| api_key | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The API key to use for authentication. You generally should not supply this directly, but instead set the `AZURE_OPENAI_API_KEY` environment variable. | `None` |
| system_prompt | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A system prompt to set the behavior of the assistant. | `None` |
| reasoning | 'Optional\[ReasoningEffort \| Reasoning\]' | The reasoning effort (e.g., `"low"`, `"medium"`, `"high"`) for reasoning-capable models like the o and gpt-5 series. To use the default reasoning settings in a way that will work for multi-turn conversations, set this to an empty dictionary `{}`. | `None` |
| service_tier | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['auto', 'default', 'flex', 'scale', 'priority'\]\] | Request a specific service tier. Options: - `"auto"` (default): uses the service tier configured in Project settings. - `"default"`: standard pricing and performance. - `"flex"`: slower and cheaper. - `"scale"`: batch-like pricing for high-volume use. - `"priority"`: faster and more expensive. | `None` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\['ChatAzureClientArgs'\] | Additional arguments to pass to the `openai.AzureOpenAI()` client constructor. | `None` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | [Chat](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat) | A Chat object. |
