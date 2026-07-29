# ChatOpenAICompletions

``` python
ChatOpenAICompletions(
    base_url='https://api.openai.com/v1',
    system_prompt=None,
    model=None,
    api_key=None,
    seed=MISSING,
    preserve_thinking=False,
    kwargs=None,
)
```

Chat with an OpenAI-compatible model (via the Chat Completions API).

Use this function to connect to any OpenAI-compatible backend, including third-party inference engines like vLLM, Ollama, LiteLLM, and others. The Chat Completions API (`/v1/chat/completions`) is the universally adopted standard across the open-source ecosystem.

For the OpenAI-specific Responses API, use [`ChatOpenAI`](https://posit-dev.github.io/chatlas/reference/ChatOpenAI.html#chatlas.ChatOpenAI) instead.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| base_url | [str](https://docs.python.org/3/library/stdtypes.html#str) | The base URL to the endpoint; the default uses OpenAI. Set this to your server’s URL for third-party backends (e.g., `"http://localhost:8000/v1"` for a local vLLM server). | `'https://api.openai.com/v1'` |
| system_prompt | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A system prompt to set the behavior of the assistant. | `None` |
| model | 'Optional\[ChatModel \| str\]' | The model to use for the chat. | `None` |
| api_key | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The API key to use for authentication. You generally should not supply this directly, but instead set the `OPENAI_API_KEY` environment variable. | `None` |
| seed | [int](https://docs.python.org/3/library/functions.html#int) \| None \| [MISSING_TYPE](https://posit-dev.github.io/chatlas/reference/types.MISSING_TYPE.html#chatlas.types.MISSING_TYPE) | Optional seed for reproducible output. | `MISSING` |
| preserve_thinking | [bool](https://docs.python.org/3/library/functions.html#bool) | If True, reasoning content returned by the model is included when sending conversation history back to the API. If False (the default), reasoning content is still captured in the turn but dropped from subsequent requests. Set to True if your provider requires or benefits from seeing prior reasoning in multi-turn conversations. | `False` |
