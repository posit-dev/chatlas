# parallel_chat

``` python
parallel_chat(
    chat,
    prompts,
    *,
    max_active=10,
    rpm=500,
    on_error='return',
    kwargs=None,
)
```

Submit multiple chat prompts in parallel.

If you have multiple prompts, you can submit them in parallel. This is typically considerably faster than submitting them in sequence, especially with providers like OpenAI and Google.

If using [`ChatOpenAI`](https://posit-dev.github.io/chatlas/reference/ChatOpenAI.html#chatlas.ChatOpenAI) or [`ChatAnthropic`](https://posit-dev.github.io/chatlas/reference/ChatAnthropic.html#chatlas.ChatAnthropic) and if you’re willing to wait longer, you might want to use `batch_chat()` instead, as it comes with a 50% discount in return for taking up to 24 hours.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| chat | `ChatT` | A base chat object. | *required* |
| prompts | [list](https://docs.python.org/3/library/stdtypes.html#list)\[`ContentT`\] \| [list](https://docs.python.org/3/library/stdtypes.html#list)\[[list](https://docs.python.org/3/library/stdtypes.html#list)\[`ContentT`\]\] | A list of prompts. Each prompt can be a string or a list of string/Content objects. | *required* |
| max_active | [int](https://docs.python.org/3/library/functions.html#int) | The maximum number of simultaneous requests to send. For Anthropic, note that the number of active connections is limited primarily by the output tokens per minute limit (OTPM) which is estimated from the `max_tokens` parameter (defaults to 4096). If your usage tier limits you to 16,000 OTPM, you should either set `max_active = 4` (16,000 / 4096) or reduce `max_tokens` via `set_model_params()`. | `10` |
| rpm | [int](https://docs.python.org/3/library/functions.html#int) | Maximum number of requests per minute. Default is 500. | `500` |
| on_error | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['return', 'continue', 'stop'\] | What to do when a request fails. One of: \* `"return"` (the default): stop processing new requests, wait for in-flight requests to finish, then return. \* `"continue"`: keep going, performing every request. \* `"stop"`: stop processing and throw an error. | `'return'` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[dict](https://docs.python.org/3/library/stdtypes.html#dict)\[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)\]\] | Additional keyword arguments to pass to the chat method. | `None` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | A list with one element for each prompt. Each element is either a Chat |  |
|  | object (if successful), None (if the request wasn't submitted), or an |  |
|  | error object (if it failed). |  |

## Examples

Basic usage with multiple prompts:

``` python
import asyncio
import chatlas as ctl

chat = ctl.ChatOpenAI()
countries = ["Canada", "New Zealand", "Jamaica", "United States"]
prompts = [f"What's the capital of {country}?" for country in countries]

# NOTE: if running from a script, you'd need to wrap this in an async function
# and call asyncio.run(main())
chats = await ctl.parallel_chat(chat, prompts)
```

Using with interpolation:

``` python
import chatlas as ctl

chat = ctl.ChatOpenAI()
template = "What's the capital of {{ country }}?"

countries = ["Canada", "New Zealand", "Jamaica"]
prompts = [ctl.interpolate(template, variables={"country": c}) for c in countries]

chats = await ctl.parallel_chat(chat, prompts, max_active=5)
```

## See Also

- [`parallel_chat_text`](https://posit-dev.github.io/chatlas/reference/parallel_chat_text.html#chatlas.parallel_chat_text) : Get just the text responses
- [`parallel_chat_structured`](https://posit-dev.github.io/chatlas/reference/parallel_chat_structured.html#chatlas.parallel_chat_structured) : Extract structured data
- [`batch_chat`](https://posit-dev.github.io/chatlas/reference/batch_chat.html#chatlas.batch_chat) : Batch API for discounted processing
