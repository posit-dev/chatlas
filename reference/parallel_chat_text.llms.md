# parallel_chat_text

``` python
parallel_chat_text(
    chat,
    prompts,
    *,
    max_active=10,
    rpm=500,
    on_error='return',
    kwargs=None,
)
```

Submit multiple chat prompts in parallel and return text responses.

This is a convenience function that wraps `parallel_chat()` and extracts just the text content from each response.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| chat | [Chat](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat) | A base chat object. | *required* |
| prompts | [list](https://docs.python.org/3/library/stdtypes.html#list)\[`ContentT`\] \| [list](https://docs.python.org/3/library/stdtypes.html#list)\[[list](https://docs.python.org/3/library/stdtypes.html#list)\[`ContentT`\]\] | A list of prompts. Each prompt can be a string or a list of string/Content objects. | *required* |
| max_active | [int](https://docs.python.org/3/library/functions.html#int) | The maximum number of simultaneous requests to send. | `10` |
| rpm | [int](https://docs.python.org/3/library/functions.html#int) | Maximum number of requests per minute. Default is 500. | `500` |
| on_error | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['return', 'continue', 'stop'\] | What to do when a request fails. One of: \* `"return"` (the default): stop processing new requests, wait for in-flight requests to finish, then return. \* `"continue"`: keep going, performing every request. \* `"stop"`: stop processing and throw an error. | `'return'` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[dict](https://docs.python.org/3/library/stdtypes.html#dict)\[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)\]\] | Additional keyword arguments to pass to the chat method. | `None` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | A list with one element for each prompt. Each element is either a string (if |  |
|  | successful), None (if the request wasn't submitted), or an error object (if |  |
|  | it failed). |  |

## Examples

``` python
import chatlas as ctl

chat = ctl.ChatOpenAI()

countries = ["Canada", "New Zealand", "Jamaica", "United States"]
prompts = [f"What's the capital of {country}?" for country in countries]

# NOTE: if running from a script, you'd need to wrap this in an async function
# and call asyncio.run(main())
responses = await ctl.parallel_chat_text(chat, prompts)
for country, response in zip(countries, responses):
    print(f"{country}: {response}")
```

## See Also

- [`parallel_chat`](https://posit-dev.github.io/chatlas/reference/parallel_chat.html#chatlas.parallel_chat) : Get full Chat objects
- [`parallel_chat_structured`](https://posit-dev.github.io/chatlas/reference/parallel_chat_structured.html#chatlas.parallel_chat_structured) : Extract structured data
