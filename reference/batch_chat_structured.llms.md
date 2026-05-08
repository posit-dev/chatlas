# batch_chat_structured

``` python
batch_chat_structured(chat, prompts, path, data_model, wait=True)
```

Submit multiple structured data requests in a batch.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| chat | [Chat](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat) | Chat instance to use for the batch | *required* |
| prompts | [list](https://docs.python.org/3/library/stdtypes.html#list)\[`ContentT`\] \| [list](https://docs.python.org/3/library/stdtypes.html#list)\[[list](https://docs.python.org/3/library/stdtypes.html#list)\[`ContentT`\]\] | List of prompts to process | *required* |
| path | [Union](https://docs.python.org/3/library/typing.html#typing.Union)\[[str](https://docs.python.org/3/library/stdtypes.html#str), [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)\] | Path to file (with .json extension) to store batch state | *required* |
| data_model | [type](https://docs.python.org/3/library/functions.html#type)\[`BaseModelT`\] | Pydantic model class for structured responses | *required* |
| wait | [bool](https://docs.python.org/3/library/functions.html#bool) | If True, wait for batch to complete | `True` |

## Return

List of structured data objects (or None for failed requests)
