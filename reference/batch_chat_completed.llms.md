# batch_chat_completed

``` python
batch_chat_completed(chat, prompts, path)
```

Check if a batch job is completed without waiting.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| chat | [Chat](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat) | Chat instance used for the batch | *required* |
| prompts | [list](https://docs.python.org/3/library/stdtypes.html#list)\[`ContentT`\] \| [list](https://docs.python.org/3/library/stdtypes.html#list)\[[list](https://docs.python.org/3/library/stdtypes.html#list)\[`ContentT`\]\] | List of prompts used for the batch | *required* |
| path | [Union](https://docs.python.org/3/library/typing.html#typing.Union)\[[str](https://docs.python.org/3/library/stdtypes.html#str), [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)\] | Path to batch state file | *required* |

## Returns

| Name | Type                                       | Description |
|------|--------------------------------------------|-------------|
|      | True if batch is complete, False otherwise |             |
