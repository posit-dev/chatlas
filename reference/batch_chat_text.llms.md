# batch_chat_text

``` python
batch_chat_text(chat, prompts, path, wait=True)
```

Submit multiple chat requests in a batch and return text responses.

This is a convenience function that returns just the text of the responses rather than full Chat objects.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| chat | [Chat](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat) | Chat instance to use for the batch | *required* |
| prompts | [list](https://docs.python.org/3/library/stdtypes.html#list)\[`ContentT`\] \| [list](https://docs.python.org/3/library/stdtypes.html#list)\[[list](https://docs.python.org/3/library/stdtypes.html#list)\[`ContentT`\]\] | List of prompts to process | *required* |
| path | [Union](https://docs.python.org/3/library/typing.html#typing.Union)\[[str](https://docs.python.org/3/library/stdtypes.html#str), [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)\] | Path to file (with .json extension) to store batch state | *required* |
| wait | [bool](https://docs.python.org/3/library/functions.html#bool) | If True, wait for batch to complete. If False, return None if incomplete. | `True` |

## Return

List of text responses (or None for failed requests), or None if wait=False and incomplete.
