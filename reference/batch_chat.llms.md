# batch_chat

``` python
batch_chat(chat, prompts, path, wait=True)
```

Submit multiple chat requests in a batch.

This function allows you to submit multiple chat requests simultaneously using provider batch APIs (currently OpenAI, Anthropic, Google, and Groq). Batch processing can take up to 24 hours but offers significant cost savings.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| chat | `ChatT` | Chat instance to use for the batch | *required* |
| prompts | [list](https://docs.python.org/3/library/stdtypes.html#list)\[`ContentT`\] \| [list](https://docs.python.org/3/library/stdtypes.html#list)\[[list](https://docs.python.org/3/library/stdtypes.html#list)\[`ContentT`\]\] | List of prompts to process. Each can be a string or list of strings. | *required* |
| path | [Union](https://docs.python.org/3/library/typing.html#typing.Union)\[[str](https://docs.python.org/3/library/stdtypes.html#str), [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)\] | Path to file (with .json extension) to store batch state | *required* |
| wait | [bool](https://docs.python.org/3/library/functions.html#bool) | If True, wait for batch to complete. If False, return None if incomplete. | `True` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | List of Chat objects (one per prompt) if complete, None if wait=False and incomplete. |  |
|  | Individual Chat objects may be None if their request failed. |  |

## Example

``` python
from chatlas import ChatOpenAI

chat = ChatOpenAI()
prompts = [
    "What's the capital of France?",
    "What's the capital of Germany?",
    "What's the capital of Italy?",
]

chats = batch_chat(chat, prompts, "capitals.json")
for i, result_chat in enumerate(chats):
    if result_chat:
        print(f"Prompt {i + 1}: {result_chat.get_last_turn().text}")
```
