# parallel_chat_structured

``` python
parallel_chat_structured(
    chat,
    prompts,
    data_model,
    *,
    max_active=10,
    rpm=500,
    on_error='return',
    kwargs=None,
)
```

Submit multiple chat prompts in parallel and extract structured data.

This function processes multiple prompts concurrently and extracts structured data from each response according to the specified Pydantic model type.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| chat | `ChatT` | A base chat object. | *required* |
| prompts | [list](https://docs.python.org/3/library/stdtypes.html#list)\[`ContentT`\] \| [list](https://docs.python.org/3/library/stdtypes.html#list)\[[list](https://docs.python.org/3/library/stdtypes.html#list)\[`ContentT`\]\] | A list of prompts. Each prompt can be a string or a list of string/Content objects. | *required* |
| data_model | [type](https://docs.python.org/3/library/functions.html#type)\[`BaseModelT`\] | A Pydantic model class defining the structure to extract. | *required* |
| max_active | [int](https://docs.python.org/3/library/functions.html#int) | The maximum number of simultaneous requests to send. | `10` |
| rpm | [int](https://docs.python.org/3/library/functions.html#int) | Maximum number of requests per minute. Default is 500. | `500` |
| on_error | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['return', 'continue', 'stop'\] | What to do when a request fails. One of: \* `"return"` (the default): stop processing new requests, wait for in-flight requests to finish, then return. \* `"continue"`: keep going, performing every request. \* `"stop"`: stop processing and throw an error. | `'return'` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[dict](https://docs.python.org/3/library/stdtypes.html#dict)\[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)\]\] | Additional keyword arguments to pass to the chat method. | `None` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | A list with one element for each prompt. Each element is either a |  |
|  | `~chatlas.types.StructuredChatResult` (if successful), `None` (if the |  |
|  | request wasn't submitted), or an error object (if it failed). Note that the |  |
|  | `StructuredChatResult` contains both the extracted data (for convenience) |  |
|  | and the full Chat object (for completeness). |  |

## Examples

Extract structured data from multiple prompts:

``` python
import chatlas as ctl
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int


chat = ctl.ChatOpenAI()

prompts = [
    "I go by Alex. 42 years on this planet and counting.",
    "Pleased to meet you! I'm Jamal, age 27.",
    "They call me Li Wei. Nineteen years young.",
    "Fatima here. Just celebrated my 35th birthday last week.",
]

# NOTE: if running from a script, you'd need to wrap this in an async
# function and call asyncio.run(main())
people = await ctl.parallel_chat_structured(chat, prompts, Person)
for person in people:
    print(f"{person.data.name} is {person.data.age} years old")
```

## See Also

- [`parallel_chat`](https://posit-dev.github.io/chatlas/reference/parallel_chat.html#chatlas.parallel_chat) : Get full Chat objects
- [`parallel_chat_text`](https://posit-dev.github.io/chatlas/reference/parallel_chat_text.html#chatlas.parallel_chat_text) : Get just the text responses
- `structured_data` : Extract data from a single chat
