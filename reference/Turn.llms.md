# Turn

``` python
Turn(contents, **kwargs)
```

Base turn class

Every conversation with a chatbot consists of pairs of user and assistant turns, corresponding to an HTTP request and response. These turns are represented by `Turn` objects (or their subclasses `UserTurn`, `SystemTurn`, `AssistantTurn`), which contain a list of [`Content`](https://posit-dev.github.io/chatlas/reference/types.Content.html#chatlas.types.Content)s representing the individual messages within the turn. These might be text, images, tool requests (assistant only), or tool responses (user only).

Note that a call to `.chat()` and related functions may result in multiple user-assistant turn cycles. For example, if you have registered tools, chatlas will automatically handle the tool calling loop, which may result in any number of additional cycles.

## Examples

``` python
from chatlas import UserTurn, AssistantTurn, ChatOpenAI, ChatAnthropic

chat = ChatOpenAI()
str(chat.chat("What is the capital of France?"))
turns = chat.get_turns()
assert len(turns) == 2
assert isinstance(turns[0], UserTurn)
assert turns[0].role == "user"
assert isinstance(turns[1], AssistantTurn)
assert turns[1].role == "assistant"

# Load context into a new chat instance
chat2 = ChatAnthropic()
chat2.set_turns(turns)
turns2 = chat2.get_turns()
assert turns == turns2
```

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| contents | [str](https://docs.python.org/3/library/stdtypes.html#str) \| [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)\[[Content](https://posit-dev.github.io/chatlas/reference/types.Content.html#chatlas.types.Content) \| [str](https://docs.python.org/3/library/stdtypes.html#str)\] | A list of [`Content`](https://posit-dev.github.io/chatlas/reference/types.Content.html#chatlas.types.Content) objects. | *required* |

## Methods

| Name | Description |
|----|----|
| [to_inspect_messages](#chatlas.Turn.to_inspect_messages) | Transform this turn into a list of Inspect AI `ChatMessage` objects. |

### to_inspect_messages

``` python
Turn.to_inspect_messages(model=None)
```

Transform this turn into a list of Inspect AI `ChatMessage` objects.

Most users will not need to call this method directly. See the `.export_eval()` method on `Chat` for a higher level interface to exporting chat history for evaluation purposes.
