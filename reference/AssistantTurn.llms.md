# AssistantTurn

``` python
AssistantTurn(
    contents,
    *,
    tokens=None,
    finish_reason=None,
    completion=None,
    cost=None,
    partial_reason=None,
    **kwargs,
)
```

Assistant turn - represents model response with additional metadata

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| contents | [str](https://docs.python.org/3/library/stdtypes.html#str) \| [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)\[[Content](https://posit-dev.github.io/chatlas/reference/types.Content.html#chatlas.types.Content) \| [str](https://docs.python.org/3/library/stdtypes.html#str)\] | A list of [`Content`](https://posit-dev.github.io/chatlas/reference/types.Content.html#chatlas.types.Content) objects. | *required* |
| tokens | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)\[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int)\] \| [list](https://docs.python.org/3/library/stdtypes.html#list)\[[int](https://docs.python.org/3/library/functions.html#int)\]\] | A numeric vector of length 3 representing the number of input, output, and cached tokens (respectively) used in this turn. | `None` |
| finish_reason | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A string indicating the reason why the conversation ended. | `None` |
| completion | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[`CompletionT`\] | The completion object returned by the provider. This is useful if there’s information returned by the provider that chatlas doesn’t otherwise expose. | `None` |
| cost | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[float](https://docs.python.org/3/library/functions.html#float)\] | The cost of this turn in USD. This is computed when the turn is created based on the token usage and pricing information (including service tier). | `None` |
| partial_reason | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | If set, indicates this turn is incomplete (e.g., the stream was interrupted or cancelled). The value describes the reason for the partial state. | `None` |

## See Also

- [`Turn`](https://posit-dev.github.io/chatlas/reference/Turn.html#chatlas.Turn): The base class for all turn types.

## Attributes

| Name | Description |
|----|----|
| [is_partial](#chatlas.AssistantTurn.is_partial) | Whether this turn is a partial (interrupted/cancelled) turn. |

## Methods

| Name | Description |
|----|----|
| [validate_tokens](#chatlas.AssistantTurn.validate_tokens) | Convert list to tuple for JSON deserialization compatibility. |

### validate_tokens

``` python
AssistantTurn.validate_tokens(v)
```

Convert list to tuple for JSON deserialization compatibility.
