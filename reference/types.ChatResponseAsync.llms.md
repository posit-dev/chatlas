# types.ChatResponseAsync

``` python
types.ChatResponseAsync(generator)
```

Chat response (async) object.

An object that, when displayed, will simulatenously consume (if not already consumed) and display the response in a streaming fashion.

This is useful for interactive use: if the object is displayed, it can be viewed as it is being generated. And, if the object is not displayed, it can act like an iterator that can be consumed by something else.

## Attributes

| Name | Type | Description |
|----|----|----|
| content | [str](https://docs.python.org/3/library/stdtypes.html#str) | The content of the chat response. |

## Properties

consumed Whether the response has been consumed. If the response has been fully consumed, then it can no longer be iterated over, but the content can still be retrieved (via the `content` attribute).

## Methods

| Name | Description |
|----|----|
| [get_content](#chatlas.types.ChatResponseAsync.get_content) | Get the chat response content as a string. |

### get_content

``` python
types.ChatResponseAsync.get_content()
```

Get the chat response content as a string.
