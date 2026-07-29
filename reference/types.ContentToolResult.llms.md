# types.ContentToolResult

``` python
types.ContentToolResult()
```

The result of calling a tool/function

A content type representing the result of a tool function call. When a model requests a tool function, [`Chat`](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat) will create, (optionally) echo, (optionally) yield, and store this content type in the chat history.

A tool function may also construct an instance of this class and return it. This is useful for a tool that wishes to customize how the result is handled (e.g., the format of the value sent to the model).

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| value |  | The return value of the tool/function. | *required* |
| model_format |  | The format used for sending the value to the model. The default, `"auto"`, first attempts to format the value as a JSON string. If that fails, it gets converted to a string via `str()`. To force `orjson.dumps()` or `str()`, set to `"json"` or `"str"`. Finally, `"as_is"` is useful for doing your own formatting and/or passing a non-string value (e.g., a list or dict) straight to the model. Non-string values are useful for tools that return images or other ‘known’ non-text content types. | *required* |
| error |  | An exception that occurred while invoking the tool. If this is set, the error message sent to the model and the value is ignored. | *required* |
| extra |  | Additional data associated with the tool result that isn’t sent to the model. | *required* |
| request |  | Not intended to be used directly. It will be set when the :class:`~chatlas.Chat` invokes the tool. | *required* |

## Note

When `model_format` is `"json"` (or `"auto"`), and the value has a `.to_json()`/`.to_dict()` method, those methods are called to obtain the JSON representation of the value. This is convenient for classes, like `pandas.DataFrame`, that have a `.to_json()` method, but don’t necessarily dump to JSON directly. If this happens to not be the desired behavior, set `model_format="as_is"` return the desired value as-is.

## Methods

| Name | Description |
|----|----|
| [get_model_value](#chatlas.types.ContentToolResult.get_model_value) | Get the actual value sent to the model. |
| [serialize_error](#chatlas.types.ContentToolResult.serialize_error) | Serialize Exception to string for JSON compatibility. |
| [tagify](#chatlas.types.ContentToolResult.tagify) | A method for rendering this object via htmltools/shiny. |
| [to_display_markdown](#chatlas.types.ContentToolResult.to_display_markdown) | Render as a fenced code block, optionally capping the value’s height. |
| [to_html](#chatlas.types.ContentToolResult.to_html) | Render as an HTML string. |
| [validate_error](#chatlas.types.ContentToolResult.validate_error) | Accept string or Exception for error field. |

### get_model_value

``` python
types.ContentToolResult.get_model_value()
```

Get the actual value sent to the model.

### serialize_error

``` python
types.ContentToolResult.serialize_error(v)
```

Serialize Exception to string for JSON compatibility.

### tagify

``` python
types.ContentToolResult.tagify()
```

A method for rendering this object via htmltools/shiny.

### to_display_markdown

``` python
types.ContentToolResult.to_display_markdown(max_lines=None)
```

Render as a fenced code block, optionally capping the value’s height.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| max_lines | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[int](https://docs.python.org/3/library/functions.html#int)\] | Truncate the value to this many lines, replacing the remainder with a count of what was dropped. `None` (the default) emits the full value. | `None` |

### to_html

``` python
types.ContentToolResult.to_html()
```

Render as an HTML string.

Shared by `.tagify()` (shinychat) and the notebook echo display, so the two can’t drift. Requires `TOOL_CSS` to be present on the page. The result is collapsed; `TOOL_CSS` bounds its height once expanded.

### validate_error

``` python
types.ContentToolResult.validate_error(v)
```

Accept string or Exception for error field.
