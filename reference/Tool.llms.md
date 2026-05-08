# Tool

``` python
Tool(func, name, description, parameters, annotations=None, strict=None)
```

Define a tool

Define a Python function for use by a chatbot. The function will always be invoked in the current Python process.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| func | [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)\[…, [Any](https://docs.python.org/3/library/typing.html#typing.Any)\] \| [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)\[…, [Awaitable](https://docs.python.org/3/library/typing.html#typing.Awaitable)\[[Any](https://docs.python.org/3/library/typing.html#typing.Any)\]\] | The function to be invoked when the tool is called. | *required* |
| name | [str](https://docs.python.org/3/library/stdtypes.html#str) | The name of the tool. | *required* |
| description | [str](https://docs.python.org/3/library/stdtypes.html#str) | A description of what the tool does. | *required* |
| parameters | [dict](https://docs.python.org/3/library/stdtypes.html#dict)\[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)\] | A dictionary describing the input parameters and their types. | *required* |
| annotations | 'Optional\[ToolAnnotations\]' | Additional properties that describe the tool and its behavior. | `None` |
| strict | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[bool](https://docs.python.org/3/library/functions.html#bool)\] | Whether to enable strict mode. | `None` |

## Methods

| Name                                 | Description                          |
|--------------------------------------|--------------------------------------|
| [from_func](#chatlas.Tool.from_func) | Create a Tool from a Python function |
| [from_mcp](#chatlas.Tool.from_mcp)   | Create a Tool from an MCP tool       |

### from_func

``` python
Tool.from_func(func, *, name=None, model=None, annotations=None)
```

Create a Tool from a Python function

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| func | [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)\[…, [Any](https://docs.python.org/3/library/typing.html#typing.Any)\] \| [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)\[…, [Awaitable](https://docs.python.org/3/library/typing.html#typing.Awaitable)\[[Any](https://docs.python.org/3/library/typing.html#typing.Any)\]\] | The function to wrap as a tool. | *required* |
| name | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The name of the tool. If not provided, the name will be inferred from the function’s name. | `None` |
| model | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[type](https://docs.python.org/3/library/functions.html#type)\[[BaseModel](https://docs.pydantic.dev/latest/api/pydantic/base_model/#pydantic.BaseModel)\]\] | A Pydantic model that describes the input parameters for the function. If not provided, the model will be inferred from the function’s type hints. The primary reason why you might want to provide a model in Note that the name and docstring of the model takes precedence over the name and docstring of the function. | `None` |
| annotations | 'Optional\[ToolAnnotations\]' | Additional properties that describe the tool and its behavior. | `None` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | [Tool](https://posit-dev.github.io/chatlas/reference/Tool.html#chatlas.Tool) | A new Tool instance wrapping the provided function. |

#### Raises

| Name | Type | Description |
|----|----|----|
|  | [ValueError](https://docs.python.org/3/library/exceptions.html#ValueError) | If there is a mismatch between model fields and function parameters. |

### from_mcp

``` python
Tool.from_mcp(session, mcp_tool)
```

Create a Tool from an MCP tool

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| session | 'MCPClientSession' | The MCP client session to use for calling the tool. | *required* |
| mcp_tool | 'MCPTool' | The MCP tool to wrap. | *required* |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | [Tool](https://posit-dev.github.io/chatlas/reference/Tool.html#chatlas.Tool) | A new Tool instance wrapping the MCP tool. |
