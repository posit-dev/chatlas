# types.ToolInfo

``` python
types.ToolInfo()
```

Serializable tool information

This contains only the serializable parts of a Tool that are needed for ContentToolRequest to be JSON-serializable. This allows tool metadata to be preserved without including the non-serializable function reference.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| name |  | The name of the tool. | *required* |
| description |  | A description of what the tool does. | *required* |
| parameters |  | A dictionary describing the input parameters and their types. | *required* |
| annotations |  | Additional properties that describe the tool and its behavior. | *required* |

## Methods

| Name | Description |
|----|----|
| [from_tool](#chatlas.types.ToolInfo.from_tool) | Create a ToolInfo from a Tool or ToolBuiltIn instance. |

### from_tool

``` python
types.ToolInfo.from_tool(tool)
```

Create a ToolInfo from a Tool or ToolBuiltIn instance.
