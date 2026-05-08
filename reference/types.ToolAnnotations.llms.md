# types.ToolAnnotations

``` python
types.ToolAnnotations()
```

Additional properties describing a Tool to clients.

NOTE: all properties in ToolAnnotations are **hints**. They are not guaranteed to provide a faithful description of tool behavior (including descriptive properties like `title`).

Clients should never make tool use decisions based on ToolAnnotations received from untrusted servers.

## Attributes

| Name | Description |
|----|----|
| [destructiveHint](#chatlas.types.ToolAnnotations.destructiveHint) | If true, the tool may perform destructive updates to its environment. |
| [extra](#chatlas.types.ToolAnnotations.extra) | Additional metadata about the tool. |
| [idempotentHint](#chatlas.types.ToolAnnotations.idempotentHint) | If true, calling the tool repeatedly with the same arguments |
| [openWorldHint](#chatlas.types.ToolAnnotations.openWorldHint) | If true, this tool may interact with an “open world” of external |
| [readOnlyHint](#chatlas.types.ToolAnnotations.readOnlyHint) | If true, the tool does not modify its environment. |
| [title](#chatlas.types.ToolAnnotations.title) | A human-readable title for the tool. |
