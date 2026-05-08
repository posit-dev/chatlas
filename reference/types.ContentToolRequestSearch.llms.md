# types.ContentToolRequestSearch

``` python
types.ContentToolRequestSearch()
```

A web search request from the model.

This content type represents the model’s request to search the web. It’s automatically generated when a built-in web search tool is used.

## Parameters

| Name  | Type | Description                              | Default    |
|-------|------|------------------------------------------|------------|
| query |      | The search query.                        | *required* |
| extra |      | The raw provider-specific response data. | *required* |
