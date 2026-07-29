# types.ContentToolResponseSearch

``` python
types.ContentToolResponseSearch()
```

Web search results from the model.

This content type represents the results of a web search. It’s automatically generated when a built-in web search tool returns results.

## Parameters

| Name    | Type | Description                              | Default    |
|---------|------|------------------------------------------|------------|
| sources |      | The pages surfaced by the search.        | *required* |
| extra   |      | The raw provider-specific response data. | *required* |
