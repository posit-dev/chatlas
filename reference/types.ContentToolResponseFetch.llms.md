# types.ContentToolResponseFetch

``` python
types.ContentToolResponseFetch()
```

Web fetch results from the model.

This content type represents the results of fetching a URL. It’s automatically generated when a built-in web fetch tool returns results.

## Parameters

| Name  | Type | Description                              | Default    |
|-------|------|------------------------------------------|------------|
| url   |      | The URL that was fetched.                | *required* |
| extra |      | The raw provider-specific response data. | *required* |
