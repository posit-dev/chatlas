# types.ContentToolRequestFetch

``` python
types.ContentToolRequestFetch()
```

A web fetch request from the model.

This content type represents the model’s request to fetch a URL. It’s automatically generated when a built-in web fetch tool is used.

## Parameters

| Name  | Type | Description                              | Default    |
|-------|------|------------------------------------------|------------|
| url   |      | The URL to fetch.                        | *required* |
| extra |      | The raw provider-specific response data. | *required* |
