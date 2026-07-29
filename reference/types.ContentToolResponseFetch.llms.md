# types.ContentToolResponseFetch

``` python
types.ContentToolResponseFetch()
```

Web fetch results from the model.

This content type represents the results of fetching a URL. It’s automatically generated when a built-in web fetch tool returns results.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| url |  | The URL that was fetched. | *required* |
| status |  | A normalized, cross-provider outcome: `"success"` if content was retrieved, `"error"` if it was not, or `None` when the provider doesn’t report an outcome. Providers expose finer-grained, non-aligned reasons (e.g. Anthropic’s `url_not_allowed`, Google’s `PAYWALL`); those are not normalized here but remain available in `extra`. | *required* |
| extra |  | The raw provider-specific response data. | *required* |
