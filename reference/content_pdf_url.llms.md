# content_pdf_url

``` python
content_pdf_url(url)
```

Use a remote PDF for input to a chat.

Not all providers support PDF input, so check the documentation for the provider you are using.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| url | [str](https://docs.python.org/3/library/stdtypes.html#str) | A URL to a remote PDF file. | *required* |

## Returns

| Name | Type | Description |
|----|----|----|
|  | \[\](`~chatlas.types.Content`) | Content suitable for a [`Turn`](https://posit-dev.github.io/chatlas/reference/Turn.html#chatlas.Turn) object. |
