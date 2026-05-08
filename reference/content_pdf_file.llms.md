# content_pdf_file

``` python
content_pdf_file(path)
```

Prepare a local PDF for input to a chat.

Not all providers support PDF input, so check the documentation for the provider you are using.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| path | [str](https://docs.python.org/3/library/stdtypes.html#str) \| [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | A path to a local PDF file. | *required* |

## Returns

| Name | Type | Description |
|----|----|----|
|  | \[\](`~chatlas.types.Content`) | Content suitable for a [`Turn`](https://posit-dev.github.io/chatlas/reference/Turn.html#chatlas.Turn) object. |
