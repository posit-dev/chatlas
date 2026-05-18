# types.ContentPDF

``` python
types.ContentPDF()
```

PDF content

This content type primarily exists to signal PDF data extraction (i.e., data extracted via [`Chat`](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat)’s `.chat_structured()` method)

## Parameters

| Name     | Type | Description                                   | Default    |
|----------|------|-----------------------------------------------|------------|
| data     |      | The PDF data extracted                        | *required* |
| filename |      | The name of the PDF file                      | *required* |
| url      |      | An optional URL where the PDF can be accessed | *required* |
