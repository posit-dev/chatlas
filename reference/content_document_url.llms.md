# content_document_url

``` python
content_document_url(url, mime_type='auto')
```

Prepare a remote text/data file for input to a chat.

Use this for plain text, Markdown, CSV, code, and other non-PDF documents hosted at a URL (use [`content_pdf_url`](https://posit-dev.github.io/chatlas/reference/content_pdf_url.html#chatlas.content_pdf_url) for PDFs). Not all providers accept every document type, so check the documentation for the provider you are using.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| url | [str](https://docs.python.org/3/library/stdtypes.html#str) | A URL to a remote file, or a `data:` URL carrying the document inline. | *required* |
| mime_type | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['auto'\] \| [str](https://docs.python.org/3/library/stdtypes.html#str) | The document’s MIME type. If `"auto"`, it’s guessed from the URL’s file extension, falling back to `"text/plain"`. `data:` URLs always use the type declared in the URL itself. | `'auto'` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | \[\](`~chatlas.types.Content`) | Content suitable for a [`Turn`](https://posit-dev.github.io/chatlas/reference/Turn.html#chatlas.Turn) object. |

## Raises

| Name | Type | Description |
|----|----|----|
|  | [ValueError](https://docs.python.org/3/library/exceptions.html#ValueError) | If the URL points to a PDF (use `content_pdf_url()` instead), or is a malformed `data:` URL. |
