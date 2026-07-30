# content_document_file

``` python
content_document_file(path, mime_type='auto')
```

Prepare a local text/data file for input to a chat.

Use this for plain text, Markdown, CSV, code, and other document files that aren’t PDFs (use [`content_pdf_file`](https://posit-dev.github.io/chatlas/reference/content_pdf_file.html#chatlas.content_pdf_file) for those). Not all providers accept every document type – e.g. `.docx`/`.xlsx` only work with `ChatOpenAI()`, and Anthropic only accepts documents it can treat as plain text – so check the documentation for the provider you are using.

This embeds the file’s bytes in every request that includes it. For a large document, or one referenced across many turns, upload it once with `chat.files.upload()` instead and pass the resulting [`ContentUploaded`](https://posit-dev.github.io/chatlas/reference/types.ContentUploaded.html#chatlas.types.ContentUploaded) – at the cost of working on fewer providers and tying the chat to the one that hosts the file.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| path | [str](https://docs.python.org/3/library/stdtypes.html#str) \| [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | A path to a local file. | *required* |
| mime_type | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['auto'\] \| [str](https://docs.python.org/3/library/stdtypes.html#str) | The file’s MIME type. If `"auto"`, it’s guessed from the file’s extension, falling back to `"text/plain"` for unrecognized extensions (e.g. most source code files). | `'auto'` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | \[\](`~chatlas.types.Content`) | Content suitable for a [`Turn`](https://posit-dev.github.io/chatlas/reference/Turn.html#chatlas.Turn) object. |

## Raises

| Name | Type | Description |
|----|----|----|
|  | [FileNotFoundError](https://docs.python.org/3/library/exceptions.html#FileNotFoundError) | If the specified file does not exist. |
|  | [ValueError](https://docs.python.org/3/library/exceptions.html#ValueError) | If `path` points to a PDF file (use `content_pdf_file()` instead). |
