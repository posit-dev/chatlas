# types.ContentDocument

``` python
types.ContentDocument()
```

Generic document content (plain text, Markdown, CSV, code, and – on providers that support it – docx/xlsx).

This is the type returned by [`content_document_file`](../reference/content_document_file.llms.md#chatlas.content_document_file). Unlike `ContentPDF`, documents carry a real `mime_type` since they span many formats; PDFs should always go through [`content_pdf_file`](https://posit-dev.github.io/chatlas/reference/content_pdf_file.html#chatlas.content_pdf_file)/[`content_pdf_url`](https://posit-dev.github.io/chatlas/reference/content_pdf_url.html#chatlas.content_pdf_url) instead, which unlock PDF-specific handling (page-image understanding on Anthropic, and URL passthrough).

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| data |  | The document’s bytes. Optional when `url` is set. | *required* |
| filename |  | The name of the document file. | *required* |
| mime_type |  | The document’s MIME type (e.g. `"text/plain"`, `"text/csv"`, `"application/vnd.openxmlformats-officedocument.wordprocessingml.document"`). Not every provider accepts every MIME type – providers that can’t accept a given type raise a clear error. | *required* |
| url |  | An optional URL where the document can be accessed. | *required* |
