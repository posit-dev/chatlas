# types.ContentUploaded

``` python
types.ContentUploaded()
```

A reference to a file already uploaded to a provider.

Returned by `chat.files.upload(...)` and usable directly in `.chat()` so the file bytes are not re-sent each turn. Can also be constructed directly to reference a file uploaded out-of-band (e.g. a Google Vertex `gs://` URI).

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| id |  | The provider’s file identifier (OpenAI/Anthropic `file_id`, or a Google/Vertex URI such as `https://.../files/abc` or `gs://bucket/obj`). | *required* |
| mime_type |  | The file’s MIME type. Determines image-vs-document serialization and is required by Google’s file references. | *required* |
| provider |  | The provider the file was uploaded to (`"openai"`, `"anthropic"`, or `"google"`). Used to detect cross-provider misuse. | *required* |
| extra |  | A plain dict of provider-native metadata (filename, size, …) when available. Unlike `FileMetadata.extra`, this can’t hold the provider’s own file object, since this type is serialized as part of a chat’s turns. | *required* |
