# types.FileMetadata

``` python
types.FileMetadata()
```

Normalized metadata for a provider-hosted file.

Can be passed straight to `.chat()` (and other chat methods) to reference the file in a turn, just like the result of `chat.files.upload()`.

## Attributes

| Name | Description |
|----|----|
| [extra](#chatlas.types.FileMetadata.extra) | The provider’s own file object, unmodified, for fields this class doesn’t |

## Methods

| Name | Description |
|----|----|
| [to_content](#chatlas.types.FileMetadata.to_content) | Build a `ContentUploaded` reference to this file. |

### to_content

``` python
types.FileMetadata.to_content()
```

Build a `ContentUploaded` reference to this file.

You rarely need to call this yourself: `.chat()` (and other chat methods) accept `FileMetadata` directly and do this conversion for you. It’s here for when you want the `ContentUploaded` itself – e.g. to build a `UserTurn` by hand or stash the reference for later.

When the provider doesn’t report a MIME type (OpenAI’s Files API never does), it’s guessed from `filename`, falling back to `"application/octet-stream"` – the same guessing `chat.files.upload()` applies. The returned reference’s `extra` is a plain, serializable dict of the normalized fields, not the provider’s own file object held on `FileMetadata.extra`.

On Google, this raises if the file failed to upload or is still processing, since Gemini can’t reference such a file in a turn.

#### Returns

| Name | Type | Description |
|----|----|----|
|  | [ContentUploaded](https://posit-dev.github.io/chatlas/reference/types.ContentUploaded.html#chatlas.types.ContentUploaded) | A reference to the file, usable as content in `.chat()`. |
