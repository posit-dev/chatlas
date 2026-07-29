# FileManager

``` python
FileManager(provider)
```

Manage files hosted by a chat’s provider. Accessed via `chat.files`.

Upload a file once with `.upload()`, then pass the returned `ContentUploaded` to `.chat()` (and other chat methods) like any other content, instead of re-sending the file’s bytes on every turn.

Supported for `ChatOpenAI()` (the Responses API), `ChatAnthropic()`, and `ChatGoogle()` (the Gemini Developer API). Every other provider – including `ChatOpenAICompletions()`, OpenAI-compatible third parties (`ChatGroq()`, `ChatMistral()`, `ChatOllama()`, etc.), `ChatAzureOpenAI()`, `ChatBedrockAnthropic()`, `ChatPosit()`, and Vertex-backed chats (`ChatVertex()`, or `ChatGoogle()` configured for Vertex) – raises `NotImplementedError` from every method here.

## Notes

- Anthropic’s Files API is still in beta. chatlas automatically adds the required `anthropic-beta: files-api-2025-04-14` header whenever a turn references an uploaded file.
- Gemini Developer API files expire automatically after 48 hours, and the Files API isn’t available on Vertex AI at all. To reference a file already in Cloud Storage on a Vertex-backed chat, construct `ContentUploaded` directly with a `gs://` URI instead of calling `.upload()`.
- `ChatOpenAICompletions()` can reference an uploaded PDF or text document, but not an uploaded image – the Chat Completions API doesn’t support it. Use `ChatOpenAI()` (Responses API) for uploaded images, or pass the image inline with `content_image_file()`.

## Methods

| Name | Description |
|----|----|
| [delete](#chatlas.FileManager.delete) | Delete a previously uploaded file from the provider. |
| [delete_async](#chatlas.FileManager.delete_async) | Async version of `.delete()`. |
| [download](#chatlas.FileManager.download) | Download a file’s raw bytes, optionally writing them to `path`. |
| [download_async](#chatlas.FileManager.download_async) | Async version of `.download()`. |
| [get](#chatlas.FileManager.get) | Get metadata for a previously uploaded file. |
| [get_async](#chatlas.FileManager.get_async) | Async version of `.get()`. |
| [list](#chatlas.FileManager.list) | List files previously uploaded to the chat’s provider. |
| [list_async](#chatlas.FileManager.list_async) | Async version of `.list()`. |
| [upload](#chatlas.FileManager.upload) | Upload a file to the chat’s provider, once, for reuse across turns. |
| [upload_async](#chatlas.FileManager.upload_async) | Async version of `.upload()`. |

### delete

``` python
FileManager.delete(id)
```

Delete a previously uploaded file from the provider.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| id | [str](https://docs.python.org/3/library/stdtypes.html#str) | The provider-assigned file id (e.g., `ContentUploaded.id`). | *required* |

### delete_async

``` python
FileManager.delete_async(id)
```

Async version of `.delete()`.

### download

``` python
FileManager.download(id, path=None)
```

Download a file’s raw bytes, optionally writing them to `path`.

Whether a file is downloadable depends on how it was created, so this raises a provider error for anything `.upload()` produced:

- OpenAI refuses `purpose="user_data"` (what `.upload()` uses), but allows `purpose="batch"`/`"batch_output"` — i.e. the input and result files behind `batch_chat()`.
- Anthropic marks uploaded files as not downloadable.
- Google only serves bytes back for model-generated files (e.g. Veo video output).

### download_async

``` python
FileManager.download_async(id, path=None)
```

Async version of `.download()`.

### get

``` python
FileManager.get(id)
```

Get metadata for a previously uploaded file.

Only supported for ChatOpenAI, ChatAnthropic, and ChatGoogle; other providers raise `NotImplementedError`.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| id | [str](https://docs.python.org/3/library/stdtypes.html#str) | The provider-assigned file id (e.g., `ContentUploaded.id`). | *required* |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | [FileMetadata](https://posit-dev.github.io/chatlas/reference/types.FileMetadata.html#chatlas.types.FileMetadata) | Metadata for the file. |

### get_async

``` python
FileManager.get_async(id)
```

Async version of `.get()`.

### list

``` python
FileManager.list()
```

List files previously uploaded to the chat’s provider.

Only supported for ChatOpenAI, ChatAnthropic, and ChatGoogle; other providers raise `NotImplementedError`.

#### Returns

| Name | Type | Description |
|----|----|----|
|  | [list](https://posit-dev.github.io/chatlas/reference/FileManager.html#chatlas.FileManager.list)\[[FileMetadata](https://posit-dev.github.io/chatlas/reference/types.FileMetadata.html#chatlas.types.FileMetadata)\] | Metadata for each file hosted by the provider. |

### list_async

``` python
FileManager.list_async()
```

Async version of `.list()`.

### upload

``` python
FileManager.upload(file, *, mime_type=None)
```

Upload a file to the chat’s provider, once, for reuse across turns.

Only supported for ChatOpenAI, ChatAnthropic, and ChatGoogle; other providers raise `NotImplementedError`.

On `ChatGoogle()`, this blocks until Gemini finishes processing the file: large media (video, audio, multi-GB uploads) is processed asynchronously, and the API refuses to reference a file that isn’t yet `ACTIVE`. That means this call can take a while for a large video, and raises if the file fails to process – but the reference you get back is always ready to use.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| file | [str](https://docs.python.org/3/library/stdtypes.html#str) \| [os](https://docs.python.org/3/library/os.html#module-os).[PathLike](https://docs.python.org/3/library/os.html#os.PathLike)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] \| [IO](https://docs.python.org/3/library/typing.html#typing.IO)\[[bytes](https://docs.python.org/3/library/stdtypes.html#bytes)\] | A path to a file, or a binary file-like object, to upload. | *required* |
| mime_type | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The file’s MIME type. If not provided, it’s guessed from `file`. | `None` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | [ContentUploaded](https://posit-dev.github.io/chatlas/reference/types.ContentUploaded.html#chatlas.types.ContentUploaded) | A reference to the uploaded file that can be passed to `.chat()` (and other chat methods) in place of the file’s raw bytes. |

### upload_async

``` python
FileManager.upload_async(file, *, mime_type=None)
```

Async version of `.upload()`.
