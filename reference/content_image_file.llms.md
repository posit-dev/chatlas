# content_image_file

``` python
content_image_file(path, content_type='auto', resize=MISSING)
```

Encode image content from a file for chat input.

This function is used to prepare image files for input to the chatbot. It can handle various image formats and provides options for resizing.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| path | [str](https://docs.python.org/3/library/stdtypes.html#str) | The path to the image file to include in the chat input. | *required* |
| content_type | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['auto', [ImageContentTypes](https://posit-dev.github.io/chatlas/reference/types.ImageContentTypes.html#chatlas.types.ImageContentTypes)\] | The content type of the image (e.g., `"image/png"`). If `"auto"`, the content type is inferred from the file extension. | `'auto'` |
| resize | [Union](https://docs.python.org/3/library/typing.html#typing.Union)\[[Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['low', 'high', 'none'\], [str](https://docs.python.org/3/library/stdtypes.html#str), [MISSING_TYPE](https://posit-dev.github.io/chatlas/reference/types.MISSING_TYPE.html#chatlas.types.MISSING_TYPE)\] | Resizing option for the image. Can be: - `"low"`: Resize to fit within 512x512 - `"high"`: Resize to fit within 2000x768 or 768x2000 - `"none"`: No resizing - Custom string (e.g., `"200x200"`, `"300x200>!"`, etc.) | `MISSING` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | \[\](`~chatlas.types.Content`) | Content suitable for a [`Turn`](https://posit-dev.github.io/chatlas/reference/Turn.html#chatlas.Turn) object. |

## Examples

``` python
from chatlas import ChatOpenAI, content_image_file

chat = ChatOpenAI()
chat.chat(
    "What do you see in this image?",
    content_image_file("path/to/image.png"),
)
```

## Raises

| Name | Type | Description |
|----|----|----|
|  | [FileNotFoundError](https://docs.python.org/3/library/exceptions.html#FileNotFoundError) | If the specified file does not exist. |
|  | [ValueError](https://docs.python.org/3/library/exceptions.html#ValueError) | If the file extension is unsupported or the resize option is invalid. |
