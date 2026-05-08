# content_image_url

``` python
content_image_url(url, detail='auto')
```

Encode image content from a URL for chat input.

This function is used to prepare image URLs for input to the chatbot. It can handle both regular URLs and data URLs.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| url | [str](https://docs.python.org/3/library/stdtypes.html#str) | The URL of the image to include in the chat input. Can be a data: URL or a regular URL. | *required* |
| detail | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['auto', 'low', 'high'\] | The detail setting for this image. Can be `"auto"`, `"low"`, or `"high"`. | `'auto'` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | \[\](`~chatlas.types.Content`) | Content suitable for a [`Turn`](https://posit-dev.github.io/chatlas/reference/Turn.html#chatlas.Turn) object. |

## Examples

``` python
from chatlas import ChatOpenAI, content_image_url

chat = ChatOpenAI()
chat.chat(
    "What do you see in this image?",
    content_image_url("https://www.python.org/static/img/python-logo.png"),
)
```

## Raises

| Name | Type | Description |
|----|----|----|
|  | [ValueError](https://docs.python.org/3/library/exceptions.html#ValueError) | If the URL is not valid or the detail setting is invalid. |
