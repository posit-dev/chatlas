# types.ContentImageRemote

``` python
types.ContentImageRemote()
```

Image content from a URL.

This is the return type for [`content_image_url`](https://posit-dev.github.io/chatlas/reference/content_image_url.html#chatlas.content_image_url). It’s not meant to be used directly.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| url |  | The URL of the image. | *required* |
| detail |  | A detail setting for the image. Can be `"auto"`, `"low"`, or `"high"`. | *required* |
