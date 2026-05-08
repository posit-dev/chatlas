# types.ContentImageInline

``` python
types.ContentImageInline()
```

Inline image content.

This is the return type for [`content_image_file`](https://posit-dev.github.io/chatlas/reference/content_image_file.html#chatlas.content_image_file) and [`content_image_plot`](https://posit-dev.github.io/chatlas/reference/content_image_plot.html#chatlas.content_image_plot). It’s not meant to be used directly.

## Parameters

| Name               | Type | Description                    | Default    |
|--------------------|------|--------------------------------|------------|
| image_content_type |      | The content type of the image. | *required* |
| data               |      | The base64-encoded image data. | *required* |
