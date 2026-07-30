# types.ImageContentTypes

`types.ImageContentTypes`

Allowable content types for images.

Note that not every provider accepts every type here: `image/heic` and `image/heif` are only supported by `ChatGoogle()` today. Providers that can’t accept a given type raise a clear error rather than silently sending it.
