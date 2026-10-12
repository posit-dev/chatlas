# types.SupportsContent

``` python
types.SupportsContent()
```

Structural type for objects that can be coerced to `Content`.

Any object implementing `to_content() -> Content` is accepted anywhere user-facing APIs take content (`.chat()`, `.stream()`, `.extract_data()`, etc.) – no need to inherit from a chatlas class. `FileMetadata` (returned by `chat.files.get()`/`.list()`) is the built-in example; third-party packages can opt their own objects in the same way.

The coercion happens at the turn boundary: the returned `Content` is what enters the chat’s turns (and its serialization), never the original object.

## Methods

| Name | Description |
|----|----|
| [to_content](#chatlas.types.SupportsContent.to_content) | Return this object as a `Content` object. |

### to_content

``` python
types.SupportsContent.to_content()
```

Return this object as a `Content` object.
