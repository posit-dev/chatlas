# types.ContentThinkingDelta

``` python
types.ContentThinkingDelta()
```

A streaming fragment of thinking/reasoning content.

Emitted during streaming to represent a chunk of the model’s thinking. The `phase` attribute communicates block boundaries to downstream consumers.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| thinking |  | The thinking/reasoning text fragment. | *required* |
| phase |  | The phase of the thinking delta: `"start"`, `"body"`, or `"end"`. | *required* |
