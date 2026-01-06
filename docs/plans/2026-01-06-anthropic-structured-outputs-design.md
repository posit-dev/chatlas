# Migration Plan: Anthropic Structured Outputs

## Current State

The current implementation in `_provider_anthropic.py` uses a **tool-based workaround**:
1. Creates a fake tool `_structured_tool_call` with the Pydantic schema as parameters
2. Forces Claude to call this tool via `tool_choice`
3. Extracts structured data from `content.input["data"]`
4. **Streaming is disabled** with a warning (line 432-437)

## New Anthropic API

Anthropic now provides native structured outputs via:
- **`output_format`** parameter with `type: "json_schema"`
- **Beta header**: `structured-outputs-2025-11-13`
- **`client.beta.messages.create()`** or **`.parse()`** methods
- **Streaming is supported!**
- Response is plain JSON text in `response.content[0].text`

Documentation: https://platform.claude.com/docs/en/build-with-claude/structured-outputs

## Proposed Changes

### 1. Update `_chat_perform_args()` to use `output_format`

**Before:**
```python
if data_model is not None:
    # Create fake tool...
    data_model_tool = Tool.from_func(_structured_tool_call)
    # ... add to tool_schemas, set tool_choice
    if stream:
        stream = False  # Disable streaming
```

**After:**
```python
if data_model is not None:
    from anthropic import transform_schema
    kwargs_full["output_format"] = {
        "type": "json_schema",
        "schema": transform_schema(data_model),
    }
    # Streaming now works!
```

### 2. Switch to beta client for structured outputs

Use `client.beta.messages.create()` when `data_model` is provided:

```python
def chat_perform(self, stream, turns, tools, data_model, kwargs):
    kwargs = self._chat_perform_args(stream, turns, tools, data_model, kwargs)

    if data_model is not None:
        # Use beta endpoint with structured outputs header
        return self._client.beta.messages.create(
            betas=["structured-outputs-2025-11-13"],
            **kwargs
        )
    else:
        return self._client.messages.create(**kwargs)
```

### 3. Update `_as_turn()` to handle new response format

**Before:** Extract from `content.input["data"]` of tool call
**After:** Parse JSON from `content.text` when `has_data_model=True`

```python
def _as_turn(self, completion: Message, has_data_model=False) -> AssistantTurn:
    contents = []
    for content in completion.content:
        if content.type == "text":
            if has_data_model:
                # New: JSON response is in text content
                contents.append(ContentJson(value=orjson.loads(content.text)))
            else:
                contents.append(ContentText(text=content.text))
        elif content.type == "tool_use":
            # Remove special handling for _structured_tool_call
            contents.append(ContentToolRequest(...))
```

### 4. Update streaming support

Remove the warning and allow streaming:

```python
# DELETE these lines:
if stream:
    stream = False
    warnings.warn(
        "Anthropic does not support structured data extraction in streaming mode.",
        stacklevel=2,
    )
```

### 5. Model compatibility check

The new API only supports certain models. Add validation:

```python
STRUCTURED_OUTPUT_MODELS = {
    "claude-sonnet-4-5", "claude-opus-4-1",
    "claude-opus-4-5", "claude-haiku-4-5"
}

if data_model is not None:
    base_model = self.model.split("-")[0:4]  # Handle dated versions
    if "-".join(base_model) not in STRUCTURED_OUTPUT_MODELS:
        # Fall back to old tool-based approach for older models
        ...
```

## Migration Strategy

| Phase | Description |
|-------|-------------|
| **Phase 1** | Add new `output_format` support alongside existing tool-based approach |
| **Phase 2** | Use new API for supported models, fallback for older models |
| **Phase 3** | Remove tool-based workaround once older models are deprecated |

## Benefits

1. **Streaming works** - No more disabling streaming for structured outputs
2. **Cleaner response** - JSON in text content, not fake tool calls
3. **Better validation** - Anthropic's constrained decoding guarantees schema compliance
4. **SDK integration** - Can use `transform_schema()` for Pydantic compatibility

## Risks & Considerations

1. **Beta API** - Feature is in public beta, may change
2. **Model restrictions** - Only works with newer Claude models
3. **Breaking change** - Response structure changes (text vs tool_use)
