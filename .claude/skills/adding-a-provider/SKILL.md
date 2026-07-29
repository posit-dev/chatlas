---
name: adding-a-provider
description: Use when adding a new LLM provider to chatlas (a new `Chat<Name>()` / `<Name>Provider` in `chatlas/_provider_*.py`) — covers the research phase, class and function structure, the standard test set, VCR cassette recording, and package integration.
---

# Adding New Providers

When implementing a new LLM provider, follow this systematic approach:

## 1. Research Phase
- **Check ellmer first**: Look in `../ellmer/R/provider-*.R` for existing implementations
- **Identify base provider**: Most providers inherit from either `OpenAIProvider` (for OpenAI-compatible APIs) or implement `Provider` directly
- **Check existing patterns**: Review similar providers in `chatlas/_provider_*.py`

## 2. Implementation Steps
1. **Create provider file**: `chatlas/_provider_[name].py`
   - Use PascalCase for class names (e.g., `MistralProvider`)
   - Use snake_case for function names (e.g., `ChatMistral`)
   - Follow existing docstring patterns with Prerequisites, Examples, Parameters, Returns sections

2. **Provider class structure**:
   ```python
   class [Name]Provider(OpenAIProvider):  # or Provider if custom
       def __init__(self, ...):
           super().__init__(...)
           # Provider-specific initialization

       def _chat_perform_args(self, ...):
           # Customize request parameters if needed
           kwargs = super()._chat_perform_args(...)
           # Apply provider-specific modifications
           return kwargs
   ```

3. **Chat function signature**:
   ```python
   def Chat[Name](
       *,
       system_prompt: Optional[str] = None,
       model: Optional[str] = None,
       api_key: Optional[str] = None,
       base_url: str = "https://...",
       seed: int | None | MISSING_TYPE = MISSING,
       kwargs: Optional["ChatClientArgs"] = None,
   ) -> Chat["SubmitInputArgs", ChatCompletion]:
   ```

## 3. Testing Setup
1. **Create test file**: `tests/test_provider_[name].py`
2. **Add environment variable skip pattern**:
   ```python
   import os
   import pytest

   do_test = os.getenv("TEST_[NAME]", "true")
   if do_test.lower() == "false":
       pytest.skip("Skipping [Name] tests", allow_module_level=True)
   ```
3. **Add VCR support** (for most providers):
   ```python
   @pytest.mark.vcr
   def test_[name]_simple_request():
       ...
   ```
   For async tests, put `@pytest.mark.vcr` before `@pytest.mark.asyncio`.
4. **Use standard test patterns**:
   - `test_[name]_simple_request()`
   - `test_[name]_simple_streaming_request()`
   - `test_[name]_respects_turns_interface()`
   - `test_[name]_tool_variations()` (if supported)
   - `test_data_extraction()`
   - `test_[name]_images()` (if vision supported)
5. **Record VCR cassettes**:
   ```bash
   # Set real API key, then record
   export [PROVIDER]_API_KEY="..."
   uv run pytest tests/test_provider_[name].py -v --record-mode=rewrite
   ```

## 4. Package Integration
1. **Update `chatlas/__init__.py`**:
   - Add import: `from ._provider_[name] import Chat[Name]`
   - Add to `__all__` tuple: `"Chat[Name]"`

2. **Run validation**:
   ```bash
   uv run pyright chatlas/_provider_[name].py
   uv run pytest tests/test_provider_[name].py -v  # Replays VCR cassettes
   uv run python -c "from chatlas import Chat[Name]; print('Import successful')"
   make check-vcr-secrets  # Ensure no secrets leaked in cassettes
   ```

## 5. Provider-Specific Customizations

**OpenAI-Compatible Providers**:
- Inherit from `OpenAIProvider`
- Override `_chat_perform_args()` for API differences
- Common customizations: remove `stream_options`, adjust parameter names, modify headers

**Custom API Providers**:
- Inherit from `Provider` directly
- Implement all abstract methods: `chat_perform()`, `chat_perform_async()`, `stream_content()`, `stream_merge_chunks()`, etc.
- Handle model-specific response formats

## 6. Common Patterns
- **Environment variables**: Use `[PROVIDER]_API_KEY` format
- **Default models**: Use provider's recommended general-purpose model
- **Seed handling**: `seed = 1014 if is_testing() else None` when MISSING
- **Error handling**: Provider APIs often return different error formats
- **Rate limiting**: Consider implementing client-side throttling for providers that need it

## 7. Documentation Requirements
- Include provider description and prerequisites
- Document known limitations (tool calling, vision support, etc.)
- Provide working examples with environment variable usage
- Note any special model requirements (e.g., vision models for images)
