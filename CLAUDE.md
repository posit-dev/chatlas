# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

The project uses `uv` for package management and Make for common tasks:

- **Setup environment**: `make setup` (installs all dependencies with `uv sync --all-extras`)
- **Run tests**: `make check-tests` or `uv run pytest` (uses VCR cassettes by default)
- **Type checking**: `make check-types` or `uv run pyright`
- **Linting/formatting**: `make check-format` (check) or `make format` (fix)
- **Full checks**: `make check` (runs format, type, and test checks)
- **Build package**: `make build` (creates dist/ with built package)
- **Run single test**: `uv run pytest tests/test_specific_file.py::TestClass::test_method -v`
- **Update snapshots**: `make update-snaps` (for syrupy snapshot tests)
- **Update VCR cassettes**: `make update-snaps-vcr` (re-records HTTP interactions, requires API keys)
- **Check VCR secrets**: `make check-vcr-secrets` (scans cassettes for leaked credentials)
- **Documentation**: `make docs` (build) or `make docs-preview` (serve locally)

## Project Architecture

### Core Components

**Chat System**: The main `Chat` class in `_chat.py` manages conversation state and provider interactions. It's a generic class that works with different providers through the `Provider` abstract base class.

**Provider Pattern**: All LLM providers (OpenAI, Anthropic, Google, etc.) inherit from `Provider` in `_provider.py`. Each provider (e.g., `_provider_openai.py`) implements:
- Model-specific parameter handling 
- API client configuration
- Request/response transformation
- Tool calling integration

**Content System**: The `_content.py` module defines structured content types:
- `ContentText`: Plain text messages
- `ContentImage`: Image content (inline, remote, or file-based)
- `ContentToolRequest`/`ContentToolResult`: Tool interaction messages
- `ContentJson`: Structured data responses

**Tool System**: Tools are defined in `_tools.py` and allow LLMs to call Python functions. The system supports:
- Function registration with automatic schema generation
- Tool approval workflows
- MCP (Model Context Protocol) server integration via `_mcp_manager.py`

**Turn Management**: `Turn` objects in `_turn.py` represent individual conversation exchanges, containing sequences of `Content` objects.

### Key Patterns

1. **Provider Abstraction**: All providers implement the same interface but handle model-specific details internally
2. **Generic Typing**: Heavy use of TypeVars and generics for type safety across providers
3. **Streaming Support**: Both sync and async streaming responses via `ChatResponse`/`ChatResponseAsync`
4. **Content-Based Messaging**: All communication uses structured `Content` objects rather than raw strings
5. **Tool Integration**: Seamless function calling with automatic JSON schema generation from Python type hints

### Testing Structure

- Tests are organized by component (e.g., `test_provider_openai.py`, `test_tools.py`)
- Snapshot testing with `syrupy` for response validation
- MCP server tests use local test servers in `tests/mcp_servers/`
- Async tests configured via `pytest.ini` with `asyncio_mode=strict`

### VCR Testing (HTTP Recording/Replay)

Tests use [pytest-recording](https://github.com/kiwicom/pytest-recording) (wrapping vcrpy) to record and replay HTTP interactions:

- **Cassettes**: YAML files stored in `tests/_vcr/` organized by test module
- **Default mode**: Tests replay cassettes without making live API calls
- **Recording**: Use `make update-snaps-vcr` or `uv run pytest --record-mode=rewrite` (requires real API keys)
- **Dummy credentials**: Auto-set by `conftest.py` when env vars are missing, enabling VCR replay without secrets

**Adding VCR to tests**:
```python
from .conftest import make_vcr_config, VCR_MATCH_ON_WITHOUT_BODY

# Most tests use default config (matches on request body)
@pytest.mark.vcr
def test_provider_simple():
    ...

# For tests with dynamic request bodies (temp files, generated IDs)
@pytest.fixture(scope="module")
def vcr_config():
    return make_vcr_config(match_on=VCR_MATCH_ON_WITHOUT_BODY)
```

**Tests requiring live API** (skip in VCR mode):
```python
from .conftest import is_dummy_credential

@pytest.mark.skipif(
    is_dummy_credential("ANTHROPIC_API_KEY"),
    reason="This test requires live API calls",
)
def test_multi_sample():
    ...
```

**Providers incompatible with VCR**: Bedrock and Snowflake require live API tests due to auth mechanisms.

See `docs/dev/vcr-tests.md` for comprehensive documentation.

### Documentation

Documentation is built with Quarto and quartodoc:
- API reference generated from docstrings in `chatlas/` modules
- Guides and examples in `docs/` as `.qmd` files
- Type definitions in `chatlas/types/` provide provider-specific parameter types


## Adding New Providers

When implementing a new LLM provider, follow this systematic approach:

### 1. Research Phase
- **Check ellmer first**: Look in `../ellmer/R/provider-*.R` for existing implementations
- **Identify base provider**: Most providers inherit from either `OpenAIProvider` (for OpenAI-compatible APIs) or implement `Provider` directly
- **Check existing patterns**: Review similar providers in `chatlas/_provider_*.py`

### 2. Implementation Steps
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

### 3. Testing Setup
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

### 4. Package Integration
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

### 5. Provider-Specific Customizations

**OpenAI-Compatible Providers**:
- Inherit from `OpenAIProvider`
- Override `_chat_perform_args()` for API differences
- Common customizations: remove `stream_options`, adjust parameter names, modify headers

**Custom API Providers**:
- Inherit from `Provider` directly
- Implement all abstract methods: `chat_perform()`, `chat_perform_async()`, `stream_text()`, etc.
- Handle model-specific response formats

### 6. Common Patterns
- **Environment variables**: Use `[PROVIDER]_API_KEY` format
- **Default models**: Use provider's recommended general-purpose model
- **Seed handling**: `seed = 1014 if is_testing() else None` when MISSING
- **Error handling**: Provider APIs often return different error formats
- **Rate limiting**: Consider implementing client-side throttling for providers that need it

### 7. Documentation Requirements
- Include provider description and prerequisites
- Document known limitations (tool calling, vision support, etc.)
- Provide working examples with environment variable usage
- Note any special model requirements (e.g., vision models for images)

## Connections to ellmer

This project is the Python equivalent of the R package ellmer. The source code for ellmer is available in a sibling directory to this project. Before implementing new features or bug fixes in chatlas, it may be useful to consult  the ellmer codebase to: (1) check whether the feature/fix already exists on the R side and (2) make sure the projects are aligned in terms of stylistic approaches. Note also that ellmer itself has a CLAUDE.md file which has a useful overview of the project. 

## Differences from ellmer

One important difference to note is that in `ellmer::chat_openai()` uses the completions API, while `chatlas.ChatOpenAI()` uses the responses API. Look to `ellmer::chat_openai_responses()` and `chatlas.ChatOpenAICompletions()` for the "non-default" API.

## Access to gh CLI

If ellmer or chatlas issues or PRs are referenced, try using the `gh` CLI tool to gain necessary context. They live at `tidyverse/ellmer` and `posit-dev/chatlas`