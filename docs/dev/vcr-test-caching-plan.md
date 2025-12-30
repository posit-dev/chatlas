# VCR Test Caching Implementation Plan

## Overview

This document outlines the plan to add VCR-style HTTP recording/replay to chatlas tests, inspired by ellmer's use of R's `vcr` package. This enables:

- **Offline testing**: No API keys needed after initial recording
- **Deterministic tests**: Same responses every time
- **Faster CI**: No network latency or rate limiting
- **Cost savings**: No API charges for replayed tests

## Decisions

| Question | Decision |
|----------|----------|
| Scope | Start with OpenAI, then Anthropic, then Google |
| Cassette format | YAML (human-readable, matches ellmer) |
| Library | `pytest-recording` (wraps vcrpy with pytest integration) |
| Streaming | Skip for now, return to later |
| CI behavior | Fail on missing cassettes with informative message |

## Dependencies

Add to `pyproject.toml` under `[project.optional-dependencies].test`:

```toml
test = [
    "pyright>=1.1.379",
    "pytest>=8.3.2",
    "pytest-asyncio",
    "syrupy>=4",
    "vcrpy>=6.0.0",           # HTTP recording
    "pytest-recording>=0.13", # pytest integration
]
```

## Directory Structure

```
tests/
├── _vcr/                           # Cassette directory
│   ├── test_provider_openai/       # Per-test-file directories
│   │   ├── test_openai_simple_request.yaml
│   │   ├── test_openai_respects_turns_interface.yaml
│   │   └── ...
│   ├── test_provider_anthropic/
│   │   └── ...
│   └── test_provider_google/
│       └── ...
├── conftest.py                     # VCR configuration
└── test_provider_*.py
```

## Configuration

### conftest.py additions

```python
import os
import pytest

# VCR configuration via pytest-recording
@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [
            "authorization",
            "x-api-key",
            "api-key",
            "openai-organization",
            "x-goog-api-key",
        ],
        "filter_post_data_parameters": ["api_key"],
        "decode_compressed_response": True,
        "record_mode": "once",
        "match_on": ["method", "scheme", "host", "port", "path", "body"],
    }

@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    """Store cassettes in per-module directories."""
    module_name = request.module.__name__.split(".")[-1]
    return os.path.join(os.path.dirname(__file__), "_vcr", module_name)
```

### pytest.ini / pyproject.toml additions

```toml
[tool.pytest.ini_options]
# ... existing options ...
# Block network access when cassette exists (fail-safe)
# addopts = "--block-network"  # Optional: stricter mode
```

## Usage Pattern

Tests use the `@pytest.mark.vcr` decorator:

```python
import pytest
from chatlas import ChatOpenAI

@pytest.mark.vcr
def test_openai_simple_request():
    chat = ChatOpenAI(system_prompt="Be terse")
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    # ... assertions ...
```

For tests that should NOT use VCR (e.g., streaming tests for now):

```python
def test_openai_streaming():  # No @pytest.mark.vcr
    # This test will always make real API calls
    ...
```

## Makefile Commands

Add to `Makefile`:

```makefile
# Record new VCR cassettes (requires API keys)
record-vcr:
	uv run pytest --vcr-record=all tests/test_provider_openai.py tests/test_provider_anthropic.py tests/test_provider_google.py -v

# Record cassettes for a specific provider
record-vcr-openai:
	uv run pytest --vcr-record=all tests/test_provider_openai.py -v

record-vcr-anthropic:
	uv run pytest --vcr-record=all tests/test_provider_anthropic.py -v

record-vcr-google:
	uv run pytest --vcr-record=all tests/test_provider_google.py -v

# Re-record all cassettes from scratch
rerecord-vcr:
	rm -rf tests/_vcr/
	$(MAKE) record-vcr
```

## CI Configuration

When a cassette is missing, pytest-recording will fail with an error. We should add a helpful message. Options:

1. **Custom pytest hook** in conftest.py to catch VCR errors and print help
2. **CI job comment** explaining how to update cassettes
3. **README section** documenting the process

Suggested error handling in conftest.py:

```python
def pytest_exception_interact(node, call, report):
    """Provide helpful message when VCR cassette is missing."""
    if "CannotOverwriteExistingCassetteException" in str(call.excinfo) or \
       "Can't find cassette" in str(call.excinfo):
        print("\n" + "=" * 60)
        print("VCR CASSETTE MISSING OR OUTDATED")
        print("=" * 60)
        print("To record/update cassettes, run locally with API keys:")
        print("  make record-vcr")
        print("Or for a specific provider:")
        print("  make record-vcr-openai")
        print("=" * 60 + "\n")
```

## Implementation Phases

### Phase 1: Infrastructure Setup - COMPLETED
- [x] Add dependencies to pyproject.toml (vcrpy>=6.0.0, pytest-recording>=0.13)
- [x] Create `tests/_vcr/` directory (with .gitkeep)
- [x] Add VCR configuration to conftest.py
- [x] Add Makefile commands (record-vcr, record-vcr-openai, etc.)
- [x] Add session-scoped fixture for dummy API keys (Anthropic/Google require keys even for replay)

### Phase 2: OpenAI Provider - COMPLETED
- [x] Add `@pytest.mark.vcr` to non-streaming tests
- [x] Record initial cassettes (8 cassettes)
- [x] Verify tests pass with cassettes
- [x] Verify tests pass without API key (replay mode) - ~1.3s
- [x] Skip streaming tests from VCR for now

### Phase 3: Anthropic Provider - COMPLETED
- [x] Add `@pytest.mark.vcr` to non-streaming tests
- [x] Record cassettes (10 cassettes)
- [x] Verify replay works - ~2s without API key

### Phase 4: Google Provider - COMPLETED
- [x] Add `@pytest.mark.vcr` to non-streaming tests
- [x] Record cassettes (11 cassettes)
- [x] Verify replay works (slower due to SDK initialization overhead)
- [x] Fixed token assertion that was too strict

### Phase 5: Documentation & CI
- [x] Helpful error message when cassette missing (in conftest.py)
- [ ] Update README with VCR instructions (optional)

### Streaming Support - COMPLETED (OpenAI, Anthropic)
- [x] Investigate VCR handling of SSE streams - Works for OpenAI and Anthropic
- [x] Test with OpenAI streaming endpoints - Works
- [x] Test with Anthropic streaming endpoints - Works
- [x] Add streaming tests to VCR for OpenAI and Anthropic
- [ ] Google streaming - Does NOT work with VCR (SDK response handling incompatibility)

### Phase 6: Additional Test Files - COMPLETED
- [x] Add VCR to test_chat.py (12 tests with cassettes)
- [x] Add VCR to test_chat_dangling_tools.py (1 test with cassette)
- [x] Add VCR to test_parallel_chat.py (4 tests with cassettes)
- [x] Add VCR to test_parallel_chat_improved.py (9 tests with cassettes)
- [x] Add VCR to test_parallel_chat_errors.py (5 tests with cassettes)
- [x] Add VCR to test_parallel_chat_ordering.py (4 tests with cassettes)
- [x] Total: 68 cassettes across all test files

## Sensitive Data Handling

VCR will automatically filter:
- `Authorization` headers (API keys)
- `x-api-key` headers
- `openai-organization` headers
- `x-goog-api-key` headers

For response body scrubbing (if needed), add `before_record_response` hook:

```python
def scrub_response(response):
    """Remove sensitive data from response bodies."""
    # Implementation if needed
    return response

@pytest.fixture(scope="module")
def vcr_config():
    return {
        # ... other config ...
        "before_record_response": scrub_response,
    }
```

## Testing the Implementation

1. **With API key**: `OPENAI_API_KEY=xxx pytest tests/test_provider_openai.py -v`
2. **Without API key** (replay): `unset OPENAI_API_KEY && pytest tests/test_provider_openai.py -v`
3. **Force re-record**: `pytest --vcr-record=all tests/test_provider_openai.py -v`

## References

- [vcrpy documentation](https://vcrpy.readthedocs.io/)
- [pytest-recording documentation](https://github.com/kiwicom/pytest-recording)
- [ellmer vcr usage](../ellmer/tests/testthat/_vcr/)
