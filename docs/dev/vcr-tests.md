# VCR Test Recording Guide

This document explains how HTTP recording/replay works for chatlas tests using [pytest-recording](https://github.com/kiwicom/pytest-recording) (which wraps [vcrpy](https://vcrpy.readthedocs.io/)).

## Overview

VCR records HTTP interactions during test runs and saves them as "cassettes" (YAML files). On subsequent runs, these cassettes are replayed instead of making live API calls. This enables:

- **Fast CI**: Tests run in seconds without API calls
- **No secrets in CI**: VCR replay mode uses dummy API keys
- **Deterministic tests**: Same responses every time

## Directory Structure

```
tests/
├── _vcr/
│   ├── test_provider_openai/
│   │   ├── test_openai_simple_request.yaml
│   │   └── ...
│   ├── test_provider_anthropic/
│   └── ...
├── conftest.py          # VCR configuration
└── test_provider_*.py   # Test files
```

## Recording Cassettes

### Record all cassettes

```bash
# Set your API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="..."

# Record with --record-mode=rewrite (overwrites existing)
make update-snaps-vcr

# Or directly with pytest
uv run pytest --record-mode=rewrite
```

### Record cassettes for a specific test file

```bash
uv run pytest tests/test_provider_openai.py -v --record-mode=rewrite
```

### Record a single test

```bash
uv run pytest tests/test_provider_openai.py::test_openai_simple_request -v --record-mode=rewrite
```

### Record modes

- `rewrite` - Delete and re-record all cassettes (recommended for updates)
- `new_episodes` - Record new interactions, keep existing
- `none` - Only replay, fail if cassette missing (CI default)
- `all` - Record everything, even if cassette exists

## VCR Configuration

The VCR configuration is centralized in `tests/conftest.py` using a helper function:

```python
from .conftest import make_vcr_config, VCR_MATCH_ON_DEFAULT, VCR_MATCH_ON_WITHOUT_BODY

# Default config - matches on body (most tests)
@pytest.fixture(scope="module")
def vcr_config():
    return make_vcr_config()

# Skip body matching for tests with dynamic request bodies
@pytest.fixture(scope="module")
def vcr_config():
    return make_vcr_config(match_on=VCR_MATCH_ON_WITHOUT_BODY)

# Allow specific hosts to bypass VCR (e.g., for tiktoken downloads)
@pytest.fixture(scope="module")
def vcr_config():
    config = make_vcr_config()
    config["ignore_hosts"] = ["openaipublic.blob.core.windows.net"]
    return config
```

### Match-on options

- `VCR_MATCH_ON_DEFAULT` = `["method", "scheme", "host", "port", "path", "body"]`
  - Use for most tests where request bodies are deterministic
- `VCR_MATCH_ON_WITHOUT_BODY` = `["method", "scheme", "host", "port", "path"]`
  - Use when request bodies contain dynamic data (temp filenames, generated IDs)

## Adding VCR to Tests

### Basic usage

```python
@pytest.mark.vcr
def test_provider_simple_request():
    chat = ChatProvider()
    chat.chat("Hello")
```

### For async tests, put `@pytest.mark.vcr` first

```python
@pytest.mark.vcr
@pytest.mark.asyncio
async def test_provider_async():
    ...
```

### For tests with dynamic request bodies

Some tests have request bodies that change between runs (e.g., temp filenames, framework-generated IDs). Use a module-level VCR config override:

```python
from .conftest import VCR_MATCH_ON_WITHOUT_BODY, make_vcr_config

# Don't match on body - temp file names are dynamic
@pytest.fixture(scope="module")
def vcr_config():
    return make_vcr_config(match_on=VCR_MATCH_ON_WITHOUT_BODY)
```

### For tests that need external downloads

Some tests require downloading external resources (e.g., tiktoken encodings). Allow specific hosts to bypass VCR:

```python
from .conftest import make_vcr_config

@pytest.fixture(scope="module")
def vcr_config():
    config = make_vcr_config()
    config["ignore_hosts"] = ["openaipublic.blob.core.windows.net"]
    return config
```

### For multi-sample tests that are flaky with VCR

Some tests involve multiple API calls where response ordering matters. These should skip VCR and run live only:

```python
from .conftest import is_dummy_credential

@pytest.mark.skipif(
    is_dummy_credential("ANTHROPIC_API_KEY"),
    reason="Multi-sample tests require live API (VCR response ordering is unreliable)",
)
def test_multiple_samples():
    ...
```

### For flaky API tests

Some API tests may fail intermittently due to rate limiting or transient errors. Use the `retry_api_call` decorator:

```python
from .conftest import retry_api_call

@pytest.mark.vcr
@retry_api_call
def test_flaky_api_call():
    # This will retry up to 3 times with exponential backoff
    ...
```

The decorator uses exponential backoff (1-60 seconds) and stops after 3 attempts.

## Provider Status

| Provider | VCR Status | Notes |
|----------|------------|-------|
| OpenAI | Supported | |
| Anthropic | Supported | |
| Google | Supported | |
| Azure | Supported | |
| Databricks | Supported | |
| DeepSeek | Supported | |
| GitHub | Supported | |
| HuggingFace | Supported | |
| Mistral | Supported | |
| OpenAI Completions | Supported | |
| OpenRouter | Supported | |
| Cloudflare | Supported | |
| Bedrock | **Live only** | AWS SSO credential fetching is incompatible with VCR |
| Snowflake | **Live only** | Requires special auth setup |

## Security: Filtered Data

The VCR configuration automatically filters sensitive data:

### Filtered request headers
- `authorization`
- `x-api-key`, `api-key`
- `x-goog-api-key`
- `user-agent`
- Various `x-stainless-*` headers
- AWS headers (`x-amz-sso_bearer_token`, etc.)

### Verify before committing

Always check for leaked secrets before committing cassettes:

```bash
# Quick check using make target
make check-vcr-secrets

# Or manually
grep -rE "sk-[a-zA-Z0-9]{20,}" tests/_vcr/
grep -rE "key-[a-zA-Z0-9]{20,}" tests/_vcr/
```

## CI Workflows

### `test.yml` (VCR replay)
- Runs on every PR/push
- Uses dummy API keys (set automatically by `conftest.py` if not in environment)
- Replays cassettes, no live API calls

### `test-live.yml` (live API)
- Runs on pushes to main and PRs to main
- Uses real API keys from GitHub secrets
- Makes actual API calls

## Troubleshooting

### "Can't find cassette" error

The cassette doesn't exist or the request doesn't match. Record it:

```bash
uv run pytest tests/test_provider_x.py::test_name -v --record-mode=rewrite
```

### Request not matching existing cassette

VCR matches on: `method`, `scheme`, `host`, `port`, `path`, `body` (by default)

If the request body changed (e.g., different model parameters), re-record:

```bash
uv run pytest tests/test_provider_x.py -v --record-mode=rewrite
```

### Provider can't use VCR

Some providers have authentication that happens before HTTP requests (e.g., AWS SSO for Bedrock). These must be live-only:

1. Remove `@pytest.mark.vcr` markers
2. Add skip logic at top of test file:
   ```python
   import os
   import pytest

   if not os.getenv("PROVIDER_API_KEY"):
       pytest.skip("PROVIDER_API_KEY not set", allow_module_level=True)
   ```
