# VCR Test Recording Guide

This document explains how HTTP recording/replay works for chatlas provider tests using [pytest-recording](https://github.com/kiwicom/pytest-recording) (which wraps [vcrpy](https://vcrpy.readthedocs.io/)).

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
| Portkey | **Live only** | API issues during recording |

## Recording Cassettes

### Record all cassettes for a provider

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Record with --record-mode=rewrite (overwrites existing)
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

## Adding VCR to a New Provider

1. **Add `@pytest.mark.vcr` to each test function**:

```python
@pytest.mark.vcr
def test_provider_simple_request():
    chat = ChatProvider()
    chat.chat("Hello")
```

2. **For async tests, put `@pytest.mark.vcr` first**:

```python
@pytest.mark.vcr
@pytest.mark.asyncio
async def test_provider_async():
    ...
```

3. **Add dummy API key to `conftest.py`** (if needed for client initialization):

```python
dummy_keys = {
    ...
    "NEW_PROVIDER_API_KEY": "dummy-key-for-vcr-replay",
}
```

4. **Record the cassettes**:

```bash
export NEW_PROVIDER_API_KEY="real-key"
uv run pytest tests/test_provider_new.py -v --record-mode=rewrite
```

5. **Verify no sensitive data** in cassettes (see Security section)

6. **Commit the cassettes**:

```bash
git add tests/_vcr/test_provider_new/
```

## Security: Filtered Data

The VCR configuration in `conftest.py` automatically filters sensitive data:

### Filtered request headers
- `authorization`
- `x-api-key`, `api-key`
- `x-goog-api-key`
- `user-agent`
- Various `x-stainless-*` headers

### Filtered response headers
- `openai-organization`, `openai-project`
- `anthropic-organization-id`
- `set-cookie`
- `cf-ray`, `x-request-id`, `request-id`

### Always verify before committing

```bash
# Check for API keys
grep -r "sk-" tests/_vcr/test_provider_new/
grep -r "api_key" tests/_vcr/test_provider_new/

# Check for account/org IDs
grep -ri "org-" tests/_vcr/test_provider_new/
grep -ri "account" tests/_vcr/test_provider_new/
```

## CI Workflows

### `test.yml` (VCR replay)
- Runs on every PR/push
- Uses dummy API keys from `conftest.py`
- Replays cassettes, no live API calls
- Skips providers without VCR support via env vars:
  ```yaml
  env:
    TEST_BEDROCK: "false"
    TEST_SNOWFLAKE: "false"
    PORTKEY_API_KEY: ""
  ```

### `test-live.yml` (live API)
- Runs on demand or for releases
- Uses real API keys from GitHub secrets
- Makes actual API calls
- Tests providers that can't use VCR

## Troubleshooting

### "Can't find cassette" error

The cassette doesn't exist or the request doesn't match. Record it:

```bash
uv run pytest tests/test_provider_x.py::test_name -v --record-mode=rewrite
```

### Request not matching existing cassette

VCR matches on: `method`, `scheme`, `host`, `port`, `path`, `body`

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
3. Add to CI skip list in `test.yml`

### Cassettes contain sensitive data

If you accidentally committed sensitive data:

1. Delete the cassettes
2. Add filtering to `conftest.py` if needed
3. Re-record with filtering in place
4. Consider rotating the exposed credentials

## Makefile Targets

```bash
# Record all providers (requires all API keys set)
make record-vcr-providers

# Record specific provider
make record-vcr-openai
make record-vcr-anthropic
make record-vcr-google
# ... etc
```

## Configuration Reference

The VCR configuration lives in `tests/conftest.py`:

```python
@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [...],           # Headers to redact
        "filter_post_data_parameters": [], # POST params to redact
        "decode_compressed_response": True,
        "record_mode": "once",             # Default mode
        "match_on": ["method", "scheme", "host", "port", "path", "body"],
        "before_record_response": _filter_response_headers,
    }

@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    # Stores cassettes in tests/_vcr/{module_name}/
    module_name = request.module.__name__.split(".")[-1]
    return os.path.join(os.path.dirname(__file__), "_vcr", module_name)
```
