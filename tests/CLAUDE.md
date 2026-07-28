# VCR Testing (HTTP Recording/Replay)

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
