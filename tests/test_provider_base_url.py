"""Tests for the `Provider.base_url` attribute used by telemetry."""

from chatlas import ChatAnthropic, ChatGoogle, ChatGroq, ChatOpenAI
from chatlas._provider import Provider


def test_base_url_defaults_to_none():
    class MinimalProvider(Provider):
        pass

    # Not a usable provider; just needs to be constructible
    MinimalProvider.__abstractmethods__ = frozenset()
    provider = MinimalProvider(name="Test", model="test-model")
    assert provider.base_url is None

    provider.base_url = "https://example.com"
    assert provider.base_url == "https://example.com"


def test_openai_base_url():
    assert ChatOpenAI().provider.base_url == "https://api.openai.com/v1/"


def test_openai_completions_base_url():
    assert ChatGroq().provider.base_url == "https://api.groq.com/openai/v1/"


def test_anthropic_base_url():
    assert ChatAnthropic().provider.base_url == "https://api.anthropic.com"


def test_google_base_url():
    assert (
        ChatGoogle().provider.base_url
        == "https://generativelanguage.googleapis.com/"
    )
