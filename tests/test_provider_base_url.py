"""Tests for the `Provider.base_url` attribute used by telemetry."""

import pytest
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


def test_bedrock_converse_base_url(monkeypatch: pytest.MonkeyPatch):
    # Construction is offline-safe by design (credentials resolve lazily),
    # but the region must be supplied explicitly in a config-less environment.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "dummy")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "dummy")
    from chatlas import ChatBedrock

    assert (
        ChatBedrock(aws_region="us-east-1").provider.base_url
        == "https://bedrock-runtime.us-east-1.amazonaws.com"
    )
