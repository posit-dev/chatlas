import pytest
from chatlas import ChatAnthropic, ChatGoogle, ChatOpenAI, Turn
from chatlas._openai import OpenAIAzureProvider, OpenAIProvider
from chatlas._tokens import (
    compute_cost,
    get_token_pricing,
    token_usage,
    tokens_log,
    tokens_reset,
)


def test_tokens_method():
    chat = ChatOpenAI(api_key="fake_key")
    assert len(chat.get_tokens()) == 0

    chat = ChatOpenAI()
    chat.set_turns(
        [
            Turn(role="user", contents="Hi"),
            Turn(role="assistant", contents="Hello", tokens=(2, 10)),
        ]
    )

    assert chat.get_tokens() == [
        {"role": "user", "tokens": 2, "tokens_total": 2},
        {"role": "assistant", "tokens": 10, "tokens_total": 10},
    ]

    chat = ChatOpenAI()
    chat.set_turns(
        [
            Turn(role="user", contents="Hi"),
            Turn(role="assistant", contents="Hello", tokens=(2, 10)),
            Turn(role="user", contents="Hi"),
            Turn(role="assistant", contents="Hello", tokens=(14, 10)),
        ],
    )

    assert chat.get_tokens() == [
        {"role": "user", "tokens": 2, "tokens_total": 2},
        {"role": "assistant", "tokens": 10, "tokens_total": 10},
        {"role": "user", "tokens": 2, "tokens_total": 14},
        {"role": "assistant", "tokens": 10, "tokens_total": 10},
    ]


def test_token_count_method():
    chat = ChatOpenAI(model="gpt-4o-mini")
    assert chat.token_count("What is 1 + 1?") == 31

    chat = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    assert chat.token_count("What is 1 + 1?") == 16

    chat = ChatGoogle(model="gemini-1.5-flash")
    assert chat.token_count("What is 1 + 1?") == 9


def test_get_token_prices():
    chat = ChatOpenAI(model="o1-mini")
    pricing = get_token_pricing(chat.provider.name, chat.provider.model)
    assert pricing is not None
    assert pricing["provider"] == "OpenAI"
    assert pricing["model"] == "o1-mini"
    assert isinstance(pricing["cached_input"], float)
    assert isinstance(pricing["input"], float)
    assert isinstance(pricing["output"], float)


def test_compute_cost():
    chat = ChatOpenAI(model="o1-mini")
    price = compute_cost(chat.provider.name, chat.provider.model, 10, 50)
    assert isinstance(price, float)
    assert price > 0

    chat = ChatOpenAI(model="ABCD")
    price = compute_cost(chat.provider.name, chat.provider.model, 10, 50)
    assert price is None


def test_usage_is_none():
    tokens_reset()
    assert token_usage() is None


def test_can_retrieve_and_log_tokens():
    tokens_reset()

    provider = OpenAIProvider(api_key="fake_key", model="gpt-4.1")
    tokens_log(provider, (10, 50))
    tokens_log(provider, (0, 10))
    usage = token_usage()
    assert usage is not None
    assert len(usage) == 1
    assert usage[0]["name"] == "OpenAI"
    assert usage[0]["input"] == 10
    assert usage[0]["output"] == 60
    assert usage[0]["cost"] is not None

    provider2 = OpenAIAzureProvider(
        api_key="fake_key", endpoint="foo", deployment_id="test", api_version="bar"
    )

    tokens_log(provider2, (5, 25))
    usage = token_usage()
    assert usage is not None
    assert len(usage) == 2
    assert usage[1]["name"] == "OpenAIAzure"
    assert usage[1]["input"] == 5
    assert usage[1]["output"] == 25
    assert usage[1]["cost"] is None

    tokens_reset()
