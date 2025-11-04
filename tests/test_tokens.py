from typing import Optional

from pydantic import BaseModel

from chatlas import ChatAnthropic, ChatGoogle, ChatOpenAI, Turn
from chatlas._provider_openai import OpenAIProvider
from chatlas._provider_openai_azure import OpenAIAzureProvider
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
            Turn(role="assistant", contents="Hello", tokens=(2, 10, 0)),
        ]
    )

    assert chat.get_tokens() == [
        {"role": "user", "tokens": 2, "tokens_cached": 0, "tokens_total": 2},
        {"role": "assistant", "tokens": 10, "tokens_cached": 0, "tokens_total": 10},
    ]

    chat = ChatOpenAI()
    chat.set_turns(
        [
            Turn(role="user", contents="Hi"),
            Turn(role="assistant", contents="Hello", tokens=(2, 10, 0)),
            Turn(role="user", contents="Hi"),
            Turn(role="assistant", contents="Hello", tokens=(14, 10, 0)),
        ],
    )

    assert chat.get_tokens() == [
        {"role": "user", "tokens": 2, "tokens_cached": 0, "tokens_total": 2},
        {"role": "assistant", "tokens": 10, "tokens_cached": 0, "tokens_total": 10},
        {"role": "user", "tokens": 2, "tokens_cached": 0, "tokens_total": 14},
        {"role": "assistant", "tokens": 10, "tokens_cached": 0, "tokens_total": 10},
    ]

    chat2 = ChatOpenAI()
    chat2.set_turns(
        [
            Turn(role="user", contents="Hi"),
            Turn(role="assistant", contents="Hello", tokens=(2, 10, 0)),
            Turn(role="user", contents="Hi"),
            Turn(role="assistant", contents="Hello", tokens=(14, 10, 2)),
        ],
    )
    assert chat2.get_tokens() == [
        {"role": "user", "tokens": 2, "tokens_cached": 0, "tokens_total": 2},
        {"role": "assistant", "tokens": 10, "tokens_cached": 0, "tokens_total": 10},
        {"role": "user", "tokens": 2, "tokens_cached": 2, "tokens_total": 14},
        {"role": "assistant", "tokens": 10, "tokens_cached": 0, "tokens_total": 10},
    ]


def test_token_count_method():
    chat = ChatOpenAI(model="gpt-4o-mini")
    assert chat.token_count("What is 1 + 1?") == 32

    chat = ChatAnthropic(model="claude-haiku-4-5-20251001")
    assert chat.token_count("What is 1 + 1?") == 16

    chat = ChatGoogle(model="gemini-2.5-flash")
    assert chat.token_count("What is 1 + 1?") == 9


def test_get_token_prices():
    chat = ChatOpenAI(model="o1-mini")
    pricing = get_token_pricing(chat.provider.name, chat.provider.model)
    assert pricing is not None
    assert pricing["provider"] == "OpenAI"
    assert pricing["model"] == "o1-mini"
    assert isinstance(pricing["input"], float)
    # cached_input and output might be optional
    if "cached_input" in pricing:
        assert isinstance(pricing["cached_input"], float)
    if "output" in pricing:
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
    tokens_log(provider, (10, 50, 0))
    tokens_log(provider, (0, 10, 0))
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

    tokens_log(provider2, (5, 25, 0))
    usage = token_usage()
    assert usage is not None
    assert len(usage) == 2
    assert usage[1]["name"] == "Azure/OpenAI"
    assert usage[1]["input"] == 5
    assert usage[1]["output"] == 25
    assert usage[1]["cost"] is None

    tokens_reset()


class TokenPricePydantic(BaseModel):
    """
    Pydantic model that corresponds to the TokenPrice TypedDict.
    Used for validation of the prices.json data.
    """

    provider: str
    model: str
    cached_input: Optional[float] = None  # Not all models have cached input
    input: Optional[float] = None
    output: Optional[float] = None  # Made optional for embedding models
    variant: Optional[str] = None


def test_prices_json_validates_against_typeddict():
    from chatlas._tokens import pricing_list

    try:
        validated_entries = [TokenPricePydantic(**entry) for entry in pricing_list]
    except Exception as e:
        raise AssertionError(f"Validation failed for prices.json: {e}")

    assert len(validated_entries) == len(pricing_list)
