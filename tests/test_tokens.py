from typing import Optional

import pytest
from chatlas import AssistantTurn, ChatAnthropic, ChatGoogle, ChatOpenAI, UserTurn
from chatlas._provider_openai import OpenAIProvider
from chatlas._provider_openai_azure import OpenAIAzureProvider
from chatlas._tokens import (
    get_price_info,
    get_token_cost,
    token_usage,
    tokens_log,
    tokens_reset,
)
from pydantic import BaseModel

from .conftest import make_vcr_config


# Allow tiktoken to download encoding files from openaipublic.blob.core.windows.net
@pytest.fixture(scope="module")
def vcr_config():
    config = make_vcr_config()
    config["ignore_hosts"] = ["openaipublic.blob.core.windows.net"]
    return config


def test_tokens_method():
    chat = ChatOpenAI(api_key="fake_key")
    assert len(chat.get_tokens()) == 0

    chat = ChatOpenAI()
    chat.set_turns(
        [
            UserTurn("Hi"),
            AssistantTurn("Hello", tokens=(2, 10, 0)),
        ]
    )

    assert chat.get_tokens() == [
        {"role": "user", "tokens": 2, "tokens_cached": 0, "tokens_total": 2},
        {"role": "assistant", "tokens": 10, "tokens_cached": 0, "tokens_total": 10},
    ]

    chat = ChatOpenAI()
    chat.set_turns(
        [
            UserTurn("Hi"),
            AssistantTurn("Hello", tokens=(2, 10, 0)),
            UserTurn("Hi"),
            AssistantTurn("Hello", tokens=(14, 10, 0)),
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
            UserTurn("Hi"),
            AssistantTurn("Hello", tokens=(2, 10, 0)),
            UserTurn("Hi"),
            AssistantTurn("Hello", tokens=(14, 10, 2)),
        ],
    )
    assert chat2.get_tokens() == [
        {"role": "user", "tokens": 2, "tokens_cached": 0, "tokens_total": 2},
        {"role": "assistant", "tokens": 10, "tokens_cached": 0, "tokens_total": 10},
        {"role": "user", "tokens": 2, "tokens_cached": 2, "tokens_total": 14},
        {"role": "assistant", "tokens": 10, "tokens_cached": 0, "tokens_total": 10},
    ]


@pytest.mark.vcr
def test_token_count_method():
    chat = ChatOpenAI(model="gpt-4o-mini")
    assert chat.token_count("What is 1 + 1?") == 32

    chat = ChatAnthropic(model="claude-haiku-4-5-20251001")
    assert chat.token_count("What is 1 + 1?") == 16

    chat = ChatGoogle(model="gemini-2.5-flash")
    assert chat.token_count("What is 1 + 1?") == 9


def test_get_token_prices():
    chat = ChatOpenAI(model="o1-mini")
    pricing = get_price_info(chat.provider.name, chat.provider.model)
    assert pricing is not None
    assert pricing["provider"] == "OpenAI"
    assert pricing["model"] == "o1-mini"
    assert isinstance(pricing["input"], float)
    # cached_input and output might be optional
    if "cached_input" in pricing:
        assert isinstance(pricing["cached_input"], float)
    if "output" in pricing:
        assert isinstance(pricing["output"], float)


def test_get_token_cost():
    chat = ChatOpenAI(model="o1-mini")
    price = get_token_cost(chat.provider.name, chat.provider.model, (10, 50, 0))
    assert isinstance(price, float)
    assert price > 0

    chat = ChatOpenAI(model="ABCD")
    price = get_token_cost(chat.provider.name, chat.provider.model, (10, 50, 0))
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


def test_get_price_info_with_variants():
    """Test pricing variant lookup and fallback behavior."""
    # gpt-4o has baseline, 'batches', and 'priority' variants

    # Test baseline (empty variant)
    baseline = get_price_info("OpenAI", "gpt-4o", variant="")
    assert baseline is not None
    assert baseline["input"] == 2.5
    assert baseline["output"] == 10
    assert baseline.get("variant", "") == ""

    # Test specific variant
    batches = get_price_info("OpenAI", "gpt-4o", variant="batches")
    assert batches is not None
    assert batches["input"] == 1.25  # 50% discount
    assert batches["output"] == 5
    assert batches.get("variant") == "batches"

    priority = get_price_info("OpenAI", "gpt-4o", variant="priority")
    assert priority is not None
    assert priority["input"] == 4.25  # Higher price
    assert priority["output"] == 17
    assert priority.get("variant") == "priority"

    # Test fallback: non-existent variant should fall back to baseline
    fallback = get_price_info("OpenAI", "gpt-4o", variant="nonexistent_variant")
    assert fallback is not None
    assert fallback["input"] == 2.5  # Should match baseline
    assert fallback.get("variant", "") == ""

    # Test no fallback when variant not specified and model not found
    unknown = get_price_info("OpenAI", "unknown-model", variant="")
    assert unknown is None


def test_get_token_cost_with_variants():
    """Test cost calculation with different pricing variants."""
    tokens = (1_000_000, 1_000_000, 0)  # 1M input, 1M output, 0 cached

    # Baseline pricing
    baseline_cost = get_token_cost("OpenAI", "gpt-4o", tokens, variant="")
    assert baseline_cost is not None
    expected_baseline = (1_000_000 * 2.5 / 1e6) + (1_000_000 * 10 / 1e6)
    assert baseline_cost == expected_baseline  # $2.50 + $10 = $12.50

    # Batches pricing (cheaper)
    batches_cost = get_token_cost("OpenAI", "gpt-4o", tokens, variant="batches")
    assert batches_cost is not None
    expected_batches = (1_000_000 * 1.25 / 1e6) + (1_000_000 * 5 / 1e6)
    assert batches_cost == expected_batches  # $1.25 + $5 = $6.25

    # Priority pricing (more expensive)
    priority_cost = get_token_cost("OpenAI", "gpt-4o", tokens, variant="priority")
    assert priority_cost is not None
    expected_priority = (1_000_000 * 4.25 / 1e6) + (1_000_000 * 17 / 1e6)
    assert priority_cost == expected_priority  # $4.25 + $17 = $21.25

    # Verify ordering: batches < baseline < priority
    assert batches_cost < baseline_cost < priority_cost


def test_tokens_log_with_variant():
    """Test that tokens_log accepts variant parameter."""
    tokens_reset()

    provider = OpenAIProvider(api_key="fake_key", model="gpt-4o")

    # Log tokens with variant
    tokens_log(provider, (100, 50, 0), variant="batches")

    usage = token_usage()
    assert usage is not None
    assert len(usage) == 1
    assert usage[0]["input"] == 100
    assert usage[0]["output"] == 50
    # Cost should be calculated using batches pricing
    assert usage[0]["cost"] is not None

    tokens_reset()


def test_provider_value_cost():
    """Test Provider.value_cost() method."""
    provider = OpenAIProvider(api_key="fake_key", model="gpt-4o")

    # Test with pre-computed tokens (completion=None is valid when tokens provided)
    tokens = (1000, 500, 0)
    cost = provider.value_cost(completion=None, tokens=tokens)
    assert cost is not None
    expected = (1000 * 2.5 / 1e6) + (500 * 10 / 1e6)
    assert cost == expected

    # Test with unknown model returns None
    provider_unknown = OpenAIProvider(api_key="fake_key", model="unknown-model")
    cost_unknown = provider_unknown.value_cost(completion=None, tokens=tokens)
    assert cost_unknown is None
