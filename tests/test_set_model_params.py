"""Tests for the set_model_params() feature."""

import pytest
from chatlas import ChatAnthropic, ChatGoogle, ChatOpenAI
from chatlas._provider import StandardModelParams
from chatlas._utils import MISSING


def test_set_model_params_basic():
    """Test basic functionality of set_model_params()."""
    chat = ChatOpenAI()

    # Set some basic parameters
    chat.set_model_params(temperature=0.5, max_tokens=100, top_p=0.9)

    # Verify parameters are stored internally
    params = getattr(chat, "_standard_model_params", {})
    assert params["temperature"] == 0.5
    assert params["max_tokens"] == 100
    assert params["top_p"] == 0.9


def test_set_model_params_missing_values():
    """Test that MISSING values are not stored."""
    chat = ChatOpenAI()

    # Set only some parameters, leaving others as MISSING
    chat.set_model_params(
        temperature=0.7,
        top_p=MISSING,  # Should not be stored
        max_tokens=200,
    )

    params = getattr(chat, "_standard_model_params", {})
    assert params["temperature"] == 0.7
    assert params["max_tokens"] == 200
    assert "top_p" not in params  # Should not be there


def test_set_model_params_none_reset():
    """Test that None values reset parameters."""
    chat = ChatOpenAI()

    # First set some parameters
    chat.set_model_params(temperature=0.8, top_p=0.95)
    params = getattr(chat, "_standard_model_params", {})
    assert "temperature" in params
    assert "top_p" in params

    # Reset temperature to None
    chat.set_model_params(temperature=None)
    assert "temperature" not in params
    assert "top_p" in params  # Should still be there


def test_set_model_params_kwargs():
    """Test setting provider-specific kwargs."""
    chat = ChatOpenAI()

    chat.set_model_params(kwargs={"frequency_penalty": 0.1, "presence_penalty": 0.2})
    kwargs = getattr(chat, "_submit_input_kwargs", {})
    assert kwargs == {"frequency_penalty": 0.1, "presence_penalty": 0.2}

    # Reset kwargs to None
    chat.set_model_params(kwargs=None)
    kwargs = getattr(chat, "_submit_input_kwargs", {})
    assert kwargs is None


def test_set_model_params_all_parameters():
    """Test setting all supported standard parameters."""
    chat = ChatOpenAI()

    chat.set_model_params(
        temperature=0.5,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        seed=42,
        max_tokens=150,
        log_probs=True,
        stop_sequences=["END", "STOP"],
    )

    expected = {
        "temperature": 0.5,
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.2,
        "seed": 42,
        "max_tokens": 150,
        "log_probs": True,
        "stop_sequences": ["END", "STOP"],
    }

    assert chat._standard_model_params == expected


def test_set_model_params_unsupported_parameter():
    """Test validation of unsupported parameters."""
    chat = ChatOpenAI()

    # OpenAI doesn't support top_k, should raise warning
    with pytest.warns(UserWarning, match="not supported by the provider"):
        chat.set_model_params(top_k=50)


def test_set_model_params_anthropic_supports_top_k():
    """Test that Anthropic provider supports top_k."""
    chat = ChatAnthropic()

    # Anthropic supports top_k, should not raise
    chat.set_model_params(top_k=50, temperature=0.7)
    params = getattr(chat, "_standard_model_params", {})
    assert params["top_k"] == 50
    assert params["temperature"] == 0.7


def test_set_model_params_anthropic_unsupported():
    """Test Anthropic doesn't support frequency_penalty."""
    chat = ChatAnthropic()

    # Anthropic doesn't support frequency_penalty
    with pytest.warns(UserWarning, match="not supported by the provider"):
        chat.set_model_params(frequency_penalty=0.1)


def test_translate_model_params_openai():
    """Test OpenAI provider's translate_model_params method."""
    chat = ChatOpenAI()
    provider = chat.provider

    params: StandardModelParams = {
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 100,
        "log_probs": True,
        "stop_sequences": ["END"],
    }

    result = provider.translate_model_params(params)

    expected = {
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 100,
        "logprobs": True,  # Note: OpenAI uses "logprobs" not "log_probs"
        "stop": ["END"],  # Note: OpenAI uses "stop" not "stop_sequences"
    }

    assert result == expected


def test_translate_model_params_anthropic():
    """Test Anthropic provider's translate_model_params method."""
    chat = ChatAnthropic()
    provider = chat.provider

    params: StandardModelParams = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": 200,
        "stop_sequences": ["STOP"],
    }

    result = provider.translate_model_params(params)

    expected = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": 200,
        "stop_sequences": ["STOP"],
    }

    assert result == expected


def test_supported_model_params_openai():
    """Test OpenAI provider's supported_model_params method."""
    chat = ChatOpenAI()
    supported = chat.provider.supported_model_params()

    expected = {
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "max_tokens",
        "log_probs",
        "stop_sequences",
    }

    assert supported == expected


def test_supported_model_params_anthropic():
    """Test Anthropic provider's supported_model_params method."""
    chat = ChatAnthropic()
    supported = chat.provider.supported_model_params()

    expected = {"temperature", "top_p", "top_k", "max_tokens", "stop_sequences"}

    assert supported == expected


def test_model_params_integration_with_provider():
    """Test that model parameters are integrated with provider arguments."""
    chat = ChatOpenAI()

    # Set model parameters
    chat.set_model_params(temperature=0.3, max_tokens=50, top_p=0.8)

    # Test that provider.translate_model_params converts them correctly
    provider_args = chat.provider.translate_model_params(chat._standard_model_params)

    assert provider_args["temperature"] == 0.3
    assert provider_args["max_tokens"] == 50
    assert provider_args["top_p"] == 0.8


def test_model_params_preserved_across_calls():
    """Test that model parameters are preserved across multiple chat calls."""
    chat = ChatOpenAI()

    # Set model parameters
    chat.set_model_params(temperature=0.7, max_tokens=150)

    # Verify parameters are stored
    params = getattr(chat, "_standard_model_params", {})
    assert params["temperature"] == 0.7
    assert params["max_tokens"] == 150

    # Simulate multiple calls - parameters should remain
    assert params["temperature"] == 0.7
    assert params["max_tokens"] == 150

    # Update parameters and verify they change
    chat.set_model_params(temperature=0.5)
    assert params["temperature"] == 0.5
    assert params["max_tokens"] == 150  # Should remain


def test_set_model_params_updates_existing():
    """Test that calling set_model_params multiple times updates existing parameters."""
    chat = ChatOpenAI()

    # Initial parameters
    chat.set_model_params(temperature=0.5, max_tokens=100)
    params = getattr(chat, "_standard_model_params", {})
    assert params["temperature"] == 0.5
    assert params["max_tokens"] == 100

    # Update some parameters
    chat.set_model_params(temperature=0.8, top_p=0.9)
    assert params["temperature"] == 0.8  # Updated
    assert params["max_tokens"] == 100  # Unchanged
    assert params["top_p"] == 0.9  # New


def test_set_model_params_kwargs_replacement():
    """Test that kwargs are completely replaced, not merged."""
    chat = ChatOpenAI()

    # Set initial kwargs
    chat.set_model_params(kwargs={"frequency_penalty": 0.1, "presence_penalty": 0.2})
    kwargs = getattr(chat, "_submit_input_kwargs", {})
    assert kwargs == {"frequency_penalty": 0.1, "presence_penalty": 0.2}

    # Set new kwargs - should completely replace
    chat.set_model_params(kwargs={"seed": 42})
    kwargs = getattr(chat, "_submit_input_kwargs", {})
    assert kwargs == {"seed": 42}
    assert "frequency_penalty" not in kwargs
    assert "presence_penalty" not in kwargs


def test_set_model_params_invalid_temperature():
    """Test validation of temperature parameter ranges."""
    chat = ChatOpenAI()

    # These should work fine (no validation in the method itself)
    chat.set_model_params(temperature=0.0)
    chat.set_model_params(temperature=2.0)
    chat.set_model_params(temperature=-1.0)  # Invalid but not validated


def test_set_model_params_invalid_top_p():
    """Test validation of top_p parameter ranges."""
    chat = ChatOpenAI()

    # These should work fine (no validation in the method itself)
    chat.set_model_params(top_p=0.0)
    chat.set_model_params(top_p=1.0)
    chat.set_model_params(top_p=-0.1)  # Invalid but not validated
    chat.set_model_params(top_p=1.1)  # Invalid but not validated


def test_set_model_params_multiple_unsupported():
    """Test error message when multiple parameters are unsupported."""
    chat = ChatOpenAI()

    # OpenAI doesn't support top_k or frequency_penalty is actually supported
    # Let's use a parameter that definitely doesn't exist
    with pytest.warns(UserWarning) as warn_info:
        chat.set_model_params(top_k=50)  # Not supported by OpenAI

    assert "not supported by the provider" in str(warn_info[0].message)
    assert "top_k" in str(warn_info[0].message)


def test_set_model_params_empty_call():
    """Test calling set_model_params with no arguments."""
    chat = ChatOpenAI()

    # Should not raise any errors
    chat.set_model_params()

    # Should not change anything
    assert chat._standard_model_params == {}
    assert chat._submit_input_kwargs is None


def test_set_model_params_type_validation():
    """Test that parameters accept the correct types."""
    chat = ChatOpenAI()

    # These should work
    chat.set_model_params(
        temperature=0.5,  # float
        top_p=0.9,  # float
        max_tokens=100,  # int
        log_probs=True,  # bool
        stop_sequences=["END"],  # list[str]
        seed=42,  # int
    )

    # Test with different valid types
    chat.set_model_params(
        temperature=1,  # int should work for float
        max_tokens=100,  # int
        log_probs=False,  # bool
        stop_sequences=[],  # empty list
    )


def test_set_model_params_incremental_updates():
    """Test that incremental updates work correctly."""
    chat = ChatOpenAI()

    # Build up parameters incrementally
    chat.set_model_params(temperature=0.5)
    params = getattr(chat, "_standard_model_params", {})
    assert len(params) == 1

    chat.set_model_params(max_tokens=100)
    assert len(params) == 2
    assert params["temperature"] == 0.5
    assert params["max_tokens"] == 100

    chat.set_model_params(top_p=0.9)
    assert len(params) == 3


def test_set_model_params_reset_specific_param():
    """Test resetting specific parameters while keeping others."""
    chat = ChatOpenAI()

    # Set multiple parameters
    chat.set_model_params(temperature=0.7, max_tokens=150, top_p=0.95, seed=42)
    params = getattr(chat, "_standard_model_params", {})
    assert len(params) == 4

    # Reset only temperature
    chat.set_model_params(temperature=None)

    assert len(params) == 3
    assert "temperature" not in params
    assert params["max_tokens"] == 150

    # Reset multiple parameters
    chat.set_model_params(max_tokens=None, seed=None)
    assert len(params) == 1
    assert "max_tokens" not in params
    assert "seed" not in params
    assert params["top_p"] == 0.95


def test_model_params_kwargs_priority():
    """Test that submit_input_kwargs override model_params."""
    chat = ChatOpenAI()

    # Set model parameters
    chat.set_model_params(temperature=0.1, max_tokens=100)

    # Set submit input kwargs that override some model params
    chat.set_model_params(kwargs={"temperature": 0.5, "seed": 42})

    # Verify that kwargs are set correctly
    kwargs = getattr(chat, "_submit_input_kwargs", {})
    params = getattr(chat, "_standard_model_params", {})
    assert kwargs["temperature"] == 0.5
    assert kwargs["seed"] == 42

    # Model params should still be stored
    assert params["temperature"] == 0.1
    assert params["max_tokens"] == 100


def test_is_present_function():
    """Test the is_present helper function used in set_model_params."""
    from chatlas._chat import is_present

    # Test various values
    assert is_present(0.5) is True
    assert is_present(0) is True
    assert is_present("") is True
    assert is_present([]) is True
    assert is_present(False) is True

    assert is_present(None) is False
    assert is_present(MISSING) is False


def test_set_model_params_with_stop_sequences():
    """Test stop_sequences parameter specifically."""
    chat = ChatOpenAI()

    # Test with various stop sequence formats
    chat.set_model_params(stop_sequences=["END", "STOP", "FINISH"])
    params = getattr(chat, "_standard_model_params", {})
    assert params["stop_sequences"] == ["END", "STOP", "FINISH"]

    # Test with empty list
    chat.set_model_params(stop_sequences=[])
    assert params["stop_sequences"] == []

    # Test reset to None
    chat.set_model_params(stop_sequences=None)
    assert "stop_sequences" not in params


def test_google_provider_model_params():
    """Test Google provider's model parameter support."""
    chat = ChatGoogle()

    # Test Google-specific supported parameters
    chat.set_model_params(
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        max_tokens=150,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        seed=42,
        log_probs=True,
        stop_sequences=["END"],
    )

    # Verify Google's supported parameters
    supported = chat.provider.supported_model_params()
    expected_google_params = {
        "temperature",
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "max_tokens",
        "log_probs",
        "stop_sequences",
    }
    assert supported == expected_google_params


def test_google_provider_parameter_mapping():
    """Test Google provider's parameter mapping to API format."""
    chat = ChatGoogle()
    provider = chat.provider

    params: StandardModelParams = {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": 200,
        "seed": 123,
        "stop_sequences": ["STOP"],
    }

    result = provider.translate_model_params(params)

    # Google uses a nested "config" structure
    assert "config" in result
    config = result["config"]
    assert config["temperature"] == 0.8
    assert config["top_p"] == 0.95
    assert config["top_k"] == 50
    assert config["max_output_tokens"] == 200  # Note: Google uses "max_output_tokens"
    assert config["seed"] == 123
    assert config["stop_sequences"] == ["STOP"]


def test_provider_parameter_differences():
    """Test that different providers support different parameter sets."""
    openai_chat = ChatOpenAI()
    anthropic_chat = ChatAnthropic()

    openai_supported = openai_chat.provider.supported_model_params()
    anthropic_supported = anthropic_chat.provider.supported_model_params()

    # OpenAI supports frequency_penalty but Anthropic doesn't
    assert "frequency_penalty" in openai_supported
    assert "frequency_penalty" not in anthropic_supported

    # Anthropic supports top_k but OpenAI doesn't
    assert "top_k" in anthropic_supported
    assert "top_k" not in openai_supported

    # Both should support common parameters
    common_params = {"temperature", "top_p", "max_tokens", "stop_sequences"}
    assert common_params.issubset(openai_supported)
    assert common_params.issubset(anthropic_supported)


def test_cross_provider_compatibility():
    """Test that model params work consistently across providers."""
    providers = [
        ("OpenAI", ChatOpenAI()),
        ("Anthropic", ChatAnthropic()),
        ("Google", ChatGoogle()),
    ]

    # Test common parameters across all providers (excluding Snowflake due to config requirements)
    for name, chat in providers:
        # Set common parameters that should work for all
        chat.set_model_params(temperature=0.7, max_tokens=100)
        params = getattr(chat, "_standard_model_params", {})
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 100


def test_anthropic_model_params_integration():
    """Test Anthropic-specific parameter integration."""
    chat = ChatAnthropic()

    # Set Anthropic-supported parameters including top_k
    chat.set_model_params(
        temperature=0.4, top_k=20, max_tokens=75, stop_sequences=["END", "STOP"]
    )

    # Test that provider.translate_model_params converts them correctly
    provider_args = chat.provider.translate_model_params(chat._standard_model_params)

    assert provider_args["temperature"] == 0.4
    assert provider_args["top_k"] == 20
    assert provider_args["max_tokens"] == 75
    assert provider_args["stop_sequences"] == ["END", "STOP"]


def test_parameter_validation_edge_cases():
    """Test edge cases in parameter validation."""
    chat = ChatOpenAI()

    # Test with zero values
    chat.set_model_params(temperature=0, top_p=0, max_tokens=0, seed=0)
    params = getattr(chat, "_standard_model_params", {})
    assert params["temperature"] == 0
    assert params["top_p"] == 0
    assert params["max_tokens"] == 0
    assert params["seed"] == 0

    # Test with very large values
    chat.set_model_params(max_tokens=1000000, seed=999999999)

    assert params["max_tokens"] == 1000000
    assert params["seed"] == 999999999
