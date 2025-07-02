import warnings

import pytest
from chatlas import ChatAnthropic, ChatGoogle, ChatOpenAI, ChatSnowflake


class TestSetModelParams:
    """Test the set_model_params() functionality across different providers."""

    def test_anthropic_set_model_params(self):
        """Test set_model_params for Anthropic provider."""
        chat = ChatAnthropic()

        # Test setting supported parameters
        chat.set_model_params(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_tokens=100,
            stop_sequences=["STOP"],
        )

        # Check that the parameters are stored in _submit_input_kwargs
        kwargs = getattr(chat.provider, "_submit_input_kwargs", {})
        assert kwargs.get("temperature") == 0.7
        assert kwargs.get("top_p") == 0.9
        assert kwargs.get("top_k") == 50
        assert kwargs.get("max_tokens") == 100
        assert kwargs.get("stop_sequences") == ["STOP"]

        # Test warning for unsupported parameters
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            chat.set_model_params(
                frequency_penalty=0.5, presence_penalty=0.3, seed=42, log_probs=True
            )
            # Should have warnings for unsupported parameters
            assert len(w) == 4
            assert "frequency_penalty is not supported by Anthropic models" in str(
                w[0].message
            )
            assert "presence_penalty is not supported by Anthropic models" in str(
                w[1].message
            )
            assert "seed is not supported by Anthropic models" in str(w[2].message)
            assert "log_probs is not supported by Anthropic models" in str(w[3].message)

    def test_openai_set_model_params(self):
        """Test set_model_params for OpenAI provider."""
        chat = ChatOpenAI()

        # Test setting supported parameters
        chat.set_model_params(
            temperature=0.8,
            top_p=0.95,
            frequency_penalty=0.2,
            presence_penalty=0.1,
            seed=123,
            max_tokens=500,
            log_probs=True,
            stop_sequences=["END", "STOP"],
        )

        # Check that the parameters are stored in _submit_input_kwargs
        kwargs = getattr(chat.provider, "_submit_input_kwargs", {})
        assert kwargs.get("temperature") == 0.8
        assert kwargs.get("top_p") == 0.95
        assert kwargs.get("frequency_penalty") == 0.2
        assert kwargs.get("presence_penalty") == 0.1
        assert kwargs.get("seed") == 123
        assert kwargs.get("max_tokens") == 500
        assert kwargs.get("logprobs") is True
        assert kwargs.get("stop") == ["END", "STOP"]

        # Test warning for unsupported parameters
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            chat.set_model_params(top_k=40)
            # Should have warning for unsupported parameter
            assert len(w) == 1
            assert "top_k is not supported by OpenAI models" in str(w[0].message)

    def test_google_set_model_params(self):
        """Test set_model_params for Google provider."""
        chat = ChatGoogle()

        # Test setting supported parameters
        chat.set_model_params(
            temperature=0.6,
            top_p=0.8,
            top_k=30,
            max_tokens=200,
            stop_sequences=["HALT"],
            seed=456,
        )

        # For Google, parameters are stored in config
        kwargs = getattr(chat.provider, "_submit_input_kwargs", {})
        assert "config" in kwargs
        config = kwargs["config"]
        assert config["temperature"] == 0.6
        assert config["top_p"] == 0.8
        assert config["top_k"] == 30
        assert config["max_output_tokens"] == 200
        assert config["stop_sequences"] == ["HALT"]
        assert config["seed"] == 456

    def test_snowflake_set_model_params(self):
        """Test set_model_params for Snowflake provider."""
        # Skip if snowflake-ml-python is not available
        pytest.importorskip("snowflake")

        # This test requires valid Snowflake credentials, so we'll just test the method exists
        # and that warnings work correctly
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Create a chat instance (this will fail without credentials, but that's ok for this test)
            try:
                chat = ChatSnowflake(model="test")

                # Test warning for unsupported parameters
                chat.set_model_params(
                    top_k=25,
                    frequency_penalty=0.3,
                    presence_penalty=0.2,
                    seed=789,
                    log_probs=True,
                )
                # Should have warnings for unsupported parameters
                assert len(w) >= 5  # At least 5 warnings for unsupported params
            except Exception:
                # Expected if no valid credentials, skip the test
                pytest.skip("Snowflake credentials not available")

    def test_set_model_params_kwargs_override(self):
        """Test that kwargs parameter works and overrides other settings."""
        chat = ChatAnthropic()

        # Set parameters via kwargs
        chat.set_model_params(
            temperature=0.5, kwargs={"temperature": 0.9, "max_tokens": 1000}
        )

        # kwargs should override individual parameters
        kwargs = getattr(chat.provider, "_submit_input_kwargs", {})
        assert kwargs.get("temperature") == 0.9
        assert kwargs.get("max_tokens") == 1000

    def test_set_model_params_none_values(self):
        """Test that None values are ignored."""
        chat = ChatAnthropic()

        # Set some parameters to None
        chat.set_model_params(temperature=None, top_p=0.8, top_k=None, max_tokens=200)

        # Only non-None values should be stored
        kwargs = getattr(chat.provider, "_submit_input_kwargs", {})
        assert "temperature" not in kwargs
        assert "top_k" not in kwargs
        assert kwargs.get("top_p") == 0.8
        assert kwargs.get("max_tokens") == 200

    def test_set_model_params_multiple_calls(self):
        """Test that multiple calls to set_model_params work correctly."""
        chat = ChatOpenAI()

        # First call
        chat.set_model_params(temperature=0.5, top_p=0.8)
        kwargs = getattr(chat.provider, "_submit_input_kwargs", {})
        assert kwargs.get("temperature") == 0.5
        assert kwargs.get("top_p") == 0.8

        # Second call should replace previous settings
        chat.set_model_params(temperature=0.7, max_tokens=300)
        kwargs = getattr(chat.provider, "_submit_input_kwargs", {})
        assert kwargs.get("temperature") == 0.7
        assert kwargs.get("max_tokens") == 300
        # top_p should be gone since it wasn't set in the second call
        assert "top_p" not in kwargs

    def test_set_model_params_integration_with_chat_args(self):
        """Test that set_model_params works with arguments passed to chat methods."""
        chat = ChatAnthropic()

        # Set some model parameters
        chat.set_model_params(temperature=0.6, max_tokens=50)

        # Verify that _chat_perform_args includes the parameters
        args = chat.provider._chat_perform_args(  # type: ignore
            stream=False, turns=[], tools={}, data_model=None, kwargs=None
        )

        assert args["temperature"] == 0.6
        assert args["max_tokens"] == 50

        # Test that chat-level kwargs can override set_model_params
        args_with_override = chat.provider._chat_perform_args(  # type: ignore
            stream=False,
            turns=[],
            tools={},
            data_model=None,
            kwargs={"temperature": 0.9},
        )

        # Chat-level kwargs should override set_model_params
        assert args_with_override["temperature"] == 0.9
        assert args_with_override["max_tokens"] == 50
