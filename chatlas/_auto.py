from __future__ import annotations

import os
import warnings
from typing import Callable, Literal, Optional

import orjson

from ._chat import Chat
from ._provider_anthropic import ChatAnthropic, ChatBedrockAnthropic
from ._provider_databricks import ChatDatabricks
from ._provider_github import ChatGithub
from ._provider_google import ChatGoogle, ChatVertex
from ._provider_groq import ChatGroq
from ._provider_ollama import ChatOllama
from ._provider_openai import ChatAzureOpenAI, ChatOpenAI
from ._provider_perplexity import ChatPerplexity
from ._provider_snowflake import ChatSnowflake
from ._utils import MISSING_TYPE as DEPRECATED_TYPE

AutoProviders = Literal[
    "anthropic",
    "bedrock-anthropic",
    "databricks",
    "github",
    "google",
    "groq",
    "ollama",
    "openai",
    "azure-openai",
    "perplexity",
    "snowflake",
    "vertex",
]

_provider_chat_model_map: dict[AutoProviders, Callable[..., Chat]] = {
    "anthropic": ChatAnthropic,
    "bedrock-anthropic": ChatBedrockAnthropic,
    "databricks": ChatDatabricks,
    "github": ChatGithub,
    "google": ChatGoogle,
    "groq": ChatGroq,
    "ollama": ChatOllama,
    "openai": ChatOpenAI,
    "azure-openai": ChatAzureOpenAI,
    "perplexity": ChatPerplexity,
    "snowflake": ChatSnowflake,
    "vertex": ChatVertex,
}

DEPRECATED = DEPRECATED_TYPE()


def ChatAuto(
    provider_model: Optional[str] = None,
    *,
    system_prompt: Optional[str] = None,
    provider: Optional[AutoProviders | DEPRECATED_TYPE] = DEPRECATED,
    model: Optional[str | DEPRECATED_TYPE] = DEPRECATED,
    **kwargs,
) -> Chat:
    """
    Use environment variables (env vars) to configure the Chat provider and model.

    Creates a :class:`~chatlas.Chat` instance based on the specified provider
    and model, which can be set directly or through environment variables. The
    `provider_model` parameter expects a string that specifies both the provider
    and model in the format `{provider}/{model}`, e.g. `"openai/gpt-4o"` or
    `"anthropic/claude-3-7-sonnet-20250219"`. Alternatively, you can specify
    only the provider to use the default model for that provider.

    If not provided explicitly, chatlas will use the
    `CHATLAS_CHAT_PROVIDER_MODEL` environment variable. Additional configuration
    may be provided through the `kwargs` parameter and/or the
    `CHATLAS_CHAT_ARGS` env var (as a JSON string).

    `ChatAuto()` always uses the values of arguments passed to it directly
    over the values in environment variables.

    If neither the `provider_model` parameter nor the env var are set, chatlas
    will fall back to using the default model from :class:`~chatlas.ChatOpenAI`.

    In applications or programs that extend chatlas, you may want to introduce
    an application-specific environment variables and model default. To do this,
    you can pass your own environment variable values to `provider_model` and
    `kwargs` (with some pre-processing to unserialize JSON strings if needed).

    ```python
    import json
    import os

    from chatlas import ChatAuto

    provider_model = os.environ.get("MYAPP_PROVIDER_MODEL", "anthropic")
    provider_args = json.loads(os.environ.get("MYAPP_ARGS", "{}"))

    chat = ChatAuto(provider_model, **provider_args)
    ```

    Prerequisites
    -------------

    ::: {.callout-note}
    ## API key

    Follow the instructions for the specific provider to obtain an API key.
    :::

    ::: {.callout-note}
    ## Python requirements

    Follow the instructions for the specific provider to install the required
    Python packages.
    :::


    Examples
    --------
    First, set the environment variables for the provider, arguments, and API key:

    ```bash
    export CHATLAS_CHAT_PROVIDER_MODEL="anthropic/claude-3-haiku-20240229"
    export CHATLAS_CHAT_ARGS='{"kwargs": {"max_retries": 3}}'
    export ANTHROPIC_API_KEY=your_api_key
    ```

    Then, you can use the `ChatAuto` function to create a Chat instance:

    ```python
    from chatlas import ChatAuto

    chat = ChatAuto()
    chat.chat("What is the capital of France?")
    ```

    Parameters
    ----------
    system_prompt
        A system prompt to set the behavior of the assistant.
    provider_model
        The name of the provider and model to use in the format
        `"{provider}/{model}"`. Providers are strings formatted in kebab-case,
        e.g. to use `ChatBedrockAnthropic` set `provider="bedrock-anthropic"`,
        and models are the provider-specific model names, e.g.
        `"claude-3-7-sonnet-20250219"`. If only the provider is specified,
        the default model for that provider will be used.

        This value can also be provided via the `CHATLAS_CHAT_PROVIDER_MODEL`
        environment variable, as long as `provider_model` is not provided when
        calling `ChatAuto()`.
    provider
        Deprecated; use `provider_model` instead.
    model
        Deprecated; use `provider_model` instead.
    **kwargs
        Additional keyword arguments to pass to the Chat constructor. See the
        documentation for each provider for more details on the available
        options.

        These arguments can also be provided via the `CHATLAS_CHAT_ARGS`
        environment variable as a JSON string. When any additional arguments are
        provided to `ChatAuto()`, the env var is ignored.

        Note that `system_prompt` and `turns` in `kwargs` or in
        `CHATLAS_CHAT_ARGS` are ignored, as is `model` in `CHATLAS_CHAT_ARGS`.

    Returns
    -------
    Chat
        A chat instance using the specified provider.

    Raises
    ------
    ValueError
        If no valid provider is specified either through parameters or
        environment variables.
    """
    if provider is not DEPRECATED:
        _warn_deprecated_param("provider")

    if model is not DEPRECATED:
        if provider is DEPRECATED:
            raise ValueError(
                "The `model` parameter is deprecated and cannot be used without the `provider` parameter. "
                "Use `provider_model` instead."
            )
        _warn_deprecated_param("model")

    if provider_model is None:
        provider_model = os.environ.get("CHATLAS_CHAT_PROVIDER_MODEL")

    if provider_model is None:
        # Backwards compatibility: construct from old env vars as a fallback
        env_provider = _get_legacy_env_var("CHATLAS_CHAT_PROVIDER", provider)
        env_model = _get_legacy_env_var("CHATLAS_CHAT_MODEL", model)

        if env_provider:
            provider_model = env_provider
            if env_model:
                provider_model += f"/{env_model}"

    if provider_model is None:
        # Fall back to OpenAI if nothing is specified
        provider_model = "openai"

    the_provider, the_model = _parse_provider_model(provider_model)

    if the_provider not in _provider_chat_model_map:
        raise ValueError(
            f"Provider name '{the_provider}' is not a known chatlas provider: "
            f"{', '.join(_provider_chat_model_map.keys())}"
        )

    # `system_prompt`, `turns` and `model` always come from `ChatAuto()`
    base_args = {"system_prompt": system_prompt, "turns": None, "model": the_model}

    # Environment kwargs, used only if no kwargs provided
    env_kwargs = {}
    if not kwargs:
        if env_kwargs_str := os.environ.get("CHATLAS_CHAT_ARGS"):
            env_kwargs = orjson.loads(env_kwargs_str)

    final_kwargs = {**env_kwargs, **kwargs, **base_args}
    final_kwargs = {k: v for k, v in final_kwargs.items() if v is not None}

    return _provider_chat_model_map[the_provider](**final_kwargs)


def _value_if_not_deprecated(value: str | None | DEPRECATED_TYPE) -> str | None:
    return value if not isinstance(value, DEPRECATED_TYPE) else None


def _parse_provider_model(provider_model: str) -> tuple[str, Optional[str]]:
    """Parse provider_model string into provider and model components.

    Splits on the first '/' to separate provider from model.

    Args:
        provider_model: String in format "provider" or "provider/model"

    Returns:
        Tuple of (provider, model) where model may be None if not specified
    """
    if "/" in provider_model:
        provider, model = provider_model.split("/", 1)
        return provider, model
    else:
        return provider_model, None


def _get_legacy_env_var(
    env_var_name: str,
    default: str | None | DEPRECATED_TYPE,
) -> str | None:
    """Get legacy environment variable with deprecation warning, fallback to default."""
    env_value = os.environ.get(env_var_name)
    if env_value:
        warnings.warn(
            f"The '{env_var_name}' environment variable is deprecated. "
            "Use 'CHATLAS_CHAT_PROVIDER_MODEL' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return env_value
    else:
        return _value_if_not_deprecated(default)


def _warn_deprecated_param(param_name: str, stacklevel: int = 3) -> None:
    """Issue deprecation warning for old parameters."""
    warnings.warn(
        f"The '{param_name}' parameter is deprecated. Use 'provider_model' instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
