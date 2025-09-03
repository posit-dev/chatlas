from __future__ import annotations

import os
import warnings
from typing import Callable, Literal, Optional

import orjson

from ._chat import Chat
from ._provider_anthropic import ChatAnthropic, ChatBedrockAnthropic
from ._provider_cloudflare import ChatCloudflare
from ._provider_databricks import ChatDatabricks
from ._provider_deepseek import ChatDeepSeek
from ._provider_github import ChatGithub
from ._provider_google import ChatGoogle, ChatVertex
from ._provider_groq import ChatGroq
from ._provider_huggingface import ChatHuggingFace
from ._provider_mistral import ChatMistral
from ._provider_ollama import ChatOllama
from ._provider_openai import ChatAzureOpenAI, ChatOpenAI
from ._provider_openrouter import ChatOpenRouter
from ._provider_perplexity import ChatPerplexity
from ._provider_portkey import ChatPortkey
from ._provider_snowflake import ChatSnowflake
from ._utils import MISSING_TYPE as DEPRECATED_TYPE

AutoProviders = Literal[
    "anthropic",
    "bedrock-anthropic",
    "cloudflare",
    "databricks",
    "deep-seek",
    "github",
    "google",
    "groq",
    "hugging-face",
    "mistral",
    "ollama",
    "openai",
    "azure-openai",
    "open-router",
    "perplexity",
    "portkey",
    "snowflake",
    "vertex",
]

_provider_chat_model_map: dict[AutoProviders, Callable[..., Chat]] = {
    "anthropic": ChatAnthropic,
    "bedrock-anthropic": ChatBedrockAnthropic,
    "cloudflare": ChatCloudflare,
    "databricks": ChatDatabricks,
    "deep-seek": ChatDeepSeek,
    "github": ChatGithub,
    "google": ChatGoogle,
    "groq": ChatGroq,
    "hugging-face": ChatHuggingFace,
    "mistral": ChatMistral,
    "ollama": ChatOllama,
    "openai": ChatOpenAI,
    "azure-openai": ChatAzureOpenAI,
    "open-router": ChatOpenRouter,
    "perplexity": ChatPerplexity,
    "portkey": ChatPortkey,
    "snowflake": ChatSnowflake,
    "vertex": ChatVertex,
}

DEPRECATED = DEPRECATED_TYPE()


def ChatAuto(
    provider_model: Optional[str] = None,
    *,
    system_prompt: Optional[str] = None,
    provider: AutoProviders | DEPRECATED_TYPE = DEPRECATED,
    model: str | DEPRECATED_TYPE = DEPRECATED,
    **kwargs,
) -> Chat:
    """
    Chat with any provider.

    This is a generic interface to all the other `Chat*()` functions, allowing
    you to pick the provider (and model) with a simple string.

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
    export CHATLAS_CHAT_PROVIDER_MODEL="anthropic/claude-sonnet-4-0"
    export CHATLAS_CHAT_ARGS='{"kwargs": {"max_retries": 3}}'
    export ANTHROPIC_API_KEY=your_api_key
    ```

    Then, you can use the `ChatAuto` function to create a Chat instance:

    ```python
    from chatlas import ChatAuto

    chat = ChatAuto()
    chat.chat("What is the capital of France?")
    ```

    Note that you can also define your own environment variables and pass them
    to `ChatAuto()` as an alternative way to configure the provider/model:

    ```bash
    export MYAPP_PROVIDER_MODEL="anthropic/claude-sonnet-4-0"
    export MYAPP_ARGS='{"kwargs": {"max_retries": 3}}'
    export ANTHROPIC_API_KEY=your_api_key
    ```

    ```python
    import os
    import json
    from chatlas import ChatAuto

    chat = ChatAuto(
        provider_model=os.environ.get("MYAPP_PROVIDER_MODEL"),
        **(json.loads(os.environ.get("MYAPP_ARGS", "{}"))),
    )
    chat.chat("What is the capital of France?")
    ```

    Parameters
    ----------
    provider_model
        The name of the provider and model to use in the format
        `"{provider}/{model}"`. Providers are strings formatted in kebab-case,
        e.g. to use `ChatBedrockAnthropic` set `provider="bedrock-anthropic"`,
        and models are the provider-specific model names, e.g.
        `"claude-3-7-sonnet-20250219"`. The `/{model}` portion may also be
        omitted, in which case, the default model for that provider will be
        used.

        If no value is provided, the `CHATLAS_CHAT_PROVIDER_MODEL` environment
        variable will be consulted for a fallback value. If this variable is also
        not set, a default value of `"openai"` is used.
    system_prompt
        A system prompt to set the behavior of the assistant.
    provider
        Deprecated; use `provider_model` instead.
    model
        Deprecated; use `provider_model` instead.
    **kwargs
        Additional keyword arguments to pass to the `Chat` constructor. See the
        documentation for each provider for more details on the available
        options.

        These arguments can also be provided via the `CHATLAS_CHAT_ARGS`
        environment variable as a JSON string. When any additional arguments are
        provided to `ChatAuto()`, the env var is ignored.

        Note that `system_prompt` and `turns` can't be set via environment variables.
        They must be provided/set directly to/on `ChatAuto()`.

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
        warn_deprecated_param("provider")

    if model is not DEPRECATED:
        if provider is DEPRECATED:
            raise ValueError(
                "The `model` parameter is deprecated and cannot be used without the `provider` parameter. "
                "Use `provider_model` instead."
            )
        warn_deprecated_param("model")

    if provider_model is None:
        provider_model = os.environ.get("CHATLAS_CHAT_PROVIDER_MODEL")

    # Backwards compatibility: construct from old env vars as a fallback
    if provider_model is None:
        env_provider = get_legacy_env_var("CHATLAS_CHAT_PROVIDER", provider)
        env_model = get_legacy_env_var("CHATLAS_CHAT_MODEL", model)

        if env_provider:
            provider_model = env_provider
            if env_model:
                provider_model += f"/{env_model}"

    # Fall back to OpenAI if nothing is specified
    if provider_model is None:
        provider_model = "openai"

    if "/" in provider_model:
        the_provider, the_model = provider_model.split("/", 1)
    else:
        the_provider, the_model = provider_model, None

    if the_provider not in _provider_chat_model_map:
        raise ValueError(
            f"Provider name '{the_provider}' is not a known chatlas provider: "
            f"{', '.join(_provider_chat_model_map.keys())}"
        )

    # `system_prompt`, `turns` and `model` always come from `ChatAuto()`
    base_args = {
        "system_prompt": system_prompt,
        "turns": None,
        "model": the_model,
    }

    # Environment kwargs, used only if no kwargs provided
    env_kwargs = {}
    if not kwargs:
        env_kwargs = orjson.loads(os.environ.get("CHATLAS_CHAT_ARGS", "{}"))

    final_kwargs = {**env_kwargs, **kwargs, **base_args}
    final_kwargs = {k: v for k, v in final_kwargs.items() if v is not None}

    return _provider_chat_model_map[the_provider](**final_kwargs)


def get_legacy_env_var(
    env_var_name: str,
    default: str | DEPRECATED_TYPE,
) -> str | None:
    env_value = os.environ.get(env_var_name)
    if env_value:
        warnings.warn(
            f"The '{env_var_name}' environment variable is deprecated. "
            "Use 'CHATLAS_CHAT_PROVIDER_MODEL' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return env_value
    elif isinstance(default, DEPRECATED_TYPE):
        return None
    else:
        return default


def warn_deprecated_param(param_name: str, stacklevel: int = 3) -> None:
    warnings.warn(
        f"The '{param_name}' parameter is deprecated. Use 'provider_model' instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
