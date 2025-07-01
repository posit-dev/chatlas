from __future__ import annotations

import os
from typing import Callable, Literal, Optional

import orjson

from ._anthropic import ChatAnthropic, ChatBedrockAnthropic
from ._chat import Chat
from ._databricks import ChatDatabricks
from ._github import ChatGithub
from ._google import ChatGoogle, ChatVertex
from ._groq import ChatGroq
from ._ollama import ChatOllama
from ._openai import ChatAzureOpenAI, ChatOpenAI
from ._perplexity import ChatPerplexity
from ._snowflake import ChatSnowflake

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


def ChatAuto(
    system_prompt: Optional[str] = None,
    *,
    provider: Optional[AutoProviders] = None,
    model: Optional[str] = None,
    **kwargs,
) -> Chat:
    """
    Use environment variables (env vars) to configure the Chat provider and model.

    Creates a :class:`~chatlas.Chat` instance based on the specified provider.
    The provider may be specified through the `provider` parameter and/or the
    `CHATLAS_CHAT_PROVIDER` env var. If both are set, the env var takes
    precedence. Similarly, the provider's model may be specified through the
    `model` parameter and/or the `CHATLAS_CHAT_MODEL` env var. Also, additional
    configuration may be provided through the `kwargs` parameter and/or the
    `CHATLAS_CHAT_ARGS` env var (as a JSON string). In this case, when both are
    set, they are merged, with the env var arguments taking precedence.

    As a result, `ChatAuto()` provides a convenient way to set a default
    provider and model in your Python code, while allowing you to override
    these settings through env vars (i.e., without modifying your code).

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
    export CHATLAS_CHAT_PROVIDER=anthropic
    export CHATLAS_CHAT_MODEL=claude-3-haiku-20240229
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
    provider
        The name of the default chat provider to use. Providers are strings
        formatted in kebab-case, e.g. to use `ChatBedrockAnthropic` set
        `provider="bedrock-anthropic"`.

        This value can also be provided via the `CHATLAS_CHAT_PROVIDER`
        environment variable, which takes precedence over `provider`
        when set.
    model
        The name of the default model to use. This value can also be provided
        via the `CHATLAS_CHAT_MODEL` environment variable, which takes
        precedence over `model` when set.
    **kwargs
        Additional keyword arguments to pass to the Chat constructor. See the
        documentation for each provider for more details on the available
        options.

        These arguments can also be provided via the `CHATLAS_CHAT_ARGS`
        environment variable as a JSON string. When provided, the options
        in the `CHATLAS_CHAT_ARGS` envvar take precedence over the options
        passed to `kwargs`.

        Note that `system_prompt` and `turns` in `kwargs` or in
        `CHATLAS_CHAT_ARGS` are ignored.

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
    the_provider = os.environ.get("CHATLAS_CHAT_PROVIDER", provider)

    if the_provider is None:
        raise ValueError(
            "Provider name is required as parameter or `CHATLAS_CHAT_PROVIDER` must be set."
        )
    if the_provider not in _provider_chat_model_map:
        raise ValueError(
            f"Provider name '{the_provider}' is not a known chatlas provider: "
            f"{', '.join(_provider_chat_model_map.keys())}"
        )

    # `system_prompt` and `turns` always come from `ChatAuto()`
    base_args = {"system_prompt": system_prompt}

    if env_model := os.environ.get("CHATLAS_CHAT_MODEL"):
        model = env_model

    if model:
        base_args["model"] = model

    env_kwargs = {}
    if env_kwargs_str := os.environ.get("CHATLAS_CHAT_ARGS"):
        env_kwargs = orjson.loads(env_kwargs_str)

    kwargs = {**kwargs, **env_kwargs, **base_args}
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    return _provider_chat_model_map[the_provider](**kwargs)
