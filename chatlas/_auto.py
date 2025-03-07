from __future__ import annotations

import json
import os
from typing import Callable, Literal, Optional

from ._anthropic import ChatAnthropic, ChatBedrockAnthropic
from ._chat import Chat
from ._github import ChatGithub
from ._google import ChatGoogle, ChatVertex
from ._groq import ChatGroq
from ._ollama import ChatOllama
from ._openai import ChatAzureOpenAI, ChatOpenAI
from ._perplexity import ChatPerplexity
from ._snowflake import ChatSnowflake
from ._turn import Turn

AutoProviders = Literal[
    "anthropic",
    "bedrock-anthropic",
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
    turns: Optional[list[Turn]] = None,
    *,
    provider: Optional[AutoProviders] = None,
    model: Optional[str] = None,
    **kwargs,
) -> Chat:
    """
    Create a Chat instance using a provider determined by environment variables.

    Create a Chat instance based on the specified provider, with optional system
    prompt and conversation turns. The provider and model be specified either
    through the function parameters or via the `CHATLAS_CHAT_PROVIDER` or
    `CHATLAS_CHAT_MODEL` environment variables. Additional configuration can be
    provided through kwargs or the `CHATLAS_CHAT_ARGS` environment variable (as
    JSON).

    The design of `ChatAuto()` allows you to easily switch between different
    chat providers by changing the environment variable without modifying your
    code:

    * `system_prompt` and `turns` are always used, regardless of how
      `default_provider` or the additional options are set. These values define
      key behavior of your chat.

    * When provided to `ChatAuto()`, `default_provider` and `kwargs` serve as
      default parameters that are used when the associated `CHATLAS_CHAT_`
      environment variables are not set.

    * When `CHATLAS_CHAT_PROVIDER` or `CHATLAS_CHAT_MODEL` are set, they are
      used in place of `provider` and `model`.

    * When `CHATLAS_CHAT_ARGS` is set, it is merged with `kwargs`, where the
      values in the environment variable take precedence.

    Prerequisites
    -------------

    ::: {.callout-note}
    ## API key

    Follow the instructions for the specific provider to obtain an API key. In
    order to use the specified provider, ensure that an API key is set in the
    environment variable `CHATLAS_CHAT_API_KEY` or passed as a parameter to the
    function.
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
    import os
    from chatlas import ChatAuto

    chat = ChatAuto()
    chat.chat("What is the capital of France?")
    ```

    Parameters
    ----------
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
    system_prompt
        A system prompt to set the behavior of the assistant.
    turns
        A list of turns to start the chat with (i.e., continuing a previous
        conversation). If not provided, the conversation begins from scratch. Do
        not provide non-`None` values for both `turns` and `system_prompt`. Each
        message in the list should be a dictionary with at least `role` (usually
        `system`, `user`, or `assistant`, but `tool` is also possible). Normally
        there is also a `content` field, which is a string.
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
        A configured Chat instance for the specified provider.

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
    elif the_provider not in _provider_chat_model_map:
        raise ValueError(
            f"Provider name '{the_provider}' is not a known chatlas provider: "
            f"{', '.join(_provider_chat_model_map.keys())}"
        )

    if env_model := os.environ.get("CHATLAS_CHAT_MODEL"):
        model = env_model

    # `system_prompt` and `turns` always come from `ChatAuto()`
    base_args = {"system_prompt": system_prompt, "turns": turns, "model": model}

    env_kwargs = {}
    if env_kwargs_str := os.environ.get("CHATLAS_CHAT_ARGS"):
        env_kwargs = json.loads(env_kwargs_str)

    kwargs = {**kwargs, **env_kwargs, **base_args}
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    return _provider_chat_model_map[the_provider](**kwargs)
