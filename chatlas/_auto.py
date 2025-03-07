from __future__ import annotations

import json
import os
from typing import Optional

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

_provider_chat_model_map = {
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
    provider: Optional[str] = None,
    system_prompt: Optional[str] = None,
    turns: Optional[list[Turn]] = None,
    **kwargs,
) -> Chat:
    """
    Factory function to create a Chat instance based on a provider specified in code or in an environment variable.

    This function creates a Chat instance based on the specified provider, with optional system prompt and conversation turns.
    The provider can be specified either through the function parameter or via the CHATLAS_CHAT_PROVIDER environment variable.
    Additional configuration can be provided through kwargs or the CHATLAS_CHAT_ARGS environment variable (as JSON). This allows
    you to easily switch between different chat providers by changing the environment variable without modifying your code.

    Prerequisites
    -------------

    ::: {.callout-note}
    ## API key

    Follow the instructions for the specific provider to obtain an API key. In order to use the specified provider, ensure
    that an API key is set in the environment variable `CHATLAS_CHAT_API_KEY` or passed as a parameter to the function.
    :::

    ::: {.callout-note}
    ## Python requirements

    Follow the instructions for the specific provider to install the required Python packages.
    :::


    Examples
    --------
    First, set the environment variables for the provider, arguments, and API key:

    ```bash
    export CHATLAS_CHAT_PROVIDER=anthropic
    export CHATLAS_CHAT_ARGS='{"model": "claude-3-haiku-20240229", "kwargs": {"max_retries": 3}}'
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
        The name of the chat provider to use. Must be one of the supported providers:
        - `anthropic`
        - `bedrock:anthropic`
        - `github`
        - `google`
        - `groq`
        - `ollama`
        - `azure:openai`
        - `openai`
        - `perplexity`
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
        Additional keyword arguments to pass to the Chat constructor. These can also
        be provided via the CHATLAS_CHAT_ARGS environment variable as a JSON string.
        The values will be injected into the Chat constructor of the specified provider.
        See the documentation for each provider for more details on the available options.

    Returns
    -------
        Chat
            A configured Chat instance for the specified provider.

    Raises
    ------
        ValueError
            If no valid provider is specified either through parameters or environment variables.
    """
    provider = os.environ.get("CHATLAS_CHAT_PROVIDER", provider)

    if provider not in _provider_chat_model_map:
        raise ValueError(
            "Provider name is required as parameter or `CHATLAS_CHAT_PROVIDER` must be set."
        )

    kwargs |= dict(
        system_prompt=system_prompt,
        turns=turns,
    )

    if env_kwargs := os.environ.get("CHATLAS_CHAT_ARGS"):
        kwargs |= json.loads(env_kwargs)

    return _provider_chat_model_map[provider](**kwargs)
