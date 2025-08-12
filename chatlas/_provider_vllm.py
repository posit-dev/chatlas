from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from ._chat import Chat
from ._provider_openai import OpenAIProvider
from ._utils import MISSING, MISSING_TYPE, is_testing

if TYPE_CHECKING:
    from ._provider_openai import ChatCompletion
    from .types.openai import ChatClientArgs, SubmitInputArgs


def ChatVllm(
    *,
    base_url: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    seed: Optional[int] | MISSING_TYPE = MISSING,
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat["SubmitInputArgs", ChatCompletion]:
    """
    Chat with a model hosted by vLLM.

    [vLLM](https://docs.vllm.ai/en/latest/) is an open source library that
    provides an efficient and convenient LLMs model server. You can use
    `ChatVllm()` to connect to endpoints powered by vLLM.

    Prerequisites
    -------------

    ::: {.callout-note}
    ## vLLM Server

    You need access to a running vLLM server instance. vLLM provides
    OpenAI-compatible API endpoints, so this function works with any
    vLLM deployment that exposes the `/v1/chat/completions` endpoint.
    :::

    Examples
    --------

    ```python
    import os
    from chatlas import ChatVllm

    # Connect to a vLLM server
    chat = ChatVllm(
        base_url="http://localhost:8000/v1",
        model="meta-llama/Llama-2-7b-chat-hf",
        api_key=os.getenv("VLLM_API_KEY"),  # Optional, depends on server config
    )
    chat.chat("What is the capital of France?")
    ```

    Parameters
    ----------
    base_url
        The base URL of the vLLM server endpoint. This should include the
        `/v1` path if the server follows OpenAI API conventions.
    system_prompt
        A system prompt to set the behavior of the assistant.
    model
        The model to use for the chat. If None, you may need to specify
        the model name that's loaded on your vLLM server.
    api_key
        The API key to use for authentication. Some vLLM deployments may
        not require authentication. You can set the `VLLM_API_KEY`
        environment variable instead of passing it directly.
    seed
        Optional integer seed that vLLM uses to try and make output more
        reproducible.
    kwargs
        Additional arguments to pass to the `openai.OpenAI()` client constructor.

    Returns
    -------
    Chat
        A chat object that retains the state of the conversation.

    Note
    ----
    This function is a lightweight wrapper around [](`~chatlas.ChatOpenAI`) with
    the defaults tweaked for vLLM endpoints.

    Note
    ----
    vLLM servers are OpenAI-compatible, so this provider uses the same underlying
    client as OpenAI but configured for your vLLM endpoint. Some advanced OpenAI
    features may not be available depending on your vLLM server configuration.

    Note
    ----
    Pasting an API key into a chat constructor (e.g., `ChatVllm(api_key="...")`)
    is the simplest way to get started, and is fine for interactive use, but is
    problematic for code that may be shared with others.

    Instead, consider using environment variables or a configuration file to manage
    your credentials. One popular way to manage credentials is to use a `.env` file
    to store your credentials, and then use the `python-dotenv` package to load them
    into your environment.

    ```shell
    pip install python-dotenv
    ```

    ```shell
    # .env
    VLLM_API_KEY=...
    ```

    ```python
    from chatlas import ChatVllm
    from dotenv import load_dotenv

    load_dotenv()
    chat = ChatVllm(base_url="http://localhost:8000/v1")
    chat.console()
    ```

    Another, more general, solution is to load your environment variables into the shell
    before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

    ```shell
    export VLLM_API_KEY=...
    ```
    """
    if api_key is None:
        api_key = os.getenv("VLLM_API_KEY")

    if isinstance(seed, MISSING_TYPE):
        seed = 1014 if is_testing() else None

    if model is None:
        raise ValueError(
            "Must specify model. vLLM servers can host different models, so you need to "
            "specify which one to use. Check your vLLM server's /v1/models endpoint "
            "to see available models."
        )

    return Chat(
        provider=VllmProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            seed=seed,
            name="vLLM",
            kwargs=kwargs,
        ),
        system_prompt=system_prompt,
    )


class VllmProvider(OpenAIProvider):
    """
    Provider for vLLM endpoints.

    vLLM is OpenAI-compatible but may have some differences in tool handling
    and other advanced features.
    """

    def _chat_perform_args(self, *args, **kwargs):
        """
        Customize request arguments for vLLM compatibility.

        vLLM may not support all OpenAI features like stream_options,
        so we remove potentially unsupported parameters.
        """
        # Get the base arguments from OpenAI provider
        result = super()._chat_perform_args(*args, **kwargs)

        # Remove stream_options if present (some vLLM versions don't support it)
        if "stream_options" in result:
            del result["stream_options"]

        return result
