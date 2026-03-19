from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from ._chat import Chat
from ._logging import log_model_default
from ._provider_openai_completions import OpenAICompletionsProvider

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

    from .types.openai import ChatClientArgs, SubmitInputArgs


def ChatHuggingFace(
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat["SubmitInputArgs", ChatCompletion]:
    """
    Chat with a model hosted on Hugging Face Inference API.

    [Hugging Face](https://huggingface.co/) hosts a variety of open-source
    and proprietary AI models available via their Inference API.
    To use the Hugging Face API, you must have an Access Token, which you can obtain
    from your [Hugging Face account](https://huggingface.co/settings/tokens).
    Ensure that at least "Make calls to Inference Providers" and
    "Make calls to your Inference Endpoints" is checked.

    Prerequisites
    --------------

    ::: {.callout-note}
    ## API key

    You will need to create a Hugging Face account and generate an API token
    from your [account settings](https://huggingface.co/settings/tokens).
    Make sure to enable "Make calls to Inference Providers" permission.
    :::

    Examples
    --------
    ```python
    import os
    from chatlas import ChatHuggingFace

    chat = ChatHuggingFace(api_key=os.getenv("HUGGINGFACE_API_KEY"))
    chat.chat("What is the capital of France?")
    ```

    Parameters
    ----------
    system_prompt
        A system prompt to set the behavior of the assistant.
    model
        The model to use for the chat. The default, None, will pick a reasonable
        default, and warn you about it. We strongly recommend explicitly
        choosing a model for all but the most casual use.
    api_key
        The API key to use for authentication. You generally should not supply
        this directly, but instead set the `HUGGINGFACE_API_KEY` environment
        variable.
    kwargs
        Additional arguments to pass to the underlying OpenAI client
        constructor.

    Returns
    -------
    Chat
        A chat object that retains the state of the conversation.

    Known limitations
    -----------------

    * Some models do not support the chat interface or parts of it, for example
      `google/gemma-2-2b-it` does not support a system prompt. You will need to
      carefully choose the model.
    * Tool calling support varies by model - many models do not support it.

    Note
    ----
    This function is a lightweight wrapper around [](`~chatlas.ChatOpenAI`), with
    the defaults tweaked for Hugging Face.

    Note
    ----
    Pasting an API key into a chat constructor (e.g., `ChatHuggingFace(api_key="...")`)
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
    HUGGINGFACE_API_KEY=...
    ```

    ```python
    from chatlas import ChatHuggingFace
    from dotenv import load_dotenv

    load_dotenv()
    chat = ChatHuggingFace()
    chat.console()
    ```

    Another, more general, solution is to load your environment variables into the shell
    before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

    ```shell
    export HUGGINGFACE_API_KEY=...
    ```
    """
    if api_key is None:
        api_key = os.getenv("HUGGINGFACE_API_KEY")

    if model is None:
        model = log_model_default("meta-llama/Llama-3.1-8B-Instruct")

    return Chat(
        provider=HuggingFaceProvider(
            api_key=api_key,
            model=model,
            kwargs=kwargs,
        ),
        system_prompt=system_prompt,
    )


class HuggingFaceProvider(OpenAICompletionsProvider):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str,
        kwargs: Optional["ChatClientArgs"] = None,
    ):
        # https://huggingface.co/docs/inference-providers/en/index?python-clients=requests#http--curl
        super().__init__(
            name="HuggingFace",
            model=model,
            api_key=api_key,
            base_url="https://router.huggingface.co/v1",
            kwargs=kwargs,
        )
