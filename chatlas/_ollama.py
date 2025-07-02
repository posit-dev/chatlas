from __future__ import annotations

import re
import urllib.request
from typing import TYPE_CHECKING, Optional

import orjson

from ._chat import Chat
from ._openai import OpenAIProvider
from ._utils import MISSING_TYPE, is_testing

if TYPE_CHECKING:
    from ._openai import ChatCompletion
    from .types.openai import ChatClientArgs, SubmitInputArgs


def ChatOllama(
    model: Optional[str] = None,
    *,
    system_prompt: Optional[str] = None,
    base_url: str = "http://localhost:11434",
    seed: Optional[int] = None,
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat["SubmitInputArgs", ChatCompletion]:
    """
    Chat with a local Ollama model.

    [Ollama](https://ollama.com) makes it easy to run a wide-variety of
    open-source models locally, making it a great choice for privacy
    and security.


    Prerequisites
    -------------

    ::: {.callout-note}
    ## Ollama runtime

    `ChatOllama` requires the [ollama](https://ollama.com/download) executable
    to be installed and running on your machine.
    :::

    ::: {.callout-note}
    ## Pull model(s)

    Once ollama is running locally, download a model from the command line
    (e.g. `ollama pull llama3.2`).
    :::


    Examples
    --------

    ```python
    from chatlas import ChatOllama

    chat = ChatOllama(model="llama3.2")
    chat.chat("What is the capital of France?")
    ```

    Parameters
    ----------
    model
        The model to use for the chat. If `None`, a list of locally installed
        models will be printed.
    system_prompt
        A system prompt to set the behavior of the assistant.
    base_url
        The base URL to the endpoint; the default uses ollama's API.
    seed
        Optional integer seed that helps to make output more reproducible.
    kwargs
        Additional arguments to pass to the `openai.OpenAI()` client constructor.

    Note
    ----
    This function is a lightweight wrapper around [](`~chatlas.ChatOpenAI`) with
    the defaults tweaked for ollama.

    Limitations
    -----------
    `ChatOllama` currently doesn't work with streaming tools, and tool calling more
    generally doesn't seem to work very well with currently available models.
    """

    base_url = re.sub("/+$", "", base_url)

    if not has_ollama(base_url):
        raise RuntimeError("Can't find locally running ollama.")

    if model is None:
        models = ollama_models(base_url)
        raise ValueError(
            f"Must specify model. Locally installed models: {', '.join(models)}"
        )
    if isinstance(seed, MISSING_TYPE):
        seed = 1014 if is_testing() else None

    return Chat(
        provider=OpenAIProvider(
            api_key="ollama",  # ignored
            model=model,
            base_url=f"{base_url}/v1",
            seed=seed,
            name="Ollama",
            kwargs=kwargs,
        ),
        system_prompt=system_prompt,
    )


def ollama_models(base_url: str) -> list[str]:
    res = urllib.request.urlopen(url=f"{base_url}/api/tags")
    data = orjson.loads(res.read())
    return [re.sub(":latest$", "", x["name"]) for x in data["models"]]


def has_ollama(base_url):
    try:
        urllib.request.urlopen(url=f"{base_url}/api/tags")
        return True
    except Exception:
        return False
