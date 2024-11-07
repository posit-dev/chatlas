from __future__ import annotations

import json
import re
import urllib.request
from typing import TYPE_CHECKING, Optional

from ._chat import Chat
from ._openai import ChatOpenAI
from ._turn import Turn

if TYPE_CHECKING:
    from .provider_types._openai_client import ProviderClientArgs


def ChatOllama(
    model: Optional[str] = None,
    *,
    system_prompt: Optional[str] = None,
    turns: Optional[list[Turn]] = None,
    base_url: str = "http://localhost:11434/v1",
    seed: Optional[int] = None,
    kwargs: Optional["ProviderClientArgs"] = None,
) -> "Chat":
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
    turns
        A list of turns to start the chat with (i.e., continuing a previous
        conversation). If not provided, the conversation begins from scratch. Do
        not provide non-`None` values for both `turns` and `system_prompt`. Each
        message in the list should be a dictionary with at least `role` (usually
        `system`, `user`, or `assistant`, but `tool` is also possible). Normally
        there is also a `content` field, which is a string.
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
    """
    if not has_ollama(base_url):
        raise RuntimeError("Can't find locally running ollama.")

    if model is None:
        models = ollama_models(base_url)
        raise ValueError(
            f"Must specify model. Locally installed models: {', '.join(models)}"
        )

    return ChatOpenAI(
        system_prompt=system_prompt,
        turns=turns,
        base_url=base_url,
        model=model,
        seed=seed,
        kwargs=kwargs,
    )


def ollama_models(base_url: str) -> list[str]:
    base_url = re.sub("/v[0-9]+$", "", base_url)
    res = urllib.request.urlopen(url=f"{base_url}/api/tags")
    data = json.loads(res.read())
    return [re.sub(":latest$", "", x["name"]) for x in data["models"]]


def has_ollama(base_url):
    base_url = re.sub("/v[0-9]+$", "", base_url)
    try:
        urllib.request.urlopen(url=f"{base_url}/api/tags")
        return True
    except Exception:
        return False
