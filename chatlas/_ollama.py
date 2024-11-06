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
