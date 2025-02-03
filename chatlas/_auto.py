from __future__ import annotations

import json
import os
from typing import Optional

from ._anthropic import ChatAnthropic, ChatBedrockAnthropic
from ._chat import Chat
from ._github import ChatGithub
from ._google import ChatGoogle
from ._groq import ChatGroq
from ._ollama import ChatOllama
from ._openai import ChatAzureOpenAI, ChatOpenAI
from ._perplexity import ChatPerplexity

_provider_chat_model_map = {
    "anthropic": ChatAnthropic,
    "bedrock:anthropic": ChatBedrockAnthropic,
    "github": ChatGithub,
    "google": ChatGoogle,
    "groq": ChatGroq,
    "ollama": ChatOllama,
    "azure:openai": ChatAzureOpenAI,
    "openai": ChatOpenAI,
    "perplexity": ChatPerplexity,
}


def ChatAuto(
    provider: Optional[str] = None,
    **kwargs,
) -> Chat:
    provider = os.environ.get("CHATLAS_CHAT_PROVIDER", provider)

    if provider not in _provider_chat_model_map:
        raise ValueError("Provider name is required as parameter or `CHATLAS_CHAT_PROVIDER` must be set.")

    if env_kwargs := os.environ.get("CHATLAS_CHAT_ARGS"):
        kwargs |= json.loads(env_kwargs)
        
    return _provider_chat_model_map[provider](**kwargs)