import sys

from . import types
from ._anthropic import ChatAnthropic, ChatBedrockAnthropic
from ._chat import Chat, ChatResponse
from ._content_image import content_image_file, content_image_plot, content_image_url
from ._github import ChatGithub
from ._google import ChatGoogle
from ._groq import ChatGroq
from ._ollama import ChatOllama
from ._openai import ChatAzureOpenAI, ChatOpenAI
from ._perplexity import ChatPerplexity
from ._provider import Provider
from ._tokens import token_usage
from ._tools import Tool
from ._turn import Turn

__all__ = (
    "ChatAnthropic",
    "ChatBedrockAnthropic",
    "ChatGithub",
    "ChatGoogle",
    "ChatGroq",
    "ChatOllama",
    "ChatOpenAI",
    "ChatAzureOpenAI",
    "ChatPerplexity",
    "Chat",
    "content_image_file",
    "content_image_plot",
    "content_image_url",
    "Turn",
    "token_usage",
    "types",
    "Tool",
    "Provider",
)

# ChatResponse objects are displayed in the REPL using rich
original_displayhook = sys.displayhook


def custom_displayhook(value):
    if isinstance(value, ChatResponse):
        value.display()
    else:
        original_displayhook(value)


sys.displayhook = custom_displayhook
