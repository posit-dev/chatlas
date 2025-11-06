from . import types
from ._auto import ChatAuto
from ._batch_chat import (
    batch_chat,
    batch_chat_completed,
    batch_chat_structured,
    batch_chat_text,
)
from ._chat import Chat
from ._content import (
    ContentToolRequest,
    ContentToolResult,
    ContentToolResultImage,
    ContentToolResultResource,
)
from ._content_image import content_image_file, content_image_plot, content_image_url
from ._content_pdf import content_pdf_file, content_pdf_url
from ._interpolate import interpolate, interpolate_file
from ._parallel import parallel_chat, parallel_chat_structured, parallel_chat_text
from ._provider import Provider
from ._provider_anthropic import ChatAnthropic, ChatBedrockAnthropic
from ._provider_cloudflare import ChatCloudflare
from ._provider_databricks import ChatDatabricks
from ._provider_deepseek import ChatDeepSeek
from ._provider_github import ChatGithub
from ._provider_google import ChatGoogle, ChatVertex
from ._provider_groq import ChatGroq
from ._provider_huggingface import ChatHuggingFace
from ._provider_mistral import ChatMistral
from ._provider_ollama import ChatOllama
from ._provider_openai import ChatOpenAI
from ._provider_openai_azure import ChatAzureOpenAI, ChatAzureOpenAICompletions
from ._provider_openai_completions import ChatOpenAICompletions
from ._provider_openrouter import ChatOpenRouter
from ._provider_perplexity import ChatPerplexity
from ._provider_portkey import ChatPortkey
from ._provider_snowflake import ChatSnowflake
from ._tokens import token_usage
from ._tools import Tool, ToolRejectError
from ._turn import Turn

try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover
    __version__ = "0.0.0"  # stub value for docs

__all__ = (
    "batch_chat",
    "batch_chat_completed",
    "batch_chat_structured",
    "batch_chat_text",
    "parallel_chat",
    "parallel_chat_structured",
    "parallel_chat_text",
    "ChatAnthropic",
    "ChatAuto",
    "ChatBedrockAnthropic",
    "ChatCloudflare",
    "ChatDatabricks",
    "ChatDeepSeek",
    "ChatGithub",
    "ChatGoogle",
    "ChatGroq",
    "ChatHuggingFace",
    "ChatMistral",
    "ChatOllama",
    "ChatOpenAICompletions",
    "ChatOpenAI",
    "ChatOpenRouter",
    "ChatAzureOpenAI",
    "ChatAzureOpenAICompletions",
    "ChatPerplexity",
    "ChatPortkey",
    "ChatSnowflake",
    "ChatVertex",
    "Chat",
    "content_image_file",
    "content_image_plot",
    "content_image_url",
    "content_pdf_file",
    "content_pdf_url",
    "ContentToolRequest",
    "ContentToolResult",
    "ContentToolResultImage",
    "ContentToolResultResource",
    "interpolate",
    "interpolate_file",
    "Provider",
    "token_usage",
    "Tool",
    "ToolRejectError",
    "Turn",
    "types",
)

# Rebuild content models to resolve forward references to ToolAnnotation
ContentToolRequest.model_rebuild()
ContentToolResult.model_rebuild()
ContentToolResultImage.model_rebuild()
ContentToolResultResource.model_rebuild()
