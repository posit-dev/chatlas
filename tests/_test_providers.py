"""
Test provider wrappers that fall back to dummy credentials for VCR replay.

These wrappers allow tests to run without real API keys when using VCR cassettes.
The dummy credentials satisfy SDK initialization requirements, and VCR intercepts
the actual HTTP requests before they reach the API.

Usage:
    from tests._test_providers import TestChatOpenAI, TestChatAnthropic

    @pytest.mark.vcr
    def test_something():
        chat = TestChatOpenAI()
        chat.chat("Hello")
"""

import os
from typing import Optional

from chatlas import (
    Chat,
    ChatAnthropic,
    ChatAzureOpenAI,
    ChatBedrockAnthropic,
    ChatCloudflare,
    ChatDatabricks,
    ChatDeepSeek,
    ChatGithub,
    ChatGoogle,
    ChatHuggingFace,
    ChatMistral,
    ChatOpenAI,
    ChatOpenAICompletions,
    ChatOpenRouter,
    ChatPortkey,
)

# ---------------------------------------------------------------------------
# Dummy credentials for VCR replay mode
# ---------------------------------------------------------------------------

DUMMY_OPENAI_KEY = "sk-dummy-openai-key-for-vcr"
DUMMY_ANTHROPIC_KEY = "sk-ant-dummy-key-for-vcr"
DUMMY_GOOGLE_KEY = "dummy-google-key-for-vcr"
DUMMY_AZURE_KEY = "dummy-azure-key-for-vcr"
DUMMY_CLOUDFLARE_KEY = "dummy-cloudflare-key-for-vcr"
# Account ID from VCR cassettes - needed because account is part of the URL path
DUMMY_CLOUDFLARE_ACCOUNT = "38bc1f7f09483f93340d366546d47dcd"
DUMMY_DATABRICKS_HOST = "https://dummy-databricks.cloud.databricks.com"
DUMMY_DATABRICKS_TOKEN = "dummy-databricks-token-for-vcr"
DUMMY_DEEPSEEK_KEY = "dummy-deepseek-key-for-vcr"
DUMMY_GITHUB_TOKEN = "dummy-github-token-for-vcr"
DUMMY_HUGGINGFACE_KEY = "dummy-huggingface-key-for-vcr"
DUMMY_MISTRAL_KEY = "dummy-mistral-key-for-vcr"
DUMMY_OPENROUTER_KEY = "dummy-openrouter-key-for-vcr"


# ---------------------------------------------------------------------------
# Helper to get key with fallback
# ---------------------------------------------------------------------------


def _get_key(env_var: str, dummy: str, provided: Optional[str] = None) -> str:
    """Get API key from provided value, env var, or fall back to dummy."""
    if provided is not None:
        return provided
    return os.environ.get(env_var) or dummy


# ---------------------------------------------------------------------------
# Test provider wrappers
# ---------------------------------------------------------------------------


def TestChatOpenAI(
    *,
    api_key: Optional[str] = None,
    **kwargs,
) -> Chat:
    """ChatOpenAI with fallback to dummy credentials for VCR replay."""
    return ChatOpenAI(
        api_key=_get_key("OPENAI_API_KEY", DUMMY_OPENAI_KEY, api_key),
        **kwargs,
    )


def TestChatOpenAICompletions(
    *,
    api_key: Optional[str] = None,
    **kwargs,
) -> Chat:
    """ChatOpenAICompletions with fallback to dummy credentials for VCR replay."""
    return ChatOpenAICompletions(
        api_key=_get_key("OPENAI_API_KEY", DUMMY_OPENAI_KEY, api_key),
        **kwargs,
    )


def TestChatAnthropic(
    *,
    api_key: Optional[str] = None,
    **kwargs,
) -> Chat:
    """ChatAnthropic with fallback to dummy credentials for VCR replay."""
    # Anthropic SDK allows None, but we use dummy for consistency
    return ChatAnthropic(
        api_key=_get_key("ANTHROPIC_API_KEY", DUMMY_ANTHROPIC_KEY, api_key),
        **kwargs,
    )


def TestChatGoogle(
    *,
    api_key: Optional[str] = None,
    **kwargs,
) -> Chat:
    """ChatGoogle with fallback to dummy credentials for VCR replay."""
    return ChatGoogle(
        api_key=_get_key("GOOGLE_API_KEY", DUMMY_GOOGLE_KEY, api_key),
        **kwargs,
    )


def TestChatAzureOpenAI(
    *,
    api_key: Optional[str] = None,
    endpoint: str = "https://dummy-azure.openai.azure.com",
    deployment_id: str = "gpt-4",
    api_version: str = "2024-02-01",
    **kwargs,
) -> Chat:
    """ChatAzureOpenAI with fallback to dummy credentials for VCR replay."""
    return ChatAzureOpenAI(
        api_key=_get_key("AZURE_OPENAI_API_KEY", DUMMY_AZURE_KEY, api_key),
        endpoint=endpoint,
        deployment_id=deployment_id,
        api_version=api_version,
        **kwargs,
    )


def TestChatCloudflare(
    *,
    api_key: Optional[str] = None,
    account: Optional[str] = None,
    **kwargs,
) -> Chat:
    """ChatCloudflare with fallback to dummy credentials for VCR replay."""
    return ChatCloudflare(
        api_key=_get_key("CLOUDFLARE_API_KEY", DUMMY_CLOUDFLARE_KEY, api_key),
        account=_get_key("CLOUDFLARE_ACCOUNT_ID", DUMMY_CLOUDFLARE_ACCOUNT, account),
        **kwargs,
    )


def TestChatDatabricks(**kwargs) -> Chat:
    """ChatDatabricks with fallback to dummy credentials for VCR replay.

    Note: Databricks uses WorkspaceClient which reads DATABRICKS_HOST and
    DATABRICKS_TOKEN from env vars. We set dummy values if not present.
    """
    # Set dummy env vars if not present (WorkspaceClient reads from env)
    if not os.environ.get("DATABRICKS_HOST"):
        os.environ["DATABRICKS_HOST"] = DUMMY_DATABRICKS_HOST
    if not os.environ.get("DATABRICKS_TOKEN"):
        os.environ["DATABRICKS_TOKEN"] = DUMMY_DATABRICKS_TOKEN

    return ChatDatabricks(**kwargs)


def TestChatDeepSeek(
    *,
    api_key: Optional[str] = None,
    **kwargs,
) -> Chat:
    """ChatDeepSeek with fallback to dummy credentials for VCR replay."""
    return ChatDeepSeek(
        api_key=_get_key("DEEPSEEK_API_KEY", DUMMY_DEEPSEEK_KEY, api_key),
        **kwargs,
    )


def TestChatGithub(
    *,
    api_key: Optional[str] = None,
    **kwargs,
) -> Chat:
    """ChatGithub with fallback to dummy credentials for VCR replay."""
    # GitHub uses GITHUB_TOKEN env var
    return ChatGithub(
        api_key=_get_key("GITHUB_TOKEN", DUMMY_GITHUB_TOKEN, api_key),
        **kwargs,
    )


def TestChatHuggingFace(
    *,
    api_key: Optional[str] = None,
    **kwargs,
) -> Chat:
    """ChatHuggingFace with fallback to dummy credentials for VCR replay."""
    return ChatHuggingFace(
        api_key=_get_key("HUGGINGFACE_API_KEY", DUMMY_HUGGINGFACE_KEY, api_key),
        **kwargs,
    )


def TestChatMistral(
    *,
    api_key: Optional[str] = None,
    **kwargs,
) -> Chat:
    """ChatMistral with fallback to dummy credentials for VCR replay."""
    return ChatMistral(
        api_key=_get_key("MISTRAL_API_KEY", DUMMY_MISTRAL_KEY, api_key),
        **kwargs,
    )


def TestChatOpenRouter(
    *,
    api_key: Optional[str] = None,
    **kwargs,
) -> Chat:
    """ChatOpenRouter with fallback to dummy credentials for VCR replay."""
    return ChatOpenRouter(
        api_key=_get_key("OPENROUTER_API_KEY", DUMMY_OPENROUTER_KEY, api_key),
        **kwargs,
    )


DUMMY_PORTKEY_KEY = "dummy-portkey-key-for-vcr"


def TestChatBedrockAnthropic(**kwargs) -> Chat:
    """ChatBedrockAnthropic for VCR replay.

    Note: Bedrock uses boto3 which reads AWS credentials from env vars or
    ~/.aws/credentials. For VCR replay, the actual credentials don't matter
    since requests are intercepted, but boto3 still requires valid-looking
    credentials to be configured.
    """
    return ChatBedrockAnthropic(**kwargs)


def TestChatPortkey(
    *,
    api_key: Optional[str] = None,
    **kwargs,
) -> Chat:
    """ChatPortkey with fallback to dummy credentials for VCR replay."""
    return ChatPortkey(
        api_key=_get_key("PORTKEY_API_KEY", DUMMY_PORTKEY_KEY, api_key),
        **kwargs,
    )
