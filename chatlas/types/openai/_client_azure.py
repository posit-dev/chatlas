# ---------------------------------------------------------
# Do not modify this file. It was generated by `scripts/generate_typed_dicts.py`.
# ---------------------------------------------------------

from typing import Mapping, Optional, TypedDict

import httpx
import openai


class ChatAzureClientArgs(TypedDict, total=False):
    azure_endpoint: str | None
    azure_deployment: str | None
    api_version: str | None
    api_key: str | None
    azure_ad_token: str | None
    organization: str | None
    project: str | None
    webhook_secret: str | None
    base_url: str | None
    websocket_base_url: str | httpx.URL | None
    timeout: float | openai.Timeout | None | openai.NotGiven
    max_retries: int
    default_headers: Optional[Mapping[str, str]]
    default_query: Optional[Mapping[str, object]]
    http_client: httpx.AsyncClient
    _strict_response_validation: bool
