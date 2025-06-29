# ---------------------------------------------------------
# Do not modify this file. It was generated by `scripts/generate_typed_dicts.py`.
# ---------------------------------------------------------


from typing import Mapping, Optional, TypedDict, Union

import httpx
import openai


class ChatClientArgs(TypedDict, total=False):
    api_key: str | None
    organization: str | None
    project: str | None
    webhook_secret: str | None
    base_url: str | httpx.URL | None
    websocket_base_url: str | httpx.URL | None
    timeout: Union[float, openai.Timeout, None, openai.NotGiven]
    max_retries: int
    default_headers: Optional[Mapping[str, str]]
    default_query: Optional[Mapping[str, object]]
    http_client: httpx.AsyncClient
    _strict_response_validation: bool
