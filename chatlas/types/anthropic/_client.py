# ---------------------------------------------------------
# Do not modify this file. It was generated by `scripts/generate_typed_dicts.py`.
# ---------------------------------------------------------


from typing import Mapping, Optional, TypedDict, Union

import anthropic
import httpx


class ChatClientArgs(TypedDict, total=False):
    api_key: str | None
    auth_token: str | None
    base_url: str | httpx.URL | None
    timeout: Union[float, anthropic.Timeout, None, anthropic.NotGiven]
    max_retries: int
    default_headers: Optional[Mapping[str, str]]
    default_query: Optional[Mapping[str, object]]
    http_client: httpx.AsyncClient
    _strict_response_validation: bool
