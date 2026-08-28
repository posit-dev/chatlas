"""Forward the Shiny viewer's session token to Posit Connect's LLM gateway.

When chat traffic on Posit Connect goes through Connect's LLM gateway, the
current viewer's session token is forwarded as a
``Posit-Connect-User-Session-Token`` header so Connect can attribute the call
to the viewer. The header is only added when running on Connect, and only sent
to Connect's own gateway URL.
"""

from __future__ import annotations

import os
from typing import Optional
from urllib.parse import urlparse

CONNECT_VIEWER_TOKEN_HEADER = "Posit-Connect-User-Session-Token"


def connect_viewer_headers(url: str) -> dict[str, str]:
    """
    Headers attributing a Connect gateway request to the current Shiny viewer.

    Parameters
    ----------
    url
        The URL the request will be sent to (e.g. the provider's base URL).

    Returns
    -------
    :
        A dict with the ``Posit-Connect-User-Session-Token`` header, or an
        empty dict when not running on Connect, when there's no active Shiny
        session, or when `url` isn't Connect's LLM gateway.
    """
    if not _is_connect_gateway_url(url):
        return {}
    token = _connect_viewer_token()
    if token is None:
        return {}
    return {CONNECT_VIEWER_TOKEN_HEADER: token}


def _is_connect_gateway_url(url: str) -> bool:
    # The token is only ever sent back to the Connect server that injected it.
    server = os.environ.get("CONNECT_SERVER", "")
    if not server:
        return False
    server_parsed = urlparse(server)
    url_parsed = urlparse(url)
    return (
        url_parsed.scheme.lower() == server_parsed.scheme.lower()
        and (url_parsed.hostname or "").lower()
        == (server_parsed.hostname or "").lower()
        and url_parsed.port == server_parsed.port
        and url_parsed.path.startswith("/__gateway__/")
    )


def _connect_viewer_token() -> Optional[str]:
    # Only read the session token when actually running on Connect
    if os.environ.get("RSTUDIO_PRODUCT") != "CONNECT":
        return None
    try:
        from shiny.session import get_current_session
    except ImportError:
        return None
    session = get_current_session()
    if session is None:
        return None
    http_conn = getattr(session, "http_conn", None)
    if http_conn is None:
        return None
    return http_conn.headers.get(CONNECT_VIEWER_TOKEN_HEADER.lower())
