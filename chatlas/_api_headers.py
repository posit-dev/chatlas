from __future__ import annotations

from typing import Callable, Union

ApiHeadersValue = dict[str, str]
"""A dict of HTTP header name-value pairs."""

ApiHeaders = Union[ApiHeadersValue, Callable[[], ApiHeadersValue]]
"""
Extra HTTP headers to include with every API request.

Can be:

* A **dict** of ``{header_name: header_value}`` — sent as-is on every request.
* A **zero-argument callable** returning such a dict — called on every
  request, enabling token refresh and other dynamic auth patterns.
"""


def resolve_api_headers(api_headers: ApiHeaders | None) -> dict[str, str] | None:
    """
    Resolve api_headers into a dict of HTTP headers (or None).

    Called at request time so that callables can return fresh values.
    """
    if api_headers is None:
        return None

    value = api_headers() if callable(api_headers) else api_headers

    if isinstance(value, dict):
        return value

    raise TypeError(
        f"api_headers must be (or return) a dict, got {type(value).__name__}"
    )
