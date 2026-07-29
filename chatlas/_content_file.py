from __future__ import annotations

from typing import TYPE_CHECKING, Union

import requests

if TYPE_CHECKING:
    from ._content import ContentDocument, ContentPDF

FileContent = Union["ContentPDF", "ContentDocument"]


def download_bytes(url: str) -> bytes:
    """Download `url`'s bytes in full."""
    # Streaming keeps the connection out of the pool until the body is consumed,
    # so the response needs closing on the error paths too -- `raise_for_status()`
    # never reads the body.
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        return b"".join(response.iter_content(chunk_size=8192))


def ensure_bytes(content: FileContent, kind: str) -> bytes:
    """Return `content.data`, downloading and caching it from `content.url` if needed.

    `content_pdf_url()` (and friends) don't eagerly download bytes anymore, since
    some providers (Anthropic, `ChatOpenAI()`'s Responses API) can take the URL
    directly. Providers that can't take a URL call this to fetch the bytes on
    demand, the first time they're actually needed, and the result is cached
    back onto `content` so a file referenced across multiple turns/providers
    is only downloaded once.
    """
    if content.data is not None:
        return content.data

    if content.url is None:
        raise ValueError(f"{kind} content has neither `data` nor `url` set.")

    try:
        data = download_bytes(content.url)
    except Exception as e:
        raise ValueError(f"Failed to download {kind} from {content.url}: {e}") from e

    content.data = data
    return data
