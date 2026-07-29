from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union
from urllib.parse import urlparse

import requests

if TYPE_CHECKING:
    from ._content import ContentDocument, ContentPDF

FileContent = Union["ContentPDF", "ContentDocument"]

# `requests` applies this per socket read rather than to the transfer as a whole,
# so a large file still downloads however long it takes, as long as bytes keep
# arriving. It only bounds a connection that has gone silent -- which otherwise
# hangs the chat indefinitely, since this runs inline while building a request.
DOWNLOAD_TIMEOUT_SECONDS = 30


def download_bytes(url: str) -> bytes:
    """Download `url`'s bytes in full."""
    # Streaming keeps the connection out of the pool until the body is consumed,
    # so the response needs closing on the error paths too -- `raise_for_status()`
    # never reads the body.
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
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


def parse_data_url(url: str) -> tuple[str, str]:
    parts = url[5:].split(";", 1)
    if len(parts) != 2 or not parts[1].startswith("base64,"):
        raise ValueError("url is not a valid data URL.")
    return (parts[0], parts[1][7:])


def filename_from_url(url: str) -> str:
    """The last path segment of `url`, which may legitimately be empty."""
    return Path(urlparse(url).path).name
