from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from ._content import ContentVideoInline, ContentVideoUrl, VideoContentTypes

__all__ = (
    "content_video_file",
    "content_video_youtube",
)


def content_video_file(
    path: str | Path,
    mime_type: Optional[VideoContentTypes] = None,
) -> ContentVideoInline:
    """
    Prepare a local video clip for input to a chat.

    Only Gemini accepts video input, and only inline data that keeps the
    total request under roughly 100 MB. For larger files, upload the video
    once with `chat.files.upload()` and reuse the returned reference across
    turns instead of re-sending its bytes.

    Parameters
    ----------
    path
        A path to a local video file.
    mime_type
        The video's MIME type. If not provided, it's guessed from `path`'s
        extension.

    Returns
    -------
    [](`~chatlas.types.Content`)
        Content suitable for a [](`~chatlas.Turn`) object.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If `mime_type` isn't provided and can't be guessed from the file
        extension.
    """

    if isinstance(path, str):
        path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    if mime_type is None:
        guessed = _VIDEO_EXTENSION_MIME_TYPES.get(path.suffix.lower())
        if guessed is None:
            raise ValueError(
                f"Couldn't guess a video MIME type from extension {path.suffix!r}. "
                f"Pass `mime_type` explicitly (one of {sorted(_VIDEO_EXTENSION_MIME_TYPES.values())})."
            )
        mime_type = guessed

    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return ContentVideoInline(
        video_content_type=mime_type,
        data=data,
        filename=path.name,
    )


def content_video_youtube(url: str) -> ContentVideoUrl:
    """
    Reference a public YouTube video for input to a chat.

    Only Gemini accepts this: the URL is passed straight through with no
    upload and no MIME type (Gemini fetches and determines the video format
    itself). As of this writing, this is a free preview feature limited to
    public (not private or unlisted) videos, up to 10 per request on Gemini
    2.5+ models.

    Parameters
    ----------
    url
        A `youtube.com` or `youtu.be` video URL.

    Returns
    -------
    [](`~chatlas.types.Content`)
        Content suitable for a [](`~chatlas.Turn`) object.

    Raises
    ------
    ValueError
        If `url` doesn't look like a YouTube video URL.
    """

    if not _is_youtube_url(url):
        raise ValueError(
            f"{url!r} doesn't look like a YouTube video URL "
            "(expected a youtube.com or youtu.be URL)."
        )

    return ContentVideoUrl(url=url)


_VIDEO_EXTENSION_MIME_TYPES: dict[str, VideoContentTypes] = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpg",
    ".mov": "video/mov",
    ".avi": "video/avi",
    ".flv": "video/x-flv",
    ".webm": "video/webm",
    ".wmv": "video/wmv",
    ".3gpp": "video/3gpp",
    ".3gp": "video/3gpp",
}


def _is_youtube_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname or ""
    return host in ("youtu.be", "youtube.com", "www.youtube.com", "m.youtube.com")
