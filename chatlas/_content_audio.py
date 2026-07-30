from __future__ import annotations

from pathlib import Path
from typing import Literal

from ._content import AudioContentTypes, ContentAudio

__all__ = ("content_audio_file",)


_AUDIO_EXTENSION_MIME_TYPES: dict[str, AudioContentTypes] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mp3",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}


def content_audio_file(
    path: str | Path,
    content_type: Literal["auto", AudioContentTypes] = "auto",
) -> ContentAudio:
    """
    Encode audio content from a file for chat input.

    Not all providers support audio input, so check the documentation for the
    provider you are using. Of the providers chatlas supports, only
    [](`~chatlas.ChatGoogle`)/[](`~chatlas.ChatVertex`) (wav, mp3, aiff, aac,
    ogg, flac) and [](`~chatlas.ChatOpenAICompletions`) (wav, mp3 only) accept
    audio input.

    Parameters
    ----------
    path
        The path to the audio file to include in the chat input.
    content_type
        The content type of the audio (e.g., `"audio/wav"`). If `"auto"`, the
        content type is inferred from the file extension.

    Returns
    -------
    [](`~chatlas.types.Content`)
        Content suitable for a [](`~chatlas.Turn`) object.

    Examples
    --------
    ```python
    from chatlas import ChatGoogle, content_audio_file

    chat = ChatGoogle()
    chat.chat(
        "What's being said in this recording?",
        content_audio_file("path/to/clip.wav"),
    )
    ```

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file extension is unsupported.
    """

    if isinstance(path, str):
        path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if content_type == "auto":
        ext = path.suffix.lower()
        if ext not in _AUDIO_EXTENSION_MIME_TYPES:
            raise ValueError(f"Unsupported audio file extension: {ext}")
        content_type = _AUDIO_EXTENSION_MIME_TYPES[ext]

    with open(path, "rb") as f:
        data = f.read()

    return ContentAudio(data=data, mime_type=content_type)
