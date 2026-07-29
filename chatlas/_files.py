from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import IO, TYPE_CHECKING, Any, Iterator, Optional

from pydantic import BaseModel, ConfigDict

from ._content import ContentUploaded

if TYPE_CHECKING:
    from ._provider import Provider


class FileMetadata(BaseModel):
    """Normalized metadata for a provider-hosted file."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    provider: str

    extra: Any = None
    """
    The provider's own file object, unmodified, for fields this class doesn't
    normalize (e.g. Gemini's processing `state`). Note this differs from
    `ContentUploaded.extra`, which is a plain dict because that type gets
    serialized as part of a chat's turns.
    """


class FileManager:
    """Manage files hosted by a chat's provider. Accessed via `chat.files`.

    Upload a file once with `.upload()`, then pass the returned
    `ContentUploaded` to `.chat()` (and other chat methods) like any other
    content, instead of re-sending the file's bytes on every turn.

    Supported for `ChatOpenAI()` (the Responses API), `ChatAnthropic()`, and
    `ChatGoogle()` (the Gemini Developer API). Every other provider --
    including `ChatOpenAICompletions()`, OpenAI-compatible third parties
    (`ChatGroq()`, `ChatMistral()`, `ChatOllama()`, etc.), `ChatAzureOpenAI()`,
    `ChatBedrockAnthropic()`, `ChatPosit()`, and Vertex-backed chats
    (`ChatVertex()`, or `ChatGoogle()` configured for Vertex) -- raises
    `NotImplementedError` from every method here.

    Notes
    -----
    * Anthropic's Files API is still in beta. chatlas automatically adds the
      required `anthropic-beta: files-api-2025-04-14` header whenever a turn
      references an uploaded file.
    * Gemini Developer API files expire automatically after 48 hours, and the
      Files API isn't available on Vertex AI at all. To reference a file
      already in Cloud Storage on a Vertex-backed chat, construct
      `ContentUploaded` directly with a `gs://` URI instead of calling
      `.upload()`.
    * `ChatOpenAICompletions()` can reference an uploaded PDF or text
      document, but not an uploaded image -- the Chat Completions API doesn't
      support it. Use `ChatOpenAI()` (Responses API) for uploaded images, or
      pass the image inline with `content_image_file()`.
    """

    def __init__(self, provider: "Provider"):
        self._provider = provider

    def upload(
        self,
        file: str | os.PathLike[str] | IO[bytes],
        *,
        mime_type: Optional[str] = None,
    ) -> ContentUploaded:
        """Upload a file to the chat's provider, once, for reuse across turns.

        Only supported for ChatOpenAI, ChatAnthropic, and ChatGoogle; other
        providers raise `NotImplementedError`.

        On `ChatGoogle()`, this blocks until Gemini finishes processing the
        file: large media (video, audio, multi-GB uploads) is processed
        asynchronously, and the API refuses to reference a file that isn't
        yet `ACTIVE`. That means this call can take a while for a large
        video, and raises if the file fails to process -- but the reference
        you get back is always ready to use.

        Parameters
        ----------
        file
            A path to a file, or a binary file-like object, to upload.
        mime_type
            The file's MIME type. If not provided, it's guessed from `file`.

        Returns
        -------
        ContentUploaded
            A reference to the uploaded file that can be passed to `.chat()`
            (and other chat methods) in place of the file's raw bytes.
        """
        return self._provider.file_upload(file, mime_type=mime_type)

    async def upload_async(
        self,
        file: str | os.PathLike[str] | IO[bytes],
        *,
        mime_type: Optional[str] = None,
    ) -> ContentUploaded:
        """Async version of `.upload()`."""
        return await self._provider.file_upload_async(file, mime_type=mime_type)

    def list(self) -> list[FileMetadata]:
        """List files previously uploaded to the chat's provider.

        Only supported for ChatOpenAI, ChatAnthropic, and ChatGoogle; other
        providers raise `NotImplementedError`.

        Returns
        -------
        list[FileMetadata]
            Metadata for each file hosted by the provider.
        """
        return self._provider.file_list()

    async def list_async(self) -> list[FileMetadata]:
        """Async version of `.list()`."""
        return await self._provider.file_list_async()

    def get(self, id: str) -> FileMetadata:  # noqa: A002
        """Get metadata for a previously uploaded file.

        Only supported for ChatOpenAI, ChatAnthropic, and ChatGoogle; other
        providers raise `NotImplementedError`.

        Parameters
        ----------
        id
            The provider-assigned file id (e.g., `ContentUploaded.id`).

        Returns
        -------
        FileMetadata
            Metadata for the file.
        """
        return self._provider.file_get(id)

    async def get_async(self, id: str) -> FileMetadata:  # noqa: A002
        """Async version of `.get()`."""
        return await self._provider.file_get_async(id)

    def download(
        self,
        id: str,  # noqa: A002
        path: str | os.PathLike[str] | None = None,
    ) -> bytes:
        """Download a file's raw bytes, optionally writing them to `path`.

        Whether a file is downloadable depends on how it was created, so this
        raises a provider error for anything `.upload()` produced:

        - OpenAI refuses `purpose="user_data"` (what `.upload()` uses), but
          allows `purpose="batch"`/`"batch_output"` — i.e. the input and result
          files behind `batch_chat()`.
        - Anthropic marks uploaded files as not downloadable.
        - Google only serves bytes back for model-generated files (e.g. Veo
          video output).
        """
        return self._provider.file_download(id, path)

    async def download_async(
        self,
        id: str,  # noqa: A002
        path: str | os.PathLike[str] | None = None,
    ) -> bytes:
        """Async version of `.download()`."""
        return await self._provider.file_download_async(id, path)

    def delete(self, id: str) -> None:  # noqa: A002
        """Delete a previously uploaded file from the provider.

        Parameters
        ----------
        id
            The provider-assigned file id (e.g., `ContentUploaded.id`).
        """
        return self._provider.file_delete(id)

    async def delete_async(self, id: str) -> None:  # noqa: A002
        """Async version of `.delete()`."""
        return await self._provider.file_delete_async(id)


@contextmanager
def open_binary(file: str | os.PathLike[str] | IO[bytes]) -> Iterator[IO[bytes]]:
    """Open `file` for reading if it's a path, otherwise yield it unchanged.

    File-like objects are left open on exit since the caller may not own them.
    """
    if isinstance(file, (str, os.PathLike)):
        f = open(file, "rb")
        try:
            yield f
        finally:
            f.close()
    else:
        yield file


def maybe_write(data: bytes, path: str | os.PathLike[str] | None) -> bytes:
    """Write `data` to `path` when given, always returning `data`."""
    if path is not None:
        with open(path, "wb") as f:
            f.write(data)
    return data
