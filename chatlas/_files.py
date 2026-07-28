from __future__ import annotations

import os
from datetime import datetime
from typing import IO, TYPE_CHECKING, Any, Optional

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


class FileManager:
    """Manage files hosted by a chat's provider. Accessed via `chat.files`."""

    def __init__(self, provider: "Provider"):
        self._provider = provider

    def upload(
        self,
        file: str | os.PathLike[str] | IO[bytes],
        *,
        mime_type: Optional[str] = None,
    ) -> ContentUploaded:
        return self._provider.file_upload(file, mime_type=mime_type)

    async def upload_async(
        self,
        file: str | os.PathLike[str] | IO[bytes],
        *,
        mime_type: Optional[str] = None,
    ) -> ContentUploaded:
        return await self._provider.file_upload_async(file, mime_type=mime_type)

    def list(self) -> list[FileMetadata]:
        return self._provider.file_list()

    async def list_async(self) -> list[FileMetadata]:
        return await self._provider.file_list_async()

    def get(self, id: str) -> FileMetadata:  # noqa: A002
        return self._provider.file_get(id)

    async def get_async(self, id: str) -> FileMetadata:  # noqa: A002
        return await self._provider.file_get_async(id)

    def download(
        self,
        id: str,  # noqa: A002
        path: str | os.PathLike[str] | None = None,
    ) -> bytes:
        return self._provider.file_download(id, path)

    async def download_async(
        self,
        id: str,  # noqa: A002
        path: str | os.PathLike[str] | None = None,
    ) -> bytes:
        return await self._provider.file_download_async(id, path)

    def delete(self, id: str) -> None:  # noqa: A002
        return self._provider.file_delete(id)

    async def delete_async(self, id: str) -> None:  # noqa: A002
        return await self._provider.file_delete_async(id)
