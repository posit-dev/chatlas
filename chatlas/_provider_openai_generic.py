from __future__ import annotations

import json
import os
import re
import tempfile
from abc import abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generic, Literal, Optional

import orjson
from openai import AsyncOpenAI, OpenAI
from openai.types.batch import Batch
from pydantic import BaseModel

from ._content import Content, ContentImage, ContentImageRemote
from ._provider import (
    BatchStatus,
    ChatCompletionChunkT,
    ChatCompletionDictT,
    ChatCompletionT,
    ModelInfo,
    Provider,
    SubmitInputArgsT,
)
from ._tokens import get_token_pricing
from ._tools import Tool
from ._turn import Turn, user_turn
from ._utils import split_http_client_kwargs

if TYPE_CHECKING:
    from .types.openai import ChatClientArgs


# Seems there is no native typing support for `files.content()` results
# so mock them based on the docs here
# https://platform.openai.com/docs/guides/batch#5-retrieve-the-results
class BatchResult(BaseModel):
    id: str
    custom_id: str
    response: BatchResultResponse


class BatchResultResponse(BaseModel):
    status_code: int
    request_id: str
    body: dict[str, Any]


class OpenAIAbstractProvider(
    Provider[
        ChatCompletionT,
        ChatCompletionChunkT,
        ChatCompletionDictT,
        SubmitInputArgsT,
    ],
    Generic[
        ChatCompletionT,
        ChatCompletionChunkT,
        ChatCompletionDictT,
        SubmitInputArgsT,
    ],
):
    """
    Abstract OpenAI provider with logic shared across both the /chat/completions
    and /responses APIs.

    Note that this is an abstract class and should not be used directly. It's
    intended to be subclassed by specific OpenAI provider implementations.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        seed: Optional[int] = None,
        name: str = "OpenAI",
        kwargs: Optional["ChatClientArgs"] = None,
    ):
        super().__init__(name=name, model=model)

        self._seed = seed

        kwargs_full: "ChatClientArgs" = {
            "api_key": api_key,
            "base_url": base_url,
            **(kwargs or {}),
        }

        # Avoid passing the wrong sync/async client to the OpenAI constructor.
        sync_kwargs, async_kwargs = split_http_client_kwargs(kwargs_full)

        # TODO: worth bringing in AsyncOpenAI types?
        self._client = OpenAI(**sync_kwargs)  # type: ignore
        self._async_client = AsyncOpenAI(**async_kwargs)

    def list_models(self):
        models = self._client.models.list()

        res: list[ModelInfo] = []
        for m in models:
            pricing = get_token_pricing(self.name, m.id) or {}
            info: ModelInfo = {
                "id": m.id,
                "owned_by": m.owned_by,
                "input": pricing.get("input"),
                "output": pricing.get("output"),
                "cached_input": pricing.get("cached_input"),
            }
            # DeepSeek compatibility
            if m.created is not None:
                info["created_at"] = datetime.fromtimestamp(m.created).date()
            res.append(info)

        # More recent models first
        res.sort(
            key=lambda x: x.get("created_at", 0),
            reverse=True,
        )

        return res

    def token_count(
        self,
        *args: Content | str,
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]],
    ) -> int:
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "The tiktoken package is required for token counting. "
                "Please install it with `pip install tiktoken`."
            )

        encoding = tiktoken.encoding_for_model(self._model)

        turn = user_turn(*args)

        # Count the tokens in image contents
        image_tokens = sum(
            self._image_token_count(x)
            for x in turn.contents
            if isinstance(x, ContentImage)
        )

        # For other contents, get the token count from the actual message param
        other_contents = [x for x in turn.contents if not isinstance(x, ContentImage)]
        other_full = self._turns_as_inputs([Turn("user", other_contents)])
        other_tokens = len(encoding.encode(str(other_full)))

        return other_tokens + image_tokens

    async def token_count_async(
        self,
        *args: Content | str,
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]],
    ) -> int:
        return self.token_count(*args, tools=tools, data_model=data_model)

    @staticmethod
    def _image_token_count(image: ContentImage) -> int:
        if isinstance(image, ContentImageRemote) and image.detail == "low":
            return 85
        else:
            # This is just the max token count for an image The highest possible
            # resolution is 768 x 2048, and 8 tiles of size 512px can fit inside
            # TODO: this is obviously a very conservative estimate and could be improved
            # https://platform.openai.com/docs/guides/vision/calculating-costs
            return 170 * 8 + 85

    def has_batch_support(self) -> bool:
        return True

    def batch_submit(
        self,
        conversations: list[list[Turn]],
        data_model: Optional[type[BaseModel]] = None,
    ):
        # First put the requests in a file
        # https://platform.openai.com/docs/api-reference/batch/request-input
        # https://platform.openai.com/docs/api-reference/batch
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = f.name

            for i, turns in enumerate(conversations):
                kwargs = self._chat_perform_args(
                    stream=False,
                    turns=turns,
                    tools={},
                    data_model=data_model,
                )

                body = {
                    "messages": kwargs.get("messages", []),
                    "model": self.model,
                }

                if "response_format" in kwargs:
                    body["response_format"] = kwargs["response_format"]

                request = {
                    "custom_id": f"request-{i}",
                    "method": "POST",
                    "url": self._batch_endpoint(),
                    "body": body,
                }

                f.write(orjson.dumps(request).decode() + "\n")

        try:
            with open(temp_path, "rb") as f:
                file_response = self._client.files.create(file=f, purpose="batch")

            batch = self._client.batches.create(
                input_file_id=file_response.id,
                endpoint=self._batch_endpoint(),
                completion_window="24h",
            )

            return batch.model_dump()
        finally:
            os.unlink(temp_path)

    def batch_poll(self, batch):
        batch = Batch.model_validate(batch)
        b = self._client.batches.retrieve(batch.id)
        return b.model_dump()

    def batch_status(self, batch):
        batch = Batch.model_validate(batch)
        counts = batch.request_counts
        total, completed, failed = 0, 0, 0
        if counts is not None:
            total = counts.total
            completed = counts.completed
            failed = counts.failed

        return BatchStatus(
            working=batch.status not in ["completed", "failed", "cancelled"],
            n_processing=total - completed - failed,
            n_succeeded=completed,
            n_failed=failed,
        )

    def batch_retrieve(self, batch):
        batch = Batch.model_validate(batch)
        if batch.output_file_id is None:
            raise ValueError("Batch has no output file")

        # Download and parse JSONL results
        response = self._client.files.content(batch.output_file_id)
        results: list[dict[str, Any]] = []
        for line in response.text.splitlines():
            results.append(json.loads(line))

        # Sort by custom_id to maintain order
        def extract_id(x: str):
            match = re.search(r"-(\d+)$", x)
            return int(match.group(1)) if match else 0

        results.sort(key=lambda x: int(extract_id(x.get("custom_id", ""))))

        return results

    @staticmethod
    @abstractmethod
    def _batch_endpoint() -> Literal["/v1/responses", "/v1/chat/completions"]: ...

    @abstractmethod
    def _chat_perform_args(
        self,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]],
    ) -> SubmitInputArgsT: ...

    @staticmethod
    @abstractmethod
    def _turns_as_inputs(turns: list[Turn]) -> list[Any]: ...

    @staticmethod
    @abstractmethod
    def _response_as_turn(
        completion: ChatCompletionT, has_data_model: bool
    ) -> Turn: ...
