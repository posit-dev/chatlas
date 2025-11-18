from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import (
    Any,
    AsyncIterable,
    Generic,
    Iterable,
    Literal,
    Optional,
    TypeVar,
    overload,
)

from pydantic import BaseModel

from ._content import Content
from ._tools import Tool
from ._turn import AssistantTurn, Turn
from ._typing_extensions import NotRequired, TypedDict

ChatCompletionT = TypeVar("ChatCompletionT")
ChatCompletionChunkT = TypeVar("ChatCompletionChunkT")
# A dictionary representation of a chat completion
ChatCompletionDictT = TypeVar("ChatCompletionDictT")


class AnyTypeDict(TypedDict, total=False):
    pass


SubmitInputArgsT = TypeVar("SubmitInputArgsT", bound=AnyTypeDict)
"""
A TypedDict representing the provider specific arguments that can specified when
submitting input to a model provider.
"""


class ModelInfo(TypedDict):
    "Information returned from the `.list_models()` method"

    id: str
    "The model ID (this gets passed to the `model` parameter of the `Chat` constructor)"

    cached_input: NotRequired[float | None]
    "The cost per user token in USD per million tokens for cached input"

    input: NotRequired[float | None]
    "The cost per user token in USD per million tokens"

    output: NotRequired[float | None]
    "The cost per assistant token in USD per million tokens"

    created_at: NotRequired[date]
    "The date the model was created"

    name: NotRequired[str]
    "The model name"

    owned_by: NotRequired[str]
    "The owner of the model"

    size: NotRequired[int]
    "The size of the model in bytes"

    provider: NotRequired[str]
    "The provider of the model"

    url: NotRequired[str]
    "A URL to learn more about the model"


class StandardModelParams(TypedDict, total=False):
    """
    A TypedDict representing the standard model parameters that can be set
    when using a [](`~chatlas.Chat`) instance.
    """

    temperature: float
    top_p: float
    top_k: int
    frequency_penalty: float
    presence_penalty: float
    seed: int
    max_tokens: int
    log_probs: bool
    stop_sequences: list[str]


StandardModelParamNames = Literal[
    "temperature",
    "top_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "max_tokens",
    "log_probs",
    "stop_sequences",
]


# Provider-agnostic batch status info
class BatchStatus(BaseModel):
    """Status information for a batch job."""

    working: bool
    n_processing: int
    n_succeeded: int
    n_failed: int


class Provider(
    ABC,
    Generic[
        ChatCompletionT, ChatCompletionChunkT, ChatCompletionDictT, SubmitInputArgsT
    ],
):
    """
    A model provider interface for a [](`~chatlas.Chat`).

    This abstract class defines the interface a model provider must implement in
    order to be used with a [](`~chatlas.Chat`) instance. The provider is
    responsible for performing the actual chat completion, and for handling the
    streaming of the completion results.

    Note that this class is exposed for developers who wish to implement their
    own provider. In general, you should not need to interact with this class
    directly.
    """

    def __init__(self, *, name: str, model: str):
        self._name = name
        self._model = model

    @property
    def name(self):
        """
        Get the name of the provider
        """
        return self._name

    @property
    def model(self):
        """
        Get the model used by the provider
        """
        return self._model

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        """
        List all available models for the provider.
        """
        pass

    @overload
    @abstractmethod
    def chat_perform(
        self,
        *,
        stream: Literal[False],
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]],
        kwargs: SubmitInputArgsT,
    ) -> ChatCompletionT: ...

    @overload
    @abstractmethod
    def chat_perform(
        self,
        *,
        stream: Literal[True],
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]],
        kwargs: SubmitInputArgsT,
    ) -> Iterable[ChatCompletionChunkT]: ...

    @abstractmethod
    def chat_perform(
        self,
        *,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]],
        kwargs: SubmitInputArgsT,
    ) -> Iterable[ChatCompletionChunkT] | ChatCompletionT: ...

    @overload
    @abstractmethod
    async def chat_perform_async(
        self,
        *,
        stream: Literal[False],
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]],
        kwargs: SubmitInputArgsT,
    ) -> ChatCompletionT: ...

    @overload
    @abstractmethod
    async def chat_perform_async(
        self,
        *,
        stream: Literal[True],
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]],
        kwargs: SubmitInputArgsT,
    ) -> AsyncIterable[ChatCompletionChunkT]: ...

    @abstractmethod
    async def chat_perform_async(
        self,
        *,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]],
        kwargs: SubmitInputArgsT,
    ) -> AsyncIterable[ChatCompletionChunkT] | ChatCompletionT: ...

    @abstractmethod
    def stream_text(self, chunk: ChatCompletionChunkT) -> Optional[str]: ...

    @abstractmethod
    def stream_merge_chunks(
        self,
        completion: Optional[ChatCompletionDictT],
        chunk: ChatCompletionChunkT,
    ) -> ChatCompletionDictT: ...

    @abstractmethod
    def stream_turn(
        self,
        completion: ChatCompletionDictT,
        has_data_model: bool,
    ) -> AssistantTurn[ChatCompletionT]: ...

    @abstractmethod
    def value_turn(
        self,
        completion: ChatCompletionT,
        has_data_model: bool,
    ) -> AssistantTurn[ChatCompletionT]: ...

    @abstractmethod
    def value_tokens(
        self,
        completion: ChatCompletionT,
    ) -> tuple[int, int, int] | None: ...

    @abstractmethod
    def token_count(
        self,
        *args: Content | str,
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]],
    ) -> int: ...

    @abstractmethod
    async def token_count_async(
        self,
        *args: Content | str,
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]],
    ) -> int: ...

    @abstractmethod
    def translate_model_params(
        self, params: StandardModelParams
    ) -> SubmitInputArgsT: ...

    @abstractmethod
    def supported_model_params(self) -> set[StandardModelParamNames]: ...

    def has_batch_support(self) -> bool:
        """
        Returns whether this provider supports batch processing.
        Override this method to return True for providers that implement batch methods.
        """
        return False

    def batch_submit(
        self,
        conversations: list[list[Turn]],
        data_model: Optional[type[BaseModel]] = None,
    ) -> dict[str, Any]:
        """
        Submit a batch of conversations for processing.

        Args:
            conversations: List of conversation histories (each is a list of Turns)
            data_model: Optional structured data model for responses

        Returns:
            BatchInfo containing batch job information
        """
        raise NotImplementedError("This provider does not support batch processing")

    def batch_poll(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Poll the status of a submitted batch.

        Args:
            batch: Batch information returned from batch_submit

        Returns:
            Updated batch information
        """
        raise NotImplementedError("This provider does not support batch processing")

    def batch_status(self, batch: dict[str, Any]) -> BatchStatus:
        """
        Get the status of a batch.

        Args:
            batch: Batch information

        Returns:
            BatchStatus with processing status information
        """
        raise NotImplementedError("This provider does not support batch processing")

    def batch_retrieve(self, batch: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Retrieve results from a completed batch.

        Args:
            batch: Batch information

        Returns:
            List of BatchResult objects, one for each request in the batch
        """
        raise NotImplementedError("This provider does not support batch processing")

    def batch_result_turn(
        self,
        result: dict[str, Any],
        has_data_model: bool = False,
    ) -> AssistantTurn[ChatCompletionT] | None:
        """
        Convert a batch result to a Turn.

        Args:
            result: Individual BatchResult from batch_retrieve
            has_data_model: Whether the request used a structured data model

        Returns:
            Turn object or None if the result was an error
        """
        raise NotImplementedError("This provider does not support batch processing")
