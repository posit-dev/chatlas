from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
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
from ._turn import Turn
from ._typing_extensions import TypedDict

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
    ) -> Turn: ...

    @abstractmethod
    def value_turn(
        self,
        completion: ChatCompletionT,
        has_data_model: bool,
    ) -> Turn: ...

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
