from __future__ import annotations

from abc import abstractmethod
from typing import Generic, Literal, Optional, Sequence, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ._content import Content, ContentText, ContentUnion, create_content

__all__ = ("Turn", "UserTurn", "SystemTurn", "AssistantTurn")

CompletionT = TypeVar("CompletionT")


class Turn(BaseModel):
    """
    Base turn class

    Every conversation with a chatbot consists of pairs of user and assistant
    turns, corresponding to an HTTP request and response. These turns are
    represented by `Turn` objects (or their subclasses `UserTurn`, `SystemTurn`,
    `AssistantTurn`), which contain a list of [](`~chatlas.types.Content`)s
    representing the individual messages within the turn. These might be text,
    images, tool requests (assistant only), or tool responses (user only).

    Note that a call to `.chat()` and related functions may result in multiple
    user-assistant turn cycles. For example, if you have registered tools, chatlas
    will automatically handle the tool calling loop, which may result in any
    number of additional cycles.

    Examples
    --------

    ```python
    from chatlas import UserTurn, AssistantTurn, ChatOpenAI, ChatAnthropic

    chat = ChatOpenAI()
    str(chat.chat("What is the capital of France?"))
    turns = chat.get_turns()
    assert len(turns) == 2
    assert isinstance(turns[0], UserTurn)
    assert turns[0].role == "user"
    assert isinstance(turns[1], AssistantTurn)
    assert turns[1].role == "assistant"

    # Load context into a new chat instance
    chat2 = ChatAnthropic()
    chat2.set_turns(turns)
    turns2 = chat2.get_turns()
    assert turns == turns2
    ```

    Parameters
    ----------
    contents
        A list of [](`~chatlas.types.Content`) objects.
    """

    contents: list[ContentUnion] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        contents: str | Sequence[Content | str],
        **kwargs,
    ):
        if isinstance(contents, str):
            contents = [ContentText(text=contents)]

        contents2: list[Content] = []
        for x in contents:
            if isinstance(x, Content):
                contents2.append(x)
            elif isinstance(x, str):
                contents2.append(ContentText(text=x))
            elif isinstance(x, dict):
                contents2.append(create_content(x))
            else:
                raise ValueError("All contents must be Content objects or str.")

        super().__init__(
            contents=contents2,
            **kwargs,
        )

    @property
    @abstractmethod
    def role(self) -> str:
        """The role of the turn (e.g., 'user', 'assistant', or 'system')."""
        pass

    @property
    def text(self) -> str:
        return "".join(x.text for x in self.contents if isinstance(x, ContentText))

    def __str__(self) -> str:
        return self.text

    def __repr__(self, indent: int = 0) -> str:
        res = " " * indent + f"<{self.__class__.__name__}>"
        for content in self.contents:
            res += "\n" + content.__repr__(indent=indent + 2)
        return res + "\n"

    def to_inspect_messages(self, model: Optional[str] = None):
        """
        Transform this turn into a list of Inspect AI `ChatMessage` objects.

        Most users will not need to call this method directly. See the
        `.export_eval()` method on `Chat` for a higher level interface to
        exporting chat history for evaluation purposes.
        """

        from ._inspect import try_import_inspect, turn_as_inspect_messages

        try_import_inspect()
        return turn_as_inspect_messages(self, model=model)


class UserTurn(Turn):
    """
    User turn - represents user input

    Parameters
    ----------
    contents
        A list of [](`~chatlas.types.Content`) objects, or strings.
    """

    def __init__(
        self,
        contents: str | Sequence[Content | str],
        **kwargs,
    ):
        super().__init__(contents, **kwargs)

    @property
    def role(self) -> Literal["user"]:
        return "user"


class SystemTurn(Turn):
    """
    System turn - represents system prompt

    Parameters
    ----------
    contents
        A list of [](`~chatlas.types.Content`) objects, or strings.
    """

    def __init__(
        self,
        contents: str | Sequence[Content | str],
        **kwargs,
    ):
        super().__init__(contents, **kwargs)

    @property
    def role(self) -> Literal["system"]:
        return "system"


class AssistantTurn(Turn, Generic[CompletionT]):
    """
    Assistant turn - represents model response with additional metadata

    Parameters
    ----------
    contents
        A list of [](`~chatlas.types.Content`) objects.
    tokens
        A numeric vector of length 3 representing the number of input, output, and cached
        tokens (respectively) used in this turn.
    finish_reason
        A string indicating the reason why the conversation ended.
    completion
        The completion object returned by the provider. This is useful if there's
        information returned by the provider that chatlas doesn't otherwise expose.
    """

    tokens: Optional[tuple[int, int, int]] = None
    finish_reason: Optional[str] = None
    completion: Optional[CompletionT] = Field(default=None, exclude=True)

    @field_validator("tokens", mode="before")
    @classmethod
    def validate_tokens(cls, v):
        """Convert list to tuple for JSON deserialization compatibility."""
        if isinstance(v, list):
            return tuple(v)
        return v

    def __init__(
        self,
        contents: str | Sequence[Content | str],
        *,
        tokens: Optional[tuple[int, int, int] | list[int]] = None,
        finish_reason: Optional[str] = None,
        completion: Optional[CompletionT] = None,
        **kwargs,
    ):
        if isinstance(tokens, list):
            tokens = cast(tuple[int, int, int], tuple(tokens))

        # Pass assistant-specific fields to parent constructor
        if tokens is not None:
            kwargs["tokens"] = tokens
        if finish_reason is not None:
            kwargs["finish_reason"] = finish_reason
        if completion is not None:
            kwargs["completion"] = completion

        super().__init__(contents, **kwargs)

    @property
    def role(self) -> Literal["assistant"]:
        return "assistant"

    def __repr__(self, indent: int = 0) -> str:
        res = " " * indent + f"<{self.__class__.__name__} role='{self.role}'"
        if self.tokens:
            res += f" tokens={self.tokens}"
        if self.finish_reason:
            res += f" finish_reason='{self.finish_reason}'"
        if self.completion:
            res += f" completion={self.completion}"
        res += ">"
        for content in self.contents:
            res += "\n" + content.__repr__(indent=indent + 2)
        return res + "\n"


def user_turn(*args: Content | str) -> UserTurn:
    if len(args) == 0:
        raise ValueError("Must supply at least one input.")

    return UserTurn(args)
