from __future__ import annotations

from typing import Any, Generic, Literal, Optional, Sequence, TypeVar

from ._content import Content, ContentText

__all__ = ("Turn",)

CompletionT = TypeVar("CompletionT")


class Turn(Generic[CompletionT]):
    """
    A user or assistant turn

    Every conversation with a chatbot consists of pairs of user and assistant
    turns, corresponding to an HTTP request and response. These turns are
    represented by the `Turn` object, which contains a list of
    [](`~chatlas.types.Content`)s representing the individual messages within the
    turn. These might be text, images, tool requests (assistant only), or tool
    responses (user only).

    Note that a call to `.chat()` and related functions may result in multiple
    user-assistant turn cycles. For example, if you have registered tools, chatlas
    will automatically handle the tool calling loop, which may result in any
    number of additional cycles.

    Examples
    --------

    ```python
    from chatlas import Turn, ChatOpenAI, ChatAnthropic

    chat = ChatOpenAI()
    str(chat.chat("What is the capital of France?"))
    turns = chat.get_turns()
    assert len(turns) == 2
    assert isinstance(turns[0], Turn)
    assert turns[0].role == "user"
    assert turns[1].role == "assistant"

    # Load context into a new chat instance
    chat2 = ChatAnthropic(turns=turns)
    turns2 = chat2.get_turns()
    assert turns == turns2
    ```

    Parameters
    ----------
    role
        Either "user", "assistant", or "system".
    contents
        A list of [](`~chatlas.types.Content`) objects.
    tokens
        A numeric vector of length 2 representing the number of input and output
        tokens (respectively) used in this turn. Currently only recorded for
        assistant turns.
    finish_reason
        A string indicating the reason why the conversation ended. This is only
        relevant for assistant turns.
    completion
        The completion object returned by the provider. This is useful if there's
        information returned by the provider that chatlas doesn't otherwise expose.
        This is only relevant for assistant turns.
    """

    def __init__(
        self,
        role: Literal["user", "assistant", "system"],
        contents: str | Sequence[Content | str],
        *,
        tokens: Optional[tuple[int, int]] = None,
        finish_reason: Optional[str] = None,
        completion: Optional[CompletionT] = None,
    ):
        self.role = role

        if isinstance(contents, str):
            contents = [ContentText(contents)]

        contents2: list[Content] = []
        for x in contents:
            if isinstance(x, Content):
                contents2.append(x)
            elif isinstance(x, str):
                contents2.append(ContentText(x))
            else:
                raise ValueError("All contents must be Content objects or str.")

        self.contents = contents2
        self.text = "".join(x.text for x in self.contents if isinstance(x, ContentText))
        self.tokens = tokens
        self.finish_reason = finish_reason
        self.completion = completion

    def __str__(self) -> str:
        return self.text

    def __repr__(self, indent: int = 0) -> str:
        res = " " * indent + f"<Turn role='{self.role}'"
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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Turn):
            return False
        res = (
            self.role == other.role
            and self.contents == other.contents
            and self.tokens == other.tokens
            and self.finish_reason == other.finish_reason
            and self.completion == other.completion
        )
        return res


def user_turn(*args: Content | str) -> Turn:
    if len(args) == 0:
        raise ValueError("Must supply at least one input.")

    return Turn("user", args)


def normalize_turns(turns: list[Turn], system_prompt: str | None = None) -> list[Turn]:
    if system_prompt is not None:
        system_turn = Turn("system", system_prompt)

        if not turns:
            turns = [system_turn]
        elif turns[0].role != "system":
            turns = [system_turn] + turns
        elif turns[0] == system_turn:
            pass  # Duplicate system prompt; don't need to do anything
        else:
            raise ValueError(
                "system_prompt and turns[0] can't contain conflicting system prompts."
            )

    return turns
