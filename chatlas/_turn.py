from __future__ import annotations

from typing import Any, Optional, Sequence

from ._content import Content, ContentText

__all__ = ("Turn",)


class Turn:
    def __init__(
        self,
        role: str,
        contents: str | Sequence[Content | str],
        json_data: Optional[dict[str, Any]] = None,
        tokens: tuple[int, int] = (0, 0),
    ):
        self.role = role
        if isinstance(contents, str):
            contents = [ContentText(contents)]
        contents = [ContentText(x) if isinstance(x, str) else x for x in contents]
        if any(not isinstance(x, Content) for x in contents):
            raise ValueError("All contents must be Content objects or str.")
        self.contents = contents
        self.json_data = json_data or {}
        self.tokens = tokens
        self.text = "".join(x.text for x in self.contents if isinstance(x, ContentText))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Turn):
            return False
        res = (
            self.role == other.role
            and self.contents == other.contents
            and self.json_data == other.json_data
            and self.tokens == other.tokens
        )
        return res


def user_turn(*args: Content | str) -> Turn:
    if len(args) == 0:
        raise ValueError("Must supply at least one input.")

    return Turn("user", args)


def is_system_prompt(turn: Turn) -> bool:
    return turn.role == "system"


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
