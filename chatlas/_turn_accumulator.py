from __future__ import annotations

from typing import Callable, Literal, Sequence

from ._content import (
    Content,
    ContentText,
    ContentThinking,
    ContentThinkingDelta,
    ContentUnion,
)
from ._stream_controller import StreamController
from ._turn import AssistantTurn, Turn, UserTurn


def merge_content_text(contents: list[ContentUnion]) -> list[ContentUnion]:
    """Merge adjacent ContentText (and ContentThinking) fragments."""
    if not contents:
        return []
    merged: list[ContentUnion] = [contents[0]]
    for item in contents[1:]:
        last = merged[-1]
        if isinstance(last, ContentText) and isinstance(item, ContentText):
            merged[-1] = ContentText.model_construct(text=last.text + item.text)
        elif isinstance(last, ContentThinking) and isinstance(item, ContentThinking):
            merged[-1] = ContentThinking.model_construct(
                thinking=last.thinking + item.thinking,
                extra=item.extra or last.extra,
            )
        else:
            merged.append(item)
    return merged


class TurnAccumulator:
    """
    Manages the lifecycle of one streaming assistant turn.

    Mirrors ellmer's TurnAccumulator R6 class. The stages are:

    1. ``begin_turn(user_turn)`` — insert user + partial assistant into turns
    2. ``process_content(content, ...)`` — append content, handle thinking
       phase boundaries, emit display text, and return items to yield
    3. ``flush_thinking(...)`` — emit closing thinking tags after the loop
    4. ``complete_turn(turn)`` — replace partial with the full turn (skipped if cancelled)
    5. ``finalize_turn()`` — called from ``finally``; merges text fragments
       and stamps the cancellation reason if the turn is still partial
    """

    def __init__(
        self,
        turns: list[Turn],
        controller: StreamController,
    ):
        self._turns = turns
        self._controller = controller
        self._turn_idx: int | None = None
        self._inside_thinking: bool = False

    def begin_turn(self, user_turn: UserTurn) -> None:
        """Insert user turn and a partial assistant placeholder."""
        partial: AssistantTurn = AssistantTurn([], partial_reason="interrupted")
        self._turns.extend([user_turn, partial])
        self._turn_idx = len(self._turns) - 1

    def process_content(
        self,
        content: Content,
        content_mode: Literal["text", "all"],
        emit: Callable[[str | Content], None],
    ) -> Sequence[str | Content]:
        """Append content to the turn, emit display text, return items to yield."""
        self._update_turn(content)

        items: list[str | Content] = []

        if isinstance(content, ContentThinkingDelta) and not self._inside_thinking:
            content = ContentThinkingDelta(
                thinking=content.thinking, phase="start"
            )
            emit("<thinking>\n")
            self._inside_thinking = True
        elif not isinstance(content, ContentThinkingDelta) and self._inside_thinking:
            emit("\n</thinking>\n\n")
            if content_mode == "all":
                items.append(ContentThinkingDelta(thinking="", phase="end"))
            self._inside_thinking = False

        if isinstance(content, ContentThinkingDelta):
            emit(content.thinking)
            if content_mode == "all":
                items.append(content)
        else:
            text = _content_text(content)
            if text:
                emit(text)
                items.append(text)

        return items

    def flush_thinking(
        self,
        content_mode: Literal["text", "all"],
        emit: Callable[[str | Content], None],
    ) -> Sequence[str | Content]:
        """Emit closing thinking tags if the stream ended mid-thinking."""
        if not self._inside_thinking:
            return []
        self._inside_thinking = False
        emit("\n</thinking>\n\n")
        if content_mode == "all":
            return [ContentThinkingDelta(thinking="", phase="end")]
        return []

    def complete_turn(self, turn: AssistantTurn) -> None:
        """Replace the partial turn with the completed turn (no-op if cancelled)."""
        if self._turn_idx is None:
            raise RuntimeError("complete_turn called before begin_turn")
        if self._controller.cancelled:
            return
        self._turns[self._turn_idx] = turn

    def finalize_turn(self) -> None:
        """
        Safety net — called from ``finally``.

        If the turn is still partial (i.e., ``complete_turn`` was never called
        or was skipped because of cancellation), merge adjacent text fragments
        and stamp the cancellation reason.
        """
        if self._turn_idx is None:
            return
        turn = self._turns[self._turn_idx]
        if not isinstance(turn, AssistantTurn) or not turn.is_partial:
            return
        if self._controller.cancelled:
            turn.partial_reason = self._controller.reason
        turn.contents = merge_content_text(turn.contents)

    def _update_turn(self, content: Content) -> None:
        """Append streamed content to the partial turn."""
        if self._turn_idx is None:
            raise RuntimeError("_update_turn called before begin_turn")
        # Content is the base class; contents is typed as list[ContentUnion]
        # (discriminated union). At runtime all Content subclasses are ContentUnion
        # members, so the append is safe.
        self._turns[self._turn_idx].contents.append(content)  # type: ignore[arg-type]


def _content_text(content: Content) -> str:
    """Extract displayable text from a Content object."""
    if isinstance(content, ContentThinkingDelta):
        return content.thinking
    if isinstance(content, ContentThinking):
        return content.thinking
    if isinstance(content, ContentText):
        return content.text
    return str(content)
