from __future__ import annotations

from typing import cast

from ._content import Content, ContentText, ContentThinking, ContentUnion
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
            merged[-1] = ContentThinking(thinking=last.thinking + item.thinking)
        else:
            merged.append(item)
    return merged


class TurnAccumulator:
    """
    Manages the lifecycle of one streaming assistant turn.

    Mirrors ellmer's TurnAccumulator R6 class. The four stages are:

    1. ``begin_turn(user_turn)`` — insert user + partial assistant into turns
    2. ``update_turn(content)`` — append streamed content to the partial turn
    3. ``complete_turn(turn)`` — replace partial with the full turn (skipped if cancelled)
    4. ``finalize_turn()`` — called from ``finally``; merges text fragments
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

    def begin_turn(self, user_turn: UserTurn) -> None:
        """Insert user turn and a partial assistant placeholder."""
        partial: AssistantTurn = AssistantTurn([], partial_reason="interrupted")
        self._turns.extend([user_turn, partial])
        self._turn_idx = len(self._turns) - 1

    def update_turn(self, content: Content) -> None:
        """Append streamed content to the partial turn."""
        assert self._turn_idx is not None
        self._turns[self._turn_idx].contents.append(cast(ContentUnion, content))

    def complete_turn(self, turn: AssistantTurn) -> None:
        """Replace the partial turn with the completed turn (no-op if cancelled)."""
        assert self._turn_idx is not None
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
