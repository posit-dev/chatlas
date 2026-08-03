from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ._chat import Chat
from ._utils import MISSING, MISSING_TYPE

if TYPE_CHECKING:
    from ._provider_openai_completions import ChatCompletion
    from .types.openai import ChatClientArgs, SubmitInputArgs


def ChatGithub(
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: str = "https://models.github.ai/inference/",
    seed: Optional[int] | MISSING_TYPE = MISSING,
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat["SubmitInputArgs", ChatCompletion]:
    """
    Deprecated: chat with a model hosted on the GitHub model marketplace.

    `ChatGithub()` is defunct because GitHub Models was retired on
    2026-07-30. Use [](`~chatlas.ChatGoogle`) (offers a free tier) or
    [](`~chatlas.ChatPosit`) (offers a free trial) instead.

    Parameters
    ----------
    system_prompt
        Unused.
    model
        Unused.
    api_key
        Unused.
    base_url
        Unused.
    seed
        Unused.
    kwargs
        Unused.

    Raises
    ------
    RuntimeError
        Always, since GitHub Models is no longer available.
    """
    raise RuntimeError(
        "ChatGithub() is defunct because GitHub Models was retired on 2026-07-30. "
        "Use ChatGoogle() (offers a free tier) or ChatPosit() (offers a free trial) instead."
    )
