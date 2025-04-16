import os
from typing import TYPE_CHECKING, Optional

from ._chat import Chat
from ._logging import log_model_default
from ._openai import OpenAIProvider
from ._turn import Turn, normalize_turns
from ._utils import MISSING, MISSING_TYPE, is_testing

if TYPE_CHECKING:
    from ._openai import ChatCompletion
    from .types.openai import ChatClientArgs, SubmitInputArgs


def ChatDeepSeek(
    *,
    system_prompt: Optional[str] = None,
    turns: Optional[list[Turn]] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    seed: Optional[int] | MISSING_TYPE = MISSING,
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat["SubmitInputArgs", ChatCompletion]:
    """
    Chat with a model hosted on DeepSeek

    Prerequisites
    -------------

    ::: {.callout-note}
    ## API key

    Sign up at <https://platform.deepseek.com> to get an API key.
    :::

    ::: {.callout-note}
    ## Python requirements

    `ChatDeepSeek` requires the `openai` package (e.g., `pip install openai`).
    :::

    Known limitations
    -----------------

    - Structured data extraction is not supported..
    - Function calling is currently unstable.
    - Images are not supported.

    Parameters
    ----------


    """

    if api_key is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")

    if model is None:
        model = log_model_default("deepseek-chat")

    if isinstance(seed, MISSING_TYPE):
        seed = 1014 if is_testing() else None

    return Chat(
        provider=DeepSeekProvider(
            base_url="https://api.deepseek.com",
            api_key=api_key,
            model=model,
            seed=seed,
            kwargs=kwargs,
        ),
        turns=normalize_turns(
            turns or [],
            system_prompt,
        ),
    )


class DeepSeekProvider(OpenAIProvider):
    pass


# method(as_json, list(ProviderDeepSeek, ContentText)) <- function(provider, x) {
#   x@text
# }
#
# method(as_json, list(ProviderDeepSeek, Turn)) <- function(provider, x) {
#   if (x@role == "user") {
#     # Text and tool results go in separate messages
#     texts <- keep(x@contents, S7_inherits, ContentText)
#     texts_out <- lapply(texts, function(text) {
#       list(role = "user", content = as_json(provider, text))
#     })
#
#     tools <- keep(x@contents, S7_inherits, ContentToolResult)
#     tools_out <- lapply(tools, function(tool) {
#       list(role = "tool", content = tool_string(tool), tool_call_id = tool@id)
#     })
#
#     c(texts_out, tools_out)
#   } else if (x@role == "assistant") {
#     # Tool requests come out of content and go into own argument
#     text <- detect(x@contents, S7_inherits, ContentText)
#     tools <- keep(x@contents, S7_inherits, ContentToolRequest)
#
#     list(compact(list(
#       role = "assistant",
#       content = as_json(provider, text),
#       tool_calls = as_json(provider, tools)
#     )))
#   } else {
#     as_json(super(provider, ProviderOpenAI), x)
#   }
# }
