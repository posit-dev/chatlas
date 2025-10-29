from pathlib import Path

from _utils import generate_typeddict_code, write_code_to_file
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.resources.chat import Completions
from openai.resources.responses import Responses

types_dir = Path(__file__).parent.parent / "chatlas" / "types"
provider_dir = types_dir / "openai"

for file in provider_dir.glob("*.py"):
    file.unlink()

create_args = generate_typeddict_code(
    Completions.create,
    "SubmitInputArgs",
    excluded_fields={"self"},
)

responses_create_args = generate_typeddict_code(
    Responses.create,
    "ResponsesSubmitInputArgs",
    excluded_fields={"self"},
)

write_code_to_file(
    create_args + "\n\n" + responses_create_args,
    provider_dir / "_submit.py",
)

init_args = generate_typeddict_code(
    AsyncOpenAI.__init__,
    "ChatClientArgs",
    excluded_fields={"self"},
)


# Temporary workaround for an issue where a type like
#   Callable[[], Awaitable[str]]
# is getting incorrectly transpiled as
#   Callable[Awaitable[str]]
def fix_callable_types(text: str):
    return text.replace("Callable[Awaitable[str]]", "Callable[[], Awaitable[str]]")


init_args = fix_callable_types(init_args)

write_code_to_file(
    init_args,
    provider_dir / "_client.py",
)

init_args = generate_typeddict_code(
    AsyncAzureOpenAI.__init__,
    "ChatAzureClientArgs",
    excluded_fields={
        "self",
        # TODO: for some reason the generated is off for this field
        "azure_ad_token_provider",
    },
)

init_args = fix_callable_types(init_args)

write_code_to_file(
    init_args,
    provider_dir / "_client_azure.py",
    setup_code="import openai",
)

init = """
from ._client import ChatClientArgs
from ._client_azure import ChatAzureClientArgs
from ._submit import SubmitInputArgs, ResponsesSubmitInputArgs

__all__ = (
    "ChatClientArgs",
    "ChatAzureClientArgs",
    "SubmitInputArgs",
    "ResponsesSubmitInputArgs",
)
"""

write_code_to_file(
    init,
    provider_dir / "__init__.py",
)
