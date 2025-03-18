from pathlib import Path

from _utils import generate_typeddict_code, write_code_to_file
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.resources.chat import Completions

types_dir = Path(__file__).parent.parent / "chatlas" / "types"
provider_dir = types_dir / "openai"

for file in provider_dir.glob("*.py"):
    file.unlink()

create_args = generate_typeddict_code(
    Completions.create,
    "SubmitInputArgs",
    # For some reason web_search_options is not being generated correctly
    excluded_fields={"self", "web_search_options"},
)

write_code_to_file(
    create_args,
    provider_dir / "_submit.py",
)

init_args = generate_typeddict_code(
    AsyncOpenAI.__init__,
    "ChatClientArgs",
    excluded_fields={"self"},
)

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

write_code_to_file(
    init_args,
    provider_dir / "_client_azure.py",
    setup_code="import openai",
)

init = """
from ._client import ChatClientArgs
from ._client_azure import ChatAzureClientArgs
from ._submit import SubmitInputArgs

__all__ = (
    "ChatClientArgs",
    "ChatAzureClientArgs",
    "SubmitInputArgs",
)
"""

write_code_to_file(
    init,
    provider_dir / "__init__.py",
)
