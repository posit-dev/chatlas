from pathlib import Path

import httpx
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock
from anthropic.resources import AsyncMessages

from _utils import generate_typeddict_code, write_code_to_file

types_dir = Path(__file__).parent.parent / "chatlas" / "types"
provider_dir = types_dir / "anthropic"

for file in provider_dir.glob("*.py"):
    file.unlink()

anthropic_src = generate_typeddict_code(
    AsyncMessages.create,
    "SubmitInputArgs",
    excluded_fields={
        "self",
        # TODO: for some reason the generated is off for metadata
        "metadata",
    },
)

write_code_to_file(
    anthropic_src,
    provider_dir / "_submit.py",
)

init_args = generate_typeddict_code(
    AsyncAnthropic.__init__,
    "ChatClientArgs",
    excluded_fields={"self"},
    localns={"URL": httpx.URL},
)

write_code_to_file(
    init_args,
    provider_dir / "_client.py",
)


init_args = generate_typeddict_code(
    AsyncAnthropicBedrock.__init__,
    "ChatBedrockClientArgs",
    excluded_fields={"self"},
    localns={"URL": httpx.URL},
)

write_code_to_file(
    init_args,
    provider_dir / "_client_bedrock.py",
    setup_code="import anthropic",
)


init = """
from ._client import ChatClientArgs
from ._client_bedrock import ChatBedrockClientArgs
from ._submit import SubmitInputArgs

__all__ = (
    "ChatClientArgs",
    "ChatBedrockClientArgs",
    "SubmitInputArgs",
)
"""

write_code_to_file(
    init,
    provider_dir / "__init__.py",
)
