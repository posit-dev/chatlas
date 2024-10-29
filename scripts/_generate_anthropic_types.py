from pathlib import Path

import httpx
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock
from anthropic.resources import AsyncMessages

from _utils import generate_typeddict_code, write_code_to_file

src_dir = Path(__file__).parent.parent / "chatlas"

anthropic_src = generate_typeddict_code(
    AsyncMessages.create,
    "CreateCompletionArgs",
    excluded_fields={
        "self",
        # TODO: for some reason the generated is off for metadata
        "metadata",
    },
)

write_code_to_file(
    anthropic_src,
    src_dir / "types" / "_anthropic_create.py",
)

init_args = generate_typeddict_code(
    AsyncAnthropic.__init__,
    "ProviderClientArgs",
    excluded_fields={"self"},
    localns={"URL": httpx.URL},
)

write_code_to_file(
    init_args,
    src_dir / "types" / "_anthropic_client.py",
)


init_args = generate_typeddict_code(
    AsyncAnthropicBedrock.__init__,
    "ProviderClientArgs",
    excluded_fields={"self"},
    localns={"URL": httpx.URL},
)

write_code_to_file(
    init_args,
    src_dir / "types" / "_anthropic_client_bedrock.py",
    setup_code="import anthropic",
)
