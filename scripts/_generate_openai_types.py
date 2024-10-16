from pathlib import Path

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.resources.chat import Completions

from _utils import generate_typeddict_code, write_code_to_file

src_dir = Path(__file__).parent.parent / "chatlas"

create_args = generate_typeddict_code(
    Completions.create,
    "ChatCompletionArgs",
    excluded_fields={"self"},
)

write_code_to_file(
    create_args,
    src_dir / "types" / "_openai_create.py",
)

init_args = generate_typeddict_code(
    AsyncOpenAI.__init__,
    "ProviderClientArgs",
    excluded_fields={"self"},
)

write_code_to_file(
    init_args,
    src_dir / "types" / "_openai_client.py",
)


init_args = generate_typeddict_code(
    AsyncAzureOpenAI.__init__,
    "ProviderClientArgs",
    excluded_fields={
        "self",
        # TODO: for some reason the generated is off for this field
        "azure_ad_token_provider",
    },
)

write_code_to_file(
    init_args,
    src_dir / "types" / "_openai_client_azure.py",
    setup_code="import openai",
)
