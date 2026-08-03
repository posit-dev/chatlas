import ssl
from pathlib import Path

import httpx

from _utils import generate_typeddict_code, write_code_to_file

types_dir = Path(__file__).parent.parent / "chatlas" / "types"
provider_dir = types_dir / "bedrock"

for file in provider_dir.glob("*.py"):
    file.unlink()

# `ChatBedrock(api="converse")` speaks over raw httpx (there's no vendor SDK
# client to introspect), so its client args are derived from
# `httpx.AsyncClient.__init__` instead. `auth` and `base_url` are excluded
# because `BedrockConverseProvider` sets both itself.
client_args = generate_typeddict_code(
    httpx.AsyncClient.__init__,
    "ChatClientArgs",
    excluded_fields={"self", "auth", "base_url"},
    localns={"ssl": ssl},
)


# `BedrockConverseProvider` builds a sync `httpx.Client` and an async
# `httpx.AsyncClient` from the same kwargs (see `split_httpx_client_kwargs`),
# so `mounts`/`transport` need to accept either transport type, not just the
# async one `httpx.AsyncClient.__init__` declares.
def widen_transport_types(text: str) -> str:
    text = text.replace(
        "mounts: Optional[Mapping[str, httpx.AsyncBaseTransport | None]]",
        "mounts: Optional[Mapping[str, httpx.BaseTransport | httpx.AsyncBaseTransport | None]]",
    )
    return text.replace(
        "transport: httpx.AsyncBaseTransport | None",
        "transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None",
    )


client_args = widen_transport_types(client_args)

# `api_key` isn't an `httpx.Client`/`httpx.AsyncClient` parameter -- it's
# chatlas's own hook for passing an AWS Bedrock bearer token straight through
# to `BedrockBearerAuth`, bypassing the botocore credential chain.
client_args = client_args.replace(
    "class ChatClientArgs(TypedDict, total=False):\n",
    "class ChatClientArgs(TypedDict, total=False):\n    api_key: str | None\n",
)

write_code_to_file(
    client_args,
    provider_dir / "_client.py",
)

init = """
from ._client import ChatClientArgs

__all__ = ("ChatClientArgs",)
"""

write_code_to_file(
    init,
    provider_dir / "__init__.py",
)
