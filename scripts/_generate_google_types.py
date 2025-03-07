from pathlib import Path

from _utils import generate_typeddict_code, write_code_to_file
from google.genai import Client
from google.genai.models import Models

types_dir = Path(__file__).parent.parent / "chatlas" / "types"
provider_dir = types_dir / "google"

for file in provider_dir.glob("*.py"):
    file.unlink()

google_src = generate_typeddict_code(
    Models.generate_content,
    "SubmitInputArgs",
    excluded_fields={"self"},
)

write_code_to_file(
    google_src,
    provider_dir / "_submit.py",
)

init_args = generate_typeddict_code(
    Client.__init__,
    "ChatClientArgs",
    excluded_fields={"self"},
)

write_code_to_file(
    init_args,
    provider_dir / "_client.py",
)


init = """
from ._client import ChatClientArgs
from ._submit import SubmitInputArgs

__all__ = (
    "ChatClientArgs",
    "SubmitInputArgs",
)
"""

write_code_to_file(
    init,
    provider_dir / "__init__.py",
)
