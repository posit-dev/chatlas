from pathlib import Path

from google.generativeai import GenerativeModel

from _utils import generate_typeddict_code, write_code_to_file

types_dir = Path(__file__).parent.parent / "chatlas" / "types"
provider_dir = types_dir / "google"

for file in provider_dir.glob("*.py"):
    file.unlink()

google_src = generate_typeddict_code(
    GenerativeModel.generate_content,
    "SubmitInputArgs",
    excluded_fields={
        "self",
        # TODO: the generated code for this field is incorrect
        "safety_settings",
    },
)

write_code_to_file(
    google_src,
    provider_dir / "_submit.py",
)

init_args = generate_typeddict_code(
    GenerativeModel.__init__,
    "ChatClientArgs",
    excluded_fields={
        "self",
        # TODO: the generated code for this field is incorrect
        "safety_settings",
    },
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
