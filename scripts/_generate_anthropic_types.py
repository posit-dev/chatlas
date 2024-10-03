from pathlib import Path

from anthropic.resources import AsyncMessages

from _utils import generate_typeddict_code, write_code_to_file

src_dir = Path(__file__).parent.parent / "chatlas"

anthropic_src = generate_typeddict_code(
    AsyncMessages.create,
    "CreateCompletion",
    excluded_fields={"self", "messages", "stream"},
)

write_code_to_file(
    anthropic_src,
    src_dir / "_anthropic_types.py",
)
