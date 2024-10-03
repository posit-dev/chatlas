from pathlib import Path

from _utils import generate_typeddict_code, write_code_to_file
from openai.resources.chat import Completions

src_dir = Path(__file__).parent.parent / "chatlas"

openai_src = generate_typeddict_code(
    Completions.create,
    "CreateCompletion",
    excluded_fields={"self", "messages", "stream"},
)

write_code_to_file(
    openai_src,
    src_dir / "_openai_types.py",
)
