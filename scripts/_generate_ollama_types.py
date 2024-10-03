from pathlib import Path

from _utils import generate_typeddict_code, write_code_to_file
from ollama import Client

src_dir = Path(__file__).parent.parent / "chatlas"

openai_src = generate_typeddict_code(
    Client.chat,
    "ChatCompletion",
    excluded_fields={"self", "messages", "stream"},
)

write_code_to_file(
    openai_src,
    src_dir / "_ollama_types.py",
)
