from pathlib import Path

from _utils import generate_typeddict_code, write_code_to_file
from snowflake.cortex import complete

types_dir = Path(__file__).parent.parent / "chatlas" / "types"
provider_dir = types_dir / "snowflake"

for file in provider_dir.glob("*.py"):
    file.unlink()

create_args = generate_typeddict_code(
    complete,
    "SubmitInputArgs",
)

write_code_to_file(
    create_args,
    provider_dir / "_submit.py",
)

init = """
from ._submit import SubmitInputArgs

__all__ = (
    "SubmitInputArgs",
)
"""

write_code_to_file(
    init,
    provider_dir / "__init__.py",
)
