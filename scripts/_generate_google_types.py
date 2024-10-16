from pathlib import Path

from google.generativeai import GenerativeModel

from _utils import generate_typeddict_code, write_code_to_file

src_dir = Path(__file__).parent.parent / "chatlas"

google_src = generate_typeddict_code(
    GenerativeModel.generate_content,
    "SendMessageArgs",
    excluded_fields={
        "self",
        # TODO: the generated code for this field is incorrect
        "safety_settings",
    },
)

write_code_to_file(
    google_src,
    src_dir / "types" / "_google_create.py",
)

init_args = generate_typeddict_code(
    GenerativeModel.__init__,
    "ProviderClientArgs",
    excluded_fields={
        "self",
        # TODO: the generated code for this field is incorrect
        "safety_settings",
    },
)

write_code_to_file(
    init_args,
    src_dir / "types" / "_google_client.py",
)
