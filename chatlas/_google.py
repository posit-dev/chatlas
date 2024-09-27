import sys
from typing import TYPE_CHECKING

from ._abc import LLMClient

if TYPE_CHECKING:
    if sys.version_info >= (3, 9):
        import google.generativeai.types as gtypes  # pyright: ignore[reportMissingTypeStubs]

        ContentDict = gtypes.ContentDict
    else:
        ContentDict = object


class Google(LLMClient["ContentDict"]):
    pass
