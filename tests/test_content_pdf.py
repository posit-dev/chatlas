from pathlib import Path

from chatlas import content_pdf_file
from chatlas._content import ContentPDF


def test_can_create_pdf_from_local_file():
    apples = Path(__file__).parent / "apples.pdf"
    obj = content_pdf_file(apples)
    assert isinstance(obj, ContentPDF)
